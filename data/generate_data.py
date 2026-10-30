"""Regenerate the bundled parameter files and cross-language golden values.

Run:  python3 generate_data.py   (from the data/ directory or anywhere)

Steps:
  1. Validate the Almgren-Chriss closed forms against seeded Monte Carlo
     (assert |MC - closed form| <= 4 SE) before pinning anything.
  2. Write params_liquid.json / params_illiquid.json / params_fx.json and
     parent_order.json.
  3. Write golden/golden.json (19 cases).  Deterministic cases carry a
     relative tolerance (|a - e| <= tol * max(1, |e|)); simulation cases are
     statistical: expect holds the 20k-path Python reference plus a *_se key
     per statistic, tol = 4.0 is the SE-band multiplier (ports use their own
     RNG and compare within 4 * sqrt(se_ref^2 + se_own^2)).

Everything is seeded and deterministic; flat scalar keys only in golden.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA_DIR.parent / "python" / "src"))

import numpy as np

import etrade as et

# ---------------------------------------------------------------- parameters

PARAMS = {
    "liquid": {
        # Almgren-Chriss (2000) paper example: large-cap, $/share/day units.
        "ac_total_shares": 1.0e6,
        "ac_n_intervals": 5,
        "ac_tau": 1.0,
        "ac_sigma": 0.95,
        "ac_eta": 2.5e-6,
        "ac_gamma_p": 2.5e-7,
        "ac_lambda_ra": 2.0e-6,
        # Avellaneda-Stoikov (2008) paper parameters (Table 1 row gamma=0.1);
        # fills use the exact thinned probability 1-exp(-lambda*dt), not the
        # paper's first-order lambda*dt, so mean PnL sits ~12% below Table 1.
        "as_s0": 100.0,
        "as_sigma": 2.0,
        "as_horizon": 1.0,
        "as_dt": 0.005,
        "as_gamma": 0.1,
        "as_k": 1.5,
        "as_a_rate": 140.0,
        "as_q_max": 20,
        "as_eta_liq": 0.5,
        # POV baseline inputs.
        "pov_market_volume_rate": 2.5e6,
        "pov_participation": 0.15,
    },
    "illiquid": {
        "ac_total_shares": 5.0e5,
        "ac_n_intervals": 10,
        "ac_tau": 0.5,
        "ac_sigma": 1.5,
        "ac_eta": 1.0e-5,
        "ac_gamma_p": 5.0e-7,
        "ac_lambda_ra": 5.0e-6,
        "as_s0": 50.0,
        "as_sigma": 3.0,
        "as_horizon": 1.0,
        "as_dt": 0.005,
        "as_gamma": 0.5,
        "as_k": 0.8,
        "as_a_rate": 20.0,
        "as_q_max": 10,
        "as_eta_liq": 1.0,
        "pov_market_volume_rate": 4.0e5,
        "pov_participation": 0.10,
    },
    "fx": {
        # FX-like: tight spread, high k, price ~1.10.
        "ac_total_shares": 5.0e7,
        "ac_n_intervals": 10,
        "ac_tau": 0.1,
        "ac_sigma": 0.006,
        "ac_eta": 2.0e-9,
        "ac_gamma_p": 5.0e-11,
        "ac_lambda_ra": 1.0e-4,
        "as_s0": 1.10,
        "as_sigma": 0.004,
        "as_horizon": 1.0,
        "as_dt": 0.005,
        "as_gamma": 100.0,
        "as_k": 2000.0,
        "as_a_rate": 500.0,
        "as_q_max": 50,
        "as_eta_liq": 2.0e-4,
        "pov_market_volume_rate": 2.0e8,
        "pov_participation": 0.20,
    },
}

PARENT_ORDER = {
    "total_shares": 1.0e6,
    "n_intervals": 5,
    "tau": 1.0,
    "side": "sell",
    "lambda_grid": [0.0, 1e-8, 1e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5],
    # Pinned deterministic IS-decomposition example (also golden).
    "decision_price": 100.0,
    "arrival_price": 99.5,
    "close_price": 98.8,
    "is_total_qty": 1000.0,
    "is_fill_qtys": [300.0, 400.0, 200.0],
    "is_fill_prices": [99.4, 99.2, 99.1],
}

AS_SIM_SEED_LIQUID = 42
AS_SIM_SEED_FX = 7
AC_MC_SEED = 123
N_PATHS_REF = 20_000


def ac_params(cfg: dict, lambda_ra: float | None = None) -> et.ACParams:
    return et.ACParams(
        cfg["ac_total_shares"],
        cfg["ac_n_intervals"],
        cfg["ac_tau"],
        cfg["ac_sigma"],
        cfg["ac_eta"],
        cfg["ac_gamma_p"],
        cfg["ac_lambda_ra"] if lambda_ra is None else lambda_ra,
    )


def as_params(cfg: dict) -> et.ASParams:
    return et.ASParams(
        cfg["as_s0"],
        cfg["as_sigma"],
        cfg["as_horizon"],
        cfg["as_dt"],
        cfg["as_gamma"],
        cfg["as_k"],
        cfg["as_a_rate"],
        cfg["as_q_max"],
        cfg["as_eta_liq"],
    )


def validate_ac_vs_mc() -> None:
    """Assert AC closed-form E/Var match Monte Carlo within 4 SE."""
    for name in ("liquid", "illiquid"):
        p = ac_params(PARAMS[name])
        e, v = et.expected_cost(p), et.variance_cost(p)
        mc = et.simulate_is(p, n_paths=N_PATHS_REF, seed=AC_MC_SEED)
        assert abs(mc.mean - e) <= 4.0 * mc.se_mean, (name, mc.mean, e, mc.se_mean)
        assert abs(mc.var - v) <= 4.0 * mc.se_var, (name, mc.var, v, mc.se_var)
        print(
            f"[MC check {name}] E: closed={e:.6g} mc={mc.mean:.6g} "
            f"(se={mc.se_mean:.3g}) | Var: closed={v:.6g} mc={mc.var:.6g} "
            f"(se={mc.se_var:.3g})  OK"
        )


def build_cases() -> list[dict]:
    cases: list[dict] = []

    def exact(name: str, inputs: dict, expect: dict, tol: float) -> None:
        cases.append(
            {
                "name": name,
                "kind": "exact",
                "inputs": inputs,
                "expect": expect,
                "tol": tol,
            }
        )

    def statistical(name: str, inputs: dict, expect: dict) -> None:
        cases.append(
            {
                "name": name,
                "kind": "statistical",
                "inputs": inputs,
                "expect": expect,
                "tol": 4.0,
            }
        )

    def ac_inputs(cfg: dict, **over) -> dict:
        d = {
            k: cfg[k]
            for k in (
                "ac_total_shares",
                "ac_n_intervals",
                "ac_tau",
                "ac_sigma",
                "ac_eta",
                "ac_gamma_p",
                "ac_lambda_ra",
            )
        }
        d.update(over)
        return d

    def as_inputs(cfg: dict, **over) -> dict:
        d = {
            k: cfg[k]
            for k in (
                "as_s0",
                "as_sigma",
                "as_horizon",
                "as_dt",
                "as_gamma",
                "as_k",
                "as_a_rate",
                "as_q_max",
                "as_eta_liq",
            )
        }
        d.update(over)
        return d

    # ---- AC deterministic solutions (2 parameter sets) ----
    for pname, xj_index in (("liquid", 2), ("illiquid", 4)):
        cfg = PARAMS[pname]
        p = ac_params(cfg)
        x = et.ac_trajectory(p)
        n = et.ac_trade_list(p)
        exact(
            f"ac_{pname}_solution",
            ac_inputs(cfg, x_index=float(xj_index)),
            {
                "kappa": et.ac_kappa(p),
                "half_life": et.half_life(p),
                "x_j": float(x[xj_index]),
                "n_1": float(n[0]),
                "e_cost": et.expected_cost(p),
                "var_cost": et.variance_cost(p),
            },
            1e-10,
        )

    cfg = PARAMS["liquid"]
    p_liq = ac_params(cfg)

    # ---- TWAP baseline ----
    twap = et.twap_holdings(cfg["ac_total_shares"], cfg["ac_n_intervals"])
    exact(
        "ac_twap_liquid",
        ac_inputs(cfg),
        {
            "e_cost": et.expected_cost(p_liq, twap),
            "var_cost": et.variance_cost(p_liq, twap),
        },
        1e-10,
    )

    # ---- POV baseline ----
    pov = et.pov_holdings(
        cfg["ac_total_shares"],
        cfg["ac_tau"],
        cfg["pov_market_volume_rate"],
        cfg["pov_participation"],
    )
    exact(
        "ac_pov_liquid",
        ac_inputs(
            cfg,
            pov_market_volume_rate=cfg["pov_market_volume_rate"],
            pov_participation=cfg["pov_participation"],
        ),
        {
            "e_cost": et.expected_cost(p_liq, pov),
            "var_cost": et.variance_cost(p_liq, pov),
            "n_intervals_used": float(pov.size - 1),
        },
        1e-10,
    )

    # ---- frontier point at pinned lambda ----
    lam_pin = 5e-7
    fp = et.efficient_frontier(p_liq, [lam_pin])[0]
    exact(
        "ac_frontier_point_liquid",
        ac_inputs(cfg, ac_lambda_ra=lam_pin),
        {"e_cost": fp.e_cost, "std_cost": fp.std_cost, "kappa": fp.kappa},
        1e-10,
    )

    # ---- kappa at small kappa_tilde*tau (log1p form; acosh(1+x) cancels) ----
    cfg_small = ac_inputs(cfg, ac_tau=1e-3, ac_n_intervals=5000, ac_lambda_ra=1e-8)
    p_small = ac_params(cfg_small)
    exact(
        "ac_kappa_small_tau",
        cfg_small,
        {
            "kappa": et.ac_kappa(p_small),
            "half_life": et.half_life(p_small),
            "x_1": float(et.ac_trajectory(p_small)[1]),
        },
        1e-10,
    )

    # ---- lambda_ra = 0 limit == TWAP (identity) ----
    p0 = p_liq.with_lambda(0.0)
    diff = float(np.max(np.abs(et.ac_trajectory(p0) - twap)))
    exact(
        "ac_lambda_zero_limit_liquid",
        ac_inputs(cfg, ac_lambda_ra=0.0),
        {"max_abs_diff_vs_twap": diff},
        1e-12,
    )

    # ---- instantaneous liquidation ----
    e_inst, v_inst = et.instantaneous_cost(p_liq)
    exact(
        "ac_instant_liquidation_liquid",
        ac_inputs(cfg),
        {"e_cost": e_inst, "var_cost": v_inst},
        1e-10,
    )

    # ---- AC Monte Carlo (statistical reference) ----
    mc = et.simulate_is(p_liq, n_paths=N_PATHS_REF, seed=AC_MC_SEED)
    statistical(
        "ac_mc_liquid",
        ac_inputs(cfg, seed=float(AC_MC_SEED), n_paths_ref=float(N_PATHS_REF)),
        {
            "mean_cost": mc.mean,
            "mean_cost_se": mc.se_mean,
            "var_cost": mc.var,
            "var_cost_se": mc.se_var,
        },
    )

    # ---- AS closed-form quantities (liquid) ----
    ap = as_params(cfg)
    s_pin, q_pin, t_pin, delta_pin = 100.0, 3.0, 0.3, 0.6
    exact(
        "as_reservation_liquid",
        as_inputs(cfg, s=s_pin, q=q_pin, t=t_pin),
        {"reservation_price": et.reservation_price(ap, s_pin, q_pin, t_pin)},
        1e-12,
    )
    exact(
        "as_spread_liquid",
        as_inputs(cfg, t=t_pin),
        {
            "spread": et.optimal_spread(ap, t_pin),
            "spread_terminal": et.optimal_spread(ap, ap.horizon),
        },
        1e-12,
    )
    exact(
        "as_fill_prob_liquid",
        as_inputs(cfg, delta=delta_pin),
        {
            "intensity": et.intensity(ap, delta_pin),
            "fill_probability": et.fill_probability(ap, delta_pin),
        },
        1e-12,
    )

    # ---- AS simulations (statistical) ----
    def sim_expect(res: et.ASSimResult, keys: list[str]) -> dict:
        vals = {
            "mean_pnl": res.mean_pnl,
            "std_pnl": res.std_pnl,
            "mean_abs_q": res.mean_abs_q,
            "mean_max_abs_q": res.mean_max_abs_q,
            "mean_fills": res.mean_fills,
        }
        out: dict = {}
        for kk in keys:
            out[kk] = vals[kk]
            out[kk + "_se"] = res.se(kk)
        return out

    res_liq = et.simulate(ap, N_PATHS_REF, AS_SIM_SEED_LIQUID, agent="as")
    statistical(
        "as_sim_liquid",
        as_inputs(cfg, seed=float(AS_SIM_SEED_LIQUID), n_paths_ref=float(N_PATHS_REF)),
        sim_expect(
            res_liq,
            ["mean_pnl", "std_pnl", "mean_abs_q", "mean_max_abs_q", "mean_fills"],
        ),
    )

    cfg_fx = PARAMS["fx"]
    ap_fx = as_params(cfg_fx)
    res_fx = et.simulate(ap_fx, N_PATHS_REF, AS_SIM_SEED_FX, agent="as")
    statistical(
        "as_sim_fx",
        as_inputs(cfg_fx, seed=float(AS_SIM_SEED_FX), n_paths_ref=float(N_PATHS_REF)),
        sim_expect(res_fx, ["mean_pnl", "std_pnl", "mean_abs_q"]),
    )

    res_fix = et.simulate(ap, N_PATHS_REF, AS_SIM_SEED_LIQUID, agent="fixed")
    statistical(
        "as_fixed_comparison_liquid",
        as_inputs(
            cfg,
            seed=float(AS_SIM_SEED_LIQUID),
            n_paths_ref=float(N_PATHS_REF),
            fixed_spread=et.average_as_spread(ap),
        ),
        {
            "as_rms_q": res_liq.rms_q,
            "as_rms_q_se": res_liq.se("rms_q"),
            "fixed_rms_q": res_fix.rms_q,
            "fixed_rms_q_se": res_fix.se("rms_q"),
        },
    )

    # ---- IS decomposition (pinned deterministic paths) ----
    po = PARENT_ORDER
    d_sell = et.decompose_shortfall(
        "sell",
        po["decision_price"],
        po["arrival_price"],
        po["close_price"],
        po["is_total_qty"],
        po["is_fill_qtys"],
        po["is_fill_prices"],
    )
    exact(
        "is_decomposition_sell",
        {
            "decision_price": po["decision_price"],
            "arrival_price": po["arrival_price"],
            "close_price": po["close_price"],
            "total_qty": po["is_total_qty"],
            "q1": 300.0,
            "p1": 99.4,
            "q2": 400.0,
            "p2": 99.2,
            "q3": 200.0,
            "p3": 99.1,
        },
        {
            "total": d_sell.total,
            "delay": d_sell.delay,
            "trading": d_sell.trading,
            "opportunity": d_sell.opportunity,
        },
        1e-10,
    )

    d_buy = et.decompose_shortfall(
        "buy", 100.0, 100.5, 101.2, 1000.0, [300.0, 400.0, 200.0], [100.6, 100.8, 100.9]
    )
    exact(
        "is_decomposition_buy",
        {
            "decision_price": 100.0,
            "arrival_price": 100.5,
            "close_price": 101.2,
            "total_qty": 1000.0,
            "q1": 300.0,
            "p1": 100.6,
            "q2": 400.0,
            "p2": 100.8,
            "q3": 200.0,
            "p3": 100.9,
        },
        {
            "total": d_buy.total,
            "delay": d_buy.delay,
            "trading": d_buy.trading,
            "opportunity": d_buy.opportunity,
        },
        1e-10,
    )

    # ---- participation rate ----
    exact(
        "participation_rate",
        {"executed_volume": 120000.0, "market_volume": 1.5e6},
        {"participation": et.participation_rate(120000.0, 1.5e6)},
        1e-12,
    )

    # ---- impact regression on pinned synthetic fills ----
    part, cost = et.synthetic_impact_fills(25, 0.5, 8.0, 0.05)
    reg = et.estimate_impact(part, cost)
    exact(
        "impact_regression",
        {"n_obs": 25.0, "true_intercept": 0.5, "true_slope": 8.0, "noise_amp": 0.05},
        {"slope": reg.slope, "intercept": reg.intercept, "r_squared": reg.r_squared},
        1e-10,
    )

    return cases


def main() -> None:
    validate_ac_vs_mc()
    for name, cfg in PARAMS.items():
        path = DATA_DIR / f"params_{name}.json"
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"wrote {path.name}")
    (DATA_DIR / "parent_order.json").write_text(json.dumps(PARENT_ORDER, indent=2) + "\n")
    print("wrote parent_order.json")
    cases = build_cases()
    golden = {"cases": cases}
    gpath = DATA_DIR / "golden" / "golden.json"
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text(json.dumps(golden, indent=2) + "\n")
    print(f"wrote golden/golden.json ({len(cases)} cases)")


if __name__ == "__main__":
    main()
