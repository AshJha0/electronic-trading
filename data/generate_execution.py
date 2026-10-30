"""Write the execution-extension golden cases (API_SPEC Section 12).

Run:  python data/generate_execution.py   ->  data/golden/execution_golden.json

Kinds
  transient_schedule  exact: xfrac_01..xfrac_{N-1} (x_j / X) of the static
                      optimum, e_cost (its E[IS]) and twap_e_cost
  internalise         exact: crossed_shares, residual_shares, baseline_cost,
                      internal_cost, saved_cost, saved_bps, buy{i}_saved_bps,
                      sell{i}_saved_bps; inputs carry n_buys/buy_{i} and
                      n_sells/sell_{i}; volume_curve = 1.0 selects
                      US_EQUITY_VOLUME_CURVE_13, 0.0 uniform weights
  adaptive_sim        statistical (tol = 4 SE): static_mean, adaptive_mean and,
                      when re-planning can help, saving_mean (+ _se); inputs
                      are the ac_* keys of golden.json plus phi, signal_sd,
                      fill_risk, seed and n_paths_ref
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA_DIR.parent / "python" / "src"))

import etrade as et
from etrade import execution as ex
from etrade import impact as im

TOL = 1e-9
N_PATHS_REF = 20_000

AC_BASE = {
    # Almgren-Chriss liquid example on a finer grid (10 half-day intervals).
    "ac_total_shares": 1.0e6,
    "ac_n_intervals": 10.0,
    "ac_tau": 0.5,
    "ac_sigma": 0.95,
    "ac_eta": 2.5e-6,
    "ac_gamma_p": 2.5e-7,
    "ac_lambda_ra": 2.0e-6,
}

MARKET = {
    "price": 100.0,
    "spread": 0.01,
    "adv": 5.0e6,
    "daily_vol": 0.015,
    "n_intervals": 13.0,
    "tau": 1.0 / 13.0,
}


def transient_case(name: str, inputs: dict) -> dict:
    p = ex.TransientParams(
        inputs["total_shares"],
        int(inputs["n_intervals"]),
        inputs["tau"],
        inputs["kappa"],
        inputs["rho"],
        inputs["gamma_p"],
    )
    x = ex.transient_optimal_schedule(p)
    expect = {f"xfrac_{j:02d}": float(x[j]) / p.total_shares for j in range(1, p.n_intervals)}
    expect["e_cost"] = ex.transient_expected_cost(p, x)
    expect["twap_e_cost"] = ex.transient_expected_cost(
        p, et.twap_holdings(p.total_shares, p.n_intervals)
    )
    return {
        "name": name,
        "kind": "transient_schedule",
        "inputs": inputs,
        "expect": expect,
        "tol": TOL,
    }


def internalise_case(name: str, market: dict, buys: list[float], sells: list[float]) -> dict:
    kw = dict(market)
    weights = im.US_EQUITY_VOLUME_CURVE_13 if kw.pop("volume_curve") == 1.0 else None
    kw["n_intervals"] = int(kw["n_intervals"])
    r = ex.internalise(buys, sells, weights=weights, **kw)
    expect = {
        "crossed_shares": r.crossed_shares,
        "residual_shares": r.residual_shares,
        "baseline_cost": r.baseline_cost,
        "internal_cost": r.internal_cost,
        "saved_cost": r.saved_cost,
        "saved_bps": r.saved_bps,
    }
    for i, v in enumerate(r.buy_saved_bps):
        expect[f"buy{i}_saved_bps"] = v
    for i, v in enumerate(r.sell_saved_bps):
        expect[f"sell{i}_saved_bps"] = v
    inputs = dict(market)
    inputs["n_buys"] = float(len(buys))
    for i, q in enumerate(buys):
        inputs[f"buy_{i}"] = q
    inputs["n_sells"] = float(len(sells))
    for i, q in enumerate(sells):
        inputs[f"sell_{i}"] = q
    return {"name": name, "kind": "internalise", "inputs": inputs, "expect": expect, "tol": TOL}


def adaptive_case(name: str, sig: dict, seed: int, with_saving: bool) -> dict:
    inputs = {**AC_BASE, **sig, "seed": float(seed), "n_paths_ref": float(N_PATHS_REF)}
    ac = et.ACParams(
        AC_BASE["ac_total_shares"],
        int(AC_BASE["ac_n_intervals"]),
        AC_BASE["ac_tau"],
        AC_BASE["ac_sigma"],
        AC_BASE["ac_eta"],
        AC_BASE["ac_gamma_p"],
        AC_BASE["ac_lambda_ra"],
    )
    r = ex.simulate_adaptive(
        ac, ex.SignalParams(sig["phi"], sig["signal_sd"], sig["fill_risk"]), N_PATHS_REF, seed
    )
    expect = {
        "static_mean": r.static_mean,
        "static_mean_se": r.static_se,
        "adaptive_mean": r.adaptive_mean,
        "adaptive_mean_se": r.adaptive_se,
    }
    if with_saving:
        expect["saving_mean"] = r.saving_mean
        expect["saving_mean_se"] = r.saving_se
    return {"name": name, "kind": "adaptive_sim", "inputs": inputs, "expect": expect, "tol": 4.0}


def build_cases() -> list[dict]:
    return [
        transient_case(
            "transient_ow_no_permanent",
            {
                "total_shares": 1.0e6,
                "n_intervals": 10.0,
                "tau": 0.1,
                "kappa": 2.0e-6,
                "rho": 5.0,
                "gamma_p": 0.0,
            },
        ),
        transient_case(
            "transient_with_permanent",
            {
                "total_shares": 1.0e6,
                "n_intervals": 10.0,
                "tau": 0.1,
                "kappa": 2.0e-6,
                "rho": 5.0,
                "gamma_p": 2.0e-7,
            },
        ),
        transient_case(
            "transient_slow_resilience",
            {
                "total_shares": 2.5e5,
                "n_intervals": 26.0,
                "tau": 1.0 / 26.0,
                "kappa": 4.0e-6,
                "rho": 2.0,
                "gamma_p": 1.0e-8,
            },
        ),
        internalise_case(
            "internalise_buy_heavy", {**MARKET, "volume_curve": 0.0}, [3.0e5, 1.0e5], [2.5e5]
        ),
        internalise_case(
            "internalise_balanced_curve",
            {**MARKET, "volume_curve": 1.0, "alpha": 0.6, "y": 0.8},
            [1.5e5],
            [5.0e4, 1.0e5],
        ),
        adaptive_case(
            "adaptive_no_signal", {"phi": 0.0, "signal_sd": 0.0, "fill_risk": 0.0}, 101, False
        ),
        adaptive_case(
            "adaptive_signal", {"phi": 0.7, "signal_sd": 0.5, "fill_risk": 0.0}, 102, True
        ),
        adaptive_case(
            "adaptive_fill_risk", {"phi": 0.0, "signal_sd": 0.0, "fill_risk": 0.5}, 103, True
        ),
    ]


def main() -> None:
    cases = build_cases()
    path = DATA_DIR / "golden" / "execution_golden.json"
    path.write_text(json.dumps({"cases": cases}, indent=2) + "\n")
    print(f"wrote {path.name} ({len(cases)} cases)")
    for c in cases:
        print(f"  {c['name']:30s} {c['kind']}")


if __name__ == "__main__":
    main()
