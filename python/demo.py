"""End-to-end demo of the etrade package.

Prints: Almgren-Chriss optimal trajectories across risk aversions, the
efficient frontier of execution, TWAP/POV/instantaneous baselines with an MC
cross-check, an Avellaneda-Stoikov market-making simulation (A-S vs
fixed-spread agent), and an implementation-shortfall decomposition with a
market-impact regression.

Run:  cd python && PYTHONPATH=src python3 demo.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


import etrade as et

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load(name: str) -> dict:
    return json.loads((DATA_DIR / name).read_text())


def hdr(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 66 - len(title)))


def ac_section(cfg: dict, order: dict) -> None:
    p = et.ACParams(
        cfg["ac_total_shares"],
        cfg["ac_n_intervals"],
        cfg["ac_tau"],
        cfg["ac_sigma"],
        cfg["ac_eta"],
        cfg["ac_gamma_p"],
        cfg["ac_lambda_ra"],
    )
    hdr("Almgren-Chriss optimal trajectories (sell 1.0e6 shares, N=5)")
    lambdas = [0.0, 5e-7, 2e-6, 1e-5]
    print(f"{'lambda_ra':>10} {'kappa':>9} {'half-life':>10} | holdings x_j (thousands)")
    for lam in lambdas:
        pl = p.with_lambda(lam)
        x = et.ac_trajectory(pl) / 1e3
        kap = et.ac_kappa(pl)
        hl = et.half_life(pl)
        hl_s = f"{hl:10.3f}" if math.isfinite(hl) else f"{'inf':>10}"
        print(f"{lam:10.1e} {kap:9.4f} {hl_s} | " + " ".join(f"{v:8.1f}" for v in x))

    hdr("Efficient frontier of execution (E[cost] vs sqrt(Var[cost]))")
    print(f"{'lambda_ra':>10} {'E[cost] $':>14} {'sqrt(Var) $':>14} {'kappa':>9}")
    for fp in et.efficient_frontier(p, order["lambda_grid"]):
        print(f"{fp.lambda_ra:10.1e} {fp.e_cost:14,.0f} {fp.std_cost:14,.0f} {fp.kappa:9.4f}")

    hdr("Baselines: optimal vs TWAP vs POV vs instantaneous")
    twap = et.twap_holdings(p.total_shares, p.n_intervals)
    pov = et.pov_holdings(
        p.total_shares, p.tau, cfg["pov_market_volume_rate"], cfg["pov_participation"]
    )
    e_i, v_i = et.instantaneous_cost(p)
    rows = [
        ("AC optimal", et.expected_cost(p), et.variance_cost(p)),
        ("TWAP", et.expected_cost(p, twap), et.variance_cost(p, twap)),
        (f"POV {cfg['pov_participation']:.0%}", et.expected_cost(p, pov), et.variance_cost(p, pov)),
        ("Instant", e_i, v_i),
    ]
    print(f"{'strategy':>12} {'E[cost] $':>14} {'sqrt(Var) $':>14}")
    for name, e, v in rows:
        print(f"{name:>12} {e:14,.0f} {math.sqrt(v):14,.0f}")
    mc = et.simulate_is(p, n_paths=20_000, seed=123)
    print(
        f"MC check (20k paths): E={mc.mean:,.0f} (closed {et.expected_cost(p):,.0f}, "
        f"se {mc.se_mean:,.0f}); Var/1e9={mc.var / 1e9:,.1f} "
        f"(closed {et.variance_cost(p) / 1e9:,.1f})"
    )


def as_section(cfg: dict, label: str, seed: int) -> None:
    ap = et.ASParams(
        cfg["as_s0"],
        cfg["as_sigma"],
        cfg["as_horizon"],
        cfg["as_dt"],
        cfg["as_gamma"],
        cfg["as_k"],
        cfg["as_a_rate"],
        int(cfg["as_q_max"]),
        cfg["as_eta_liq"],
    )
    hdr(f"Avellaneda-Stoikov market making ({label})")
    bid, ask = et.quotes(ap, ap.s0, 3.0, 0.0)
    print(
        f"params: s0={ap.s0} sigma={ap.sigma} gamma={ap.gamma} k={ap.k} "
        f"A={ap.a_rate} T={ap.horizon} dt={ap.dt} q_max={ap.q_max}"
    )
    print(
        f"example quote at t=0, q=+3: bid={bid:.4f} ask={ask:.4f} "
        f"(mid {ap.s0}, reservation {et.reservation_price(ap, ap.s0, 3.0, 0.0):.4f})"
    )
    n_paths = 2000
    res_as = et.simulate(ap, n_paths, seed, agent="as")
    res_fx = et.simulate(ap, n_paths, seed, agent="fixed")
    print(
        f"{'agent':>14} {'mean PnL':>10} {'std PnL':>9} {'Sharpe':>7} "
        f"{'mean|q|':>8} {'mean max|q|':>11} {'rms q':>7} {'fills':>7}"
    )
    for name, r in (("A-S", res_as), ("fixed-spread", res_fx)):
        print(
            f"{name:>14} {r.mean_pnl:10.4f} {r.std_pnl:9.4f} {r.sharpe:7.2f} "
            f"{r.mean_abs_q:8.3f} {r.mean_max_abs_q:11.3f} {r.rms_q:7.3f} "
            f"{r.mean_fills:7.1f}"
        )
    print(
        f"inventory control: A-S rms q = {res_as.rms_q:.3f} vs fixed "
        f"{res_fx.rms_q:.3f} (similar PnL, much tighter inventory)"
    )
    stress = et.simulate(ap, n_paths, seed, stress_factor=2.0)
    print(f"stress (vol x2 after T/2): std PnL {res_as.std_pnl:.4f} -> {stress.std_pnl:.4f}")


def analytics_section(order: dict) -> None:
    hdr("Implementation-shortfall decomposition (sell parent order)")
    d = et.decompose_shortfall(
        order["side"],
        order["decision_price"],
        order["arrival_price"],
        order["close_price"],
        order["is_total_qty"],
        order["is_fill_qtys"],
        order["is_fill_prices"],
    )
    print(
        f"decision {order['decision_price']:.2f}, arrival "
        f"{order['arrival_price']:.2f}, close {order['close_price']:.2f}; "
        f"executed {d.executed_qty:.0f}/{order['is_total_qty']:.0f}"
    )
    for name, v in (
        ("delay", d.delay),
        ("trading", d.trading),
        ("opportunity", d.opportunity),
        ("TOTAL", d.total),
    ):
        print(f"  {name:>12}: {v:10.2f} $")
    part = et.participation_rate(120_000.0, 1.5e6)
    print(f"participation rate example: 120k / 1.5m = {part:.2%}")

    hdr("Market-impact regression (cost/share vs participation, native OLS)")
    parts, costs = et.synthetic_impact_fills(25, 0.5, 8.0, 0.05)
    reg = et.estimate_impact(parts, costs)
    print(
        f"fitted: cost = {reg.intercept:.4f} + {reg.slope:.4f} * participation "
        f"(true 0.50 + 8.00), R^2 = {reg.r_squared:.4f}, n = {reg.n_obs}"
    )


def market_section() -> None:
    """Presets in market units, results in basis points (API_SPEC Section 7)."""
    hdr("Market presets in bps (execution: sell order; market making: A-S spread)")
    presets = load("presets.json")
    print(
        f"{'preset':>26} {'order %ADV':>10} {'E[IS]':>8} {'sd':>8} {'half-sprd':>9}"
        f" {'A-S spread open/close':>22}"
    )
    for name, body in presets.items():
        ac = body["ac"]
        p = et.market.ac_params_from_market(**ac)
        notional = ac["total_shares"] * ac["price"]
        e = et.market.bps(et.expected_cost(p), notional)
        sd = et.market.bps(math.sqrt(et.variance_cost(p)), notional)
        hs = et.market.bps(et.market.half_spread_cost(ac["spread"], ac["total_shares"]), notional)
        mm = "-"
        if "as" in body:
            a = et.market.as_params_from_market(**body["as"])
            o = et.market.bps(et.optimal_spread(a, 0.0), a.s0)
            c = et.market.bps(et.optimal_spread(a, a.horizon), a.s0)
            mm = f"{o:8.2f} / {c:6.2f} bps"
        print(
            f"{name:>26} {ac['total_shares'] / ac['adv'] * 100:9.1f}% {e:6.2f}bp"
            f" {sd:6.2f}bp {hs:7.2f}bp {mm:>22}"
        )
    print("execution costs use the paper's linear impact calibration (one spread per 1% ADV);")
    print("it under-prices small orders against the square-root law.")


def main() -> int:
    print("etrade demo — Almgren-Chriss execution & Avellaneda-Stoikov market making")
    liq = load("params_liquid.json")
    fx = load("params_fx.json")
    order = load("parent_order.json")
    ac_section(liq, order)
    as_section(liq, "liquid large-cap", seed=42)
    as_section(fx, "FX-like: tight spread, high k", seed=7)
    analytics_section(order)
    market_section()
    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
