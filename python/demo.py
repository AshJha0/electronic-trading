"""End-to-end demo of the etrade package.

Prints: Almgren-Chriss optimal trajectories across risk aversions, the
efficient frontier of execution, TWAP/POV/instantaneous baselines with an MC
cross-check, an Avellaneda-Stoikov market-making simulation (A-S vs
fixed-spread agent), and an implementation-shortfall decomposition with a
market-impact regression, presets in bps, a pre-trade report, post-trade
TCA, calibration with the fitted ITCH presets, execution extensions
(transient impact, adaptive vs static, internalisation), the market-making
friction ledger and limit-order placement. Deterministic lines match the
C++, Rust and Java demos; simulated lines are each language's own sample.

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


def pretrade_section() -> None:
    """Pre-trade report on the power-law model (API_SPEC Section 8)."""
    hdr("Pre-trade report: US large cap, sell 5% ADV in a day (sqrt impact, U-curve)")
    kw = {
        "price": 100.0,
        "spread": 0.01,
        "adv": 5e6,
        "daily_vol": 0.015,
        "total_shares": 250_000.0,
        "n_intervals": 13,
        "tau": 1.0 / 13.0,
        "weights": et.impact.US_EQUITY_VOLUME_CURVE_13,
    }
    rep = et.market.pretrade_report(lambdas=[0.0, 1e-7, 1e-6, 1e-5], **kw)
    print(
        f"square-root-law estimate {rep.sqrt_law_bps:.2f} bps; half spread "
        f"{rep.half_spread_bps:.2f} bps"
    )
    print(f"{'strategy':>10} {'lambda':>9} {'impact':>9} {'sd':>9} {'total':>9} {'horizon':>8}")
    for r in rep.rows:
        print(
            f"{r.label:>10} {r.lambda_ra:9.1e} {r.impact_bps:7.2f}bp {r.sd_bps:7.2f}bp"
            f" {r.total_bps:7.2f}bp {r.horizon:8d}"
        )
    p = et.market.impact_params_from_market(lambda_ra=0.0, drift_bps_per_day=50.0, **kw)
    notional = kw["total_shares"] * kw["price"]
    vwap = et.impact.vwap_holdings(kw["total_shares"], kw["weights"])
    s = et.impact.optimal_schedule(p)
    e_v = et.market.bps(et.impact.impact_expected_cost(p, vwap), notional)
    e_o = et.market.bps(et.impact.impact_expected_cost(p, s.holdings), notional)
    print(
        f"alpha-aware: with 50 bps/day adverse drift, VWAP costs {e_v:.2f} bps and the "
        f"drift-aware schedule {e_o:.2f} bps (saves {e_v - e_o:.2f} bps)"
    )


def tca_section() -> None:
    """Post-trade TCA (API_SPEC Section 10); deterministic."""
    hdr("Post-trade TCA: benchmarks, markouts, participation, A/B test")
    fill_bins = [0, 1, 1, 2]
    fill_qtys = [20_000.0, 15_000.0, 10_000.0, 25_000.0]
    fill_prices = [99.96, 99.91, 99.89, 99.84]
    bin_volumes = [400_000.0, 300_000.0, 300_000.0, 500_000.0]
    bin_vwaps = [99.95, 99.90, 99.85, 99.82]
    r = et.tca.slippage(
        "sell", 100.0, 99.80, fill_bins, fill_qtys, fill_prices, bin_volumes, bin_vwaps
    )
    print(
        f"sell 70,000 at avg {r.avg_price:.4f}: arrival {r.arrival_bps:.4f} bps, "
        f"full-day VWAP {r.vwap_bps:.4f} bps,"
    )
    print(
        f"  interval VWAP {r.interval_vwap_bps:.4f} bps, close {r.close_bps:.4f} bps"
        " (positive = cost)"
    )
    part = et.tca.participation(fill_bins, fill_qtys, bin_volumes)
    mids = [100.00, 99.98, 99.95, 99.93, 99.90, 99.91, 99.88, 99.86, 99.85, 99.84]
    mo = et.tca.markouts("sell", [0, 2, 4, 6], fill_qtys, mids, [1, 3])
    print(
        f"participation {100.0 * part.overall:.4f}%; markout per share after 1 step "
        f"{mo.per_share[0]:.6f} $, after 3 steps {mo.per_share[1]:.6f} $"
    )
    w = et.tca.welch_ttest(
        [1.2, 3.4, 2.2, 5.1, 0.7, 2.9, 4.0, 3.3], [0.4, 1.1, -0.2, 0.9, 1.6, 0.3]
    )
    print(
        f"algo A vs B (8 vs 6 orders): diff {w.mean_diff:.4f} bps, t = {w.t_stat:.4f}, "
        f"df = {w.df:.4f}, p = {w.p_value:.6f}"
    )


#: Fitted presets of API_SPEC 11.6 (data/presets.json).
ITCH_PRESETS = ("itch_aapl_20191230", "itch_msft_20191230", "itch_qqq_20191230")


def calibration_section() -> None:
    """Calibration estimators and the fitted ITCH presets (API_SPEC Section 11)."""
    hdr("Calibration: fill intensity from 100 market orders; fitted ITCH presets")
    dist = [0.005] * 60 + [0.015] * 25 + [0.025] * 10 + [0.035] * 5
    fi = et.calibration.fill_intensity(dist, tick=0.01, n_levels=4, days=1.0, sides=2.0)
    print(
        f"toy fit: A = {fi.a_rate:.4f}/day, k = {fi.k:.4f} per $, touch fills "
        f"{fi.touch_fills_per_day:.4f}/day, halving {fi.halving_ticks:.4f} ticks"
    )
    presets = load("presets.json")
    print("fitted to one NASDAQ ITCH day, 2019-12-30 (one venue; caveats in API_SPEC 11.6):")
    print(
        f"{'preset':>20} {'price':>7} {'spread':>8} {'vol':>7} {'A /day':>10} {'k /$':>8}"
        f" {'A-S open':>9} {'E[IS] 5% ADV':>12}"
    )
    for name in ITCH_PRESETS:
        ac, asm = presets[name]["ac"], presets[name]["as"]
        p = et.market.ac_params_from_market(**ac)
        a = et.market.as_params_from_market(**asm)
        e = et.market.bps(et.expected_cost(p), ac["total_shares"] * ac["price"])
        o = et.market.bps(et.optimal_spread(a, 0.0), a.s0)
        sp = et.market.bps(ac["spread"], ac["price"])
        print(
            f"{name:>20} {ac['price']:7.2f} {sp:6.3f}bp {100.0 * ac['daily_vol']:6.3f}%"
            f" {a.a_rate:10.1f} {a.k:8.3f} {o:7.4f}bp {e:10.4f}bp"
        )


def execution_ext_section() -> None:
    """Transient impact, adaptive re-planning, internalisation (API_SPEC Section 12)."""
    hdr("Execution extensions: transient impact, adaptive vs static, internalisation")
    tp = et.execution.TransientParams(
        total_shares=1e6, n_intervals=10, tau=0.1, kappa=1e-6, rho=5.0
    )
    x = et.execution.transient_optimal_schedule(tp)
    e_opt = et.execution.transient_expected_cost(tp, x)
    e_twap = et.execution.transient_expected_cost(tp, et.twap_holdings(1e6, 10))
    print(
        f"Obizhaeva-Wang, rho*tau = 0.5: first / middle / last trade "
        f"{x[0] - x[1]:.2f} / {x[4] - x[5]:.2f} / {x[9] - x[10]:.2f} shares"
    )
    print(
        f"  E[IS] optimal {e_opt:.2f} $ vs TWAP {e_twap:.2f} $ "
        f"(saves {100.0 * (1.0 - e_opt / e_twap):.4f}%)"
    )
    p = et.ACParams(1_000_000.0, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6)
    print("adaptive re-planning, liquid set, 20k paths (each language samples its own paths):")
    print(f"{'case':>24} {'static E[IS]':>13} {'adaptive':>11} {'saving':>9} {'se':>6}")
    for label, sig in (
        ("no signal, full fills", et.execution.SignalParams()),
        ("AR(1) signal 0.7, 0.5", et.execution.SignalParams(phi=0.7, signal_sd=0.5)),
        ("fill risk 0.5", et.execution.SignalParams(fill_risk=0.5)),
    ):
        s = et.execution.simulate_adaptive(p, sig, n_paths=20_000, seed=1)
        print(
            f"{label:>24} {s.static_mean:13.0f} {s.adaptive_mean:11.0f} "
            f"{s.saving_mean:9.0f} {s.saving_se:6.0f}"
        )
    r = et.execution.internalise(
        [200_000.0, 50_000.0],
        [150_000.0],
        price=100.0,
        spread=0.01,
        adv=5e6,
        daily_vol=0.015,
        n_intervals=13,
        tau=1.0 / 13.0,
        weights=et.impact.US_EQUITY_VOLUME_CURVE_13,
    )
    print(
        f"internalise buys 200k + 50k against a 150k sell: cross {r.crossed_shares:.0f}, "
        f"send {r.residual_shares:.0f} ({r.residual_side})"
    )
    print(
        f"  impact {r.baseline_cost:.2f} $ -> {r.internal_cost:.2f} $, "
        f"saves {r.saved_bps:.4f} bps of gross notional"
    )


#: Friction ledger rows: label, MMConfig fields, seed. Seeds 101-106 are the
#: mm_sim golden cases of API_SPEC 9.7; the fee row is extra.
FRICTIONS = (
    ("frictionless", {}, 101),
    ("tick + 0.2c rebate", {"tick": 0.01, "maker_fee": 0.002}, 102),
    ("tick + 0.2c fee", {"tick": 0.01, "maker_fee": -0.002}, 107),
    ("+ 15% informed flow", {"tick": 0.01, "maker_fee": 0.002, "informed_prob": 0.15}, 103),
    (
        "+ 60% signal",
        {"tick": 0.01, "maker_fee": 0.002, "informed_prob": 0.15, "signal_accuracy": 0.6},
        104,
    ),
    ("quotes 5 s stale", {"tick": 0.01, "maker_fee": 0.002, "latency_steps": 1}, 105),
    ("GLFT, tick + rebate", {"agent": "glft", "tick": 0.01, "maker_fee": 0.002}, 106),
)


def mm_frictions_section() -> None:
    """Market making with frictions: the friction ledger (API_SPEC Section 9)."""
    hdr("Market making with frictions: US large-cap A-S preset, one-share quotes")
    a = et.market.as_params_from_market(**load("presets.json")["us_large_cap"]["as"])
    print(
        f"A = {a.a_rate:.4f}/day, k = {a.k:.4f} per $, gamma = {a.gamma:.6f}, "
        f"{a.n_steps} steps of 5 s"
    )
    for q in (0.0, 3.0):
        db, da = et.mm.glft_half_spreads(a, q)
        print(f"GLFT half spreads at q = {q:+.0f}: bid {db:.8f} $, ask {da:.8f} $")
    b, k = et.mm.round_to_tick(99.9934, 100.0066, 0.01)
    wm = et.mm.weighted_mid(99.99, 100.01, 900.0, 100.0)
    print(
        f"tick rounding away from fair: 99.9934 / 100.0066 -> {b:.2f} / {k:.2f}; "
        f"weighted mid {wm:.3f}"
    )
    print("friction ledger, $ per day, 20k paths (each language samples its own paths):")
    print(
        f"{'case':>20} {'PnL':>7} {'se':>5} | {'spread':>7} {'fees':>6} {'invent':>7}"
        f" {'markout':>8} | {'fills':>6} {'informed':>8}"
    )
    for label, cfg, seed in FRICTIONS:
        r = et.mm.simulate_mm(a, 20_000, seed, et.mm.MMConfig(**cfg))
        print(
            f"{label:>20} {r.mean('pnl'):7.2f} {r.se('pnl'):5.2f} | {r.mean('spread'):7.2f}"
            f" {r.mean('fees'):6.2f} {r.mean('inventory'):7.2f} {r.mean('markout'):8.2f} |"
            f" {r.mean('fills'):6.1f} {r.mean('informed_fills'):8.1f}"
        )
    print("PnL = spread + fees + inventory - liquidation on every path; the markout is part")
    print("of inventory (negative = adverse selection).")


def placement_section() -> None:
    """Queue position, passive or aggressive, liquidation (API_SPEC Section 13)."""
    hdr("Limit-order placement: queue fill probability, post or cross, liquidation")
    print(f"{'q_ahead':>8} {'size':>5} {'mu':>6} {'theta':>6} {'horizon':>8} {'P(full fill)':>14}")
    for qa, sz, mu, th, h in (
        (0, 5, 10.0, 0.1, 0.5),
        (30, 5, 10.0, 0.0, 2.0),
        (50, 10, 20.0, 0.05, 3.0),
        (400, 50, 100.0, 0.2, 4.0),
    ):
        pf = et.placement.queue_fill_probability(qa, sz, mu, th, h)
        print(f"{qa:8d} {sz:5d} {mu:6.1f} {th:6.2f} {h:8.1f} {pf:14.10f}")
    for label, qa, theta, horizon, adverse in (
        ("near the front", 5, 0.1, 1.0, 0.01),
        ("deep in queue", 60, 0.05, 3.0, 0.03),
    ):
        pf = et.placement.queue_fill_probability(qa, 2, 10.0, theta, horizon)
        d = et.placement.place_or_cross(2000.0, 0.02, pf, adverse, 1e-4, 0.5, 1.0 / 13.0)
        print(
            f"sell 2,000 {label}: P(fill) {pf:.6f}, post {d.passive_cost:.6f} vs cross "
            f"{d.aggressive_cost:.6f} $/sh, breakeven {d.breakeven_fill_prob:.4f}: "
            f"{'post' if d.passive else 'cross'}"
        )
    print("liquidate 10 units over 50 periods (patient) or 20 with penalty phi = 0.05 (urgent):")
    print(
        f"{'case':>8} {'DP cost':>9} {'depth':>8} {'mkt':>4} {'all-mkt':>7} |"
        f" {'optimal':>8} {'all-mkt':>8} {'touch':>8} {'saving vs all-mkt':>17}"
    )
    for label, lp, seed in (
        ("patient", et.placement.LiquidationParams(10, 50, 0.02, 0.5, 0.02, 30.0, 0.3, 0.005), 201),
        (
            "urgent",
            et.placement.LiquidationParams(20, 50, 0.02, 0.5, 0.02, 50.0, 0.4, 0.005, 0.05),
            202,
        ),
    ):
        pol = et.placement.liquidation_policy(lp)
        n_mkt = int(pol.market[:, 1:].sum())
        s = et.placement.simulate_liquidation(lp, n_paths=20_000, seed=seed)
        print(
            f"{label:>8} {pol.expected_cost:9.5f} {pol.depth[0, lp.total_units]:8.5f}"
            f" {n_mkt:4d} {et.placement.aggressive_cost(lp):7.3f} | {s.optimal_mean:8.4f}"
            f" {s.aggressive_mean:8.4f} {s.passive_mean:8.4f}"
            f" {s.saving_vs_aggressive_mean:9.4f} +- {s.saving_vs_aggressive_se:.4f}"
        )
    print("costs in $ per position against the starting mid (negative = sold above it); the")
    print("DP cost adds the urgency penalty, so it exceeds the simulated shortfall when phi > 0;")
    print("depth = first posting depth in $; mkt = states where the policy sends a market order;")
    print("simulations 20k paths, each language samples its own paths.")


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
    pretrade_section()
    tca_section()
    calibration_section()
    execution_ext_section()
    mm_frictions_section()
    placement_section()
    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
