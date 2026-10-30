"""Execution extensions (API_SPEC Section 12): transient impact and its
reduction to the linear case, the Obizhaeva-Wang schedule shape, adaptive
re-planning, internalisation, goldens, validation."""

from __future__ import annotations

import math

import numpy as np
import pytest

import etrade as et
from conftest import ac_from_cfg, load_json
from etrade import execution as ex
from etrade import impact as im

GOLDEN = load_json("golden/execution_golden.json")
LIQ = load_json("params_liquid.json")

MARKET = {
    "price": 100.0,
    "spread": 0.01,
    "adv": 5e6,
    "daily_vol": 0.015,
    "n_intervals": 13,
    "tau": 1.0 / 13.0,
}

# ------------------------------------------------------------ transient impact


def test_cost_recursion_equals_quadratic_form() -> None:
    p = ex.TransientParams(1e6, 8, 0.25, 3e-6, 1.5, 4e-7)
    rng = np.random.default_rng(1)
    n = rng.standard_normal(8) * 1e5 + 1.25e5
    x = np.concatenate(([n.sum()], n.sum() - np.cumsum(n)))
    x[-1] = 0.0
    a = p.decay()
    k = a ** np.abs(np.subtract.outer(np.arange(8), np.arange(8)))
    big_a = p.kappa * k - p.gamma_p * np.eye(8)
    want = 0.5 * p.gamma_p * x[0] ** 2 + 0.5 * n @ big_a @ n
    assert ex.transient_expected_cost(p, x) == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize("gamma_p", [0.0, 4e-7])
def test_optimum_solves_the_dense_kkt_system(gamma_p: float) -> None:
    p = ex.TransientParams(1e6, 12, 0.25, 3e-6, 1.5, gamma_p)
    a = p.decay()
    k = a ** np.abs(np.subtract.outer(np.arange(12), np.arange(12)))
    y = np.linalg.solve(p.kappa * k - p.gamma_p * np.eye(12), np.ones(12))
    want = p.total_shares * y / y.sum()
    got = -np.diff(ex.transient_optimal_schedule(p))
    assert np.max(np.abs(got - want)) <= 1e-9 * p.total_shares


def test_obizhaeva_wang_shape_blocks_at_both_ends_constant_between() -> None:
    p = ex.TransientParams(1e6, 10, 0.1, 2e-6, 5.0, 0.0)
    n = -np.diff(ex.transient_optimal_schedule(p))
    assert n[0] == pytest.approx(n[-1], rel=1e-12)
    assert np.ptp(n[1:-1]) <= 1e-12 * p.total_shares
    # interior rate / block = 1 - a (Obizhaeva and Wang 2013, discrete time)
    assert n[1] / n[0] == pytest.approx(1.0 - p.decay(), rel=1e-12)
    assert np.all(n > 0.0)


def test_permanent_impact_keeps_symmetry() -> None:
    p = ex.TransientParams(1e6, 10, 0.1, 2e-6, 5.0, 2e-7)
    n = -np.diff(ex.transient_optimal_schedule(p))
    assert np.max(np.abs(n - n[::-1])) <= 1e-9 * p.total_shares
    assert n[0] > n[1] and np.all(n > 0.0)


def test_optimum_beats_twap_and_perturbations() -> None:
    p = ex.TransientParams(2.5e5, 26, 1 / 26, 4e-6, 2.0, 1e-8)
    x = ex.transient_optimal_schedule(p)
    e_opt = ex.transient_expected_cost(p, x)
    assert e_opt < ex.transient_expected_cost(p, et.twap_holdings(2.5e5, 26))
    rng = np.random.default_rng(3)
    for _ in range(100):
        z = x.copy()
        z[1:-1] += rng.standard_normal(25) * 1e3
        assert ex.transient_expected_cost(p, z) >= e_opt


@pytest.mark.parametrize("lam", [0.0])
def test_fast_resilience_reduces_to_linear_temporary_impact(lam: float) -> None:
    """rho -> infinity with kappa = 2 eta / tau is Section 1 (alpha = 1)."""
    ac = ac_from_cfg(LIQ, lambda_ra=lam)
    kappa = 2.0 * ac.eta / ac.tau
    fast = ex.TransientParams(ac.total_shares, ac.n_intervals, ac.tau, kappa, 1e3, ac.gamma_p)
    assert fast.decay() == 0.0
    x_ow = ex.transient_optimal_schedule(fast)
    assert np.max(np.abs(x_ow - et.ac_trajectory(ac))) <= 1e-9 * ac.total_shares
    for x in (
        x_ow,
        et.twap_holdings(ac.total_shares, ac.n_intervals),
        [ac.total_shares] + [0.0] * 5,
    ):
        assert ex.transient_expected_cost(fast, x) == pytest.approx(
            et.expected_cost(ac, x), rel=1e-12
        )
    # and the transient cost converges to it as rho grows
    x = et.twap_holdings(ac.total_shares, ac.n_intervals)
    gaps = []
    for rho in (0.5, 2.0, 8.0):
        p = ex.TransientParams(ac.total_shares, ac.n_intervals, ac.tau, kappa, rho, ac.gamma_p)
        gaps.append(ex.transient_expected_cost(p, x) - et.expected_cost(ac, x))
    assert gaps[0] > gaps[1] > gaps[2] > 0.0
    # and the alpha = 1 Section 8 model agrees
    ip = im.ImpactParams(
        ac.total_shares, ac.n_intervals, ac.tau, ac.sigma, ac.eta, 1.0, ac.gamma_p, 0.0
    )
    assert ex.transient_expected_cost(fast, x) == pytest.approx(
        im.impact_expected_cost(ip, x), rel=1e-12
    )


def test_single_interval_and_zero_shares() -> None:
    p = ex.TransientParams(5.0, 1, 1.0, 1.0, 1.0)
    assert list(ex.transient_optimal_schedule(p)) == [5.0, 0.0]
    assert ex.transient_expected_cost(p, [5.0, 0.0]) == pytest.approx(12.5)
    z = ex.TransientParams(0.0, 4, 1.0, 1.0, 1.0)
    assert np.all(ex.transient_optimal_schedule(z) == 0.0)


# ------------------------------------------------------------ adaptive


def _ac10() -> et.ACParams:
    return et.ACParams(1e6, 10, 0.5, 0.95, 2.5e-6, 2.5e-7, 2e-6)


def test_static_plan_is_the_almgren_chriss_trajectory() -> None:
    ac = _ac10()
    plan = ex.replan(ac, ex.SignalParams(), ac.total_shares, 0.0, 0)
    assert np.max(np.abs(np.array(plan) - et.ac_trajectory(ac))) <= 1e-9 * ac.total_shares


def test_replan_is_time_consistent_without_signal() -> None:
    ac = _ac10()
    sig = ex.SignalParams()
    plan = ex.replan(ac, sig, ac.total_shares, 0.0, 0)
    for k in range(1, 10):
        tail = ex.replan(ac, sig, plan[k], 0.0, k)
        assert np.max(np.abs(np.array(tail) - np.array(plan[k:]))) <= 1e-9 * ac.total_shares


def test_feedback_coefficients_match_replan() -> None:
    ac = _ac10()
    sig = ex.SignalParams(0.6, 0.4, 0.0)
    c, d = ex.replan_coefficients(ac, sig)
    for k in range(10):
        m = 10 - k
        x = ex.replan(ac, sig, 3e5, 0.7, k)
        assert x[0] - x[1] == pytest.approx(c[m] * 3e5 - d[m] * 0.7, rel=1e-12)


def test_signal_makes_the_trader_hold_when_price_is_expected_to_rise() -> None:
    ac = _ac10()
    sig = ex.SignalParams(0.7, 0.5, 0.0)
    base = ex.replan(ac, sig, 5e5, 0.0, 3)
    up = ex.replan(ac, sig, 5e5, 1.0, 3)
    assert up[1] > base[1]


def test_adaptive_equals_static_without_signal_or_fill_risk() -> None:
    """Almgren-Chriss, linear impact, no signal: the static schedule is already
    dynamically optimal, so re-planning changes nothing path by path."""
    r = ex.simulate_adaptive(_ac10(), ex.SignalParams(), 4000, 7)
    assert abs(r.saving_mean) <= 1e-9 * r.static_mean
    assert r.static_std == pytest.approx(r.adaptive_std, rel=1e-9)
    # and the static mean agrees with the Section 1 closed form within 4 SE
    assert abs(r.static_mean - et.expected_cost(_ac10())) <= 4.0 * r.static_se


@pytest.mark.parametrize("sig", [ex.SignalParams(0.7, 0.5, 0.0), ex.SignalParams(0.0, 0.0, 0.5)])
def test_adaptive_helps_with_a_signal_or_fill_risk(sig: ex.SignalParams) -> None:
    r = ex.simulate_adaptive(_ac10(), sig, 4000, 8)
    assert r.saving_mean > 4.0 * r.saving_se


def test_adaptive_is_deterministic() -> None:
    a = ex.simulate_adaptive(_ac10(), ex.SignalParams(0.5, 0.3, 0.2), 100, 5)
    b = ex.simulate_adaptive(_ac10(), ex.SignalParams(0.5, 0.3, 0.2), 100, 5)
    assert a == b


# ------------------------------------------------------------ internalisation


def test_crossing_never_increases_total_cost() -> None:
    rng = np.random.default_rng(11)
    for _ in range(200):
        buys = list(rng.uniform(1e3, 3e5, rng.integers(0, 4)))
        sells = list(rng.uniform(1e3, 3e5, rng.integers(1, 4)))
        r = ex.internalise(buys, sells, **MARKET)
        assert r.internal_cost <= r.baseline_cost
        assert r.saved_cost >= 0.0
        assert all(v >= 0.0 for v in r.buy_saved_bps + r.sell_saved_bps)


def test_internalisation_accounting() -> None:
    r = ex.internalise([3e5, 1e5], [2.5e5], **MARKET)
    assert r.crossed_shares == 2.5e5 and r.residual_shares == 1.5e5
    assert r.residual_side == "buy"
    # per-order savings add up to the total saving
    q = [3e5, 1e5, 2.5e5]
    per = r.buy_saved_bps + r.sell_saved_bps
    total = sum(b * qi * MARKET["price"] / 1e4 for b, qi in zip(per, q, strict=True))
    assert total == pytest.approx(r.saved_cost, rel=1e-12)
    assert r.saved_bps == pytest.approx(1e4 * r.saved_cost / (6.5e5 * 100.0), rel=1e-15)
    # one-sided flow saves nothing
    one = ex.internalise([], [1e5, 2e5], **MARKET)
    assert one.saved_cost == 0.0 and one.residual_side == "sell"
    # a perfect cross saves everything
    even = ex.internalise([2e5], [1e5, 1e5], **MARKET)
    assert even.internal_cost == 0.0 and even.residual_side == "none"


# ------------------------------------------------------------ goldens


def _ac_inputs(inp: dict) -> et.ACParams:
    return ac_from_cfg(inp)


def _actual(case: dict) -> tuple[dict, dict]:
    inp = dict(case["inputs"])
    kind = case["kind"]
    if kind == "transient_schedule":
        p = ex.TransientParams(
            inp["total_shares"],
            int(inp["n_intervals"]),
            inp["tau"],
            inp["kappa"],
            inp["rho"],
            inp["gamma_p"],
        )
        x = ex.transient_optimal_schedule(p)
        out = {f"xfrac_{j:02d}": float(x[j]) / p.total_shares for j in range(1, p.n_intervals)}
        out["e_cost"] = ex.transient_expected_cost(p, x)
        out["twap_e_cost"] = ex.transient_expected_cost(
            p, et.twap_holdings(p.total_shares, p.n_intervals)
        )
        return out, {}
    if kind == "internalise":
        buys = [inp.pop(f"buy_{i}") for i in range(int(inp.pop("n_buys")))]
        sells = [inp.pop(f"sell_{i}") for i in range(int(inp.pop("n_sells")))]
        weights = im.US_EQUITY_VOLUME_CURVE_13 if inp.pop("volume_curve") == 1.0 else None
        inp["n_intervals"] = int(inp["n_intervals"])
        r = ex.internalise(buys, sells, weights=weights, **inp)
        out = {
            "crossed_shares": r.crossed_shares,
            "residual_shares": r.residual_shares,
            "baseline_cost": r.baseline_cost,
            "internal_cost": r.internal_cost,
            "saved_cost": r.saved_cost,
            "saved_bps": r.saved_bps,
        }
        for i, v in enumerate(r.buy_saved_bps):
            out[f"buy{i}_saved_bps"] = v
        for i, v in enumerate(r.sell_saved_bps):
            out[f"sell{i}_saved_bps"] = v
        return out, {}
    sig = ex.SignalParams(inp["phi"], inp["signal_sd"], inp["fill_risk"])
    r = ex.simulate_adaptive(_ac_inputs(inp), sig, int(inp["n_paths_ref"]), int(inp["seed"]))
    out = {
        "static_mean": r.static_mean,
        "adaptive_mean": r.adaptive_mean,
        "saving_mean": r.saving_mean,
    }
    se = {"static_mean": r.static_se, "adaptive_mean": r.adaptive_se, "saving_mean": r.saving_se}
    return out, se


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["name"])
def test_execution_golden(case: dict) -> None:
    actual, se = _actual(case)
    if case["kind"] == "adaptive_sim":
        keys = [k for k in case["expect"] if not k.endswith("_se")]
        assert keys and all(k in actual for k in keys)
        for key in keys:
            ref, se_ref = case["expect"][key], case["expect"][key + "_se"]
            band = case["tol"] * math.sqrt(se_ref**2 + se[key] ** 2)
            assert abs(actual[key] - ref) <= band, f"{case['name']}.{key}"
        return
    assert set(actual) == set(case["expect"])
    for key, want in case["expect"].items():
        assert abs(actual[key] - want) <= case["tol"] * max(1.0, abs(want)), f"{case['name']}.{key}"


def test_execution_golden_count() -> None:
    assert len(GOLDEN["cases"]) == 8


# ------------------------------------------------------------ validation


@pytest.mark.parametrize(
    "field,bad",
    [
        ("total_shares", -1.0),
        ("n_intervals", 0),
        ("n_intervals", ex.MAX_EXEC_INTERVALS + 1),
        ("tau", 0.0),
        ("kappa", 0.0),
        ("rho", 0.0),
        ("rho", math.inf),
        ("gamma_p", -1.0),
        ("gamma_p", 1e-6),  # >= kappa (1-a)/(1+a): not convex
    ],
)
def test_transient_params_validation(field: str, bad: object) -> None:
    kw: dict[str, object] = {
        "total_shares": 1e6,
        "n_intervals": 10,
        "tau": 0.1,
        "kappa": 2e-6,
        "rho": 5.0,
        "gamma_p": 0.0,
    }
    kw[field] = bad
    with pytest.raises(ValueError, match=field.split("_")[0] if field != "gamma_p" else "gamma_p"):
        ex.TransientParams(**kw)  # type: ignore[arg-type]


def test_transient_cost_validates_holdings() -> None:
    p = ex.TransientParams(1e6, 4, 0.25, 2e-6, 5.0)
    with pytest.raises(ValueError, match="length"):
        ex.transient_expected_cost(p, [1.0, 0.0])
    with pytest.raises(ValueError, match="liquidate"):
        ex.transient_expected_cost(p, [1.0, 0.8, 0.6, 0.4, 0.2])
    with pytest.raises(ValueError, match="finite"):
        ex.transient_expected_cost(p, [1.0, math.nan, 0.6, 0.4, 0.0])


@pytest.mark.parametrize(
    "kw,match",
    [
        ({"phi": 1.0}, "phi"),
        ({"phi": -0.1}, "phi"),
        ({"signal_sd": -1.0}, "signal_sd"),
        ({"fill_risk": 1.0}, "fill_risk"),
        ({"fill_risk": math.nan}, "fill_risk"),
    ],
)
def test_signal_params_validation(kw: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ex.SignalParams(**kw)


def test_adaptive_validation() -> None:
    ac = _ac10()
    with pytest.raises(ValueError, match="n_paths"):
        ex.simulate_adaptive(ac, ex.SignalParams(), 1, 0)
    with pytest.raises(ValueError, match="k must"):
        ex.replan(ac, ex.SignalParams(), 1.0, 0.0, 10)
    with pytest.raises(ValueError, match="holding"):
        ex.replan(ac, ex.SignalParams(), math.inf, 0.0, 0)
    with pytest.raises(ValueError, match="n_intervals"):
        ex.simulate_adaptive(ac.__class__(1.0, 1001, 1.0, 1.0, 1.0, 0.0, 0.0), ex.SignalParams())


def test_internalise_validation() -> None:
    with pytest.raises(ValueError, match="buys"):
        ex.internalise([0.0], [1.0], **MARKET)
    with pytest.raises(ValueError, match="sells"):
        ex.internalise([1.0], [math.nan], **MARKET)
    with pytest.raises(ValueError, match="at least one"):
        ex.internalise([], [], **MARKET)
    with pytest.raises(ValueError, match="spread"):
        ex.internalise([1.0], [2.0], **{**MARKET, "spread": 0.0})
