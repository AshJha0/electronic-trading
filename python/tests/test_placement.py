"""Limit-order placement (API_SPEC Section 13): queue fill probability against
closed forms, monotonicity, the passive-or-aggressive rule, the liquidation
policy against its Monte Carlo, goldens, validation."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy import special, stats

from conftest import load_json
from etrade import impact as im
from etrade import placement as pl

GOLDEN = load_json("golden/placement_golden.json")


def _liq(**kw: float) -> pl.LiquidationParams:
    base: dict = {
        "total_units": 10,
        "n_periods": 50,
        "dt": 0.02,
        "sigma": 0.5,
        "spread": 0.02,
        "kappa": 30.0,
        "p_touch": 0.3,
        "beta": 0.005,
        "phi": 0.0,
    }
    base.update(kw)
    return pl.LiquidationParams(**base)


# ------------------------------------------------------------ fill probability


@pytest.mark.parametrize("q,n,mu,h", [(30, 5, 10.0, 2.0), (0, 1, 3.0, 0.4), (200, 20, 50.0, 5.0)])
def test_no_cancellations_is_the_poisson_tail(q: int, n: int, mu: float, h: float) -> None:
    want = float(stats.poisson.sf(q + n - 1, mu * h))
    assert pl.queue_fill_probability(q, n, mu, 0.0, h) == pytest.approx(want, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize(
    "q,mu,theta,h", [(1, 4.0, 0.5, 0.7), (12, 6.0, 0.3, 1.5), (40, 2.0, 1.0, 3.0)]
)
def test_queue_depletion_is_an_incomplete_beta(q: int, mu: float, theta: float, h: float) -> None:
    # sum_{k=1}^{q} Exp(mu + theta k) <= h  <=>  I_{1-exp(-theta h)}(q, mu/theta + 1); a size-1
    # order then needs one more Exp(mu).  Check the depletion part via a tiny mu tail:
    # P(T_ahead + Exp(mu) <= h) = int f_ahead(t) (1 - exp(-mu (h - t))) dt.
    def cdf_ahead(t: float) -> float:
        return float(special.betainc(q, mu / theta + 1.0, 1.0 - math.exp(-theta * t)))

    ts = np.linspace(0.0, h, 20001)
    cdf = np.array([cdf_ahead(t) for t in ts])
    integrand = mu * np.exp(-mu * (h - ts)) * cdf  # integration by parts of the convolution
    want = float(np.trapezoid(integrand, ts))
    assert pl.queue_fill_probability(q, 1, mu, theta, h) == pytest.approx(want, rel=1e-6)


def test_fill_probability_decreases_with_queue_position() -> None:
    probs = [pl.queue_fill_probability(q, 10, 20.0, 0.05, 3.0) for q in range(0, 120, 5)]
    assert all(b < a for a, b in itertools.pairwise(probs))
    sizes = [pl.queue_fill_probability(20, n, 20.0, 0.05, 3.0) for n in range(1, 40, 3)]
    assert all(b < a for a, b in itertools.pairwise(sizes))


def test_fill_probability_increases_with_horizon_and_cancellations() -> None:
    probs = [pl.queue_fill_probability(50, 10, 20.0, 0.05, h) for h in np.linspace(1.0, 4.0, 30)]
    assert all(b > a for a, b in itertools.pairwise(probs))
    thetas = [pl.queue_fill_probability(50, 10, 20.0, t, 2.0) for t in (0.0, 0.05, 0.2, 1.0)]
    assert all(b > a for a, b in itertools.pairwise(thetas))


def test_fill_probability_edges() -> None:
    assert pl.queue_fill_probability(10, 3, 5.0, 0.2, 0.0) == 0.0
    assert pl.queue_fill_probability(0, 1, 5.0, 0.0, 0.3) == pytest.approx(1.0 - math.exp(-1.5))
    assert 0.0 <= pl.queue_fill_probability(4999, 1, 1.0, 0.0, 1.0) <= 1e-300
    assert pl.queue_fill_probability(0, 1, 1e3, 0.0, 100.0) == 1.0


def test_uniformisation_window() -> None:
    assert pl.uniformisation_window(0.5) == (0, 18)
    assert pl.uniformisation_window(100.0) == (0, 210)
    assert pl.uniformisation_window(10_000.0) == (8990, 11010)


# ------------------------------------------------------------ passive or aggressive


def test_decision_matches_section_8_and_closed_form_difference() -> None:
    d = pl.place_or_cross(2000.0, 0.02, 0.7, 0.01, 1e-4, 0.5, 1.0 / 13.0)
    ip = im.ImpactParams(2000.0, 1, 1.0 / 13.0, 0.0, 1e-4, 0.5, 0.0, 0.0)
    assert d.impact_cost == pytest.approx(im.impact_expected_cost(ip, [2000.0, 0.0]) / 2000.0)
    assert d.aggressive_cost == pytest.approx(0.01 + d.impact_cost, rel=1e-15)
    diff = 0.3 * 0.01 - 2.0 * 0.7 * 0.01
    assert d.passive_cost - d.aggressive_cost == pytest.approx(diff, rel=1e-12)
    assert d.passive


@pytest.mark.parametrize("fp", [0.0, 0.2, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("adv", [0.0, 0.004, 0.05])
def test_aggressive_chosen_when_spread_is_negligible(fp: float, adv: float) -> None:
    assert not pl.place_or_cross(500.0, 0.0, fp, adv, 1e-4, 0.5, 0.1).passive


def test_breakeven_fill_probability_separates_the_choices() -> None:
    adv, spread = 0.015, 0.01
    d0 = pl.place_or_cross(100.0, spread, 0.5, adv, 1e-4, 0.6, 0.2)
    assert d0.breakeven_fill_prob == pytest.approx(adv / (adv + spread))
    for fp in np.linspace(0.0, 1.0, 41):
        d = pl.place_or_cross(100.0, spread, float(fp), adv, 1e-4, 0.6, 0.2)
        if abs(fp - d.breakeven_fill_prob) > 1e-9:
            assert d.passive == (fp > d.breakeven_fill_prob)
    assert pl.place_or_cross(100.0, 0.0, 0.5, 0.0, 1e-4, 0.6, 0.2).breakeven_fill_prob == 1.0


# ------------------------------------------------------------ liquidation


def test_policy_closed_form_one_period() -> None:
    p = _liq(total_units=1, n_periods=1, kappa=30.0, p_touch=0.3)
    pol = pl.liquidation_policy(p)
    g = 0.01 + 0.015  # half spread + (C_N(1) - C_N(0)), C_N(1) = hs + beta
    d = 1.0 / 30.0 - g
    assert pol.depth[0, 1] == pytest.approx(d)
    assert pol.expected_cost == pytest.approx(0.015 - 0.3 * math.exp(-30.0 * d) / 30.0)
    assert not pol.market[0, 1]


def test_depth_falls_with_inventory_and_time() -> None:
    pol = pl.liquidation_policy(_liq())
    posted = ~pol.market
    for j in range(50):
        d = [pol.depth[j, q] for q in range(1, 11) if posted[j, q]]
        assert all(b <= a + 1e-15 for a, b in itertools.pairwise(d))
    assert np.all(np.diff(pol.cost[:, 3], axis=0) >= -1e-15)  # less time costs more


def test_monte_carlo_matches_dp_and_beats_baselines_without_penalty() -> None:
    p = _liq()
    r = pl.simulate_liquidation(p, 20_000, 5)
    assert abs(r.optimal_mean - pl.liquidation_policy(p).expected_cost) <= 4.0 * r.optimal_se
    assert abs(r.aggressive_mean - pl.aggressive_cost(p)) <= 4.0 * r.aggressive_se
    assert r.saving_vs_aggressive_mean > 4.0 * r.saving_vs_aggressive_se
    assert r.saving_vs_passive_mean > 4.0 * r.saving_vs_passive_se


def test_deterministic_without_volatility() -> None:
    p = _liq(sigma=0.0, total_units=4, n_periods=4)
    r = pl.simulate_liquidation(p, 50, 1)
    assert r.aggressive_std <= 1e-15
    assert r.aggressive_mean == pytest.approx(pl.aggressive_cost(p), rel=1e-15)


def test_simulation_is_deterministic() -> None:
    a = pl.simulate_liquidation(_liq(), 200, 9)
    b = pl.simulate_liquidation(_liq(), 200, 9)
    assert a == b


def test_aggressive_schedule() -> None:
    assert pl.aggressive_schedule(10, 4) == [2, 3, 2, 3]
    assert pl.aggressive_schedule(3, 5) == [0, 1, 0, 1, 1]
    assert sum(pl.aggressive_schedule(997, 50)) == 997


# ------------------------------------------------------------ goldens


def _liq_inputs(inp: dict) -> pl.LiquidationParams:
    return pl.LiquidationParams(
        int(inp["total_units"]),
        int(inp["n_periods"]),
        inp["dt"],
        inp["sigma"],
        inp["spread"],
        inp["kappa"],
        inp["p_touch"],
        inp["beta"],
        inp["phi"],
    )


def _actual(case: dict) -> tuple[dict, dict]:
    inp = case["inputs"]
    kind = case["kind"]
    if kind in ("fill_probability", "placement_decision"):
        fp = pl.queue_fill_probability(
            int(inp["q_ahead"]), int(inp["size"]), inp["mu"], inp["theta"], inp["horizon"]
        )
        if kind == "fill_probability":
            return {"fill_prob": fp}, {}
        d = pl.place_or_cross(
            inp["child_shares"],
            inp["spread"],
            fp,
            inp["adverse_move"],
            inp["eta"],
            inp["alpha"],
            inp["weight"],
        )
        return {
            "fill_prob": fp,
            "passive_cost": d.passive_cost,
            "aggressive_cost": d.aggressive_cost,
            "impact_cost": d.impact_cost,
            "breakeven_fill_prob": d.breakeven_fill_prob,
            "passive": 1.0 if d.passive else 0.0,
        }, {}
    p = _liq_inputs(inp)
    if kind == "liquidation_policy":
        pol = pl.liquidation_policy(p)
        return {
            "dp_cost": pol.expected_cost,
            "depth_first": float(pol.depth[0, p.total_units]),
            "depth_last": float(pol.depth[p.n_periods - 1, 1]),
            "n_market": float(pol.market[:, 1:].sum()),
            "aggressive_cost": pl.aggressive_cost(p),
        }, {}
    r = pl.simulate_liquidation(p, int(inp["n_paths_ref"]), int(inp["seed"]))
    out = {
        "optimal_mean": r.optimal_mean,
        "aggressive_mean": r.aggressive_mean,
        "passive_mean": r.passive_mean,
        "saving_vs_aggressive_mean": r.saving_vs_aggressive_mean,
        "saving_vs_passive_mean": r.saving_vs_passive_mean,
    }
    se = {
        "optimal_mean": r.optimal_se,
        "aggressive_mean": r.aggressive_se,
        "passive_mean": r.passive_se,
        "saving_vs_aggressive_mean": r.saving_vs_aggressive_se,
        "saving_vs_passive_mean": r.saving_vs_passive_se,
    }
    return out, se


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["name"])
def test_placement_golden(case: dict) -> None:
    actual, se = _actual(case)
    if case["kind"] == "liquidation_sim":
        keys = [k for k in case["expect"] if not k.endswith("_se")]
        assert set(keys) == set(actual)
        for key in keys:
            ref, se_ref = case["expect"][key], case["expect"][key + "_se"]
            band = case["tol"] * math.sqrt(se_ref**2 + se[key] ** 2)
            assert abs(actual[key] - ref) <= band, f"{case['name']}.{key}"
        return
    assert set(actual) == set(case["expect"])
    for key, want in case["expect"].items():
        assert abs(actual[key] - want) <= case["tol"] * max(1.0, abs(want)), f"{case['name']}.{key}"


def test_placement_golden_count() -> None:
    assert len(GOLDEN["cases"]) == 11


# ------------------------------------------------------------ validation


@pytest.mark.parametrize(
    "args,match",
    [
        ((-1, 1, 1.0, 0.0, 1.0), "q_ahead"),
        ((1.5, 1, 1.0, 0.0, 1.0), "q_ahead"),
        ((True, 1, 1.0, 0.0, 1.0), "q_ahead"),
        ((0, 0, 1.0, 0.0, 1.0), "size"),
        ((4999, 2, 1.0, 0.0, 1.0), "q_ahead \\+ size"),
        ((0, 1, 0.0, 0.0, 1.0), "mu"),
        ((0, 1, math.inf, 0.0, 1.0), "mu"),
        ((0, 1, 1.0, -0.1, 1.0), "theta"),
        ((0, 1, 1.0, math.nan, 1.0), "theta"),
        ((0, 1, 1.0, 0.0, -1.0), "horizon"),
        ((0, 1, 1.0, 0.0, math.inf), "horizon"),
        ((0, 1, 1e6, 0.0, 1.0), "horizon too long"),
        ((0, 1, 1e300, 0.0, 1e300), "horizon too long"),
    ],
)
def test_fill_probability_validation(args: tuple, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        pl.queue_fill_probability(*args)


@pytest.mark.parametrize(
    "idx,bad,match",
    [
        (0, 0.0, "size"),
        (0, math.inf, "size"),
        (1, -0.01, "spread"),
        (2, 1.1, "fill_prob"),
        (2, math.nan, "fill_prob"),
        (3, -1.0, "adverse_move"),
        (4, -1.0, "eta"),
        (5, 0.0, "alpha"),
        (5, 1.5, "alpha"),
        (6, 0.0, "weight"),
    ],
)
def test_decision_validation(idx: int, bad: float, match: str) -> None:
    args = [100.0, 0.01, 0.5, 0.01, 1e-4, 0.5, 0.1]
    args[idx] = bad
    with pytest.raises(ValueError, match=match):
        pl.place_or_cross(*args)


@pytest.mark.parametrize(
    "kw,match",
    [
        ({"total_units": 0}, "total_units"),
        ({"total_units": 1001}, "total_units"),
        ({"total_units": 2.0}, "total_units"),
        ({"n_periods": 0}, "n_periods"),
        ({"n_periods": 1001}, "n_periods"),
        ({"dt": 0.0}, "dt"),
        ({"sigma": -1.0}, "sigma"),
        ({"spread": math.nan}, "spread"),
        ({"kappa": 0.0}, "kappa"),
        ({"p_touch": 0.0}, "p_touch"),
        ({"p_touch": 1.01}, "p_touch"),
        ({"beta": -1.0}, "beta"),
        ({"phi": math.inf}, "phi"),
    ],
)
def test_liquidation_params_validation(kw: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _liq(**kw)


def test_simulation_validation() -> None:
    with pytest.raises(ValueError, match="n_paths"):
        pl.simulate_liquidation(_liq(), 1)
    with pytest.raises(ValueError, match="n_paths"):
        pl.simulate_liquidation(_liq(), 2.5)  # type: ignore[arg-type]
