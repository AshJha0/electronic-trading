"""Analytics tests: IS decomposition identity and pinned values,
participation rate, native OLS, impact regression."""

from __future__ import annotations

import numpy as np
import pytest

import etrade as et


def test_is_sell_pinned_example() -> None:
    d = et.decompose_shortfall("sell", 100.0, 99.5, 98.8, 1000.0,
                               [300.0, 400.0, 200.0], [99.4, 99.2, 99.1])
    assert d.delay == pytest.approx(450.0, abs=1e-10)
    assert d.trading == pytest.approx(230.0, abs=1e-10)
    assert d.opportunity == pytest.approx(120.0, abs=1e-10)
    assert d.total == pytest.approx(800.0, abs=1e-10)
    assert d.executed_qty == 900.0
    assert d.remaining_qty == 100.0
    # identity is exact
    assert d.total == pytest.approx(d.delay + d.trading + d.opportunity, abs=1e-12)


def test_is_buy_mirrors_sell() -> None:
    d = et.decompose_shortfall("buy", 100.0, 100.5, 101.2, 1000.0,
                               [300.0, 400.0, 200.0], [100.6, 100.8, 100.9])
    assert d.delay == pytest.approx(450.0, abs=1e-10)
    assert d.trading == pytest.approx(230.0, abs=1e-10)
    assert d.opportunity == pytest.approx(120.0, abs=1e-10)
    assert d.total == pytest.approx(800.0, abs=1e-10)


def test_is_identity_property_grid() -> None:
    """Property-style loop: total == delay + trading + opportunity and total
    matches its from-scratch definition for a grid of scenarios."""
    for side in ("buy", "sell"):
        for pd_ in (95.0, 100.0, 105.0):
            for pt in (90.0, 100.0, 110.0):
                qtys = [100.0, 250.0, 150.0]
                prices = [pd_ - 0.4, pd_ + 0.1, pd_ - 0.2]
                d = et.decompose_shortfall(side, pd_, pd_ - 0.3, pt, 700.0,
                                           qtys, prices)
                assert d.total == pytest.approx(
                    d.delay + d.trading + d.opportunity, abs=1e-10)
                filled = sum(q * p for q, p in zip(qtys, prices))
                r = 700.0 - sum(qtys)
                direct = (700.0 * pd_ - filled - r * pt if side == "sell"
                          else filled + r * pt - 700.0 * pd_)
                assert d.total == pytest.approx(direct, abs=1e-10)


def test_is_fully_executed_has_no_opportunity_cost() -> None:
    d = et.decompose_shortfall("sell", 100.0, 100.0, 90.0, 500.0,
                               [500.0], [99.0])
    assert d.opportunity == 0.0
    assert d.remaining_qty == 0.0


def test_is_validation() -> None:
    with pytest.raises(ValueError):
        et.decompose_shortfall("hold", 100.0, 99.5, 98.8, 100.0, [50.0], [99.0])
    with pytest.raises(ValueError):  # over-execution
        et.decompose_shortfall("sell", 100.0, 99.5, 98.8, 100.0,
                               [80.0, 40.0], [99.0, 99.1])
    with pytest.raises(ValueError):  # mismatched arrays
        et.decompose_shortfall("sell", 100.0, 99.5, 98.8, 100.0, [50.0],
                               [99.0, 99.1])
    with pytest.raises(ValueError):  # negative fill
        et.decompose_shortfall("sell", 100.0, 99.5, 98.8, 100.0, [-10.0], [99.0])
    with pytest.raises(ValueError):  # non-finite price
        et.decompose_shortfall("sell", float("inf"), 99.5, 98.8, 100.0,
                               [10.0], [99.0])


def test_participation_rate() -> None:
    assert et.participation_rate(120000.0, 1.5e6) == pytest.approx(0.08, abs=1e-15)
    assert et.participation_rate(0.0, 1.0) == 0.0
    with pytest.raises(ValueError):
        et.participation_rate(-1.0, 100.0)
    with pytest.raises(ValueError):
        et.participation_rate(10.0, 0.0)


def test_ols_recovers_exact_line() -> None:
    x = np.linspace(0.0, 1.0, 20)
    y = 3.5 * x - 1.25
    r = et.ols(x, y)
    assert r.slope == pytest.approx(3.5, abs=1e-12)
    assert r.intercept == pytest.approx(-1.25, abs=1e-12)
    assert r.r_squared == pytest.approx(1.0, abs=1e-12)
    assert r.n_obs == 20


def test_ols_validation() -> None:
    with pytest.raises(ValueError):
        et.ols([1.0], [2.0])                  # too few points
    with pytest.raises(ValueError):
        et.ols([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])  # degenerate x
    with pytest.raises(ValueError):
        et.ols([1.0, 2.0], [1.0])             # length mismatch
    with pytest.raises(ValueError):
        et.ols([1.0, float("nan")], [1.0, 2.0])


def test_impact_regression_recovers_slope() -> None:
    part, cost = et.synthetic_impact_fills(25, 0.5, 8.0, 0.05)
    r = et.estimate_impact(part, cost)
    assert abs(r.slope - 8.0) < 0.5           # deterministic noise, close fit
    assert abs(r.intercept - 0.5) < 0.05
    assert r.r_squared > 0.98
    # deterministic construction: identical on repeat, exact with zero noise
    part2, cost2 = et.synthetic_impact_fills(25, 0.5, 8.0, 0.05)
    assert np.array_equal(part, part2) and np.array_equal(cost, cost2)
    p0, c0 = et.synthetic_impact_fills(10, 0.2, 5.0, 0.0)
    r0 = et.estimate_impact(p0, c0)
    assert r0.slope == pytest.approx(5.0, abs=1e-10)
    assert r0.intercept == pytest.approx(0.2, abs=1e-10)


# ----------------------------------------------------------------------------
# Robustness tests (T-9, T-13)
# ----------------------------------------------------------------------------

def test_is_decomposition_price_shift_invariance() -> None:
    """Adding a constant c to every price leaves all components unchanged
    (delay + trading + opportunity involve only price differences; the total
    identity uses Q + R = X)."""
    c = 37.5
    qtys, prices = [300.0, 400.0, 200.0], [99.4, 99.2, 99.1]
    for side in ("sell", "buy"):
        d0 = et.decompose_shortfall(side, 100.0, 99.5, 98.8, 1000.0, qtys, prices)
        d1 = et.decompose_shortfall(side, 100.0 + c, 99.5 + c, 98.8 + c, 1000.0,
                                    qtys, [p + c for p in prices])
        for name in ("total", "delay", "trading", "opportunity"):
            assert getattr(d1, name) == pytest.approx(getattr(d0, name), abs=1e-9)


def test_is_decomposition_empty_fills() -> None:
    d = et.decompose_shortfall("sell", 100.0, 99.0, 98.0, 500.0, [], [])
    assert d.delay == 0.0 and d.trading == 0.0
    assert d.opportunity == pytest.approx(1000.0, abs=1e-12)   # 500 * (100 - 98)
    assert d.total == pytest.approx(1000.0, abs=1e-12)
    assert d.executed_qty == 0.0 and d.remaining_qty == 500.0
    b = et.decompose_shortfall("buy", 100.0, 99.0, 98.0, 500.0, [], [])
    assert b.opportunity == pytest.approx(-1000.0, abs=1e-12)  # price fell: a gain


def test_participation_rate_above_one_is_error() -> None:
    with pytest.raises(ValueError, match="exceeds market volume"):
        et.participation_rate(2e6, 1.5e6)
    assert et.participation_rate(1.5e6, 1.5e6) == 1.0      # boundary allowed
    with pytest.raises(ValueError):
        et.participation_rate(float("nan"), 1.0)
    with pytest.raises(ValueError):
        et.participation_rate(1.0, float("inf"))


def test_synthetic_fills_reject_bool_and_accept_numpy_int() -> None:
    with pytest.raises(ValueError):
        et.synthetic_impact_fills(True)
    part, cost = et.synthetic_impact_fills(np.int64(4))
    assert part.size == 4 and cost.size == 4
