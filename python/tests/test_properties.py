"""Property-based tests (hypothesis): invariants that must hold for every
valid input, not only the bundled parameter sets, and validation that never
fails with anything but ValueError on arbitrary finite or non-finite input."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

import etrade as et
from etrade import impact as im
from etrade import market as mk

SETTINGS = settings(max_examples=150, deadline=None)


def pos(lo: float, hi: float) -> st.SearchStrategy[float]:
    return st.floats(lo, hi, allow_nan=False, allow_infinity=False)


@st.composite
def ac_params(draw: st.DrawFn) -> et.ACParams:
    tau = draw(pos(1e-3, 10.0))
    eta = draw(pos(1e-8, 1e-3))
    return et.ACParams(
        total_shares=draw(pos(0.0, 1e8)),
        n_intervals=draw(st.integers(1, 200)),
        tau=tau,
        sigma=draw(pos(0.0, 10.0)),
        eta=eta,
        gamma_p=draw(pos(0.0, 0.99)) * 2.0 * eta / tau,
        lambda_ra=draw(pos(0.0, 1e-3)),
    )


def assert_liquidating_path(x: np.ndarray, total: float) -> None:
    # x_0 == X up to one rounding (the linear branch forms X*N/N).
    assert abs(x[0] - total) <= 1e-15 * total
    assert x[-1] == 0.0
    scale = max(1.0, total)
    assert np.all(np.diff(x) <= 1e-12 * scale)
    assert abs(float(np.sum(-np.diff(x))) - total) <= 1e-9 * scale


@SETTINGS
@given(ac_params())
def test_ac_trajectory_is_monotone_and_sums_to_x(p: et.ACParams) -> None:
    x = et.ac_trajectory(p)
    assert x.size == p.n_intervals + 1
    assert np.all(np.isfinite(x))
    assert_liquidating_path(x, p.total_shares)


@SETTINGS
@given(pos(0.0, 1e9), st.integers(1, 500))
def test_twap_is_monotone_and_sums_to_x(total: float, n: int) -> None:
    assert_liquidating_path(et.twap_holdings(total, n), total)


@SETTINGS
@given(
    pos(1e4, 2e6),
    st.integers(2, 26),
    pos(0.3, 1.0),
    pos(0.0, 1e-5),
    pos(0.0, 100.0),
    st.booleans(),
)
def test_impact_schedule_is_monotone_and_sums_to_x(
    total: float, n: int, alpha: float, lam: float, drift_bps: float, curve: bool
) -> None:
    weights = im.US_EQUITY_VOLUME_CURVE_13 if curve and n == 13 else None
    p = mk.impact_params_from_market(
        price=100.0,
        spread=0.01,
        adv=5e6,
        daily_vol=0.015,
        total_shares=total,
        n_intervals=n,
        tau=1.0 / n,
        lambda_ra=lam,
        alpha=alpha,
        drift_bps_per_day=drift_bps,
        weights=weights,
    )
    r = im.optimal_schedule(p)
    x = np.asarray(r.holdings, dtype=float)
    assert x.size == n + 1
    assert abs(x[0] - total) <= 1e-12 * total
    assert x[-1] == 0.0
    assert np.all(np.diff(x) <= 1e-12 * total)
    assert np.all(x[r.horizon :] == 0.0)


prices = pos(-1e4, 1e4)
qty = pos(0.0, 1e6)


@SETTINGS
@given(
    st.sampled_from(["buy", "sell"]),
    prices,
    prices,
    prices,
    st.lists(st.tuples(qty, prices), min_size=0, max_size=8),
    pos(0.0, 1e6),
    pos(-1e3, 1e3),
)
def test_is_decomposition_identity_and_price_shift_invariance(
    side: str,
    pd: float,
    p0: float,
    pt: float,
    fills: list[tuple[float, float]],
    extra: float,
    shift: float,
) -> None:
    n = [f[0] for f in fills]
    px = [f[1] for f in fills]
    total_qty = sum(n) + extra
    d = et.decompose_shortfall(side, pd, p0, pt, total_qty, n, px)
    sign = 1.0 if side == "sell" else -1.0
    scale = max(1.0, total_qty * 2e4)
    assert abs(d.total - (d.delay + d.trading + d.opportunity)) <= 1e-12 * scale
    closed = sign * (total_qty * pd - sum(a * b for a, b in fills) - d.remaining_qty * pt)
    assert abs(d.total - closed) <= 1e-9 * scale
    s = et.decompose_shortfall(
        side, pd + shift, p0 + shift, pt + shift, total_qty, n, [v + shift for v in px]
    )
    tol = 1e-9 * max(1.0, total_qty * 4e4)
    for a, b in (
        (d.total, s.total),
        (d.delay, s.delay),
        (d.trading, s.trading),
        (d.opportunity, s.opportunity),
    ):
        assert abs(a - b) <= tol


@SETTINGS
@given(pos(1e-4, 0.2), pos(0.0, 1e7), pos(0.0, 1e7), pos(1e3, 1e9))
def test_impact_cost_is_non_decreasing_in_order_size(
    vol: float, q1: float, q2: float, adv: float
) -> None:
    lo, hi = min(q1, q2), max(q1, q2)
    # Currency cost per unit notional: bps * shares.
    assert lo * im.sqrt_law_cost_bps(vol, lo, adv) <= hi * im.sqrt_law_cost_bps(vol, hi, adv)
    p_lo = et.ACParams(lo, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 0.0)
    p_hi = et.ACParams(hi, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 0.0)
    assert et.expected_cost(p_lo) <= et.expected_cost(p_hi)
    assert et.instantaneous_cost(p_lo)[0] <= et.instantaneous_cost(p_hi)[0]


@SETTINGS
@given(pos(-1e9, 1e9), pos(-1e9, 1e9), pos(1e-3, 1e12), pos(-100.0, 100.0))
def test_bps_is_linear(a: float, b: float, notional: float, k: float) -> None:
    def close(x: float, y: float) -> bool:
        return abs(x - y) <= 1e-12 * max(1.0, abs(x), abs(y), 1e4 * (abs(a) + abs(b)) / notional)

    assert close(mk.bps(a + b, notional), mk.bps(a, notional) + mk.bps(b, notional))
    assert close(mk.bps(k * a, notional), k * mk.bps(a, notional))


@SETTINGS
@given(
    pos(1.0, 1000.0),
    pos(1e-4, 1.0),
    pos(1e4, 1e9),
    pos(1e-3, 0.1),
    pos(0.001, 1.0),
    pos(0.001, 1.0),
)
def test_ac_conversion_reproduces_its_definitions(
    price: float, spread: float, adv: float, vol: float, temp: float, perm: float
) -> None:
    eta_bound = spread / (temp * adv)
    gp = spread / (perm * adv)
    if gp * 0.01 >= 2.0 * eta_bound:
        return  # model-invalid combination: rejected, covered by validation fuzz
    p = mk.ac_params_from_market(
        price=price,
        spread=spread,
        adv=adv,
        daily_vol=vol,
        total_shares=1e5,
        n_intervals=10,
        tau=0.01,
        lambda_ra=1e-6,
        temp_frac=temp,
        perm_frac=perm,
    )
    assert math.isclose(p.sigma, price * vol, rel_tol=1e-15)
    assert math.isclose(p.eta, eta_bound, rel_tol=1e-15)
    assert math.isclose(p.gamma_p, gp, rel_tol=1e-15)


@SETTINGS
@given(
    pos(1.0, 1000.0),
    pos(1e-4, 1.0),
    pos(1e-3, 0.1),
    pos(1.0, 1e5),
    pos(0.1, 50.0),
    pos(0.01, 10.0),
)
def test_as_conversion_reproduces_its_defining_properties(
    price: float, tick: float, vol: float, touch: float, halving: float, skew: float
) -> None:
    p = mk.as_params_from_market(
        price=price,
        tick=tick,
        daily_vol=vol,
        touch_fills_per_day=touch,
        halving_ticks=halving,
        skew_ticks=skew,
        q_max=5,
        liq_ticks=1.0,
        dt_seconds=60.0,
    )
    assert math.isclose(et.intensity(p, tick / 2.0), touch, rel_tol=1e-9)
    ratio = et.intensity(p, tick / 2.0 + halving * tick) / et.intensity(p, tick / 2.0)
    assert math.isclose(ratio, 0.5, rel_tol=1e-9)
    skew_px = price - et.reservation_price(p, price, 1.0, 0.0)
    # The difference cancels against the price: compare on the price scale.
    assert abs(skew_px - skew * tick) <= 1e-12 * price


# ------------------------------------------------- validation never crashes

anything = st.one_of(
    st.floats(allow_nan=True, allow_infinity=True),
    st.sampled_from([0.0, -0.0, -1.0, 1.0, 0.5, 1e-300, 1e300, math.nan, math.inf, -math.inf]),
    st.integers(-5, 10**8),
)


def only_value_error(fn: Callable[[], object]) -> None:
    try:
        fn()
    except ValueError:
        pass


@SETTINGS
@given(st.lists(anything, min_size=7, max_size=7))
def test_ac_params_validation_only_raises_value_error(v: list[float]) -> None:
    only_value_error(lambda: et.ACParams(*v))


@SETTINGS
@given(st.lists(anything, min_size=9, max_size=9))
def test_as_params_validation_only_raises_value_error(v: list[float]) -> None:
    only_value_error(lambda: et.ASParams(*v))


@SETTINGS
@given(st.lists(anything, min_size=4, max_size=4))
def test_schedules_validation_only_raises_value_error(v: list[float]) -> None:
    only_value_error(lambda: et.pov_holdings(*v))
    if not (isinstance(v[1], int) and v[1] > 10_000):  # keep valid TWAPs small
        only_value_error(lambda: et.twap_holdings(v[0], v[1]))  # type: ignore[arg-type]


@SETTINGS
@given(st.lists(anything, min_size=6, max_size=6), st.sampled_from(["buy", "sell", "hold"]))
def test_decomposition_validation_only_raises_value_error(v: list[float], side: str) -> None:
    only_value_error(lambda: et.decompose_shortfall(side, v[0], v[1], v[2], v[3], [v[4]], [v[5]]))


@SETTINGS
@given(st.lists(anything, min_size=10, max_size=10))
def test_market_constructors_only_raise_value_error(v: list[float]) -> None:
    only_value_error(lambda: mk.bps(v[0], v[1]))
    only_value_error(lambda: mk.half_spread_cost(v[0], v[1]))
    only_value_error(
        lambda: mk.ac_params_from_market(
            price=v[0],
            spread=v[1],
            adv=v[2],
            daily_vol=v[3],
            total_shares=v[4],
            n_intervals=v[5],  # type: ignore[arg-type]
            tau=v[6],
            lambda_ra=v[7],
            temp_frac=v[8],
            perm_frac=v[9],
        )
    )
    only_value_error(
        lambda: mk.as_params_from_market(
            price=v[0],
            tick=v[1],
            daily_vol=v[2],
            touch_fills_per_day=v[3],
            halving_ticks=v[4],
            skew_ticks=v[5],
            q_max=v[6],  # type: ignore[arg-type]
            liq_ticks=v[7],
            dt_seconds=v[8],
            session_seconds=v[9],
        )
    )
    only_value_error(
        lambda: im.ImpactParams(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8])  # type: ignore[arg-type]
    )
