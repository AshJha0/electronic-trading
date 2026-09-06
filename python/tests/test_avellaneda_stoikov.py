"""Avellaneda-Stoikov tests: closed-form quotes, fill mechanics, simulation
invariants (cap, determinism, A-S vs fixed spread), and error paths."""

from __future__ import annotations

import math

import numpy as np
import pytest

import etrade as et
from conftest import as_from_cfg, load_json

LIQ = load_json("params_liquid.json")


@pytest.fixture(scope="module")
def ap() -> et.ASParams:
    return as_from_cfg(LIQ)


def test_reservation_price_sign_conventions(ap: et.ASParams) -> None:
    s, t = 100.0, 0.25
    assert et.reservation_price(ap, s, 0.0, t) == s
    # long inventory => shade down (selling pressure), short => shade up
    assert et.reservation_price(ap, s, 5.0, t) < s
    assert et.reservation_price(ap, s, -5.0, t) > s
    skew = s - et.reservation_price(ap, s, 1.0, t)
    assert skew == pytest.approx(ap.gamma * ap.sigma ** 2 * (ap.horizon - t),
                                 rel=1e-12)


def test_spread_positive_decreasing_to_terminal_floor(ap: et.ASParams) -> None:
    floor = 2.0 / ap.gamma * math.log1p(ap.gamma / ap.k)
    prev = math.inf
    for t in np.linspace(0.0, ap.horizon, 21):
        sp = et.optimal_spread(ap, float(t))
        assert sp > 0.0
        assert sp >= floor - 1e-15
        assert sp < prev
        prev = sp
    assert et.optimal_spread(ap, ap.horizon) == pytest.approx(floor, rel=1e-12)


def test_terminal_time_behaviour(ap: et.ASParams) -> None:
    # T - t = 0: reservation collapses to the mid, spread to the floor.
    assert et.reservation_price(ap, 42.0, 7.0, ap.horizon) == 42.0
    with pytest.raises(ValueError):
        et.reservation_price(ap, 100.0, 0.0, ap.horizon + 0.01)
    with pytest.raises(ValueError):
        et.optimal_spread(ap, -0.01)


def test_quotes_centered_on_reservation(ap: et.ASParams) -> None:
    s, q, t = 100.0, 3.0, 0.3
    bid, ask = et.quotes(ap, s, q, t)
    r = et.reservation_price(ap, s, q, t)
    assert bid < ask
    assert 0.5 * (bid + ask) == pytest.approx(r, rel=1e-12)
    assert ask - bid == pytest.approx(et.optimal_spread(ap, t), rel=1e-12)


def test_fill_probability_pinned_formula(ap: et.ASParams) -> None:
    delta = 0.6
    lam = ap.a_rate * math.exp(-ap.k * delta)
    assert et.intensity(ap, delta) == pytest.approx(lam, rel=1e-14)
    p = et.fill_probability(ap, delta)
    assert p == pytest.approx(1.0 - math.exp(-lam * ap.dt), rel=1e-12)
    assert 0.0 < p < 1.0
    # decreasing in delta; negative delta (quote through mid) fills more often
    assert et.fill_probability(ap, 1.0) < p < et.fill_probability(ap, -0.5)
    assert et.fill_probability(ap, -0.5) < 1.0


@pytest.mark.parametrize("kwargs", [
    dict(gamma=0.0),           # gamma -> 0 limit is outside the model: floored
    dict(gamma=-0.1),
    dict(k=0.0),
    dict(a_rate=0.0),
    dict(sigma=-1.0),
    dict(dt=0.0),
    dict(dt=2.0),              # dt > horizon
    dict(dt=0.3),              # horizon not an integer multiple of dt
    dict(horizon=0.0),
    dict(q_max=-1),
    dict(eta_liq=-0.5),
])
def test_validation_errors(kwargs: dict) -> None:
    base = dict(s0=100.0, sigma=2.0, horizon=1.0, dt=0.005, gamma=0.1,
                k=1.5, a_rate=140.0, q_max=20, eta_liq=0.5)
    base.update(kwargs)
    with pytest.raises(ValueError):
        et.ASParams(**base)


def test_simulation_determinism(ap: et.ASParams) -> None:
    r1 = et.simulate(ap, n_paths=500, seed=11)
    r2 = et.simulate(ap, n_paths=500, seed=11)
    assert np.array_equal(r1.pnl, r2.pnl)
    assert r1.mean_fills == r2.mean_fills
    r3 = et.simulate(ap, n_paths=500, seed=12)
    assert not np.array_equal(r1.pnl, r3.pnl)


def test_inventory_cap_respected_every_step() -> None:
    p = et.ASParams(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, 2, 0.5)
    res = et.simulate(p, n_paths=800, seed=3)
    assert not res.cap_violated          # checked inside the loop, every step
    assert res.overall_max_abs_q <= 2.0
    assert res.mean_fills > 0.0          # one-sided quoting still trades


def test_qmax_zero_never_quotes(ap: et.ASParams) -> None:
    p = et.ASParams(ap.s0, ap.sigma, ap.horizon, ap.dt, ap.gamma, ap.k,
                    ap.a_rate, 0, ap.eta_liq)
    res = et.simulate(p, n_paths=50, seed=1)
    assert res.mean_fills == 0.0
    assert res.overall_max_abs_q == 0.0
    assert np.all(res.pnl == 0.0)


def test_as_beats_fixed_on_inventory(ap: et.ASParams) -> None:
    """A-S inventory control: RMS inventory well below the fixed-spread agent
    (same average spread, same RNG stream), with a statistical margin."""
    res_as = et.simulate(ap, n_paths=2000, seed=21, agent="as")
    res_fx = et.simulate(ap, n_paths=2000, seed=21, agent="fixed")
    assert res_as.rms_q < 0.8 * res_fx.rms_q
    assert np.std(res_as.max_abs_q_path) < np.std(res_fx.max_abs_q_path)
    # comparable gross activity: both agents actually trade
    assert res_as.mean_fills > 10.0 and res_fx.mean_fills > 10.0


def test_stress_double_vol_raises_pnl_dispersion(ap: et.ASParams) -> None:
    base = et.simulate(ap, n_paths=1500, seed=5)
    stress = et.simulate(ap, n_paths=1500, seed=5, stress_factor=2.0,
                         stress_time_frac=0.5)
    assert stress.std_pnl > base.std_pnl


def test_average_spread_matches_numeric_average(ap: et.ASParams) -> None:
    # The spread is linear in t, so its time average is the exact midpoint of
    # the endpoint values; a composite-trapezoid check (NumPy 1.x/2.x
    # compatible) is included as an independent numeric cross-check.
    exact_mid = 0.5 * (et.optimal_spread(ap, 0.0) + et.optimal_spread(ap, ap.horizon))
    assert et.average_as_spread(ap) == pytest.approx(exact_mid, rel=1e-13)
    ts = np.linspace(0.0, ap.horizon, 2001)
    vals = np.array([et.optimal_spread(ap, float(t)) for t in ts])
    numeric = float(np.sum(0.5 * (vals[1:] + vals[:-1]) * np.diff(ts))) / ap.horizon
    assert et.average_as_spread(ap) == pytest.approx(numeric, rel=1e-9)


def test_simulation_bad_args(ap: et.ASParams) -> None:
    with pytest.raises(ValueError):
        et.simulate(ap, n_paths=1, seed=0)
    with pytest.raises(ValueError):
        et.simulate(ap, n_paths=10, seed=0, agent="momentum")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        et.simulate(ap, n_paths=10, seed=0, agent="fixed", fixed_spread=-1.0)
    with pytest.raises(ValueError):
        et.simulate(ap, n_paths=10, seed=0, stress_factor=0.0)
    with pytest.raises(ValueError):
        res = et.simulate(ap, n_paths=10, seed=0)
        res.se("nope")


def test_sharpe_and_result_shapes(ap: et.ASParams) -> None:
    res = et.simulate(ap, n_paths=300, seed=2)
    assert res.n_paths == 300
    assert res.pnl.shape == (300,)
    assert res.sharpe == pytest.approx(res.mean_pnl / res.std_pnl, rel=1e-12)
    for stat in ("mean_pnl", "std_pnl", "mean_abs_q", "mean_max_abs_q",
                 "rms_q", "mean_fills"):
        assert res.se(stat) > 0.0


# ----------------------------------------------------------------------------
# Robustness tests (review findings CRIT-1, MAJ-8, MIN-16, T-8, T-11, T-14)
# ----------------------------------------------------------------------------

def _reference_loop(p: et.ASParams, n_paths: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Independent scalar re-implementation of the pinned loop (API_SPEC 2.2)
    returning (fills, pnl) per path.  Draw order: u_ask, u_bid, z per step,
    as vectors across paths."""
    rng = np.random.default_rng(seed)
    s = np.full(n_paths, p.s0)
    q = np.zeros(n_paths, dtype=int)
    cash = np.zeros(n_paths)
    fills = np.zeros(n_paths, dtype=int)
    markup = 2.0 / p.gamma * math.log1p(p.gamma / p.k)
    for i in range(p.n_steps):
        t = i * p.dt
        r = s - q * p.gamma * p.sigma ** 2 * (p.horizon - t)
        half = 0.5 * (p.gamma * p.sigma ** 2 * (p.horizon - t) + markup)
        ask, bid = r + half, r - half
        u_ask = rng.random(n_paths)
        u_bid = rng.random(n_paths)
        for m in range(n_paths):
            fa = q[m] > -p.q_max and u_ask[m] < -math.expm1(
                -p.a_rate * math.exp(-p.k * (ask[m] - s[m])) * p.dt)
            fb = q[m] < p.q_max and u_bid[m] < -math.expm1(
                -p.a_rate * math.exp(-p.k * (s[m] - bid[m])) * p.dt)
            if fa:
                q[m] -= 1
                cash[m] += ask[m]
                fills[m] += 1
            if fb:
                q[m] += 1
                cash[m] -= bid[m]
                fills[m] += 1
        s = s + p.sigma * math.sqrt(p.dt) * rng.standard_normal(n_paths)
    return fills, cash + q * s - p.eta_liq * np.abs(q)


def test_mean_fills_counts_both_sides(ap: et.ASParams) -> None:
    # 1. Saturated book: P(fill) = 1 - exp(-1e6*exp(-k*delta)*dt) rounds to 1
    #    on both sides at every step, so fills == 2*n_steps exactly.
    sat = et.ASParams(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, a_rate=1e6,
                      q_max=10_000, eta_liq=0.0)
    res = et.simulate(sat, n_paths=50, seed=1)
    assert res.mean_fills == 2.0 * sat.n_steps            # 400, not 200
    assert bool(np.all(res.fills_path == 2.0 * sat.n_steps))
    # 2. Liquid set: matches an independent scalar re-implementation exactly.
    fills_ref, pnl_ref = _reference_loop(ap, 60, 42)
    res = et.simulate(ap, n_paths=60, seed=42)
    assert np.array_equal(res.fills_path, fills_ref)
    assert np.allclose(res.pnl, pnl_ref, rtol=0.0, atol=1e-9)
    # both-side steps exist here (~4-5% of steps): sum > OR count
    assert res.mean_fills > 0.0


def test_fixed_spread_must_be_finite_and_positive(ap: et.ASParams) -> None:
    for bad in (float("inf"), float("nan"), 0.0, -1.0):
        with pytest.raises(ValueError):
            et.simulate(ap, n_paths=10, seed=0, agent="fixed", fixed_spread=bad)
    with pytest.raises(ValueError):
        et.simulate(ap, n_paths=10, seed=0, stress_time_frac=float("nan"))
    with pytest.raises(ValueError):
        et.simulate(ap, n_paths=10, seed=0, stress_factor=float("inf"))


def test_stress_edge_fractions(ap: et.ASParams) -> None:
    base = et.simulate(ap, n_paths=300, seed=9)
    # frac = 1.0: no step has t >= T (t = i*dt <= T - dt), so identical.
    late = et.simulate(ap, n_paths=300, seed=9, stress_factor=5.0, stress_time_frac=1.0)
    assert np.array_equal(base.pnl, late.pnl)
    # factor = 1.0 at any fraction: identical.
    unit = et.simulate(ap, n_paths=300, seed=9, stress_factor=1.0, stress_time_frac=0.0)
    assert np.array_equal(base.pnl, unit.pnl)
    # frac = 0.0 with factor 5: every step stressed, dispersion up.
    early = et.simulate(ap, n_paths=300, seed=9, stress_factor=5.0, stress_time_frac=0.0)
    assert early.std_pnl > base.std_pnl


def test_quotes_reject_non_finite_state(ap: et.ASParams) -> None:
    nan, inf = float("nan"), float("inf")
    with pytest.raises(ValueError):
        et.reservation_price(ap, nan, 0.0, 0.0)
    with pytest.raises(ValueError):
        et.reservation_price(ap, 100.0, inf, 0.0)
    with pytest.raises(ValueError):
        et.quotes(ap, 100.0, inf, 0.0)
    with pytest.raises(ValueError):
        et.quotes(ap, -inf, 0.0, 0.0)
    with pytest.raises(ValueError):
        et.reservation_price(ap, 100.0, 0.0, nan)
    with pytest.raises(ValueError):
        et.fill_probability(ap, nan)


def test_n_steps_integrality_float() -> None:
    p3 = et.ASParams(100.0, 2.0, 1.0, 1.0 / 3.0, 0.1, 1.5, 140.0, 20, 0.5)
    assert p3.n_steps == 3
    with pytest.raises(ValueError):
        et.ASParams(100.0, 2.0, 1.0, 0.3, 0.1, 1.5, 140.0, 20, 0.5)
    p1 = et.ASParams(100.0, 2.0, 1.0, 1.0, 0.1, 1.5, 140.0, 20, 0.5)   # dt = T
    assert p1.n_steps == 1
    res = et.simulate(p1, n_paths=200, seed=4)
    # one step: the time average of |q| is the single post-update value
    assert np.array_equal(res.mean_abs_q_path, res.max_abs_q_path)
    assert np.array_equal(res.rms_q_path, res.max_abs_q_path)
    assert bool(np.all(res.fills_path <= 2.0))
    assert res.mean_abs_q == res.mean_max_abs_q


def test_step_and_path_bounds_are_input_errors(ap: et.ASParams) -> None:
    with pytest.raises(ValueError):
        et.ASParams(100.0, 2.0, 1.0, 1.0 / (et.MAX_AS_STEPS * 10), 0.1, 1.5, 140.0, 20, 0.5)
    with pytest.raises(ValueError):
        et.simulate(ap, n_paths=et.MAX_PATHS + 1, seed=0)
    with pytest.raises(ValueError):
        et.simulate(ap, n_paths=10.0, seed=0)   # type: ignore[arg-type]
    with pytest.raises(ValueError):
        et.ASParams(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, True, 0.5)
    assert et.ASParams(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, np.int64(3), 0.5).q_max == 3


def test_fills_bounded_by_two_per_step_and_inventory_consistent(ap: et.ASParams) -> None:
    res = et.simulate(ap, n_paths=400, seed=8)
    assert bool(np.all(res.fills_path <= 2.0 * ap.n_steps))
    assert bool(np.all(res.fills_path >= res.max_abs_q_path))   # |q| needs fills
    assert bool(np.all(res.fills_path == np.round(res.fills_path)))
    assert res.se("mean_fills") > 0.0
