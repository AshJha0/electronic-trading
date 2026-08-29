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
    ts = np.linspace(0.0, ap.horizon, 200_001)
    numeric = float(np.trapezoid([et.optimal_spread(ap, float(t)) for t in ts],
                                 ts)) / ap.horizon
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
