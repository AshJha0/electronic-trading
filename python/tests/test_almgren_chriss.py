"""Almgren-Chriss tests: trajectory shape, closed forms vs MC, baselines,
limit branches, and error paths."""

from __future__ import annotations

import math

import numpy as np
import pytest

import etrade as et
from conftest import ac_from_cfg, load_json

LIQ = load_json("params_liquid.json")
ILQ = load_json("params_illiquid.json")


@pytest.fixture(scope="module")
def p_liq() -> et.ACParams:
    return ac_from_cfg(LIQ)


def test_trajectory_boundary_monotone_and_sum(p_liq: et.ACParams) -> None:
    x = et.ac_trajectory(p_liq)
    n = et.ac_trade_list(p_liq)
    assert x.size == p_liq.n_intervals + 1
    assert x[0] == pytest.approx(p_liq.total_shares, abs=0.0)
    assert x[-1] == 0.0
    assert np.all(np.diff(x) < 0.0)  # strictly decreasing sell schedule
    assert np.all(n > 0.0)
    assert float(np.sum(n)) == pytest.approx(p_liq.total_shares, rel=1e-12)


def test_kappa_solves_discrete_equation(p_liq: et.ACParams) -> None:
    kappa = et.ac_kappa(p_liq)
    kt2 = p_liq.lambda_ra * p_liq.sigma ** 2 / p_liq.eta_tilde
    lhs = 2.0 / p_liq.tau ** 2 * (math.cosh(kappa * p_liq.tau) - 1.0)
    assert lhs == pytest.approx(kt2, rel=1e-12)


def test_more_risk_aversion_front_loads_and_moves_frontier(p_liq: et.ACParams) -> None:
    """Property sweep: higher lambda_ra => faster liquidation (smaller interior
    holdings), strictly lower Var and higher E (grid loop)."""
    lambdas = [1e-8, 1e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5]
    prev_x = prev_e = prev_v = None
    for lam in lambdas:
        p = p_liq.with_lambda(lam)
        x = et.ac_trajectory(p)
        n = et.ac_trade_list(p)
        assert np.all(np.diff(x) < 0.0)
        assert float(np.sum(n)) == pytest.approx(p.total_shares, rel=1e-12)
        e, v = et.expected_cost(p), et.variance_cost(p)
        if prev_x is not None:
            assert np.all(x[1:-1] < prev_x[1:-1])  # front-loaded
            assert e > prev_e
            assert v < prev_v
        prev_x, prev_e, prev_v = x, e, v


def test_lambda_zero_is_exact_twap(p_liq: et.ACParams) -> None:
    p0 = p_liq.with_lambda(0.0)
    twap = et.twap_holdings(p0.total_shares, p0.n_intervals)
    assert np.max(np.abs(et.ac_trajectory(p0) - twap)) <= 1e-12 * p0.total_shares
    assert et.ac_kappa(p0) == 0.0
    assert math.isinf(et.half_life(p0))


def test_kappa_limit_continuity_near_threshold(p_liq: et.ACParams) -> None:
    """The sinh branch just above kappa*T = 1e-6 must agree with the linear
    limit branch to high accuracy (branch continuity)."""
    # choose lambda so that kappa*T is a hair above the 1e-6 threshold
    target_kappa = 2e-6 / p_liq.horizon
    kt2 = 2.0 / p_liq.tau ** 2 * (math.cosh(target_kappa * p_liq.tau) - 1.0)
    lam = kt2 * p_liq.eta_tilde / p_liq.sigma ** 2
    p = p_liq.with_lambda(lam)
    assert et.ac_kappa(p) * p.horizon > et.KAPPA_T_LIMIT  # sinh branch active
    twap = et.twap_holdings(p.total_shares, p.n_intervals)
    assert np.max(np.abs(et.ac_trajectory(p) - twap)) <= 1e-8 * p.total_shares


def test_closed_forms_match_monte_carlo(p_liq: et.ACParams) -> None:
    e, v = et.expected_cost(p_liq), et.variance_cost(p_liq)
    mc = et.simulate_is(p_liq, n_paths=20_000, seed=9)
    assert abs(mc.mean - e) <= 4.0 * mc.se_mean
    assert abs(mc.var - v) <= 4.0 * mc.se_var


def test_single_clip_n1() -> None:
    p = et.ACParams(1e5, 1, 1.0, 0.9, 2e-6, 1e-7, 1e-6)
    x = et.ac_trajectory(p)
    assert x.tolist() == [1e5, 0.0]
    e, v = et.instantaneous_cost(p)
    assert et.expected_cost(p) == pytest.approx(e, rel=1e-12)
    assert et.variance_cost(p) == pytest.approx(v, abs=1e-12)


def test_zero_shares_empty_trajectory(p_liq: et.ACParams) -> None:
    p = et.ACParams(0.0, p_liq.n_intervals, p_liq.tau, p_liq.sigma,
                    p_liq.eta, p_liq.gamma_p, p_liq.lambda_ra)
    assert np.all(et.ac_trajectory(p) == 0.0)
    assert et.expected_cost(p) == 0.0
    assert et.variance_cost(p) == 0.0


@pytest.mark.parametrize("kwargs", [
    dict(total_shares=-1.0),
    dict(n_intervals=0),
    dict(tau=0.0),
    dict(tau=-1.0),
    dict(sigma=-0.1),
    dict(eta=0.0),
    dict(eta=-1e-6),
    dict(gamma_p=-1e-8),
    dict(lambda_ra=-1e-6),
    dict(total_shares=float("nan")),
])
def test_invalid_params_raise(kwargs: dict) -> None:
    base = dict(total_shares=1e6, n_intervals=5, tau=1.0, sigma=0.95,
                eta=2.5e-6, gamma_p=2.5e-7, lambda_ra=1e-6)
    base.update(kwargs)
    with pytest.raises(ValueError):
        et.ACParams(**base)


def test_model_validity_gamma_p_tau() -> None:
    """gamma_p*tau >= 2*eta => eta_tilde <= 0 => invalid per paper footnote."""
    with pytest.raises(ValueError, match="gamma_p"):
        et.ACParams(1e6, 5, 1.0, 0.95, 2.5e-6, 5.0e-6, 1e-6)  # equality case
    with pytest.raises(ValueError, match="gamma_p"):
        et.ACParams(1e6, 5, 1.0, 0.95, 2.5e-6, 6.0e-6, 1e-6)


def test_twap_closed_formulas(p_liq: et.ACParams) -> None:
    x0, n_int, tau = p_liq.total_shares, p_liq.n_intervals, p_liq.tau
    twap = et.twap_holdings(x0, n_int)
    e_expect = 0.5 * p_liq.gamma_p * x0 ** 2 + p_liq.eta_tilde * x0 ** 2 / (n_int * tau)
    v_expect = (p_liq.sigma ** 2 * tau * x0 ** 2
                * (n_int - 1) * (2 * n_int - 1) / (6 * n_int))
    assert et.expected_cost(p_liq, twap) == pytest.approx(e_expect, rel=1e-12)
    assert et.variance_cost(p_liq, twap) == pytest.approx(v_expect, rel=1e-12)


def test_pov_schedule(p_liq: et.ACParams) -> None:
    v_mkt, rate = LIQ["pov_market_volume_rate"], LIQ["pov_participation"]
    x = et.pov_holdings(p_liq.total_shares, p_liq.tau, v_mkt, rate)
    n = et.trades_from_holdings(x)
    clip = rate * v_mkt * p_liq.tau
    assert np.all(n <= clip * (1.0 + 1e-12))
    assert np.all(n[:-1] == pytest.approx(clip, rel=1e-12))  # full clips first
    assert float(np.sum(n)) == pytest.approx(p_liq.total_shares, rel=1e-12)
    assert x[-1] == 0.0
    with pytest.raises(ValueError):
        et.pov_holdings(1e6, 1.0, 1e6, 1.5)  # participation out of (0,1)
    with pytest.raises(ValueError):
        et.pov_holdings(1e6, 1.0, 0.0, 0.1)


def test_instantaneous_liquidation(p_liq: et.ACParams) -> None:
    e, v = et.instantaneous_cost(p_liq)
    x0 = p_liq.total_shares
    assert v == 0.0
    assert e == pytest.approx(0.5 * p_liq.gamma_p * x0 ** 2
                              + p_liq.eta_tilde * x0 ** 2 / p_liq.tau, rel=1e-12)
    # instantaneous cost dominates any optimal E for these params
    assert e > et.expected_cost(p_liq)


def test_half_life_decreasing_in_lambda(p_liq: et.ACParams) -> None:
    hls = [et.half_life(p_liq.with_lambda(lam)) for lam in (1e-7, 1e-6, 1e-5)]
    assert hls[0] > hls[1] > hls[2] > 0.0


def test_efficient_frontier_consistency(p_liq: et.ACParams) -> None:
    grid = [1e-8, 1e-7, 1e-6, 1e-5]
    front = et.efficient_frontier(p_liq, grid)
    assert [f.lambda_ra for f in front] == grid
    for f in front:
        p = p_liq.with_lambda(f.lambda_ra)
        assert f.e_cost == pytest.approx(et.expected_cost(p), rel=1e-12)
        assert f.var_cost == pytest.approx(et.variance_cost(p), rel=1e-12)
        assert f.std_cost == pytest.approx(math.sqrt(f.var_cost), rel=1e-12)
    with pytest.raises(ValueError):
        et.efficient_frontier(p_liq, [])


def test_illiquid_set_also_valid() -> None:
    p = ac_from_cfg(ILQ)
    n = et.ac_trade_list(p)
    assert float(np.sum(n)) == pytest.approx(p.total_shares, rel=1e-12)
    assert et.variance_cost(p) > 0.0
