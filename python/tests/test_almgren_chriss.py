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


# ----------------------------------------------------------------------------
# Robustness / literature tests (review findings MAJ-1..MAJ-5, MIN-3/4/5/15)
# ----------------------------------------------------------------------------

FX = load_json("params_fx.json")


@pytest.mark.parametrize("cfg,n_int,tau,lam", [
    (LIQ, 1000, 1.0, 2e-6),       # kappa*T ~ 846  (sinh overflows at ~710)
    (FX, 20000, None, None),      # kappa*T ~ 2683
])
def test_trajectory_finite_for_large_kappa_t(cfg: dict, n_int: int, tau, lam) -> None:
    """The exp/expm1 ratio form never overflows: every x_j finite,
    non-increasing (strictly decreasing while > 0), trades sum to X and the
    first ratio x_1/X equals exp(-kappa*tau) (sinh(k(T-tau))/sinh(kT) ->
    exp(-k tau) for kappa*T >> 1)."""
    p = et.ACParams(cfg["ac_total_shares"], n_int,
                    cfg["ac_tau"] if tau is None else tau, cfg["ac_sigma"],
                    cfg["ac_eta"], cfg["ac_gamma_p"],
                    cfg["ac_lambda_ra"] if lam is None else lam)
    kappa = et.ac_kappa(p)
    assert kappa * p.horizon > 710.0            # would overflow math.sinh
    x = et.ac_trajectory(p)
    assert x.size == n_int + 1
    assert bool(np.all(np.isfinite(x)))
    assert x[0] == p.total_shares and x[-1] == 0.0
    assert bool(np.all(np.diff(x) <= 0.0))
    pos = x > np.finfo(float).tiny * x[0]        # normal-range entries
    assert pos.sum() >= 800
    assert bool(np.all(np.diff(x[pos]) < 0.0))  # strictly decreasing while normal
    assert float(np.sum(et.ac_trade_list(p))) == pytest.approx(p.total_shares, rel=1e-12)
    assert x[1] / x[0] == pytest.approx(math.exp(-kappa * p.tau), rel=1e-9)
    assert math.isfinite(et.expected_cost(p)) and math.isfinite(et.variance_cost(p))
    for fp in et.efficient_frontier(p, [1e-8, 1e-6, 1e-4, 1.0]):
        assert math.isfinite(fp.e_cost) and math.isfinite(fp.std_cost)
        assert math.isfinite(fp.kappa)


@pytest.mark.parametrize("cfg", [LIQ, ILQ, FX])
def test_paper_closed_forms_match_discrete_sums(cfg: dict) -> None:
    """Almgren-Chriss (2000) eqs. for E and V of the optimal strategy (with
    the discrete kappa) are the exact discrete sums in closed form."""
    p = ac_from_cfg(cfg)
    k, big_t, tau, x0 = et.ac_kappa(p), p.horizon, p.tau, p.total_shares
    sh_kt = math.sinh(k * big_t)
    e_paper = (0.5 * p.gamma_p * x0 ** 2
               + p.eta_tilde * x0 ** 2 * math.tanh(0.5 * k * tau)
               * (tau * math.sinh(2 * k * big_t) + 2 * big_t * math.sinh(k * tau))
               / (2 * tau ** 2 * sh_kt ** 2))
    v_paper = (0.5 * p.sigma ** 2 * x0 ** 2
               * (tau * sh_kt * math.cosh(k * (big_t - tau)) - big_t * math.sinh(k * tau))
               / (sh_kt ** 2 * math.sinh(k * tau)))
    assert et.expected_cost(p) == pytest.approx(e_paper, rel=1e-12)
    assert et.variance_cost(p) == pytest.approx(v_paper, rel=1e-12)


@pytest.mark.parametrize("lam", [1e-6, 1e-12])
def test_kappa_small_tau_accuracy(lam: float) -> None:
    """kappa = kappa_tilde*(1 - kappa_tilde^2 tau^2/24 + O(tau^4)); the log1p
    form reproduces the series to 1e-9 where acosh(1+x) loses 4-5 digits."""
    p = et.ACParams(LIQ["ac_total_shares"], 5000, 1e-3, LIQ["ac_sigma"],
                    LIQ["ac_eta"], LIQ["ac_gamma_p"], lam)
    kt2 = lam * p.sigma ** 2 / p.eta_tilde
    series = math.sqrt(kt2) * (1.0 - kt2 * p.tau ** 2 / 24.0)
    assert et.ac_kappa(p) == pytest.approx(series, rel=1e-9)
    assert et.half_life(p) == pytest.approx(1.0 / series, rel=1e-9)
    assert et.characteristic_time(p) == et.half_life(p)


def test_pov_clip_too_small_is_input_error() -> None:
    with pytest.raises(ValueError, match="POV clip too small"):
        et.pov_holdings(1e6, 1.0, 1.0, 1e-9)          # would need 1e15 intervals
    # exactly at the bound: 1e7 intervals of clip 1 for 1e7 shares is allowed
    # in principle; one share more is rejected without allocating.
    with pytest.raises(ValueError, match="POV clip too small"):
        et.pov_holdings(et.MAX_POV_INTERVALS + 1.0, 1.0, 1.0, 0.5)


def test_pov_boundary_interval_counts() -> None:
    clip = 0.15 * 2.5e6 * 1.0                          # 375,000
    x2 = et.pov_holdings(750_000.0, 1.0, 2.5e6, 0.15)  # X/clip = 2 exactly
    assert x2.size - 1 == 2
    assert x2.tolist() == [750_000.0, 375_000.0, 0.0]
    x3 = et.pov_holdings(750_000.001, 1.0, 2.5e6, 0.15)
    assert x3.size - 1 == 3
    n3 = et.trades_from_holdings(x3)
    assert n3[0] == n3[1] == clip
    assert n3[2] == pytest.approx(0.001, abs=1e-6)     # last clip partial
    assert x3[-1] == 0.0


def test_simulate_is_rejects_non_liquidating_schedule(p_liq: et.ACParams) -> None:
    with pytest.raises(ValueError, match="liquidate"):
        et.simulate_is(p_liq, [1e6, 5e5, 1e3], n_paths=10, seed=0)
    with pytest.raises(ValueError):
        et.simulate_is(p_liq, [1e6, float("nan"), 0.0], n_paths=10, seed=0)
    with pytest.raises(ValueError):
        et.simulate_is(p_liq, [1e6], n_paths=10, seed=0)


def test_simulate_is_twap_matches_closed_forms(p_liq: et.ACParams) -> None:
    twap = et.twap_holdings(p_liq.total_shares, p_liq.n_intervals)
    mc = et.simulate_is(p_liq, twap, n_paths=5000, seed=17)
    assert abs(mc.mean - et.expected_cost(p_liq, twap)) <= 4.0 * mc.se_mean
    assert abs(mc.var - et.variance_cost(p_liq, twap)) <= 4.0 * mc.se_var
    assert mc.n_paths == 5000


def test_efficient_frontier_rejects_non_finite_lambda(p_liq: et.ACParams) -> None:
    with pytest.raises(ValueError):
        et.efficient_frontier(p_liq, [1e-7, float("nan")])
    with pytest.raises(ValueError):
        et.efficient_frontier(p_liq, [float("inf")])
    with pytest.raises(ValueError):
        et.efficient_frontier(p_liq, [1e-7, -1e-7])
    # duplicate / unsorted grids are reproduced in input order
    front = et.efficient_frontier(p_liq, [2e-6, 1e-8, 2e-6])
    assert [f.lambda_ra for f in front] == [2e-6, 1e-8, 2e-6]
    assert front[0] == front[2]
    assert front[1].e_cost < front[0].e_cost


def test_twap_and_pov_reject_non_finite() -> None:
    inf, nan = float("inf"), float("nan")
    with pytest.raises(ValueError):
        et.twap_holdings(inf, 5)
    with pytest.raises(ValueError):
        et.twap_holdings(nan, 5)
    with pytest.raises(ValueError):
        et.pov_holdings(nan, 1.0, 1e6, 0.1)
    with pytest.raises(ValueError):
        et.pov_holdings(inf, 1.0, 1e6, 0.1)
    with pytest.raises(ValueError):
        et.pov_holdings(1e6, inf, 1e6, 0.1)
    with pytest.raises(ValueError):
        et.pov_holdings(1e6, 1.0, inf, 0.1)
    with pytest.raises(ValueError):
        et.pov_holdings(1e6, 1.0, 1e6, nan)


def test_integer_arguments_accept_numpy_ints_reject_bools() -> None:
    p = et.ACParams(1e6, np.int64(5), 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6)
    assert p.n_intervals == 5
    assert et.twap_holdings(1e6, np.int32(4)).size == 5
    with pytest.raises(ValueError):
        et.ACParams(1e6, True, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6)
    with pytest.raises(ValueError):
        et.ACParams(1e6, 5.0, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        et.twap_holdings(1e6, False)


def test_size_bounds_are_input_errors(p_liq: et.ACParams) -> None:
    with pytest.raises(ValueError):
        et.ACParams(1e6, et.MAX_AC_INTERVALS + 1, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6)
    with pytest.raises(ValueError):
        et.twap_holdings(1e6, et.MAX_AC_INTERVALS + 1)
    with pytest.raises(ValueError):
        et.simulate_is(p_liq, n_paths=et.MAX_PATHS + 1, seed=0)
    with pytest.raises(ValueError):
        et.simulate_is(p_liq, n_paths=2.0, seed=0)  # type: ignore[arg-type]
    # the bounds themselves are valid (cheap: N = 1e7 TWAP path)
    assert et.twap_holdings(1.0, et.MAX_AC_INTERVALS).size == et.MAX_AC_INTERVALS + 1


def test_cost_functions_validate_holdings(p_liq: et.ACParams) -> None:
    with pytest.raises(ValueError):
        et.variance_cost(p_liq, [1e6])              # length < 2
    with pytest.raises(ValueError):
        et.variance_cost(p_liq, [1e6, float("inf"), 0.0])
    with pytest.raises(ValueError):
        et.expected_cost(p_liq, [1e6, float("nan"), 0.0])
    with pytest.raises(ValueError):
        et.trades_from_holdings([[1e6, 0.0]])        # not 1-D
    # X is taken from the schedule (x_0), not from params (pinned)
    half = et.twap_holdings(5e5, 5)
    e_half = et.expected_cost(p_liq, half)
    assert e_half == pytest.approx(0.5 * p_liq.gamma_p * 5e5 ** 2
                                   + p_liq.eta_tilde * 5e5 ** 2 / 5.0, rel=1e-12)


def test_trajectory_x0_exact_and_sinh_identity_moderate(p_liq: et.ACParams) -> None:
    """For moderate kappa*T the expm1 form equals the textbook sinh ratio to
    1e-14 and x_0 == X exactly (ratio formed before the product)."""
    for lam in (1e-8, 5e-7, 2e-6, 1e-4):
        p = p_liq.with_lambda(lam)
        k, big_t = et.ac_kappa(p), p.horizon
        x = et.ac_trajectory(p)
        assert x[0] == p.total_shares
        for j in range(1, p.n_intervals):
            ref = p.total_shares * math.sinh(k * (big_t - j * p.tau)) / math.sinh(k * big_t)
            assert x[j] == pytest.approx(ref, rel=1e-14)


def test_determinism_of_simulate_is(p_liq: et.ACParams) -> None:
    a = et.simulate_is(p_liq, n_paths=300, seed=5)
    b = et.simulate_is(p_liq, n_paths=300, seed=5)
    assert (a.mean, a.var, a.se_mean, a.se_var) == (b.mean, b.var, b.se_mean, b.se_var)
    assert et.simulate_is(p_liq, n_paths=300, seed=6).mean != a.mean
