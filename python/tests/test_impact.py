"""Power-law impact execution (API_SPEC Section 8): reduction to
Almgren-Chriss, the square-root law, optimality (independent checks), drift
and early finish, volume curves, the pre-trade report, goldens, validation."""

from __future__ import annotations

import math

import numpy as np
import pytest

import etrade as et
from conftest import load_json
from etrade import impact as im
from etrade import market as mk

GOLDEN = load_json("golden/impact_golden.json")
LIQ = load_json("params_liquid.json")
ILQ = load_json("params_illiquid.json")

LARGE = {
    "price": 100.0,
    "spread": 0.01,
    "adv": 5e6,
    "daily_vol": 0.015,
    "total_shares": 250_000.0,
    "n_intervals": 13,
    "tau": 1.0 / 13.0,
}


def _u(p: im.ImpactParams, x: np.ndarray) -> float:
    return im.impact_expected_cost(p, x) + p.lambda_ra * im.impact_variance_cost(p, x)


# ------------------------------------------------------------ reduction to AC


@pytest.mark.parametrize("cfg", [LIQ, ILQ])
def test_alpha_one_reproduces_almgren_chriss(cfg: dict) -> None:
    ac = et.ACParams(
        cfg["ac_total_shares"],
        cfg["ac_n_intervals"],
        cfg["ac_tau"],
        cfg["ac_sigma"],
        cfg["ac_eta"],
        cfg["ac_gamma_p"],
        cfg["ac_lambda_ra"],
    )
    ip = im.ImpactParams(
        total_shares=ac.total_shares,
        n_intervals=ac.n_intervals,
        tau=ac.tau,
        sigma=ac.sigma,
        eta=ac.eta,
        alpha=1.0,
        gamma_p=ac.gamma_p,
        lambda_ra=ac.lambda_ra,
    )
    r = im.optimal_schedule(ip)
    assert np.max(np.abs(r.holdings - et.ac_trajectory(ac))) <= 1e-10 * ac.total_shares
    assert im.impact_expected_cost(ip, r.holdings) == pytest.approx(et.expected_cost(ac), rel=1e-12)
    assert im.impact_variance_cost(ip, r.holdings) == pytest.approx(et.variance_cost(ac), rel=1e-12)
    assert r.horizon == ac.n_intervals
    assert r.iterations <= 3  # quadratic objective: one Newton step plus the stop check


# ------------------------------------------------------------ square-root law


def test_one_day_twap_is_the_square_root_law_exactly() -> None:
    p = mk.impact_params_from_market(lambda_ra=0.0, **LARGE)
    x = et.twap_holdings(LARGE["total_shares"], LARGE["n_intervals"])
    big_x, n = LARGE["total_shares"], LARGE["n_intervals"]
    temp_plus_perm = im.impact_expected_cost(p, x)
    perm = 0.5 * p.gamma_p * big_x**2 * (1.0 - 1.0 / n)
    law = mk.bps(temp_plus_perm - perm, big_x * LARGE["price"])
    assert law == pytest.approx(im.sqrt_law_cost_bps(0.015, big_x, 5e6), rel=1e-12)


def test_cost_scales_with_square_root_of_size() -> None:
    costs = []
    for frac in (0.01, 0.04):
        kw = {**LARGE, "total_shares": frac * LARGE["adv"]}
        p = mk.impact_params_from_market(lambda_ra=0.0, perm_frac=1.0, **kw)
        x = et.twap_holdings(kw["total_shares"], kw["n_intervals"])
        costs.append(mk.bps(im.impact_expected_cost(p, x), kw["total_shares"] * kw["price"]))
    assert costs[1] / costs[0] == pytest.approx(2.0, rel=0.01)  # 4x size -> 2x cost per share


# ------------------------------------------------------------ optimality


def test_optimum_beats_twap_vwap_and_random_feasible_perturbations() -> None:
    p = mk.impact_params_from_market(lambda_ra=1e-6, weights=im.US_EQUITY_VOLUME_CURVE_13, **LARGE)
    r = im.optimal_schedule(p)
    u_opt = _u(p, r.holdings)
    assert u_opt <= _u(p, et.twap_holdings(LARGE["total_shares"], 13))
    assert u_opt <= _u(p, im.vwap_holdings(LARGE["total_shares"], im.US_EQUITY_VOLUME_CURVE_13))
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = -np.diff(r.holdings) * np.exp(0.05 * rng.standard_normal(13))
        n *= LARGE["total_shares"] / n.sum()
        x = np.concatenate(([LARGE["total_shares"]], LARGE["total_shares"] - np.cumsum(n)))
        x[-1] = 0.0
        assert _u(p, x) >= u_opt - 1e-9 * abs(u_opt)


def test_more_risk_aversion_trades_faster() -> None:
    prev_e = prev_sd = None
    for lam in (0.0, 1e-7, 1e-6, 1e-5):
        p = mk.impact_params_from_market(lambda_ra=lam, **LARGE)
        x = im.optimal_schedule(p).holdings
        e, sd = im.impact_expected_cost(p, x), math.sqrt(im.impact_variance_cost(p, x))
        if prev_e is not None:
            assert e >= prev_e - 1e-9 and sd <= prev_sd + 1e-9
        prev_e, prev_sd = e, sd


def test_matches_scipy_constrained_optimum() -> None:
    sp = pytest.importorskip("scipy.optimize")
    for drift in (0.0, 50.0, 200.0, 1000.0):
        p = mk.impact_params_from_market(lambda_ra=1e-7, drift_bps_per_day=drift, **LARGE)
        r = im.optimal_schedule(p)
        big_x = LARGE["total_shares"]

        def f(z: np.ndarray, p: im.ImpactParams = p, big_x: float = big_x) -> float:
            n = np.maximum(z, 0.0) * big_x
            x = np.maximum(np.concatenate(([big_x], big_x - np.cumsum(n))), 0.0)
            x[-1] = 0.0
            x = np.minimum.accumulate(x)
            return _u(p, x) / 1e4

        cons = [{"type": "eq", "fun": lambda z: z.sum() - 1.0}]
        z0 = np.full(13, 1.0 / 13.0)
        res = sp.minimize(
            f,
            z0,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * 13,
            constraints=cons,
            options={"ftol": 1e-14, "maxiter": 1000},
        )
        assert _u(p, r.holdings) <= res.fun * 1e4 * (1.0 + 1e-7), drift


# ------------------------------------------------------------ drift and early finish


def test_drift_shortens_the_horizon_and_satisfies_kkt() -> None:
    horizons = []
    for drift in (0.0, 50.0, 200.0, 1000.0, 5000.0):
        p = mk.impact_params_from_market(lambda_ra=0.0, drift_bps_per_day=drift, **LARGE)
        r = im.optimal_schedule(p)
        horizons.append(r.horizon)
        x = r.holdings
        assert np.all(np.diff(x) <= 0.0) and x[-1] == 0.0
        assert np.all(x[r.horizon :] == 0.0)
        if r.horizon < 13:
            n_last = x[r.horizon - 1] - x[r.horizon]
            eta_m = p.etas()[r.horizon - 1]
            marginal = (1 + p.alpha) * eta_m * n_last**p.alpha - p.gamma_p * n_last
            assert marginal <= p.drift * p.tau * (1 + 1e-9)
    assert horizons == sorted(horizons, reverse=True)
    assert horizons[0] == 13 and horizons[-1] < 13


def test_bisection_equals_linear_scan_of_feasible_horizons() -> None:
    for drift in (200.0, 1000.0):
        p = mk.impact_params_from_market(lambda_ra=0.0, drift_bps_per_day=drift, **LARGE)
        etas = p.etas()
        feasible = [m for m in range(1, 14) if im._solve_horizon(p, etas, m) is not None]
        assert feasible == list(range(1, max(feasible) + 1))  # monotone feasibility
        assert im.optimal_schedule(p).horizon == max(feasible)


def test_alpha_aware_saves_cost_against_twap_when_drift_is_real() -> None:
    p = mk.impact_params_from_market(lambda_ra=0.0, drift_bps_per_day=50.0, **LARGE)
    twap = et.twap_holdings(LARGE["total_shares"], 13)
    aware = im.optimal_schedule(p).holdings
    assert im.impact_expected_cost(p, aware) < im.impact_expected_cost(p, twap)


# ------------------------------------------------------------ volume curve


def test_volume_curve_sums_to_one_and_is_u_shaped() -> None:
    w = im.US_EQUITY_VOLUME_CURVE_13
    assert math.isclose(sum(w), 1.0, rel_tol=1e-15)
    assert w[0] > w[6] < w[-1]


def test_vwap_is_twap_under_uniform_weights_and_beats_it_under_a_curve() -> None:
    x_u = im.vwap_holdings(1e6, [1.0] * 10)
    assert np.allclose(x_u, et.twap_holdings(1e6, 10), rtol=0.0, atol=1e-6)
    p = mk.impact_params_from_market(lambda_ra=0.0, weights=im.US_EQUITY_VOLUME_CURVE_13, **LARGE)
    vwap = im.vwap_holdings(LARGE["total_shares"], im.US_EQUITY_VOLUME_CURVE_13)
    twap = et.twap_holdings(LARGE["total_shares"], 13)
    assert im.impact_expected_cost(p, vwap) < im.impact_expected_cost(p, twap)
    # risk-neutral optimum on a curve is (numerically) VWAP when gamma_p is small
    opt = im.optimal_schedule(p).holdings
    assert im.impact_expected_cost(p, opt) <= im.impact_expected_cost(p, vwap)


# ------------------------------------------------------------ report


def test_pretrade_report_structure() -> None:
    rep = mk.pretrade_report(lambdas=[0.0, 1e-6, 1e-5], **LARGE)
    assert [r.label for r in rep.rows] == ["twap", "vwap", "optimal", "optimal", "optimal"]
    assert rep.sqrt_law_bps == pytest.approx(im.sqrt_law_cost_bps(0.015, 250_000.0, 5e6))
    assert rep.half_spread_bps == pytest.approx(0.5, rel=1e-12)
    for r in rep.rows:
        assert r.total_bps == pytest.approx(r.impact_bps + rep.half_spread_bps, rel=1e-15)
    assert rep.rows[0].impact_bps == pytest.approx(rep.rows[1].impact_bps, rel=1e-12)
    sds = [r.sd_bps for r in rep.rows[2:]]
    assert sds == sorted(sds, reverse=True)
    with pytest.raises(ValueError):
        mk.pretrade_report(lambdas=[], **LARGE)


# ------------------------------------------------------------ goldens


def _actual(case: dict) -> dict:
    inp = dict(case["inputs"])
    kind = case["kind"]
    if kind == "sqrt_law":
        return {
            "bps": im.sqrt_law_cost_bps(inp["daily_vol"], inp["order_shares"], inp["adv"], inp["y"])
        }
    weights = im.US_EQUITY_VOLUME_CURVE_13 if inp.pop("volume_curve") == 1.0 else None
    inp["n_intervals"] = int(inp["n_intervals"])
    if kind == "pretrade":
        lambdas = [inp.pop(f"lambda_{i}") for i in range(int(inp.pop("n_lambdas")))]
        rep = mk.pretrade_report(lambdas=lambdas, weights=weights, **inp)
        out = {"sqrt_law_bps": rep.sqrt_law_bps, "half_spread_bps": rep.half_spread_bps}
        for i, r in enumerate(rep.rows):
            out[f"row{i}_impact_bps"] = r.impact_bps
            out[f"row{i}_sd_bps"] = r.sd_bps
            out[f"row{i}_horizon"] = float(r.horizon)
        return out
    p = mk.impact_params_from_market(weights=weights, **inp)
    if kind == "impact_vwap":
        x = im.vwap_holdings(p.total_shares, p.weights or ())
        return {"e_cost": im.impact_expected_cost(p, x), "var_cost": im.impact_variance_cost(p, x)}
    r = im.optimal_schedule(p)
    out = {f"xfrac_{j:02d}": float(r.holdings[j]) / p.total_shares for j in range(1, p.n_intervals)}
    out["e_cost"] = im.impact_expected_cost(p, r.holdings)
    out["var_cost"] = im.impact_variance_cost(p, r.holdings)
    out["horizon"] = float(r.horizon)
    return out


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["name"])
def test_impact_golden(case: dict) -> None:
    actual = _actual(case)
    assert set(actual) == set(case["expect"])
    for key, want in case["expect"].items():
        assert abs(actual[key] - want) <= case["tol"] * max(1.0, abs(want)), f"{case['name']}.{key}"


def test_impact_golden_count() -> None:
    assert len(GOLDEN["cases"]) == 8


# ------------------------------------------------------------ validation


@pytest.mark.parametrize(
    "field,bad",
    [
        ("alpha", 0.0),
        ("alpha", 1.5),
        ("eta", 0.0),
        ("drift", -1.0),
        ("gamma_p", -1e-9),
        ("lambda_ra", float("nan")),
        ("n_intervals", im.MAX_IMPACT_INTERVALS + 1),
        ("weights", (0.1,) * 12),
        ("weights", (0.1,) * 12 + (0.0,)),
    ],
)
def test_impact_params_validation(field: str, bad: object) -> None:
    kw = {
        "total_shares": 1e5,
        "n_intervals": 13,
        "tau": 1 / 13,
        "sigma": 1.5,
        "eta": 1e-3,
        "alpha": 0.5,
        "gamma_p": 2e-8,
        "lambda_ra": 0.0,
    }
    kw[field] = bad
    with pytest.raises(ValueError):
        im.ImpactParams(**kw)  # type: ignore[arg-type]


def test_convexity_condition_is_enforced() -> None:
    with pytest.raises(ValueError, match="convex"):
        im.ImpactParams(
            total_shares=1e8,
            n_intervals=13,
            tau=1 / 13,
            sigma=1.5,
            eta=1e-6,
            alpha=0.5,
            gamma_p=1e-3,
            lambda_ra=0.0,
        )


def test_cost_functions_validate_holdings() -> None:
    p = mk.impact_params_from_market(lambda_ra=0.0, **LARGE)
    with pytest.raises(ValueError, match="length"):
        im.impact_expected_cost(p, [1.0, 0.0])
    with pytest.raises(ValueError, match="non-increasing"):
        bad = et.twap_holdings(LARGE["total_shares"], 13)
        bad[3] = bad[1] + 1.0
        im.impact_expected_cost(p, bad)
    with pytest.raises(ValueError, match="liquidate"):
        im.impact_variance_cost(p, np.linspace(1.0, 0.5, 14))


def test_market_constructor_and_sqrt_law_validation() -> None:
    with pytest.raises(ValueError, match="drift_bps_per_day"):
        mk.impact_params_from_market(lambda_ra=0.0, drift_bps_per_day=-1.0, **LARGE)
    with pytest.raises(ValueError, match="alpha"):
        mk.impact_params_from_market(lambda_ra=0.0, alpha=0.0, **LARGE)
    with pytest.raises(ValueError, match="y"):
        mk.impact_params_from_market(lambda_ra=0.0, y=0.0, **LARGE)
    with pytest.raises(ValueError):
        im.sqrt_law_cost_bps(0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        im.sqrt_law_cost_bps(0.01, -1.0, 1.0)
    with pytest.raises(ValueError):
        im.vwap_holdings(1.0, [])


def test_zero_shares_is_the_zero_schedule() -> None:
    p = mk.impact_params_from_market(lambda_ra=1e-6, **{**LARGE, "total_shares": 0.0})
    r = im.optimal_schedule(p)
    assert np.all(r.holdings == 0.0) and r.iterations == 0
