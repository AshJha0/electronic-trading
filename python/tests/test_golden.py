"""Golden-value suite: every case in data/golden/golden.json is recomputed
from its inputs and checked against the stored expectation.

Exact cases:        |actual - expected| <= tol * max(1, |expected|).
Statistical cases:  the golden reference was produced by the 20k-path Python
run; this suite re-simulates with <= 2k paths (own SEs) and asserts
|actual - ref| <= tol * sqrt(se_ref^2 + se_own^2), tol = 4 (the band ports
with their own RNG must also satisfy).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import etrade as et
from conftest import ac_from_cfg, as_from_cfg, load_json

GOLDEN = load_json("golden/golden.json")
CASES = {c["name"]: c for c in GOLDEN["cases"]}

N_PATHS_TEST = 2000
AC_MC_PATHS_TEST = 5000


def check_exact(case: dict, actual: dict) -> None:
    tol = case["tol"]
    for key, expected in case["expect"].items():
        assert key in actual, f"{case['name']}: missing key {key}"
        err = abs(actual[key] - expected)
        assert err <= tol * max(1.0, abs(expected)), (
            f"{case['name']}.{key}: got {actual[key]}, want {expected}, err {err}")


def check_statistical(case: dict, actual: dict, actual_se: dict) -> None:
    mult = case["tol"]
    for key, ref in case["expect"].items():
        if key.endswith("_se"):
            continue
        se_ref = case["expect"][key + "_se"]
        band = mult * math.sqrt(se_ref ** 2 + actual_se[key] ** 2)
        err = abs(actual[key] - ref)
        assert err <= band, (
            f"{case['name']}.{key}: got {actual[key]}, ref {ref}, "
            f"err {err} > band {band}")


def ac_solution_actual(inputs: dict) -> dict:
    p = ac_from_cfg(inputs)
    x = et.ac_trajectory(p)
    n = et.ac_trade_list(p)
    return {"kappa": et.ac_kappa(p), "half_life": et.half_life(p),
            "x_j": float(x[int(inputs["x_index"])]), "n_1": float(n[0]),
            "e_cost": et.expected_cost(p), "var_cost": et.variance_cost(p)}


@pytest.mark.parametrize("name", ["ac_liquid_solution", "ac_illiquid_solution"])
def test_ac_solutions(name: str) -> None:
    case = CASES[name]
    check_exact(case, ac_solution_actual(case["inputs"]))


def test_ac_twap() -> None:
    case = CASES["ac_twap_liquid"]
    p = ac_from_cfg(case["inputs"])
    twap = et.twap_holdings(p.total_shares, p.n_intervals)
    check_exact(case, {"e_cost": et.expected_cost(p, twap),
                       "var_cost": et.variance_cost(p, twap)})


def test_ac_pov() -> None:
    case = CASES["ac_pov_liquid"]
    inp = case["inputs"]
    p = ac_from_cfg(inp)
    pov = et.pov_holdings(p.total_shares, p.tau, inp["pov_market_volume_rate"],
                          inp["pov_participation"])
    check_exact(case, {"e_cost": et.expected_cost(p, pov),
                       "var_cost": et.variance_cost(p, pov),
                       "n_intervals_used": float(pov.size - 1)})


def test_ac_frontier_point() -> None:
    case = CASES["ac_frontier_point_liquid"]
    p = ac_from_cfg(case["inputs"])
    fp = et.efficient_frontier(p, [p.lambda_ra])[0]
    check_exact(case, {"e_cost": fp.e_cost, "std_cost": fp.std_cost,
                       "kappa": fp.kappa})


def test_ac_kappa_small_tau() -> None:
    """kappa at kappa_tilde^2*tau^2/2 ~ 2e-7: the log1p form must agree with
    the golden (which the acosh(1+x) form misses by ~1e-9 relative)."""
    case = CASES["ac_kappa_small_tau"]
    p = ac_from_cfg(case["inputs"])
    assert p.tau == 1e-3 and p.n_intervals == 5000
    check_exact(case, {"kappa": et.ac_kappa(p), "half_life": et.half_life(p),
                       "x_1": float(et.ac_trajectory(p)[1])})


def test_ac_lambda_zero_limit() -> None:
    case = CASES["ac_lambda_zero_limit_liquid"]
    p = ac_from_cfg(case["inputs"])
    assert p.lambda_ra == 0.0
    twap = et.twap_holdings(p.total_shares, p.n_intervals)
    diff = float(np.max(np.abs(et.ac_trajectory(p) - twap)))
    check_exact(case, {"max_abs_diff_vs_twap": diff})


def test_ac_instant_liquidation() -> None:
    case = CASES["ac_instant_liquidation_liquid"]
    e, v = et.instantaneous_cost(ac_from_cfg(case["inputs"]))
    check_exact(case, {"e_cost": e, "var_cost": v})


def test_ac_mc_statistical() -> None:
    case = CASES["ac_mc_liquid"]
    p = ac_from_cfg(case["inputs"])
    mc = et.simulate_is(p, n_paths=AC_MC_PATHS_TEST,
                        seed=int(case["inputs"]["seed"]))
    check_statistical(case,
                      {"mean_cost": mc.mean, "var_cost": mc.var},
                      {"mean_cost": mc.se_mean, "var_cost": mc.se_var})


def test_as_reservation() -> None:
    case = CASES["as_reservation_liquid"]
    inp = case["inputs"]
    ap = as_from_cfg(inp)
    check_exact(case, {"reservation_price": et.reservation_price(
        ap, inp["s"], inp["q"], inp["t"])})


def test_as_spread() -> None:
    case = CASES["as_spread_liquid"]
    ap = as_from_cfg(case["inputs"])
    check_exact(case, {"spread": et.optimal_spread(ap, case["inputs"]["t"]),
                       "spread_terminal": et.optimal_spread(ap, ap.horizon)})


def test_as_fill_prob() -> None:
    case = CASES["as_fill_prob_liquid"]
    ap = as_from_cfg(case["inputs"])
    delta = case["inputs"]["delta"]
    check_exact(case, {"intensity": et.intensity(ap, delta),
                       "fill_probability": et.fill_probability(ap, delta)})


def _sim_stats(res: et.ASSimResult, keys: list[str]) -> tuple[dict, dict]:
    vals = {"mean_pnl": res.mean_pnl, "std_pnl": res.std_pnl,
            "mean_abs_q": res.mean_abs_q, "mean_max_abs_q": res.mean_max_abs_q,
            "mean_fills": res.mean_fills}
    return ({k: vals[k] for k in keys}, {k: res.se(k) for k in keys})


@pytest.mark.parametrize("name,keys", [
    ("as_sim_liquid", ["mean_pnl", "std_pnl", "mean_abs_q", "mean_max_abs_q",
                       "mean_fills"]),
    ("as_sim_fx", ["mean_pnl", "std_pnl", "mean_abs_q"]),
])
def test_as_sim_statistical(name: str, keys: list[str]) -> None:
    case = CASES[name]
    ap = as_from_cfg(case["inputs"])
    res = et.simulate(ap, n_paths=N_PATHS_TEST, seed=int(case["inputs"]["seed"]))
    actual, se = _sim_stats(res, keys)
    check_statistical(case, actual, se)
    assert not res.cap_violated


def test_as_fixed_comparison_statistical() -> None:
    case = CASES["as_fixed_comparison_liquid"]
    ap = as_from_cfg(case["inputs"])
    seed = int(case["inputs"]["seed"])
    res_as = et.simulate(ap, n_paths=N_PATHS_TEST, seed=seed, agent="as")
    res_fx = et.simulate(ap, n_paths=N_PATHS_TEST, seed=seed, agent="fixed",
                         fixed_spread=case["inputs"]["fixed_spread"])
    check_statistical(case,
                      {"as_rms_q": res_as.rms_q, "fixed_rms_q": res_fx.rms_q},
                      {"as_rms_q": res_as.se("rms_q"),
                       "fixed_rms_q": res_fx.se("rms_q")})
    assert res_as.rms_q < res_fx.rms_q  # the qualitative claim itself


@pytest.mark.parametrize("name,side", [("is_decomposition_sell", "sell"),
                                       ("is_decomposition_buy", "buy")])
def test_is_decomposition(name: str, side: str) -> None:
    case = CASES[name]
    inp = case["inputs"]
    d = et.decompose_shortfall(side, inp["decision_price"], inp["arrival_price"],
                               inp["close_price"], inp["total_qty"],
                               [inp["q1"], inp["q2"], inp["q3"]],
                               [inp["p1"], inp["p2"], inp["p3"]])
    check_exact(case, {"total": d.total, "delay": d.delay,
                       "trading": d.trading, "opportunity": d.opportunity})


def test_participation_rate_golden() -> None:
    case = CASES["participation_rate"]
    inp = case["inputs"]
    check_exact(case, {"participation": et.participation_rate(
        inp["executed_volume"], inp["market_volume"])})


def test_impact_regression_golden() -> None:
    case = CASES["impact_regression"]
    inp = case["inputs"]
    part, cost = et.synthetic_impact_fills(int(inp["n_obs"]),
                                           inp["true_intercept"],
                                           inp["true_slope"], inp["noise_amp"])
    r = et.estimate_impact(part, cost)
    check_exact(case, {"slope": r.slope, "intercept": r.intercept,
                       "r_squared": r.r_squared})


def test_all_cases_covered() -> None:
    """Every golden case has a dedicated test above (guards against silent
    additions to golden.json)."""
    covered = {
        "ac_liquid_solution", "ac_illiquid_solution", "ac_twap_liquid",
        "ac_pov_liquid", "ac_frontier_point_liquid",
        "ac_lambda_zero_limit_liquid", "ac_instant_liquidation_liquid",
        "ac_kappa_small_tau", "ac_mc_liquid", "as_reservation_liquid",
        "as_spread_liquid", "as_fill_prob_liquid", "as_sim_liquid", "as_sim_fx",
        "as_fixed_comparison_liquid", "is_decomposition_sell",
        "is_decomposition_buy", "participation_rate", "impact_regression",
    }
    assert set(CASES) == covered
    assert len(GOLDEN["cases"]) == 19
