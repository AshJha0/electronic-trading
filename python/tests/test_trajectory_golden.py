"""Whole-trajectory goldens (schema v2, API_SPEC Section 5.1): every point of
each stored vector is recomputed and compared with the Section 5 rule."""

from __future__ import annotations

import pytest

import etrade as et
from conftest import ac_from_cfg, load_json
from etrade import impact as im
from etrade import market as mk

GOLDEN = load_json("golden/trajectory_golden.json")


def actual(case: dict) -> dict[str, float]:
    inp = case["inputs"]
    kind = case["kind"]
    if kind == "ac_trajectory":
        return vec("x", et.ac_trajectory(ac_from_cfg(inp)))
    if kind == "twap":
        return vec("x", et.twap_holdings(inp["ac_total_shares"], int(inp["ac_n_intervals"])))
    if kind == "pov":
        x = et.pov_holdings(
            inp["ac_total_shares"],
            inp["ac_tau"],
            inp["pov_market_volume_rate"],
            inp["pov_participation"],
        )
        return vec("x", x)
    if kind == "frontier":
        lambdas = [inp[f"lambda_{i}"] for i in range(int(inp["n_lambdas"]))]
        out: dict[str, float] = {"len": float(len(lambdas))}
        for i, fp in enumerate(et.efficient_frontier(ac_from_cfg(inp), lambdas)):
            out[f"kappa_{i}"] = fp.kappa
            out[f"e_cost_{i}"] = fp.e_cost
            out[f"std_cost_{i}"] = fp.std_cost
        return out
    if kind == "impact_schedule":
        args = dict(inp)
        weights = im.US_EQUITY_VOLUME_CURVE_13 if args.pop("volume_curve") == 1.0 else None
        args["n_intervals"] = int(args["n_intervals"])
        p = mk.impact_params_from_market(weights=weights, **args)
        h = im.optimal_schedule(p).holdings
        return vec("xfrac", [v / p.total_shares for v in h])
    raise AssertionError(f"unknown kind {kind}")


def vec(prefix: str, values: object) -> dict[str, float]:
    vals = [float(v) for v in values]  # type: ignore[attr-defined]
    out = {f"{prefix}_{j}": v for j, v in enumerate(vals)}
    out["len"] = float(len(vals))
    return out


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["name"])
def test_trajectory_golden(case: dict) -> None:
    got = actual(case)
    tol = case["tol"]
    # Same key set both ways: no stored point is skipped, none is missing.
    assert set(got) == set(case["expect"]), case["name"]
    for key, want in case["expect"].items():
        err = abs(got[key] - want)
        assert err <= tol * max(1.0, abs(want)), f"{case['name']}.{key}: {got[key]} vs {want}"


def test_trajectory_golden_schema_and_count() -> None:
    assert GOLDEN["schema"] == 2
    assert len(GOLDEN["cases"]) == 7
