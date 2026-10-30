"""Write the power-law impact golden cases (API_SPEC Section 8).

Run:  python data/generate_impact.py   ->  data/golden/impact_golden.json

Kinds
  impact_schedule  optimal schedule: xfrac_01..xfrac_{N-1} (x_j / X), e_cost,
                   var_cost, horizon
  impact_vwap      VWAP schedule costed on the model: e_cost, var_cost
  pretrade         pre-trade report: sqrt_law_bps, half_spread_bps and
                   row{i}_impact_bps / row{i}_sd_bps / row{i}_horizon
  sqrt_law         closed-form square-root-law estimate: bps
Inputs are the impact_params_from_market / pretrade_report argument names;
``volume_curve`` = 1.0 selects US_EQUITY_VOLUME_CURVE_13, 0.0 uniform weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA_DIR.parent / "python" / "src"))

from etrade import impact as im
from etrade import market as mk

TOL = 1e-9

LARGE = {
    "price": 100.0,
    "spread": 0.01,
    "adv": 5.0e6,
    "daily_vol": 0.015,
    "total_shares": 250_000.0,
    "n_intervals": 13.0,
    "tau": 1.0 / 13.0,
}

PAPER_ALPHA1 = {
    # Almgren-Chriss example restated so that alpha = 1 reproduces it:
    # eta = y*sigma/adv with y chosen so eta = 2.5e-6 (sigma 0.95, adv 5e6).
    "price": 50.0,
    "spread": 0.125,
    "adv": 5.0e6,
    "daily_vol": 0.019,
    "total_shares": 1.0e6,
    "n_intervals": 5.0,
    "tau": 1.0,
    "alpha": 1.0,
    "y": 2.5e-6 * 5.0e6 / 0.95,
    "perm_frac": 0.10,
}


def schedule_case(name: str, inputs: dict) -> dict:
    inp = dict(inputs)
    weights = im.US_EQUITY_VOLUME_CURVE_13 if inp.pop("volume_curve") == 1.0 else None
    inp["n_intervals"] = int(inp["n_intervals"])
    p = mk.impact_params_from_market(weights=weights, **inp)
    r = im.optimal_schedule(p)
    expect = {
        f"xfrac_{j:02d}": float(r.holdings[j]) / p.total_shares for j in range(1, p.n_intervals)
    }
    expect["e_cost"] = im.impact_expected_cost(p, r.holdings)
    expect["var_cost"] = im.impact_variance_cost(p, r.holdings)
    expect["horizon"] = float(r.horizon)
    return {"name": name, "kind": "impact_schedule", "inputs": inputs, "expect": expect, "tol": TOL}


def vwap_case(name: str, inputs: dict) -> dict:
    inp = dict(inputs)
    weights = im.US_EQUITY_VOLUME_CURVE_13 if inp.pop("volume_curve") == 1.0 else None
    inp["n_intervals"] = int(inp["n_intervals"])
    p = mk.impact_params_from_market(weights=weights, **inp)
    x = im.vwap_holdings(p.total_shares, p.weights or ())
    expect = {"e_cost": im.impact_expected_cost(p, x), "var_cost": im.impact_variance_cost(p, x)}
    return {"name": name, "kind": "impact_vwap", "inputs": inputs, "expect": expect, "tol": TOL}


def pretrade_case(name: str, inputs: dict, lambdas: list[float]) -> dict:
    inp = dict(inputs)
    weights = im.US_EQUITY_VOLUME_CURVE_13 if inp.pop("volume_curve") == 1.0 else None
    inp["n_intervals"] = int(inp["n_intervals"])
    rep = mk.pretrade_report(lambdas=lambdas, weights=weights, **inp)
    expect = {"sqrt_law_bps": rep.sqrt_law_bps, "half_spread_bps": rep.half_spread_bps}
    for i, r in enumerate(rep.rows):
        expect[f"row{i}_impact_bps"] = r.impact_bps
        expect[f"row{i}_sd_bps"] = r.sd_bps
        expect[f"row{i}_horizon"] = float(r.horizon)
    stored = dict(inputs)
    stored["n_lambdas"] = float(len(lambdas))
    for i, lam in enumerate(lambdas):
        stored[f"lambda_{i}"] = lam
    return {"name": name, "kind": "pretrade", "inputs": stored, "expect": expect, "tol": TOL}


def build_cases() -> list[dict]:
    return [
        schedule_case(
            "impact_alpha1_paper", {**PAPER_ALPHA1, "lambda_ra": 2.0e-6, "volume_curve": 0.0}
        ),
        schedule_case("impact_sqrt_large_cap", {**LARGE, "lambda_ra": 1.0e-6, "volume_curve": 0.0}),
        schedule_case(
            "impact_sqrt_large_cap_curve", {**LARGE, "lambda_ra": 1.0e-6, "volume_curve": 1.0}
        ),
        schedule_case(
            "impact_sqrt_drift_early_finish",
            {**LARGE, "lambda_ra": 0.0, "drift_bps_per_day": 200.0, "volume_curve": 0.0},
        ),
        schedule_case(
            "impact_alpha06_drift_curve",
            {
                **LARGE,
                "lambda_ra": 1.0e-7,
                "alpha": 0.6,
                "y": 0.8,
                "drift_bps_per_day": 50.0,
                "volume_curve": 1.0,
            },
        ),
        vwap_case("impact_vwap_curve", {**LARGE, "lambda_ra": 0.0, "volume_curve": 1.0}),
        pretrade_case(
            "pretrade_large_cap_curve",
            {**LARGE, "volume_curve": 1.0},
            [0.0, 1.0e-7, 1.0e-6, 1.0e-5],
        ),
        {
            "name": "sqrt_law_small_cap",
            "kind": "sqrt_law",
            "inputs": {"daily_vol": 0.035, "order_shares": 30_000.0, "adv": 3.0e5, "y": 1.0},
            "expect": {"bps": im.sqrt_law_cost_bps(0.035, 30_000.0, 3.0e5, 1.0)},
            "tol": TOL,
        },
    ]


def main() -> None:
    cases = build_cases()
    path = DATA_DIR / "golden" / "impact_golden.json"
    path.write_text(json.dumps({"cases": cases}, indent=2) + "\n")
    print(f"wrote {path.name} ({len(cases)} cases)")
    for c in cases:
        e = c["expect"]
        extra = f" horizon {e['horizon']:.0f}" if "horizon" in e else ""
        print(f"  {c['name']:34s} {c['kind']:16s}{extra}")


if __name__ == "__main__":
    main()
