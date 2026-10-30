"""Write the whole-trajectory golden cases (golden schema v2, API_SPEC Section 5.1).

Run:  python data/generate_trajectories.py   ->  data/golden/trajectory_golden.json

Scalar goldens pin one or two points of each schedule; these cases pin every
point, stored as indexed flat keys so the readers of the other ports need no
array support.

Kinds
  ac_trajectory   Almgren-Chriss optimum (Section 1.4): x_0..x_N
  twap            twap_holdings(X, N): x_0..x_N
  pov             pov_holdings(X, tau, V, p): x_0..x_M
  frontier        efficient_frontier over lambda_0..lambda_{n-1}:
                  kappa_i, e_cost_i, std_cost_i
  impact_schedule Section 8.2 optimal schedule as fractions of X: xfrac_0..xfrac_N
Every kind also stores ``len`` (number of points). Comparison rule as
Section 5: |actual - expected| <= tol * max(1, |expected|).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA_DIR.parent / "python" / "src"))

import etrade as et
from etrade import impact as im
from etrade import market as mk

AC_TOL = 1e-10
IMPACT_TOL = 1e-9

AC_KEYS = (
    "ac_total_shares",
    "ac_n_intervals",
    "ac_tau",
    "ac_sigma",
    "ac_eta",
    "ac_gamma_p",
    "ac_lambda_ra",
)

# Same market inputs as impact_alpha06_drift_curve in generate_impact.py: a
# non-default alpha, a volume curve and a drift that finishes early.
IMPACT_INPUTS = {
    "price": 100.0,
    "spread": 0.01,
    "adv": 5.0e6,
    "daily_vol": 0.015,
    "total_shares": 250_000.0,
    "n_intervals": 13.0,
    "tau": 1.0 / 13.0,
    "lambda_ra": 1.0e-7,
    "alpha": 0.6,
    "y": 0.8,
    "drift_bps_per_day": 50.0,
    "volume_curve": 1.0,
}


def load(name: str) -> dict:
    return json.loads((DATA_DIR / name).read_text())


def ac_params(inputs: dict) -> et.ACParams:
    return et.ACParams(
        total_shares=inputs["ac_total_shares"],
        n_intervals=int(inputs["ac_n_intervals"]),
        tau=inputs["ac_tau"],
        sigma=inputs["ac_sigma"],
        eta=inputs["ac_eta"],
        gamma_p=inputs["ac_gamma_p"],
        lambda_ra=inputs["ac_lambda_ra"],
    )


def vector(prefix: str, values: list[float]) -> dict:
    out = {f"{prefix}_{j}": float(v) for j, v in enumerate(values)}
    out["len"] = float(len(values))
    return out


def case(name: str, kind: str, inputs: dict, expect: dict, tol: float) -> dict:
    return {"name": name, "kind": kind, "inputs": inputs, "expect": expect, "tol": tol}


def build_cases() -> list[dict]:
    cases = []
    for label in ("liquid", "illiquid", "fx"):
        cfg = load(f"params_{label}.json")
        inputs = {k: float(cfg[k]) for k in AC_KEYS}
        x = et.ac_trajectory(ac_params(inputs))
        cases.append(
            case(f"ac_trajectory_{label}", "ac_trajectory", inputs, vector("x", list(x)), AC_TOL)
        )

    liq = load("params_liquid.json")
    twap_in = {"ac_total_shares": float(liq["ac_total_shares"]), "ac_n_intervals": 10.0}
    x = et.twap_holdings(twap_in["ac_total_shares"], int(twap_in["ac_n_intervals"]))
    cases.append(case("twap_liquid_10", "twap", twap_in, vector("x", list(x)), AC_TOL))

    pov_in = {
        "ac_total_shares": float(liq["ac_total_shares"]),
        "ac_tau": float(liq["ac_tau"]),
        "pov_market_volume_rate": float(liq["pov_market_volume_rate"]),
        "pov_participation": float(liq["pov_participation"]),
    }
    x = et.pov_holdings(
        pov_in["ac_total_shares"],
        pov_in["ac_tau"],
        pov_in["pov_market_volume_rate"],
        pov_in["pov_participation"],
    )
    cases.append(case("pov_liquid", "pov", pov_in, vector("x", list(x)), AC_TOL))

    grid = [float(v) for v in load("parent_order.json")["lambda_grid"]]
    fr_in = {k: float(liq[k]) for k in AC_KEYS}
    fr_in["n_lambdas"] = float(len(grid))
    for i, lam in enumerate(grid):
        fr_in[f"lambda_{i}"] = lam
    expect: dict[str, float] = {}
    for i, fp in enumerate(et.efficient_frontier(ac_params(fr_in), grid)):
        expect[f"kappa_{i}"] = fp.kappa
        expect[f"e_cost_{i}"] = fp.e_cost
        expect[f"std_cost_{i}"] = fp.std_cost
    expect["len"] = float(len(grid))
    cases.append(case("frontier_liquid_8", "frontier", fr_in, expect, AC_TOL))

    inp = dict(IMPACT_INPUTS)
    weights = im.US_EQUITY_VOLUME_CURVE_13 if inp.pop("volume_curve") == 1.0 else None
    inp["n_intervals"] = int(inp["n_intervals"])
    p = mk.impact_params_from_market(weights=weights, **inp)
    r = im.optimal_schedule(p)
    fracs = [float(v) / p.total_shares for v in r.holdings]
    cases.append(
        case(
            "impact_alpha06_drift_curve_path",
            "impact_schedule",
            dict(IMPACT_INPUTS),
            vector("xfrac", fracs),
            IMPACT_TOL,
        )
    )
    return cases


def main() -> None:
    out = {"schema": 2, "cases": build_cases()}
    path = DATA_DIR / "golden" / "trajectory_golden.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {len(out['cases'])} cases to {path}")


if __name__ == "__main__":
    main()
