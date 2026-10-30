"""Write the limit-order placement golden cases (API_SPEC Section 13).

Run:  python data/generate_placement.py   ->  data/golden/placement_golden.json

Kinds
  fill_probability    exact: fill_prob of queue_fill_probability(q_ahead, size,
                      mu, theta, horizon)
  placement_decision  exact: fill_prob (from the queue inputs) and
                      place_or_cross with it: passive_cost, aggressive_cost,
                      impact_cost, breakeven_fill_prob, passive (1 or 0)
  liquidation_policy  exact: dp_cost (C_0(Q)), depth_first (j = 0, q = Q),
                      depth_last (j = N-1, q = 1), n_market (market states
                      with q >= 1), aggressive_cost
  liquidation_sim     statistical (tol = 4 SE): optimal_mean, aggressive_mean,
                      passive_mean, saving_vs_aggressive_mean,
                      saving_vs_passive_mean (+ _se); inputs carry seed and
                      n_paths_ref
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA_DIR.parent / "python" / "src"))

from etrade import placement as pl

TOL = 1e-10
N_PATHS_REF = 20_000


def _fill(inputs: dict) -> float:
    return pl.queue_fill_probability(
        int(inputs["q_ahead"]),
        int(inputs["size"]),
        inputs["mu"],
        inputs["theta"],
        inputs["horizon"],
    )


def fill_case(name: str, inputs: dict) -> dict:
    return {
        "name": name,
        "kind": "fill_probability",
        "inputs": inputs,
        "expect": {"fill_prob": _fill(inputs)},
        "tol": TOL,
    }


def decision_case(name: str, inputs: dict) -> dict:
    fp = _fill(inputs)
    d = pl.place_or_cross(
        inputs["child_shares"],
        inputs["spread"],
        fp,
        inputs["adverse_move"],
        inputs["eta"],
        inputs["alpha"],
        inputs["weight"],
    )
    expect = {
        "fill_prob": fp,
        "passive_cost": d.passive_cost,
        "aggressive_cost": d.aggressive_cost,
        "impact_cost": d.impact_cost,
        "breakeven_fill_prob": d.breakeven_fill_prob,
        "passive": 1.0 if d.passive else 0.0,
    }
    return {
        "name": name,
        "kind": "placement_decision",
        "inputs": inputs,
        "expect": expect,
        "tol": TOL,
    }


def _liq(inputs: dict) -> pl.LiquidationParams:
    return pl.LiquidationParams(
        int(inputs["total_units"]),
        int(inputs["n_periods"]),
        inputs["dt"],
        inputs["sigma"],
        inputs["spread"],
        inputs["kappa"],
        inputs["p_touch"],
        inputs["beta"],
        inputs["phi"],
    )


def policy_case(name: str, inputs: dict) -> dict:
    p = _liq(inputs)
    pol = pl.liquidation_policy(p)
    expect = {
        "dp_cost": pol.expected_cost,
        "depth_first": float(pol.depth[0, p.total_units]),
        "depth_last": float(pol.depth[p.n_periods - 1, 1]),
        "n_market": float(pol.market[:, 1:].sum()),
        "aggressive_cost": pl.aggressive_cost(p),
    }
    return {
        "name": name,
        "kind": "liquidation_policy",
        "inputs": inputs,
        "expect": expect,
        "tol": TOL,
    }


def sim_case(name: str, liq: dict, seed: int) -> dict:
    inputs = {**liq, "seed": float(seed), "n_paths_ref": float(N_PATHS_REF)}
    r = pl.simulate_liquidation(_liq(inputs), N_PATHS_REF, seed)
    expect = {
        "optimal_mean": r.optimal_mean,
        "optimal_mean_se": r.optimal_se,
        "aggressive_mean": r.aggressive_mean,
        "aggressive_mean_se": r.aggressive_se,
        "passive_mean": r.passive_mean,
        "passive_mean_se": r.passive_se,
        "saving_vs_aggressive_mean": r.saving_vs_aggressive_mean,
        "saving_vs_aggressive_mean_se": r.saving_vs_aggressive_se,
        "saving_vs_passive_mean": r.saving_vs_passive_mean,
        "saving_vs_passive_mean_se": r.saving_vs_passive_se,
    }
    return {"name": name, "kind": "liquidation_sim", "inputs": inputs, "expect": expect, "tol": 4.0}


LIQ_PATIENT = {
    # 10 units over 50 periods: time to post away from the touch.
    "total_units": 10.0,
    "n_periods": 50.0,
    "dt": 0.02,
    "sigma": 0.5,
    "spread": 0.02,
    "kappa": 30.0,
    "p_touch": 0.3,
    "beta": 0.005,
    "phi": 0.0,
}

LIQ_URGENT = {
    # 20 units over 50 periods with a running inventory penalty.
    "total_units": 20.0,
    "n_periods": 50.0,
    "dt": 0.02,
    "sigma": 0.5,
    "spread": 0.02,
    "kappa": 50.0,
    "p_touch": 0.4,
    "beta": 0.005,
    "phi": 0.05,
}

DECISION_BASE = {
    "child_shares": 2000.0,
    "spread": 0.02,
    "eta": 1.0e-4,
    "alpha": 0.5,
    "weight": 1.0 / 13.0,
}


def build_cases() -> list[dict]:
    return [
        fill_case(
            "fill_front_of_queue",
            {"q_ahead": 0.0, "size": 5.0, "mu": 10.0, "theta": 0.1, "horizon": 0.5},
        ),
        fill_case(
            "fill_no_cancellations",
            {"q_ahead": 30.0, "size": 5.0, "mu": 10.0, "theta": 0.0, "horizon": 2.0},
        ),
        fill_case(
            "fill_mid_queue_with_cancellations",
            {"q_ahead": 50.0, "size": 10.0, "mu": 20.0, "theta": 0.05, "horizon": 3.0},
        ),
        fill_case(
            "fill_deep_queue_long_horizon",
            {"q_ahead": 400.0, "size": 50.0, "mu": 100.0, "theta": 0.2, "horizon": 4.0},
        ),
        decision_case(
            "decision_passive_near_front",
            {
                **DECISION_BASE,
                "adverse_move": 0.01,
                "q_ahead": 5.0,
                "size": 2.0,
                "mu": 10.0,
                "theta": 0.1,
                "horizon": 1.0,
            },
        ),
        decision_case(
            "decision_aggressive_deep_queue",
            {
                **DECISION_BASE,
                "adverse_move": 0.03,
                "q_ahead": 60.0,
                "size": 2.0,
                "mu": 10.0,
                "theta": 0.05,
                "horizon": 3.0,
            },
        ),
        decision_case(
            "decision_aggressive_zero_spread",
            {
                **DECISION_BASE,
                "spread": 0.0,
                "adverse_move": 0.005,
                "q_ahead": 5.0,
                "size": 2.0,
                "mu": 10.0,
                "theta": 0.1,
                "horizon": 1.0,
            },
        ),
        policy_case("liquidation_policy_patient", LIQ_PATIENT),
        policy_case("liquidation_policy_urgent", LIQ_URGENT),
        sim_case("liquidation_sim_patient", LIQ_PATIENT, 201),
        sim_case("liquidation_sim_urgent", LIQ_URGENT, 202),
    ]


def main() -> None:
    cases = build_cases()
    path = DATA_DIR / "golden" / "placement_golden.json"
    path.write_text(json.dumps({"cases": cases}, indent=2) + "\n")
    print(f"wrote {path.name} ({len(cases)} cases)")
    for c in cases:
        print(f"  {c['name']:36s} {c['kind']:20s} {c['expect']}")


if __name__ == "__main__":
    main()
