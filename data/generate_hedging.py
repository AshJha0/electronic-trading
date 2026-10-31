"""Write the hedged market-making and risk-control golden cases (API_SPEC Section 14).

Run:  python data/generate_hedging.py   ->  data/golden/hedging_golden.json

Exact kinds
  hedge_trade     hedge_trade(q, y, hedge_ratio, hedge_threshold, hedge_target): dy
  hedge_quantile  quantile of a fixed 10-value array at p: q
Statistical kind
  hedge_sim       simulate_hedged on the us_large_cap preset (market units,
                  Section 7): 20,000-path Python reference with *_se per
                  statistic and pnl_var with its delta-method pnl_var_se;
                  ports run their own RNG with >= 2000 paths and must land
                  within 4*sqrt(se_ref^2 + se_own^2) (Section 4).
Inputs for hedge_sim: the preset's as_params_from_market arguments, the
MMConfig fields as in mm_sim (agent_code, quote_tick, maker_fee,
latency_steps, informed_prob, signal_accuracy), the HedgeConfig and
RiskConfig fields (integers as floats), seed and n_paths_ref.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA_DIR.parent / "python" / "src"))

from etrade import hedging as hg
from etrade import market as mk
from etrade import mm

N_PATHS_REF = 20_000
EXACT_TOL = 1e-12
STATS = (
    "pnl",
    "spread",
    "fees",
    "inventory",
    "hedge_pnl",
    "hedge_costs",
    "risk_costs",
    "fills",
    "rms_q",
    "rms_x",
    "rms_skew",
    "quote_updates",
    "loss_stop",
    "kill_stop",
    "pos_flattens",
)
AGENTS = {"as": 0.0, "glft": 1.0, "fixed": 2.0}

LARGE_AS = json.loads((DATA_DIR / "presets.json").read_text())["us_large_cap"]["as"]

# One-tick grid, a 0.2-cent rebate and 15% informed arrivals: the Section 9
# book that can lose money.  B is a 0.9-correlated future or basket; hedging
# it costs half a cent per share (half spread plus fee).
BOOK = {"tick": 0.01, "maker_fee": 0.002, "informed_prob": 0.15}
HEDGE = {"rho": 0.9, "hedge_ratio": 0.9, "hedge_cost": 0.005}

SIMS = [
    ("hedge_sim_as_unhedged", {**BOOK}, {}, {}, 201),
    ("hedge_sim_as_hedged", {**BOOK}, {**HEDGE}, {}, 202),
    ("hedge_sim_as_band", {**BOOK}, {**HEDGE, "hedge_threshold": 3.0}, {}, 203),
    ("hedge_sim_fixed_unhedged", {**BOOK, "agent": "fixed"}, {}, {}, 204),
    ("hedge_sim_fixed_hedged", {**BOOK, "agent": "fixed"}, {**HEDGE}, {}, 205),
    (
        "risk_sim_loss_limit",
        {**BOOK, "agent": "fixed"},
        {},
        {"loss_limit": 5.0, "flatten_penalty": 0.01},
        206,
    ),
    (
        "risk_sim_position_limit",
        {**BOOK, "agent": "fixed"},
        {},
        {"position_limit": 10, "flatten_penalty": 0.01},
        207,
    ),
    ("risk_sim_throttle", {**BOOK}, {}, {"throttle_updates": 1, "throttle_window": 3}, 208),
    (
        "risk_sim_kill_switch",
        {**BOOK, "agent": "fixed"},
        {},
        {"kill_step": 3510, "flatten_penalty": 0.01},
        209,
    ),
    (
        "risk_sim_all_controls_hedged",
        {**BOOK, "agent": "fixed"},
        {**HEDGE, "hedge_threshold": 2.0, "hedge_target": 0.25},
        {
            "loss_limit": 8.0,
            "position_limit": 20,
            "flatten_penalty": 0.01,
            "throttle_updates": 3,
            "throttle_window": 4,
            "kill_step": 4000,
        },
        210,
    ),
]


def config_inputs(cfg: dict, hedge: dict, risk: dict) -> dict:
    h = hg.HedgeConfig(**hedge)
    r = hg.RiskConfig(**risk)
    return {
        "agent_code": AGENTS[cfg.get("agent", "as")],
        "quote_tick": float(cfg.get("tick", 0.0)),
        "maker_fee": float(cfg.get("maker_fee", 0.0)),
        "latency_steps": float(cfg.get("latency_steps", 0)),
        "informed_prob": float(cfg.get("informed_prob", 0.0)),
        "signal_accuracy": float(cfg.get("signal_accuracy", 0.5)),
        "rho": h.rho,
        "hedge_ratio": h.hedge_ratio,
        "hedge_cost": h.hedge_cost,
        "hedge_threshold": h.hedge_threshold,
        "hedge_target": h.hedge_target,
        "loss_limit": r.loss_limit,
        "position_limit": float(r.position_limit),
        "flatten_penalty": r.flatten_penalty,
        "throttle_updates": float(r.throttle_updates),
        "throttle_window": float(r.throttle_window),
        "kill_step": float(r.kill_step),
    }


def sim_case(name: str, cfg: dict, hedge: dict, risk: dict, seed: int) -> dict:
    ap = mk.as_params_from_market(**LARGE_AS)
    res = hg.simulate_hedged(
        ap, N_PATHS_REF, seed, mm.MMConfig(**cfg), hg.HedgeConfig(**hedge), hg.RiskConfig(**risk)
    )
    expect: dict[str, float] = {}
    for st in STATS:
        expect[st] = res.mean(st)
        expect[st + "_se"] = res.se(st)
    expect["pnl_var"] = res.var("pnl")
    expect["pnl_var_se"] = res.var_se("pnl")
    inputs = {k: float(v) for k, v in LARGE_AS.items()}
    inputs.update(config_inputs(cfg, hedge, risk))
    inputs["seed"] = float(seed)
    inputs["n_paths_ref"] = float(N_PATHS_REF)
    print(
        f"  {name:30s} pnl {res.mean('pnl'):8.3f} sd {res.var('pnl') ** 0.5:7.3f} "
        f"q05 {res.quantile('pnl', 0.05):8.3f} rms_skew {res.mean('rms_skew'):.4f} "
        f"identity err {res.max_identity_error:.1e}"
    )
    return {"name": name, "kind": "hedge_sim", "inputs": inputs, "expect": expect, "tol": 4.0}


QUANTILE_VALUES = [3.5, -1.25, 0.0, 7.75, -4.5, 2.0, 2.0, -0.5, 10.25, 1.0]


def exact_cases() -> list[dict]:
    cases = []
    for name, q, y, ratio, thr, tgt in (
        ("hedge_trade_no_hedge", 7.0, 0.0, 0.0, 0.0, 0.0),
        ("hedge_trade_full", 7.0, 0.0, 0.9, 0.0, 0.0),
        ("hedge_trade_inside_band", 7.0, -4.5, 0.9, 2.5, 0.0),
        ("hedge_trade_partial_target", -6.0, 1.8, 0.9, 2.5, 0.25),
    ):
        cases.append(
            {
                "name": name,
                "kind": "hedge_trade",
                "inputs": {
                    "q": q,
                    "y": y,
                    "hedge_ratio": ratio,
                    "hedge_threshold": thr,
                    "hedge_target": tgt,
                },
                "expect": {"dy": hg.hedge_trade(q, y, ratio, thr, tgt)},
                "tol": EXACT_TOL,
            }
        )
    for p in (0.05, 0.5, 0.95, 1.0):
        inputs = {f"v_{i}": v for i, v in enumerate(QUANTILE_VALUES)}
        inputs["p"] = p
        cases.append(
            {
                "name": f"hedge_quantile_p{round(p * 100):03d}",
                "kind": "hedge_quantile",
                "inputs": inputs,
                "expect": {"q": hg.quantile(QUANTILE_VALUES, p)},
                "tol": EXACT_TOL,
            }
        )
    return cases


def main() -> None:
    cases = exact_cases()
    print(f"wrote {len(cases)} exact cases; simulating {len(SIMS)} statistical cases:")
    for name, cfg, hedge, risk, seed in SIMS:
        cases.append(sim_case(name, cfg, hedge, risk, seed))
    path = DATA_DIR / "golden" / "hedging_golden.json"
    path.write_text(json.dumps({"cases": cases}, indent=2) + "\n")
    print(f"wrote {path.name} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
