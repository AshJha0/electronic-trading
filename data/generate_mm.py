"""Write the market-making golden cases (API_SPEC Section 9).

Run:  python data/generate_mm.py   ->  data/golden/mm_golden.json

Exact kinds
  mm_tick      round_to_tick: bid, ask
  mm_glft      glft_half_spreads at inventory q: delta_bid, delta_ask
  mm_wmid      weighted_mid: wmid
  mm_ofi       order_flow_imbalance on a 5-state book: e_1..e_4, ofi (sum)
Statistical kind
  mm_sim       simulate_mm on the us_large_cap preset (market units, Section 7):
               20,000-path Python reference with *_se per statistic; ports run
               their own RNG with >= 2000 paths and must land within
               4*sqrt(se_ref^2 + se_own^2) (Section 4).
Inputs for mm_sim: the preset's as_params_from_market arguments, the MMConfig
fields (agent encoded as agent_code 0 = as, 1 = glft, 2 = fixed), seed and
n_paths_ref.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA_DIR.parent / "python" / "src"))

from etrade import market as mk
from etrade import mm

N_PATHS_REF = 20_000
EXACT_TOL = 1e-12
STATS = ("pnl", "spread", "fees", "inventory", "markout", "fills", "informed_fills")
AGENTS = {"as": 0.0, "glft": 1.0, "fixed": 2.0}

LARGE_AS = json.loads((DATA_DIR / "presets.json").read_text())["us_large_cap"]["as"]

SIMS = [
    ("mm_sim_frictionless", {}, 101),
    ("mm_sim_tick_rebate", {"tick": 0.01, "maker_fee": 0.002}, 102),
    ("mm_sim_informed", {"tick": 0.01, "maker_fee": 0.002, "informed_prob": 0.15}, 103),
    (
        "mm_sim_informed_signal",
        {"tick": 0.01, "maker_fee": 0.002, "informed_prob": 0.15, "signal_accuracy": 0.6},
        104,
    ),
    ("mm_sim_latency", {"tick": 0.01, "maker_fee": 0.002, "latency_steps": 1}, 105),
    ("mm_sim_glft", {"agent": "glft", "tick": 0.01, "maker_fee": 0.002}, 106),
]


def config_inputs(cfg: dict) -> dict:
    full = {
        "agent_code": AGENTS[cfg.get("agent", "as")],
        "quote_tick": float(cfg.get("tick", 0.0)),
        "maker_fee": float(cfg.get("maker_fee", 0.0)),
        "latency_steps": float(cfg.get("latency_steps", 0)),
        "informed_prob": float(cfg.get("informed_prob", 0.0)),
        "signal_accuracy": float(cfg.get("signal_accuracy", 0.5)),
    }
    return full


def sim_case(name: str, cfg: dict, seed: int) -> dict:
    ap = mk.as_params_from_market(**LARGE_AS)
    res = mm.simulate_mm(ap, N_PATHS_REF, seed, mm.MMConfig(**cfg))
    expect: dict[str, float] = {}
    for st in STATS:
        expect[st] = res.mean(st)
        expect[st + "_se"] = res.se(st)
    inputs = {k: float(v) for k, v in LARGE_AS.items()}
    inputs.update(config_inputs(cfg))
    inputs["seed"] = float(seed)
    inputs["n_paths_ref"] = float(N_PATHS_REF)
    print(
        f"  {name:26s} pnl {res.mean('pnl'):8.3f} (se {res.se('pnl'):.3f}) "
        f"identity err {res.max_identity_error:.1e}"
    )
    return {"name": name, "kind": "mm_sim", "inputs": inputs, "expect": expect, "tol": 4.0}


def exact_cases() -> list[dict]:
    ap = mk.as_params_from_market(**LARGE_AS)
    cases = []
    for name, bid, ask, tick in (
        ("mm_tick_round_away", 99.9934, 100.0066, 0.01),
        ("mm_tick_on_grid", 99.99, 100.01, 0.01),
        ("mm_tick_continuous", 99.9934, 100.0066, 0.0),
    ):
        b, a = mm.round_to_tick(bid, ask, tick)
        cases.append(
            {
                "name": name,
                "kind": "mm_tick",
                "inputs": {"bid": bid, "ask": ask, "tick": tick},
                "expect": {"bid": b, "ask": a},
                "tol": EXACT_TOL,
            }
        )
    for q in (0.0, 3.0, -2.0):
        db, da = mm.glft_half_spreads(ap, q)
        inputs = {k: float(v) for k, v in LARGE_AS.items()}
        inputs["q"] = q
        cases.append(
            {
                "name": f"mm_glft_q{int(q):+d}".replace("+", "p").replace("-", "m"),
                "kind": "mm_glft",
                "inputs": inputs,
                "expect": {"delta_bid": db, "delta_ask": da},
                "tol": EXACT_TOL,
            }
        )
    cases.append(
        {
            "name": "mm_weighted_mid",
            "kind": "mm_wmid",
            "inputs": {"bid": 99.99, "ask": 100.01, "bid_size": 900.0, "ask_size": 100.0},
            "expect": {"wmid": mm.weighted_mid(99.99, 100.01, 900.0, 100.0)},
            "tol": EXACT_TOL,
        }
    )
    pb = [100.00, 100.00, 100.01, 100.01, 100.00]
    qb = [500.0, 700.0, 200.0, 100.0, 800.0]
    pa = [100.02, 100.02, 100.02, 100.03, 100.01]
    qa = [400.0, 300.0, 600.0, 900.0, 250.0]
    e = mm.order_flow_imbalance(pb, qb, pa, qa)
    inputs: dict[str, float] = {}
    for i in range(5):
        inputs[f"bid_px_{i}"], inputs[f"bid_sz_{i}"] = pb[i], qb[i]
        inputs[f"ask_px_{i}"], inputs[f"ask_sz_{i}"] = pa[i], qa[i]
    expect = {f"e_{i + 1}": float(e[i]) for i in range(4)}
    expect["ofi"] = float(e.sum())
    cases.append(
        {
            "name": "mm_ofi_five_states",
            "kind": "mm_ofi",
            "inputs": inputs,
            "expect": expect,
            "tol": EXACT_TOL,
        }
    )
    return cases


def main() -> None:
    cases = exact_cases()
    print(f"wrote {len(cases)} exact cases; simulating {len(SIMS)} statistical cases:")
    for name, cfg, seed in SIMS:
        cases.append(sim_case(name, cfg, seed))
    path = DATA_DIR / "golden" / "mm_golden.json"
    path.write_text(json.dumps({"cases": cases}, indent=2) + "\n")
    print(f"wrote {path.name} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
