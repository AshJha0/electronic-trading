"""Write the post-trade TCA golden cases (API_SPEC Section 10).

Run:  python data/generate_tca.py

Output
  data/golden/tca_golden.json   cross-language golden cases

Schema matches data/golden/market_golden.json: {"cases": [{name, kind,
inputs, expect, tol}]}, flat scalar keys.  Arrays are stored as indexed keys
(``px_0``, ``px_1``, ...) with their length in an ``n_*`` key; ``side_buy`` is
1 for a buy and 0 for a sell.  The inputs are small hand-written orders and
samples, not market data; they exercise every branch of the formulas (both
sides, a bin with zero market volume, a constant sample, a reflected
log-gamma argument).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA_DIR.parent / "python" / "src"))

from etrade import tca

TOL = 1e-12

BIN_VOLUMES = [1000.0, 2000.0, 1500.0, 500.0]
BIN_VWAPS = [100.00, 100.05, 100.12, 100.08]
MIDS = [100.0, 100.02, 100.05, 100.03, 100.01, 99.98, 99.97, 100.00, 100.04, 100.06]


def flat(prefix: str, xs: list[float]) -> dict[str, float]:
    return {f"{prefix}_{i}": float(v) for i, v in enumerate(xs)}


def slippage_case(name: str, side: str, arrival: float, close: float, fills: list) -> dict:
    bins = [b for b, _, _ in fills]
    qtys = [q for _, q, _ in fills]
    pxs = [p for _, _, p in fills]
    inputs = {
        "side_buy": 1.0 if side == "buy" else 0.0,
        "arrival_price": arrival,
        "close_price": close,
        "n_fills": float(len(fills)),
        **flat("bin", bins),
        **flat("qty", qtys),
        **flat("px", pxs),
        "n_bins": float(len(BIN_VOLUMES)),
        **flat("vol", BIN_VOLUMES),
        **flat("vwap", BIN_VWAPS),
    }
    r = tca.slippage(side, arrival, close, bins, qtys, pxs, BIN_VOLUMES, BIN_VWAPS)
    expect = {
        "executed_qty": r.executed_qty,
        "avg_price": r.avg_price,
        "market_vwap": r.market_vwap,
        "interval_vwap": r.interval_vwap,
        "arrival_bps": r.arrival_bps,
        "vwap_bps": r.vwap_bps,
        "interval_vwap_bps": r.interval_vwap_bps,
        "close_bps": r.close_bps,
    }
    return {"name": name, "kind": "slippage", "inputs": inputs, "expect": expect, "tol": TOL}


def markout_case(name: str, side: str, steps: list[int], qtys: list[float]) -> dict:
    horizons = [0, 1, 3]
    inputs = {
        "side_buy": 1.0 if side == "buy" else 0.0,
        "n_fills": float(len(steps)),
        **flat("step", steps),
        **flat("qty", qtys),
        "n_mids": float(len(MIDS)),
        **flat("mid", MIDS),
        "n_horizons": float(len(horizons)),
        **flat("h", horizons),
    }
    r = tca.markouts(side, steps, qtys, MIDS, horizons)
    expect = {**flat("per_fill", list(r.per_fill)), **flat("per_share", list(r.per_share))}
    return {"name": name, "kind": "markout", "inputs": inputs, "expect": expect, "tol": TOL}


def participation_case() -> dict:
    vols = [1000.0, 2000.0, 0.0, 1500.0, 500.0]
    bins = [0, 1, 1, 3, 4]
    qtys = [100.0, 150.0, 50.0, 200.0, 0.0]
    inputs = {
        "n_fills": float(len(bins)),
        **flat("bin", bins),
        **flat("qty", qtys),
        "n_bins": float(len(vols)),
        **flat("vol", vols),
    }
    r = tca.participation(bins, qtys, vols)
    expect = {**flat("rate", list(r.per_bin)), "overall": r.overall}
    return {
        "name": "participation_with_empty_bin",
        "kind": "participation",
        "inputs": inputs,
        "expect": expect,
        "tol": TOL,
    }


def welch_case(name: str, a: list[float], b: list[float]) -> dict:
    inputs = {"n_a": float(len(a)), **flat("a", a), "n_b": float(len(b)), **flat("b", b)}
    r = tca.welch_ttest(a, b)
    expect = {"mean_diff": r.mean_diff, "t_stat": r.t_stat, "df": r.df, "p_value": r.p_value}
    return {"name": name, "kind": "welch", "inputs": inputs, "expect": expect, "tol": TOL}


def t_cdf_case(name: str, t: float, df: float) -> dict:
    return {
        "name": name,
        "kind": "t_cdf",
        "inputs": {"t": t, "df": df},
        "expect": {"cdf": tca.student_t_cdf(t, df)},
        "tol": TOL,
    }


def build_cases() -> list[dict]:
    return [
        slippage_case(
            "slippage_buy",
            "buy",
            99.99,
            100.10,
            [(0, 100.0, 100.02), (1, 150.0, 100.06), (1, 50.0, 100.07), (2, 200.0, 100.15)],
        ),
        slippage_case(
            "slippage_sell",
            "sell",
            100.20,
            100.00,
            [(1, 300.0, 100.03), (3, 120.0, 100.06), (3, 0.0, 100.50)],
        ),
        markout_case("markout_buy", "buy", [0, 2, 5], [100.0, 250.0, 50.0]),
        markout_case("markout_sell", "sell", [1, 4, 6], [200.0, 100.0, 300.0]),
        participation_case(),
        welch_case(
            "welch_unequal_variance",
            [1.2, 3.4, 2.2, 5.1, 0.7, 2.9, 4.0, 3.3],
            [0.4, 1.1, -0.2, 0.9, 1.6, 0.3],
        ),
        welch_case("welch_constant_sample", [2.0, 3.5, 1.0, 4.5, 3.0], [1.5, 1.5, 1.5, 1.5]),
        t_cdf_case("t_cdf_cauchy", 1.0, 1.0),
        t_cdf_case("t_cdf_negative_fractional_df", -2.0, 3.5),
        t_cdf_case("t_cdf_df30", 2.5, 30.0),
        t_cdf_case("t_cdf_small_df", 0.3, 0.5),
    ]


def main() -> None:
    cases = build_cases()
    (DATA_DIR / "golden" / "tca_golden.json").write_text(
        json.dumps({"cases": cases}, indent=2) + "\n"
    )
    print(f"wrote tca_golden.json ({len(cases)} cases)")
    for c in cases:
        print(f"  {c['name']:32s} " + "  ".join(f"{k}={v:.6g}" for k, v in c["expect"].items()))


if __name__ == "__main__":
    main()
