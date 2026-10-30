"""Write the calibration golden cases (API_SPEC Section 11.7).

Run:  python data/generate_calibration.py

Output: data/golden/calibration_golden.json, schema as data/golden/golden.json
({"cases": [{name, kind, inputs, expect, tol}]}, flat scalar keys).  The
inputs are small synthetic arrays, flattened as ``<array>_<i>`` with the
length in ``n``; no recorded data is involved.  Kinds: ``cal_volume_curve``,
``cal_volatility``, ``cal_spread``, ``cal_fill_intensity`` and
``cal_fitted_curve`` (the pinned FITTED_VOLUME_CURVE_13, no inputs).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA_DIR.parent / "python" / "src"))

from etrade import calibration as cal

TOL = 1e-12


class Lcg:
    """Deterministic uniform draws in [0, 1) (Numerical Recipes LCG)."""

    def __init__(self, seed: int) -> None:
        self.state = seed

    def uniform(self) -> float:
        self.state = (1664525 * self.state + 1013904223) % 2**32
        return self.state / 2**32


def flat(**arrays: list[float]) -> dict[str, float]:
    out: dict[str, float] = {}
    n = None
    for name, values in arrays.items():
        n = len(values) if n is None else n
        assert len(values) == n
        for i, v in enumerate(values):
            out[f"{name}_{i}"] = float(v)
    out["n"] = float(n or 0)
    return out


def volume_case(
    name: str, times: list[float], sizes: list[float], t0: float, t1: float, nb: int
) -> dict:
    w = cal.volume_curve(times, sizes, t0, t1, nb)
    inputs = flat(times=times, sizes=sizes)
    inputs.update(session_start=t0, session_end=t1, n_bins=float(nb))
    return {
        "name": name,
        "kind": "cal_volume_curve",
        "inputs": inputs,
        "expect": {f"frac_{j:02d}": v for j, v in enumerate(w)},
        "tol": TOL,
    }


def vol_case(
    name: str, times: list[float], mids: list[float], t0: float, t1: float, nb: int
) -> dict:
    v = cal.volatility_curve(times, mids, t0, t1, nb)
    inputs = flat(times=times, mids=mids)
    inputs.update(session_start=t0, session_end=t1, n_bins=float(nb))
    expect = {"daily_vol": v.daily_vol, "n_returns": float(v.n_returns)}
    expect.update({f"bin_vol_{j:02d}": x for j, x in enumerate(v.bin_vol)})
    return {"name": name, "kind": "cal_volatility", "inputs": inputs, "expect": expect, "tol": TOL}


def spread_case(name: str, spreads: list[float], mids: list[float], tick: float) -> dict:
    s = cal.spread_stats(spreads, mids, tick)
    inputs = flat(spreads=spreads, mids=mids)
    inputs["tick"] = tick
    return {
        "name": name,
        "kind": "cal_spread",
        "inputs": inputs,
        "expect": {
            "mean_ticks": s.mean_ticks,
            "median_ticks": s.median_ticks,
            "mean_bps": s.mean_bps,
            "median_bps": s.median_bps,
            "n": float(s.n),
        },
        "tol": TOL,
    }


def fill_case(
    name: str, distances: list[float], tick: float, levels: int, days: float, sides: float
) -> dict:
    f = cal.fill_intensity(distances, tick, levels, days, sides)
    inputs = flat(distances=distances)
    inputs.update(tick=tick, n_levels=float(levels), days=days, sides=sides)
    return {
        "name": name,
        "kind": "cal_fill_intensity",
        "inputs": inputs,
        "expect": {
            "a_rate": f.a_rate,
            "k": f.k,
            "touch_fills_per_day": f.touch_fills_per_day,
            "halving_ticks": f.halving_ticks,
            "n_buckets": float(f.n_buckets),
        },
        "tol": TOL,
    }


def build_cases() -> list[dict]:
    cases = []
    # Hand-checkable: one trade before the session, one after, one at each edge.
    cases.append(
        volume_case(
            "cal_volume_curve_edges",
            [-1.0, 0.0, 10.0, 24.9, 25.0, 49.0, 50.5, 74.0, 99.0, 100.0, 101.0],
            [500.0, 100.0, 200.0, 100.0, 300.0, 100.0, 400.0, 0.0, 200.0, 100.0, 900.0],
            0.0,
            100.0,
            4,
        )
    )
    rng = Lcg(20191230)
    span = 23400.0
    times = sorted(span * rng.uniform() for _ in range(40))
    sizes = [100.0 * (1 + math.floor(10 * rng.uniform())) for _ in range(40)]
    cases.append(volume_case("cal_volume_curve_13_bins", times, sizes, 0.0, span, 13))

    # Mids sampled on a clock that starts before and ends after the session.
    cases.append(
        vol_case(
            "cal_volatility_edges",
            [-10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0],
            [
                99.0,
                100.0,
                100.5,
                100.2,
                100.9,
                101.0,
                100.4,
                100.6,
                99.8,
                100.1,
                100.3,
                100.0,
                103.0,
            ],
            0.0,
            100.0,
            4,
        )
    )
    rng = Lcg(7)
    t, m = [], []
    mid = 150.0
    for i in range(53):
        t.append(450.0 * i)
        m.append(mid)
        mid *= math.exp(0.002 * (rng.uniform() - 0.5))
    cases.append(vol_case("cal_volatility_13_bins", t, m, 0.0, span, 13))

    cases.append(
        spread_case(
            "cal_spread_odd",
            [0.01, 0.02, 0.01, 0.03, 0.01],
            [100.0, 100.005, 99.99, 100.02, 100.0],
            0.01,
        )
    )
    cases.append(
        spread_case(
            "cal_spread_even_effective",
            [0.0, 0.01, 0.002, 0.02, 0.01, 0.04],
            [50.0, 50.005, 49.99, 50.01, 50.0, 50.02],
            0.01,
        )
    )

    # Counts reaching each bucket close to 1000 exp(-0.9 b); touch prints at
    # exactly half a tick exercise the edge slack.
    dist: list[float] = []
    reach = [1000, 407, 165, 67, 27, 11]
    for b, c in enumerate(reach):
        nxt = reach[b + 1] if b + 1 < len(reach) else 0
        dist += [(b + 0.5) * 0.01] * (c - nxt)
    dist += [0.002] * 50  # inside-spread prints reach no bucket
    cases.append(fill_case("cal_fill_intensity_exponential", dist, 0.01, 6, 1.0, 2.0))
    rng = Lcg(42)
    dist = [0.01 * (-math.log(1.0 - rng.uniform()) / 1.7) for _ in range(300)]
    cases.append(fill_case("cal_fill_intensity_two_days", dist, 0.01, 8, 2.0, 2.0))

    cases.append(
        {
            "name": "cal_fitted_curve_13",
            "kind": "cal_fitted_curve",
            "inputs": {},
            "expect": {f"w_{j:02d}": w for j, w in enumerate(cal.FITTED_VOLUME_CURVE_13)},
            "tol": TOL,
        }
    )
    return cases


def main() -> None:
    cases = build_cases()
    (DATA_DIR / "golden" / "calibration_golden.json").write_text(
        json.dumps({"cases": cases}, indent=2) + "\n"
    )
    print(f"wrote calibration_golden.json ({len(cases)} cases)")
    for c in cases:
        print(f"  {c['name']:34s} {json.dumps(c['expect'])[:110]}")


if __name__ == "__main__":
    main()
