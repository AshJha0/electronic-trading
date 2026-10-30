"""Calibration estimators (API_SPEC Section 11): golden cases, defining
properties, the round trip into the market-unit layer, and validation."""

from __future__ import annotations

import math

import pytest

from conftest import load_json
from etrade import calibration as cal
from etrade import impact
from etrade import market as mk

CAL = load_json("golden/calibration_golden.json")
FITTED = load_json("calibration/fitted_20191230.json")
PRESETS = load_json("presets.json")


def _array(inputs: dict, name: str) -> list[float]:
    return [inputs[f"{name}_{i}"] for i in range(int(inputs["n"]))]


def _actual(case: dict) -> dict:
    inp = case["inputs"]
    kind = case["kind"]
    if kind == "cal_volume_curve":
        w = cal.volume_curve(
            _array(inp, "times"),
            _array(inp, "sizes"),
            inp["session_start"],
            inp["session_end"],
            int(inp["n_bins"]),
        )
        return {f"frac_{j:02d}": v for j, v in enumerate(w)}
    if kind == "cal_volatility":
        v = cal.volatility_curve(
            _array(inp, "times"),
            _array(inp, "mids"),
            inp["session_start"],
            inp["session_end"],
            int(inp["n_bins"]),
        )
        out = {"daily_vol": v.daily_vol, "n_returns": float(v.n_returns)}
        out.update({f"bin_vol_{j:02d}": x for j, x in enumerate(v.bin_vol)})
        return out
    if kind == "cal_spread":
        s = cal.spread_stats(_array(inp, "spreads"), _array(inp, "mids"), inp["tick"])
        return {
            "mean_ticks": s.mean_ticks,
            "median_ticks": s.median_ticks,
            "mean_bps": s.mean_bps,
            "median_bps": s.median_bps,
            "n": float(s.n),
        }
    if kind == "cal_fill_intensity":
        f = cal.fill_intensity(
            _array(inp, "distances"), inp["tick"], int(inp["n_levels"]), inp["days"], inp["sides"]
        )
        return {
            "a_rate": f.a_rate,
            "k": f.k,
            "touch_fills_per_day": f.touch_fills_per_day,
            "halving_ticks": f.halving_ticks,
            "n_buckets": float(f.n_buckets),
        }
    assert kind == "cal_fitted_curve"
    return {f"w_{j:02d}": w for j, w in enumerate(cal.FITTED_VOLUME_CURVE_13)}


@pytest.mark.parametrize("case", CAL["cases"], ids=lambda c: c["name"])
def test_calibration_golden(case: dict) -> None:
    actual = _actual(case)
    assert set(actual) == set(case["expect"]), "every expect key must be asserted"
    for key, want in case["expect"].items():
        got = actual[key]
        assert abs(got - want) <= case["tol"] * max(1.0, abs(want)), f"{case['name']}.{key}"


def test_calibration_golden_count() -> None:
    assert len(CAL["cases"]) == 9


def test_fitted_curve_sums_to_one_and_matches_fit() -> None:
    w = cal.FITTED_VOLUME_CURVE_13
    assert len(w) == 13
    assert abs(sum(w) - 1.0) <= 1e-12
    assert list(w) == FITTED["fitted_volume_curve_13"]
    # usable as Section 8 liquidity weights
    x = impact.vwap_holdings(1000.0, w)
    assert x[0] == 1000.0 and x[-1] == 0.0


def test_volume_curve_is_a_distribution() -> None:
    w = cal.volume_curve([1.0, 2.0, 3.0, 7.0], [10.0, 20.0, 30.0, 40.0], 0.0, 8.0, 2)
    assert w == pytest.approx([0.6, 0.4], rel=1e-15)


def test_volatility_of_a_constant_return() -> None:
    r = 0.001
    mids = [100.0 * math.exp(r * i) for i in range(11)]
    v = cal.volatility_curve([float(i) for i in range(11)], mids, 0.0, 10.0, 1)
    assert v.daily_vol == pytest.approx(r * math.sqrt(10.0), rel=1e-9)
    assert v.n_returns == 10


def test_fill_intensity_recovers_an_exact_exponential() -> None:
    """Counts c_b = C q^b fit exactly: k = -ln(q) / tick."""
    tick, q = 0.01, 0.5
    counts = [1024, 512, 256, 128, 64]
    dist: list[float] = []
    for b, c in enumerate(counts):
        nxt = counts[b + 1] if b + 1 < len(counts) else 0
        dist += [(b + 0.5) * tick] * (c - nxt)
    f = cal.fill_intensity(dist, tick, 5)
    assert f.k == pytest.approx(-math.log(q) / tick, rel=1e-12)
    assert f.halving_ticks == pytest.approx(1.0, rel=1e-12)
    assert f.touch_fills_per_day == pytest.approx(512.0, rel=1e-12)


def test_fill_intensity_round_trips_through_the_market_layer() -> None:
    """as_params_from_market with the fitted market units returns the fitted A and k."""
    case = next(c for c in CAL["cases"] if c["kind"] == "cal_fill_intensity")
    inp = case["inputs"]
    f = cal.fill_intensity(_array(inp, "distances"), inp["tick"], int(inp["n_levels"]))
    a = mk.as_params_from_market(
        price=100.0,
        tick=inp["tick"],
        daily_vol=0.01,
        touch_fills_per_day=f.touch_fills_per_day,
        halving_ticks=f.halving_ticks,
        skew_ticks=1.0,
        q_max=10,
        liq_ticks=1.0,
        dt_seconds=5.0,
    )
    assert a.k == pytest.approx(f.k, rel=1e-12)
    assert a.a_rate == pytest.approx(f.a_rate, rel=1e-12)


def test_fitted_presets_carry_their_source() -> None:
    for name, body in PRESETS.items():
        assert body["source"], name
        if name.startswith("itch_"):
            assert "2019-12-30" in body["source"]
            sym = name.split("_")[1].upper()
            fit = FITTED["symbols"][sym]
            assert body["as"]["halving_ticks"] == round(fit["fill_intensity"]["halving_ticks"], 3)


@pytest.mark.parametrize(
    ("call", "msg"),
    [
        (
            lambda: cal.volume_curve([0.0], [1.0], 0.0, 0.0, 1),
            "session_end must be finite and > session_start",
        ),
        (lambda: cal.volume_curve([0.0], [1.0], 0.0, 1.0, 0), "n_bins must be an integer"),
        (
            lambda: cal.volume_curve([0.0], [1.0, 2.0], 0.0, 1.0, 1),
            "times and sizes must have the same length",
        ),
        (lambda: cal.volume_curve([0.0], [-1.0], 0.0, 1.0, 1), "sizes must be finite and >= 0"),
        (lambda: cal.volume_curve([5.0], [1.0], 0.0, 1.0, 1), "no volume inside the session"),
        (
            lambda: cal.volatility_curve([0.0, 1.0], [1.0, 0.0], 0.0, 1.0, 1),
            "mids must be finite and > 0",
        ),
        (
            lambda: cal.volatility_curve([1.0, 0.0], [1.0, 1.0], 0.0, 1.0, 1),
            "times must be non-decreasing",
        ),
        (lambda: cal.volatility_curve([0.0], [1.0], 0.0, 1.0, 1), "no return inside the session"),
        (lambda: cal.spread_stats([], [], 0.01), "spreads must not be empty"),
        (lambda: cal.spread_stats([0.01], [100.0], 0.0), "tick must be finite and > 0"),
        (lambda: cal.spread_stats([-0.01], [100.0], 0.01), "spreads must be finite and >= 0"),
        (lambda: cal.fill_intensity([0.005], 0.01, 1), "n_levels must be an integer"),
        (lambda: cal.fill_intensity([0.005], 0.01, 3, days=0.0), "days must be finite and > 0"),
        (lambda: cal.fill_intensity([0.005], 0.01, 3, sides=0.0), "sides must be finite and > 0"),
        (lambda: cal.fill_intensity([-1.0], 0.01, 3), "distances must be finite and >= 0"),
        (
            lambda: cal.fill_intensity([0.005], 0.01, 3),
            "fewer than two distance buckets have fills",
        ),
        (
            lambda: cal.fill_intensity([0.015, 0.015], 0.01, 3),
            "fitted fill intensity does not decay with distance",
        ),
    ],
)
def test_validation(call: object, msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        call()  # type: ignore[operator]
