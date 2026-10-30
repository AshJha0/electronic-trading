"""Market-unit layer (API_SPEC Section 7): golden cases, the paper identity,
the defining properties of each conversion, and validation."""

from __future__ import annotations

import math

import pytest

import etrade as et
from conftest import load_json
from etrade import market as mk

MARKET = load_json("golden/market_golden.json")
PRESETS = load_json("presets.json")
LIQ = load_json("params_liquid.json")

INT_KEYS = {"n_intervals", "q_max"}


def _kwargs(inputs: dict) -> dict:
    return {k: int(v) if k in INT_KEYS else v for k, v in inputs.items()}


def _actual(case: dict) -> dict:
    inp = _kwargs(case["inputs"])
    if case["kind"] == "market_ac":
        p = mk.ac_params_from_market(**inp)
        notional = inp["total_shares"] * inp["price"]
        return {
            "sigma": p.sigma,
            "eta": p.eta,
            "gamma_p": p.gamma_p,
            "kappa": et.ac_kappa(p),
            "e_cost_bps": mk.bps(et.expected_cost(p), notional),
            "sd_cost_bps": mk.bps(math.sqrt(et.variance_cost(p)), notional),
            "spread_cost_bps": mk.bps(
                mk.half_spread_cost(inp["spread"], inp["total_shares"]), notional
            ),
        }
    a = mk.as_params_from_market(**inp)
    return {
        "sigma": a.sigma,
        "k": a.k,
        "a_rate": a.a_rate,
        "gamma": a.gamma,
        "eta_liq": a.eta_liq,
        "dt": a.dt,
        "n_steps": float(a.n_steps),
        "spread_open_bps": mk.bps(et.optimal_spread(a, 0.0), inp["price"]),
        "spread_close_bps": mk.bps(et.optimal_spread(a, a.horizon), inp["price"]),
    }


@pytest.mark.parametrize("case", MARKET["cases"], ids=lambda c: c["name"])
def test_market_golden(case: dict) -> None:
    actual = _actual(case)
    assert set(actual) == set(case["expect"]), "every expect key must be asserted"
    for key, want in case["expect"].items():
        got = actual[key]
        assert abs(got - want) <= case["tol"] * max(1.0, abs(want)), f"{case['name']}.{key}"


def test_market_golden_count_and_presets_covered() -> None:
    assert len(MARKET["cases"]) == 9
    names = {c["name"] for c in MARKET["cases"]}
    for preset, body in PRESETS.items():
        for side in ("ac", "as"):
            if side in body:
                assert f"market_{side}_{preset}" in names


def test_paper_calibration_reproduces_liquid_set() -> None:
    """Almgren-Chriss (2000): one spread per 1% ADV (temporary) and per 10%
    ADV (permanent) gives exactly the repository's liquid AC parameters."""
    p = mk.ac_params_from_market(**_kwargs(PRESETS["paper_almgren_chriss_2000"]["ac"]))
    assert p.eta == pytest.approx(LIQ["ac_eta"], rel=1e-15)
    assert p.gamma_p == pytest.approx(LIQ["ac_gamma_p"], rel=1e-15)
    assert p.sigma == pytest.approx(LIQ["ac_sigma"], rel=1e-15)


def test_as_conversion_defining_properties() -> None:
    c = _kwargs(PRESETS["us_large_cap"]["as"])
    a = mk.as_params_from_market(**c)
    tick = c["tick"]
    # the touch (half a tick from mid) fills at the stated rate
    assert et.intensity(a, tick / 2.0) == pytest.approx(c["touch_fills_per_day"], rel=1e-12)
    # intensity halves every halving_ticks ticks
    d0 = tick / 2.0
    ratio = et.intensity(a, d0 + c["halving_ticks"] * tick) / et.intensity(a, d0)
    assert ratio == pytest.approx(0.5, rel=1e-12)
    # opening skew per unit of inventory is skew_ticks ticks
    skew = a.s0 - et.reservation_price(a, a.s0, 1.0, 0.0)
    assert skew == pytest.approx(c["skew_ticks"] * tick, rel=1e-12)
    assert a.n_steps == 4680


def test_temp_and_perm_fractions_scale_impact() -> None:
    base = _kwargs(PRESETS["us_large_cap"]["ac"])
    p1 = mk.ac_params_from_market(**base)
    p2 = mk.ac_params_from_market(**base, temp_frac=0.02, perm_frac=0.2)
    assert p2.eta == pytest.approx(p1.eta / 2.0, rel=1e-15)
    assert p2.gamma_p == pytest.approx(p1.gamma_p / 2.0, rel=1e-15)


def test_bps_and_half_spread() -> None:
    assert mk.bps(5.0, 1e4) == pytest.approx(5.0, rel=1e-15)
    assert mk.bps(-1.0, 1e4) == -1.0
    assert mk.half_spread_cost(0.01, 1000.0) == pytest.approx(5.0, rel=1e-15)
    with pytest.raises(ValueError):
        mk.bps(1.0, 0.0)
    with pytest.raises(ValueError):
        mk.bps(float("nan"), 1.0)
    with pytest.raises(ValueError):
        mk.half_spread_cost(0.0, 1.0)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("price", 0.0),
        ("spread", -0.01),
        ("adv", float("inf")),
        ("daily_vol", float("nan")),
        ("temp_frac", 0.0),
        ("temp_frac", 1.5),
        ("perm_frac", 0.0),
    ],
)
def test_ac_market_validation(field: str, bad: float) -> None:
    kw = _kwargs(PRESETS["us_large_cap"]["ac"])
    kw[field] = bad
    with pytest.raises(ValueError, match=field):
        mk.ac_params_from_market(**kw)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("price", -1.0),
        ("tick", 0.0),
        ("daily_vol", 0.0),
        ("touch_fills_per_day", 0.0),
        ("halving_ticks", float("inf")),
        ("skew_ticks", 0.0),
        ("liq_ticks", -1.0),
        ("dt_seconds", 0.0),
        ("session_seconds", float("nan")),
        ("horizon_days", 0.0),
    ],
)
def test_as_market_validation(field: str, bad: float) -> None:
    kw = _kwargs(PRESETS["us_large_cap"]["as"])
    kw[field] = bad
    with pytest.raises(ValueError, match=field):
        mk.as_params_from_market(**kw)


def test_as_market_step_not_dividing_session_is_error() -> None:
    kw = _kwargs(PRESETS["us_large_cap"]["as"])
    kw["dt_seconds"] = 7.0  # 23400 / 7 is not an integer
    with pytest.raises(ValueError, match="multiple of dt"):
        mk.as_params_from_market(**kw)
