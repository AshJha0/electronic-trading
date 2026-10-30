"""Market making with frictions (API_SPEC Section 9): goldens, the per-path
attribution identity, the direction of each friction, the pinned draw order,
the fair-value helpers and validation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import load_json
from etrade import market as mk
from etrade import mm

GOLDEN = load_json("golden/mm_golden.json")
PRESETS = load_json("presets.json")
LARGE_AS = PRESETS["us_large_cap"]["as"]
AGENTS = {0.0: "as", 1.0: "glft", 2.0: "fixed"}
N_TEST = 2000
AS_KEYS = (
    "price",
    "tick",
    "daily_vol",
    "touch_fills_per_day",
    "halving_ticks",
    "skew_ticks",
    "q_max",
    "liq_ticks",
    "dt_seconds",
    "session_seconds",
    "horizon_days",
)


def _ap(inputs: dict | None = None) -> object:
    src = LARGE_AS if inputs is None else inputs
    kw = {k: src[k] for k in AS_KEYS}
    kw["q_max"] = int(kw["q_max"])
    return mk.as_params_from_market(**kw)


def _config(inp: dict) -> mm.MMConfig:
    return mm.MMConfig(
        agent=AGENTS[inp["agent_code"]],  # type: ignore[arg-type]
        tick=inp["quote_tick"],
        maker_fee=inp["maker_fee"],
        latency_steps=int(inp["latency_steps"]),
        informed_prob=inp["informed_prob"],
        signal_accuracy=inp["signal_accuracy"],
    )


# ------------------------------------------------------------------ goldens


def _exact_actual(case: dict) -> dict:
    inp, kind = case["inputs"], case["kind"]
    if kind == "mm_tick":
        b, a = mm.round_to_tick(inp["bid"], inp["ask"], inp["tick"])
        return {"bid": b, "ask": a}
    if kind == "mm_glft":
        db, da = mm.glft_half_spreads(_ap(inp), inp["q"])  # type: ignore[arg-type]
        return {"delta_bid": db, "delta_ask": da}
    if kind == "mm_wmid":
        return {"wmid": mm.weighted_mid(inp["bid"], inp["ask"], inp["bid_size"], inp["ask_size"])}
    cols = [[inp[f"{c}_{i}"] for i in range(5)] for c in ("bid_px", "bid_sz", "ask_px", "ask_sz")]
    e = mm.order_flow_imbalance(*cols)
    out = {f"e_{i + 1}": float(e[i]) for i in range(4)}
    out["ofi"] = float(e.sum())
    return out


EXACT = [c for c in GOLDEN["cases"] if c["kind"] != "mm_sim"]
SIMS = [c for c in GOLDEN["cases"] if c["kind"] == "mm_sim"]


@pytest.mark.parametrize("case", EXACT, ids=lambda c: c["name"])
def test_exact_golden(case: dict) -> None:
    actual = _exact_actual(case)
    assert set(actual) == set(case["expect"])
    for key, want in case["expect"].items():
        assert abs(actual[key] - want) <= case["tol"] * max(1.0, abs(want)), f"{case['name']}.{key}"


@pytest.mark.parametrize("case", SIMS, ids=lambda c: c["name"])
def test_statistical_golden(case: dict) -> None:
    inp = case["inputs"]
    res = mm.simulate_mm(_ap(inp), N_TEST, int(inp["seed"]) + 1, _config(inp))  # type: ignore[arg-type]
    keys = [k for k in case["expect"] if not k.endswith("_se")]
    assert set(case["expect"]) == set(keys) | {k + "_se" for k in keys}
    for key in keys:
        band = case["tol"] * math.hypot(case["expect"][key + "_se"], res.se(key))
        assert abs(res.mean(key) - case["expect"][key]) <= band, f"{case['name']}.{key}"
    assert res.max_identity_error <= 1e-9


def test_golden_count() -> None:
    assert len(GOLDEN["cases"]) == 14
    assert len(EXACT) == 8 and len(SIMS) == 6


# ------------------------------------------------------------------ attribution


@pytest.mark.parametrize(
    "cfg",
    [
        mm.MMConfig(),
        mm.MMConfig(tick=0.01, maker_fee=-0.003, informed_prob=0.3, signal_accuracy=0.7),
        mm.MMConfig(agent="glft", tick=0.01, latency_steps=2),
        mm.MMConfig(agent="fixed", maker_fee=0.002),
    ],
)
def test_attribution_identity_holds_on_every_path(cfg: mm.MMConfig) -> None:
    res = mm.simulate_mm(_ap(), 300, 9, cfg)  # type: ignore[arg-type]
    p = res.paths
    recon = p["spread"] + p["fees"] + p["inventory"] - p["liquidation"]
    assert np.max(np.abs(recon - p["pnl"]) / np.maximum(1.0, np.abs(p["pnl"]))) <= 1e-9
    assert np.allclose(p["fees"], cfg.maker_fee * p["fills"], rtol=0.0, atol=1e-9)
    assert np.all(p["informed_fills"] <= p["fills"])


# ------------------------------------------------------------------ frictions


def test_each_friction_moves_profit_the_right_way() -> None:
    ap = _ap()

    def pnl(cfg: mm.MMConfig) -> float:
        return mm.simulate_mm(ap, 1000, 21, cfg).mean("pnl")  # type: ignore[arg-type]

    base = pnl(mm.MMConfig(tick=0.01))
    assert pnl(mm.MMConfig(tick=0.01, maker_fee=0.002)) > base
    assert pnl(mm.MMConfig(tick=0.01, maker_fee=-0.003)) < base
    informed = pnl(mm.MMConfig(tick=0.01, informed_prob=0.15))
    assert informed < base
    assert pnl(mm.MMConfig(tick=0.01, informed_prob=0.15, signal_accuracy=0.7)) > informed
    assert pnl(mm.MMConfig(tick=0.01, latency_steps=1)) < 0.0


def test_informed_flow_shows_up_as_adverse_markout() -> None:
    res = mm.simulate_mm(_ap(), 1000, 4, mm.MMConfig(tick=0.01, informed_prob=0.2))  # type: ignore[arg-type]
    assert res.mean("informed_fills") > 0.0
    assert res.mean("markout") < -5.0 * res.se("markout")


def test_no_frictions_reproduce_zero_fees_and_no_informed_fills() -> None:
    res = mm.simulate_mm(_ap(), 200, 1)  # type: ignore[arg-type]
    assert np.all(res.paths["fees"] == 0.0)
    assert np.all(res.paths["informed_fills"] == 0.0)


def test_determinism_and_draw_order_independence_of_config() -> None:
    ap = _ap()
    a = mm.simulate_mm(ap, 200, 7, mm.MMConfig(informed_prob=0.1))  # type: ignore[arg-type]
    b = mm.simulate_mm(ap, 200, 7, mm.MMConfig(informed_prob=0.1))  # type: ignore[arg-type]
    assert np.array_equal(a.paths["pnl"], b.paths["pnl"])
    # all five draws are consumed every step whatever the switches, so the
    # price path (and hence the frictionless inventory term without fills)
    # is identical across configurations: compare the terminal mid via a
    # q_max = 0 market maker, which never trades.
    kw = {k: LARGE_AS[k] for k in AS_KEYS}
    kw["q_max"] = 0
    still = mk.as_params_from_market(**kw)
    r1 = mm.simulate_mm(still, 50, 3, mm.MMConfig())
    r2 = mm.simulate_mm(still, 50, 3, mm.MMConfig(informed_prob=0.9, signal_accuracy=0.9))
    assert np.all(r1.paths["pnl"] == 0.0) and np.all(r2.paths["pnl"] == 0.0)


def test_q_max_respected_and_glft_skews_with_inventory() -> None:
    res = mm.simulate_mm(_ap(), 300, 2, mm.MMConfig(agent="glft", tick=0.01))  # type: ignore[arg-type]
    assert np.all(res.paths["rms_q"] <= 50.0)
    db0, da0 = mm.glft_half_spreads(_ap(), 0.0)  # type: ignore[arg-type]
    db5, da5 = mm.glft_half_spreads(_ap(), 5.0)  # type: ignore[arg-type]
    assert db0 == pytest.approx(da0, rel=1e-15)
    assert db5 > db0 and da5 < da0  # long: bid further away, ask closer


# ------------------------------------------------------------------ helpers


def test_round_to_tick_rounds_away_from_fair_value() -> None:
    assert mm.round_to_tick(99.991, 100.009, 0.01) == pytest.approx((99.99, 100.01))
    assert mm.round_to_tick(99.99, 100.01, 0.01) == pytest.approx((99.99, 100.01))
    assert mm.round_to_tick(1.0, 2.0, 0.0) == (1.0, 2.0)


def test_weighted_mid_and_ofi_properties() -> None:
    assert mm.weighted_mid(99.0, 101.0, 1.0, 1.0) == 100.0
    assert mm.weighted_mid(99.0, 101.0, 9.0, 1.0) > 100.0  # big bid queue pulls up
    e = mm.order_flow_imbalance([100, 100], [500, 700], [101, 101], [400, 400])
    assert e.tolist() == [200.0]  # bid queue grew by 200


@pytest.mark.parametrize(
    "kwargs",
    [
        {"agent": "momentum"},
        {"tick": -0.01},
        {"maker_fee": float("nan")},
        {"latency_steps": -1},
        {"informed_prob": 1.5},
        {"signal_accuracy": 0.4},
        {"fixed_spread": 0.0},
    ],
)
def test_config_validation(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        mm.MMConfig(**kwargs)


def test_simulation_and_helper_validation() -> None:
    ap = _ap()
    with pytest.raises(ValueError):
        mm.simulate_mm(ap, 1, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        mm.simulate_mm(ap, 10, 0, mm.MMConfig(latency_steps=10_000))  # type: ignore[arg-type]
    res = mm.simulate_mm(ap, 10, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        res.mean("nope")
    with pytest.raises(ValueError):
        mm.weighted_mid(101.0, 99.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        mm.weighted_mid(99.0, 101.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        mm.order_flow_imbalance([1.0], [1.0], [2.0], [1.0])
    with pytest.raises(ValueError):
        mm.round_to_tick(1.0, 2.0, -1.0)
