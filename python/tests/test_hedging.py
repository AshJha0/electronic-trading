"""Hedged market making and risk controls (API_SPEC Section 14): goldens, the
per-path attribution identity, no hedge = Section 9 in distribution, the
loss-limit overshoot bound, the effect of each control and validation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import load_json
from etrade import hedging as hg
from etrade import market as mk
from etrade import mm

GOLDEN = load_json("golden/hedging_golden.json")
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
BOOK = mm.MMConfig(tick=0.01, maker_fee=0.002, informed_prob=0.15)
FIXED = mm.MMConfig(agent="fixed", tick=0.01, maker_fee=0.002, informed_prob=0.15)
HEDGE = hg.HedgeConfig(rho=0.9, hedge_ratio=0.9, hedge_cost=0.005)


def _ap(inputs: dict | None = None) -> object:
    src = LARGE_AS if inputs is None else inputs
    kw = {k: src[k] for k in AS_KEYS}
    kw["q_max"] = int(kw["q_max"])
    return mk.as_params_from_market(**kw)


def _configs(inp: dict) -> tuple[mm.MMConfig, hg.HedgeConfig, hg.RiskConfig]:
    cfg = mm.MMConfig(
        agent=AGENTS[inp["agent_code"]],  # type: ignore[arg-type]
        tick=inp["quote_tick"],
        maker_fee=inp["maker_fee"],
        latency_steps=int(inp["latency_steps"]),
        informed_prob=inp["informed_prob"],
        signal_accuracy=inp["signal_accuracy"],
    )
    hedge = hg.HedgeConfig(
        rho=inp["rho"],
        hedge_ratio=inp["hedge_ratio"],
        hedge_cost=inp["hedge_cost"],
        hedge_threshold=inp["hedge_threshold"],
        hedge_target=inp["hedge_target"],
    )
    risk = hg.RiskConfig(
        loss_limit=inp["loss_limit"],
        position_limit=int(inp["position_limit"]),
        flatten_penalty=inp["flatten_penalty"],
        throttle_updates=int(inp["throttle_updates"]),
        throttle_window=int(inp["throttle_window"]),
        kill_step=int(inp["kill_step"]),
    )
    return cfg, hedge, risk


def _sim(
    cfg: mm.MMConfig,
    hedge: hg.HedgeConfig | None = None,
    risk: hg.RiskConfig | None = None,
    n: int = 600,
    seed: int = 11,
) -> hg.HedgedResult:
    return hg.simulate_hedged(_ap(), n, seed, cfg, hedge, risk)  # type: ignore[arg-type]


# ------------------------------------------------------------------ goldens


EXACT = [c for c in GOLDEN["cases"] if c["kind"] != "hedge_sim"]
SIMS = [c for c in GOLDEN["cases"] if c["kind"] == "hedge_sim"]


def _exact_actual(case: dict) -> dict:
    inp = case["inputs"]
    if case["kind"] == "hedge_trade":
        dy = hg.hedge_trade(
            inp["q"], inp["y"], inp["hedge_ratio"], inp["hedge_threshold"], inp["hedge_target"]
        )
        return {"dy": dy}
    values = [inp[f"v_{i}"] for i in range(10)]
    return {"q": hg.quantile(values, inp["p"])}


@pytest.mark.parametrize("case", EXACT, ids=lambda c: c["name"])
def test_exact_golden(case: dict) -> None:
    actual = _exact_actual(case)
    assert set(actual) == set(case["expect"])
    for key, want in case["expect"].items():
        assert abs(actual[key] - want) <= case["tol"] * max(1.0, abs(want)), f"{case['name']}.{key}"


@pytest.mark.parametrize("case", SIMS, ids=lambda c: c["name"])
def test_statistical_golden(case: dict) -> None:
    inp = case["inputs"]
    cfg, hedge, risk = _configs(inp)
    res = hg.simulate_hedged(_ap(inp), N_TEST, int(inp["seed"]) + 1, cfg, hedge, risk)  # type: ignore[arg-type]
    keys = [k for k in case["expect"] if not k.endswith("_se")]
    assert set(case["expect"]) == set(keys) | {k + "_se" for k in keys}
    assert len(keys) == 16
    for key in keys:
        if key == "pnl_var":
            got, se = res.var("pnl"), res.var_se("pnl")
        else:
            got, se = res.mean(key), res.se(key)
        band = case["tol"] * math.hypot(case["expect"][key + "_se"], se)
        assert abs(got - case["expect"][key]) <= band + 1e-12, f"{case['name']}.{key}"
    assert res.max_identity_error <= 1e-9


def test_golden_count() -> None:
    assert len(GOLDEN["cases"]) == 18
    assert len(EXACT) == 8 and len(SIMS) == 10


# ------------------------------------------------------------------ attribution


@pytest.mark.parametrize(
    ("cfg", "hedge", "risk"),
    [
        (mm.MMConfig(), None, None),
        (BOOK, HEDGE, None),
        (
            mm.MMConfig(agent="glft", tick=0.01, latency_steps=2),
            HEDGE,
            hg.RiskConfig(position_limit=3, flatten_penalty=0.02),
        ),
        (
            FIXED,
            hg.HedgeConfig(
                rho=-0.5, hedge_ratio=0.4, hedge_cost=0.01, hedge_threshold=2.0, hedge_target=0.5
            ),
            hg.RiskConfig(
                loss_limit=2.0,
                flatten_penalty=0.01,
                throttle_updates=2,
                throttle_window=5,
                kill_step=3000,
            ),
        ),
    ],
)
def test_attribution_identity_holds_on_every_path(
    cfg: mm.MMConfig, hedge: hg.HedgeConfig | None, risk: hg.RiskConfig | None
) -> None:
    res = _sim(cfg, hedge, risk, n=300, seed=9)
    p = res.paths
    recon = (
        p["spread"]
        + p["fees"]
        + p["inventory"]
        - p["liquidation"]
        + p["hedge_pnl"]
        - p["hedge_costs"]
        - p["risk_costs"]
    )
    assert np.max(np.abs(recon - p["pnl"]) / np.maximum(1.0, np.abs(p["pnl"]))) <= 1e-9
    assert res.max_identity_error <= 1e-9
    assert np.all(p["hedge_costs"] >= 0.0) and np.all(p["risk_costs"] >= 0.0)


# ------------------------------------------------------------------ properties


def test_no_hedge_no_controls_is_section_9_in_distribution() -> None:
    ap = _ap()
    for cfg in (BOOK, FIXED):
        a = hg.simulate_hedged(ap, N_TEST, 31, cfg)  # type: ignore[arg-type]
        b = mm.simulate_mm(ap, N_TEST, 32, cfg)  # type: ignore[arg-type]
        for stat in mm.MM_STATS:
            band = 4.0 * math.hypot(a.se(stat), b.se(stat))
            assert abs(a.mean(stat) - b.mean(stat)) <= band + 1e-12, stat
        p = a.paths
        for stat in ("hedge_pnl", "hedge_costs", "risk_costs", "loss_stop", "kill_stop"):
            assert np.all(p[stat] == 0.0)
        assert np.array_equal(p["rms_x"], p["rms_q"])
        assert np.all(p["quote_updates"] == ap.n_steps)  # type: ignore[attr-defined]


def test_loss_limit_overshoot_is_at_most_one_step_move() -> None:
    res = _sim(FIXED, risk=hg.RiskConfig(loss_limit=3.0, flatten_penalty=0.01), n=800, seed=5)
    p = res.paths
    stopped = p["loss_stop"] == 1.0
    assert stopped.sum() > 100
    assert np.all(p["overshoot"] <= p["stop_move"] + 1e-9)
    assert np.all(p["overshoot"][~stopped] == 0.0)
    # once stopped, the book is flat and no longer quotes
    assert np.all(p["liquidation"][stopped] == 0.0)
    assert np.all(p["quote_updates"][stopped] < 4680)


def test_hedging_cuts_variance_and_skew_at_a_cost() -> None:
    fixed_un = _sim(FIXED)
    fixed_h = _sim(FIXED, HEDGE)
    assert fixed_h.var("pnl") < 0.5 * fixed_un.var("pnl")
    assert fixed_h.mean("rms_x") < 0.5 * fixed_h.mean("rms_q")
    assert fixed_h.mean("hedge_costs") > 0.0 and fixed_h.mean("hedge_trades") > 0.0
    as_un = _sim(BOOK)
    as_h = _sim(BOOK, HEDGE)
    assert as_h.mean("rms_skew") < 0.6 * as_un.mean("rms_skew")
    assert as_h.var("pnl") < as_un.var("pnl")


def test_each_control_triggers_and_cuts_the_tail() -> None:
    base = _sim(FIXED)
    pos = _sim(FIXED, risk=hg.RiskConfig(position_limit=10, flatten_penalty=0.01))
    assert pos.mean("pos_flattens") > 1.0 and pos.mean("risk_costs") > 0.0
    assert pos.quantile("pnl", 0.05) > base.quantile("pnl", 0.05)
    loss = _sim(FIXED, risk=hg.RiskConfig(loss_limit=5.0, flatten_penalty=0.01))
    assert loss.mean("loss_stop") > 0.2
    assert loss.quantile("pnl", 0.05) > base.quantile("pnl", 0.05)
    kill = _sim(FIXED, risk=hg.RiskConfig(kill_step=1000))
    assert np.all(kill.paths["kill_stop"] == 1.0)
    assert np.all(kill.paths["quote_updates"] == 1000.0)
    thr = _sim(BOOK, risk=hg.RiskConfig(throttle_updates=1, throttle_window=3))
    assert np.all(thr.paths["quote_updates"] == 1560.0)  # ceil(4680 / 3)
    assert thr.mean("pnl") < _sim(BOOK).mean("pnl")  # stale quotes are picked off


def test_determinism() -> None:
    a = _sim(
        FIXED, HEDGE, hg.RiskConfig(loss_limit=4.0, throttle_updates=2, throttle_window=3), n=100
    )
    b = _sim(
        FIXED, HEDGE, hg.RiskConfig(loss_limit=4.0, throttle_updates=2, throttle_window=3), n=100
    )
    for stat in hg.HEDGE_STATS:
        assert np.array_equal(a.paths[stat], b.paths[stat])


# ------------------------------------------------------------------ helpers


def test_hedge_trade_leaves_target_fraction_uncovered() -> None:
    q, y, h = 9.0, -2.0, 0.8
    u = q + y / h
    dy = hg.hedge_trade(q, y, h, 1.0, 0.25)
    assert q + (y + dy) / h == pytest.approx(0.25 * u, rel=1e-14)
    assert hg.hedge_trade(1.0, 0.0, 0.9, 1.0, 0.0) == 0.0  # |u| <= threshold
    assert hg.hedge_trade(5.0, 0.0, 0.0, 0.0, 0.0) == 0.0


def test_quantile_matches_numpy_type_7() -> None:
    rng = np.random.default_rng(3)
    v = rng.standard_normal(57)
    for p in (0.0, 0.05, 0.33, 0.5, 0.95, 1.0):
        assert hg.quantile(v, p) == pytest.approx(float(np.quantile(v, p)), abs=1e-14)
    assert hg.quantile([4.0], 0.3) == 4.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rho": 1.5},
        {"hedge_ratio": -0.1},
        {"hedge_cost": float("nan")},
        {"hedge_threshold": -1.0},
        {"hedge_target": 1.0},
    ],
)
def test_hedge_config_validation(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        hg.HedgeConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"loss_limit": -1.0},
        {"loss_limit": float("inf")},
        {"position_limit": -1},
        {"position_limit": 2.5},
        {"flatten_penalty": -0.01},
        {"throttle_updates": -1},
        {"throttle_window": 0},
        {"kill_step": -2},
    ],
)
def test_risk_config_validation(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        hg.RiskConfig(**kwargs)


def test_simulation_and_helper_validation() -> None:
    ap = _ap()
    with pytest.raises(ValueError):
        hg.simulate_hedged(ap, 1, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        hg.simulate_hedged(ap, 10, 0, risk=hg.RiskConfig(kill_step=10_000))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        hg.simulate_hedged(
            ap, 10, 0, risk=hg.RiskConfig(throttle_updates=1, throttle_window=10_000)
        )  # type: ignore[arg-type]
    res = hg.simulate_hedged(ap, 10, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        res.mean("nope")
    with pytest.raises(ValueError):
        hg.quantile([], 0.5)
    with pytest.raises(ValueError):
        hg.quantile([1.0, 2.0], 1.5)
    with pytest.raises(ValueError):
        hg.hedge_trade(1.0, 0.0, 0.9, 0.0, 1.0)
