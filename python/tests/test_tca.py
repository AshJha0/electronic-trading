"""Post-trade TCA (API_SPEC Section 10): golden cases, sign conventions,
invariances, validation, and the t-test against SciPy."""

from __future__ import annotations

import math

import pytest
from scipy import special, stats

from conftest import load_json
from etrade import tca

TCA = load_json("golden/tca_golden.json")

VOLS = [1000.0, 2000.0, 1500.0, 500.0]
VWAPS = [100.00, 100.05, 100.12, 100.08]
BINS = [0, 1, 1, 2]
QTYS = [100.0, 150.0, 50.0, 200.0]
PXS = [100.02, 100.06, 100.07, 100.15]


def _arr(inp: dict, prefix: str, n_key: str) -> list[float]:
    return [inp[f"{prefix}_{i}"] for i in range(int(inp[n_key]))]


def _ints(inp: dict, prefix: str, n_key: str) -> list[int]:
    return [int(v) for v in _arr(inp, prefix, n_key)]


def _side(inp: dict) -> str:
    return "buy" if inp["side_buy"] == 1.0 else "sell"


def _actual(case: dict) -> dict[str, float]:
    inp = case["inputs"]
    kind = case["kind"]
    if kind == "slippage":
        r = tca.slippage(
            _side(inp),
            inp["arrival_price"],
            inp["close_price"],
            _ints(inp, "bin", "n_fills"),
            _arr(inp, "qty", "n_fills"),
            _arr(inp, "px", "n_fills"),
            _arr(inp, "vol", "n_bins"),
            _arr(inp, "vwap", "n_bins"),
        )
        return {
            "executed_qty": r.executed_qty,
            "avg_price": r.avg_price,
            "market_vwap": r.market_vwap,
            "interval_vwap": r.interval_vwap,
            "arrival_bps": r.arrival_bps,
            "vwap_bps": r.vwap_bps,
            "interval_vwap_bps": r.interval_vwap_bps,
            "close_bps": r.close_bps,
        }
    if kind == "markout":
        m = tca.markouts(
            _side(inp),
            _ints(inp, "step", "n_fills"),
            _arr(inp, "qty", "n_fills"),
            _arr(inp, "mid", "n_mids"),
            _ints(inp, "h", "n_horizons"),
        )
        out = {f"per_fill_{i}": v for i, v in enumerate(m.per_fill)}
        out.update({f"per_share_{i}": v for i, v in enumerate(m.per_share)})
        return out
    if kind == "participation":
        p = tca.participation(
            _ints(inp, "bin", "n_fills"), _arr(inp, "qty", "n_fills"), _arr(inp, "vol", "n_bins")
        )
        out = {f"rate_{i}": v for i, v in enumerate(p.per_bin)}
        out["overall"] = p.overall
        return out
    if kind == "welch":
        w = tca.welch_ttest(_arr(inp, "a", "n_a"), _arr(inp, "b", "n_b"))
        return {"mean_diff": w.mean_diff, "t_stat": w.t_stat, "df": w.df, "p_value": w.p_value}
    assert kind == "t_cdf"
    return {"cdf": tca.student_t_cdf(inp["t"], inp["df"])}


@pytest.mark.parametrize("case", TCA["cases"], ids=lambda c: c["name"])
def test_tca_golden(case: dict) -> None:
    actual = _actual(case)
    assert set(actual) == set(case["expect"]), "every expect key must be asserted"
    for key, want in case["expect"].items():
        got = actual[key]
        assert abs(got - want) <= case["tol"] * max(1.0, abs(want)), f"{case['name']}.{key}"


def test_tca_golden_count() -> None:
    assert len(TCA["cases"]) == 11


# ------------------------------------------------------------------ slippage


def test_slippage_hand_computed_buy() -> None:
    r = tca.slippage("buy", 100.0, 100.0, [0], [10.0], [100.10], [100.0], [100.0])
    assert r.arrival_bps == pytest.approx(10.0, rel=1e-12)
    assert r.vwap_bps == pytest.approx(10.0, rel=1e-12)


def test_buy_sell_symmetry() -> None:
    b = tca.slippage("buy", 100.01, 100.09, BINS, QTYS, PXS, VOLS, VWAPS)
    s = tca.slippage("sell", 100.01, 100.09, BINS, QTYS, PXS, VOLS, VWAPS)
    for f in ("arrival_bps", "vwap_bps", "interval_vwap_bps", "close_bps"):
        assert getattr(b, f) == pytest.approx(-getattr(s, f), abs=1e-12)
    assert b.avg_price == s.avg_price


def test_slippage_scale_invariance() -> None:
    """Multiplying every price by c leaves the bps unchanged."""
    c = 3.7
    r1 = tca.slippage("buy", 100.01, 100.09, BINS, QTYS, PXS, VOLS, VWAPS)
    r2 = tca.slippage(
        "buy",
        100.01 * c,
        100.09 * c,
        BINS,
        QTYS,
        [p * c for p in PXS],
        VOLS,
        [v * c for v in VWAPS],
    )
    for f in ("arrival_bps", "vwap_bps", "interval_vwap_bps", "close_bps"):
        assert getattr(r2, f) == pytest.approx(getattr(r1, f), rel=1e-10)


def test_slippage_shift_invariance_per_share() -> None:
    """Adding a constant to every price leaves the per-share cost unchanged."""
    k = 25.0
    r1 = tca.slippage("sell", 100.01, 100.09, BINS, QTYS, PXS, VOLS, VWAPS)
    r2 = tca.slippage(
        "sell",
        100.01 + k,
        100.09 + k,
        BINS,
        QTYS,
        [p + k for p in PXS],
        VOLS,
        [v + k for v in VWAPS],
    )
    for f, bench in (
        ("arrival_bps", (100.01, 100.01 + k)),
        ("close_bps", (100.09, 100.09 + k)),
        ("vwap_bps", (r1.market_vwap, r2.market_vwap)),
        ("interval_vwap_bps", (r1.interval_vwap, r2.interval_vwap)),
    ):
        c1 = getattr(r1, f) * bench[0] / 1e4
        c2 = getattr(r2, f) * bench[1] / 1e4
        assert c2 == pytest.approx(c1, abs=1e-9)


def test_interval_vwap_uses_only_traded_bins() -> None:
    r = tca.slippage("buy", 100.0, 100.0, [2, 2], [10.0, 0.0], [100.0, 99.0], VOLS, VWAPS)
    assert r.interval_vwap == pytest.approx(100.12, rel=1e-15)


def test_slippage_vs_decomposition_trading_component() -> None:
    """Arrival slippage times notional at arrival is the Perold trading cost."""
    from etrade import decompose_shortfall

    r = tca.slippage("buy", 100.01, 100.09, BINS, QTYS, PXS, VOLS, VWAPS)
    d = decompose_shortfall("buy", 100.01, 100.01, 100.09, 500.0, QTYS, PXS)
    assert r.arrival_bps * 100.01 * r.executed_qty / 1e4 == pytest.approx(d.trading, rel=1e-10)


@pytest.mark.parametrize(
    "kwargs,msg",
    [
        ({"side": "hold"}, "side"),
        ({"arrival_price": 0.0}, "arrival_price"),
        ({"close_price": float("nan")}, "close_price"),
        ({"fill_bins": [0, 1, 1, 4]}, "fill_bins"),
        ({"fill_bins": [0, 1, 1, 1.0]}, "fill_bins"),
        ({"fill_bins": [0, 1, 1]}, "equal length"),
        ({"fill_qtys": [100.0, -1.0, 50.0, 200.0]}, "fill_qtys"),
        ({"fill_prices": [100.0, 0.0, 1.0, 1.0]}, "fill_prices"),
        ({"fill_qtys": [0.0, 0.0, 0.0, 0.0]}, "executed quantity"),
        ({"bin_volumes": [1.0, 2.0]}, "bin arrays"),
        ({"bin_volumes": [0.0, 0.0, 0.0, 0.0]}, "total market volume"),
        ({"bin_volumes": [1.0, -2.0, 1.0, 1.0]}, "bin_volumes"),
        ({"bin_vwaps": [1.0, 1.0, float("inf"), 1.0]}, "bin_vwaps"),
        ({"bin_volumes": [0.0, 0.0, 0.0, 1.0]}, "traded bins"),
    ],
)
def test_slippage_validation(kwargs: dict, msg: str) -> None:
    base = {
        "side": "buy",
        "arrival_price": 100.0,
        "close_price": 100.0,
        "fill_bins": BINS,
        "fill_qtys": QTYS,
        "fill_prices": PXS,
        "bin_volumes": VOLS,
        "bin_vwaps": VWAPS,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=msg):
        tca.slippage(**base)


def test_empty_fills_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        tca.slippage("buy", 1.0, 1.0, [], [], [], VOLS, VWAPS)


# ------------------------------------------------------------------ markouts


def test_markout_sign_and_aggregation() -> None:
    mids = [10.0, 10.5, 9.0]
    buy = tca.markouts("buy", [0, 1], [1.0, 3.0], mids, [0, 1])
    assert buy.per_fill == (0.0, pytest.approx((0.5 - 1.5) / 2))
    assert buy.per_share[1] == pytest.approx((0.5 - 4.5) / 4)
    sell = tca.markouts("sell", [0, 1], [1.0, 3.0], mids, [0, 1])
    assert sell.per_fill[1] == pytest.approx(-buy.per_fill[1])
    assert sell.horizons == (0, 1)


@pytest.mark.parametrize(
    "kwargs,msg",
    [
        ({"side": "x"}, "side"),
        ({"mids": []}, "mids must be non-empty"),
        ({"mids": [1.0, float("nan"), 1.0, 1.0]}, "mids must be finite"),
        ({"fill_steps": [0, 4]}, "fill_steps"),
        ({"horizons": []}, "horizons must be non-empty"),
        ({"horizons": [-1]}, "horizons"),
        ({"horizons": [3]}, "exceeds mid series"),
        ({"fill_qtys": [0.0, 0.0]}, "executed quantity"),
    ],
)
def test_markout_validation(kwargs: dict, msg: str) -> None:
    base = {
        "side": "buy",
        "fill_steps": [0, 1],
        "fill_qtys": [1.0, 1.0],
        "mids": [1.0, 1.0, 1.0, 1.0],
        "horizons": [0, 1],
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=msg):
        tca.markouts(**base)


# ------------------------------------------------------------- participation


def test_participation_overall_matches_analytics() -> None:
    from etrade import participation_rate

    p = tca.participation(BINS, QTYS, VOLS)
    assert p.overall == pytest.approx(participation_rate(sum(QTYS), sum(VOLS)), rel=1e-15)
    assert p.per_bin == (0.1, 0.1, pytest.approx(200.0 / 1500.0), 0.0)


def test_participation_validation() -> None:
    with pytest.raises(ValueError, match="exceeds market volume"):
        tca.participation([0], [1001.0], VOLS)
    with pytest.raises(ValueError, match="exceeds market volume"):
        tca.participation([1], [1.0], [1.0, 0.0])
    with pytest.raises(ValueError, match="total market volume"):
        tca.participation([0], [0.0], [0.0])
    with pytest.raises(ValueError, match="bins must be non-empty"):
        tca.participation([0], [0.0], [])


# --------------------------------------------------------------- statistics


@pytest.mark.parametrize("x", [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 7.3, 30.0, 171.5, 1e4])
def test_ln_gamma_against_math(x: float) -> None:
    assert tca.ln_gamma(x) == pytest.approx(math.lgamma(x), rel=1e-13, abs=1e-14)


@pytest.mark.parametrize(
    "a,b,x",
    [(0.5, 0.5, 0.3), (2.0, 3.0, 0.7), (15.0, 0.5, 0.95), (0.25, 0.5, 0.01), (400.0, 0.5, 0.999)],
)
def test_betainc_against_scipy(a: float, b: float, x: float) -> None:
    assert tca.betainc_reg(a, b, x) == pytest.approx(special.betainc(a, b, x), rel=1e-12)


@pytest.mark.parametrize("df", [0.5, 1.0, 2.0, 3.5, 10.0, 30.0, 250.0])
@pytest.mark.parametrize("t", [-6.0, -2.0, -0.4, 0.0, 0.7, 1.96, 4.0])
def test_t_cdf_against_scipy(t: float, df: float) -> None:
    assert tca.student_t_cdf(t, df) == pytest.approx(stats.t.cdf(t, df), rel=1e-11, abs=1e-15)


def test_t_cdf_known_values() -> None:
    assert tca.student_t_cdf(1.0, 1.0) == pytest.approx(0.75, rel=1e-14)  # Cauchy
    assert tca.student_t_cdf(0.0, 4.0) == 0.5
    # df = 2 has a closed form: 1/2 + t / (2 sqrt(2 + t^2))
    assert tca.student_t_cdf(1.5, 2.0) == pytest.approx(0.5 + 1.5 / (2 * math.sqrt(4.25)), 1e-14)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_welch_against_scipy(seed: int) -> None:
    import numpy as np

    rng = np.random.default_rng(seed)
    a = list(rng.normal(3.0, 2.0, size=20 + seed))
    b = list(rng.normal(2.0, 5.0, size=12))
    w = tca.welch_ttest(a, b)
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert w.t_stat == pytest.approx(ref.statistic, rel=1e-12)
    assert w.df == pytest.approx(ref.df, rel=1e-12)
    assert w.p_value == pytest.approx(ref.pvalue, rel=1e-10)
    assert w.mean_diff == pytest.approx(float(np.mean(a) - np.mean(b)), rel=1e-12)


def test_welch_antisymmetry() -> None:
    a = [1.0, 2.0, 4.0]
    b = [0.5, 0.7, 0.2, 0.9]
    ab = tca.welch_ttest(a, b)
    ba = tca.welch_ttest(b, a)
    assert ba.t_stat == pytest.approx(-ab.t_stat, rel=1e-15)
    assert ba.p_value == pytest.approx(ab.p_value, rel=1e-15)


def test_statistics_validation() -> None:
    with pytest.raises(ValueError, match="a must have at least 2"):
        tca.welch_ttest([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="b must be finite"):
        tca.welch_ttest([1.0, 2.0], [1.0, float("nan")])
    with pytest.raises(ValueError, match="zero variance"):
        tca.welch_ttest([1.0, 1.0], [2.0, 2.0])
    with pytest.raises(ValueError, match="df"):
        tca.student_t_cdf(1.0, 0.0)
    with pytest.raises(ValueError, match="t must be finite"):
        tca.student_t_cdf(float("inf"), 1.0)
    with pytest.raises(ValueError, match="x must be in"):
        tca.betainc_reg(1.0, 1.0, 1.5)
    with pytest.raises(ValueError, match="a must be finite"):
        tca.betainc_reg(0.0, 1.0, 0.5)
    with pytest.raises(ValueError, match="x must be finite"):
        tca.ln_gamma(-1.0)
    assert tca.betainc_reg(2.0, 3.0, 0.0) == 0.0
    assert tca.betainc_reg(2.0, 3.0, 1.0) == 1.0
