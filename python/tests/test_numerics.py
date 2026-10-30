"""Numeric checks of the pinned standard-error formulas (Sections 1.7, 2.2,
9.5) and of the product bounds of Section 0.

Each SE is checked two ways: recomputed from the per-path arrays (or from
the returned sample moments) with the pinned formula, and on a case with a
closed form (deterministic paths, so every SE is exactly zero).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import etrade as et
from conftest import ac_from_cfg, as_from_cfg, load_json
from etrade import mm

LIQ = load_json("params_liquid.json")


def as_params(**over: float) -> et.ASParams:
    cfg = {**LIQ, **over}
    return as_from_cfg(cfg)


# ---------------------------------------------------------------- Section 1.7


def test_simulate_is_se_formulas_recomputed() -> None:
    p = ac_from_cfg(LIQ)
    r = et.simulate_is(p, n_paths=500, seed=3)
    assert r.se_mean == math.sqrt(r.var / 500)
    assert r.se_var == r.var * math.sqrt(2.0 / 499)


def test_simulate_is_sample_moments_recomputed_from_the_draws() -> None:
    p = ac_from_cfg(LIQ)
    x = et.ac_trajectory(p)
    n = -np.diff(x)
    m = 300
    xi = np.random.default_rng(11).standard_normal((m, n.size))
    s = np.zeros(m)
    proceeds = np.zeros(m)
    for j in range(n.size):
        proceeds += n[j] * (s - p.eta * n[j] / p.tau)
        s = s + p.sigma * math.sqrt(p.tau) * xi[:, j] - p.gamma_p * n[j]
    shortfall = -proceeds
    r = et.simulate_is(p, n_paths=m, seed=11)
    assert r.mean == pytest.approx(float(np.mean(shortfall)), rel=1e-14)
    assert r.var == pytest.approx(float(np.var(shortfall, ddof=1)), rel=1e-12)


def test_simulate_is_zero_volatility_is_the_closed_form_with_zero_se() -> None:
    p = ac_from_cfg({**LIQ, "ac_sigma": 0.0})
    r = et.simulate_is(p, n_paths=50, seed=1)
    assert r.mean == pytest.approx(et.expected_cost(p), rel=1e-12)
    assert r.var == 0.0
    assert r.se_mean == 0.0
    assert r.se_var == 0.0


# ---------------------------------------------------------------- Section 2.2


def test_as_se_formulas_recomputed_from_per_path_arrays() -> None:
    r = et.simulate(as_params(), n_paths=400, seed=5)
    m = 400
    arrays = {
        "mean_pnl": r.pnl,
        "mean_abs_q": r.mean_abs_q_path,
        "mean_max_abs_q": r.max_abs_q_path,
        "rms_q": r.rms_q_path,
        "mean_fills": r.fills_path,
    }
    for stat, arr in arrays.items():
        want = math.sqrt(float(np.sum((arr - arr.mean()) ** 2)) / (m - 1)) / math.sqrt(m)
        assert r.se(stat) == pytest.approx(want, rel=1e-12), stat
    assert r.se("std_pnl") == r.std_pnl / math.sqrt(2.0 * (m - 1))


def test_as_no_quoting_gives_zero_pnl_and_zero_se() -> None:
    r = et.simulate(as_params(as_q_max=0), n_paths=20, seed=2)
    assert np.all(r.pnl == 0.0)
    for stat in ("mean_pnl", "std_pnl", "mean_abs_q", "mean_max_abs_q", "rms_q", "mean_fills"):
        assert r.se(stat) == 0.0, stat


def test_as_saturated_fills_closed_form() -> None:
    # sigma = 0 and a huge intensity: every quoted side fills every step, so
    # q stays at 0, each step earns the full spread, and every path is the
    # same number: fills = 2*n_steps, pnl = sum of the spreads.
    p = as_params(as_sigma=0.0, as_a_rate=1e12, as_q_max=1)
    r = et.simulate(p, n_paths=10, seed=9)
    n = p.n_steps
    spread = 2.0 / p.gamma * math.log1p(p.gamma / p.k)
    assert np.all(r.fills_path == 2.0 * n)
    assert r.mean_pnl == pytest.approx(n * spread, rel=1e-12)
    # Identical paths: the sample std is zero up to the rounding of the mean.
    assert np.all(r.pnl == r.pnl[0])
    for stat in ("mean_pnl", "std_pnl", "mean_fills", "mean_abs_q"):
        assert r.se(stat) <= 1e-14 * n * spread, stat


# ---------------------------------------------------------------- Section 9.5


def test_mm_se_formula_recomputed_from_per_path_arrays() -> None:
    cfg = mm.MMConfig(agent="glft", tick=0.01, maker_fee=0.002, informed_prob=0.2)
    r = mm.simulate_mm(as_params(), n_paths=300, seed=4, config=cfg)
    m = 300
    for stat in mm.MM_STATS:
        arr = r.paths[stat]
        want = math.sqrt(float(np.sum((arr - arr.mean()) ** 2)) / (m - 1)) / math.sqrt(m)
        assert r.se(stat) == pytest.approx(want, rel=1e-12, abs=1e-300), stat


def test_mm_no_quoting_gives_zero_se() -> None:
    r = mm.simulate_mm(as_params(as_q_max=0), n_paths=10, seed=1)
    for stat in mm.MM_STATS:
        assert r.se(stat) == 0.0, stat
        assert r.mean(stat) == 0.0, stat


def test_mm_saturated_fills_closed_form() -> None:
    p = as_params(as_sigma=0.0, as_a_rate=1e12, as_q_max=1)
    r = mm.simulate_mm(p, n_paths=8, seed=6, config=mm.MMConfig(maker_fee=0.01))
    n = p.n_steps
    assert r.mean("fills") == 2.0 * n
    assert r.mean("fees") == pytest.approx(0.01 * 2 * n, rel=1e-12)
    for stat in ("pnl", "fills", "fees", "spread", "rms_q"):
        assert np.all(r.paths[stat] == r.paths[stat][0]), stat
        assert r.se(stat) <= 1e-14 * max(1.0, abs(r.mean(stat))), stat


# ---------------------------------------------------------------- Section 0


def test_product_bound_constants() -> None:
    assert et.MAX_MC_WORK == 1_000_000_000
    assert et.MAX_AS_WORK == 10_000_000_000


def test_simulate_is_rejects_total_work_above_the_bound() -> None:
    p = ac_from_cfg({**LIQ, "ac_n_intervals": 1000})
    n_paths = et.MAX_MC_WORK // 1000 + 1
    with pytest.raises(ValueError, match=r"n_paths \* n_intervals must be <= 1000000000"):
        et.simulate_is(p, n_paths=n_paths, seed=0)
    # A supplied schedule is bounded by its own length.
    x = et.twap_holdings(1.0e6, 2000)
    with pytest.raises(ValueError, match="n_paths"):
        et.simulate_is(p, holdings=x, n_paths=et.MAX_MC_WORK // 2000 + 1, seed=0)
    assert et.simulate_is(p, n_paths=2, seed=0).n_paths == 2


def test_simulate_rejects_total_work_above_the_bound() -> None:
    p = as_params(as_dt=1e-5)  # 100000 steps
    with pytest.raises(ValueError, match=r"n_paths \* n_steps must be <= 10000000000"):
        et.simulate(p, n_paths=et.MAX_AS_WORK // 100_000 + 1, seed=0)
    assert et.simulate(as_params(), n_paths=2, seed=0).n_paths == 2
