"""Execution analytics: implementation-shortfall decomposition, participation
rate, and a tiny native OLS for market-impact estimation.

Implementation-shortfall decomposition (Perold / Wagner convention)
-------------------------------------------------------------------
Parent order of ``X`` shares, decision price ``P_d`` (price when the decision
was made), arrival price ``P_0`` (price when trading started), fills
``(n_j, p_j)`` with executed quantity ``Q = sum n_j``, remainder
``R = X - Q``, closing price ``P_T``.  All components are *costs in currency*
(positive = money lost versus the paper portfolio).  For a **sell** order:

    total       = X * P_d - sum_j n_j * p_j - R * P_T
    delay       = Q * (P_d - P_0)          (drift before trading started)
    trading     = sum_j n_j * (P_0 - p_j)  (execution slippage vs arrival)
    opportunity = R * (P_d - P_T)          (unfilled shares vs decision)

and the identity  total == delay + trading + opportunity  holds *exactly*
(algebraic, not approximate).  For a **buy** order every price difference is
mirrored:

    total       = sum_j n_j * p_j + R * P_T - X * P_d
    delay       = Q * (P_0 - P_d)
    trading     = sum_j n_j * (p_j - P_0)
    opportunity = R * (P_T - P_d)

Participation rate: executed volume / market volume over the same window.
Market volume must include the trader's own fills, so a rate above 1 is an
input error (pinned, API_SPEC.md Section 3.2).

Impact estimation: ordinary least squares of per-share cost against
participation rate, ``cost = intercept + slope * participation``, with a
tiny native OLS (closed-form slope/intercept/R^2), no external fitting
library.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _is_int(v: object) -> bool:
    """True for Python/NumPy integers, False for bool and floats."""
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


@dataclass(frozen=True)
class ISDecomposition:
    """Implementation-shortfall components (currency costs; positive = loss)."""

    total: float
    delay: float
    trading: float
    opportunity: float
    executed_qty: float
    remaining_qty: float


def decompose_shortfall(
    side: str,
    decision_price: float,
    arrival_price: float,
    close_price: float,
    total_qty: float,
    fill_qtys: Sequence[float],
    fill_prices: Sequence[float],
) -> ISDecomposition:
    """Decompose implementation shortfall into delay/trading/opportunity.

    ``side`` is 'buy' or 'sell'.  Raises ValueError on inconsistent inputs
    (mismatched fill arrays, negative quantities, over-execution).
    The identity total == delay + trading + opportunity holds exactly.
    """
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    for name, price in (
        ("decision_price", decision_price),
        ("arrival_price", arrival_price),
        ("close_price", close_price),
    ):
        _require(math.isfinite(price), f"{name} must be finite")
    _require(math.isfinite(total_qty) and total_qty >= 0.0, "total_qty must be >= 0")
    n = np.asarray(fill_qtys, dtype=float)
    p = np.asarray(fill_prices, dtype=float)
    _require(
        n.shape == p.shape and n.ndim == 1,
        "fill_qtys and fill_prices must be 1-D arrays of equal length",
    )
    _require(bool(np.all(n >= 0.0)), "fill quantities must be >= 0")
    _require(bool(np.all(np.isfinite(n))) and bool(np.all(np.isfinite(p))), "fills must be finite")
    executed = float(np.sum(n))
    _require(executed <= total_qty * (1.0 + 1e-12) + 1e-12, "executed quantity exceeds total_qty")
    remaining = total_qty - executed
    sign = 1.0 if side == "sell" else -1.0
    delay = executed * sign * (decision_price - arrival_price)
    trading = float(np.sum(n * sign * (arrival_price - p)))
    opportunity = remaining * sign * (decision_price - close_price)
    total = delay + trading + opportunity
    return ISDecomposition(total, delay, trading, opportunity, executed, remaining)


def participation_rate(executed_volume: float, market_volume: float) -> float:
    """Executed volume as a fraction of market volume over the same window.

    ``market_volume`` must include the trader's own fills, so
    ``executed_volume <= market_volume`` is required (a rate above 1 is an
    input error, not a warning).
    """
    _require(
        math.isfinite(executed_volume) and executed_volume >= 0.0, "executed_volume must be >= 0"
    )
    _require(math.isfinite(market_volume) and market_volume > 0.0, "market_volume must be > 0")
    _require(
        executed_volume <= market_volume,
        "executed volume exceeds market volume — market volume must include own fills",
    )
    return executed_volume / market_volume


@dataclass(frozen=True)
class OLSResult:
    """Simple linear regression y = intercept + slope * x."""

    slope: float
    intercept: float
    r_squared: float
    n_obs: int


def ols(x: Sequence[float], y: Sequence[float]) -> OLSResult:
    """Tiny native OLS: slope = cov(x, y)/var(x) in closed form.

    Requires >= 2 observations and non-degenerate x.  R^2 = 1 - SSR/SST
    (defined as 1.0 when SST == 0 and the fit is exact).
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    _require(xa.shape == ya.shape and xa.ndim == 1, "x and y must be 1-D arrays of equal length")
    _require(xa.size >= 2, "need at least 2 observations")
    _require(
        bool(np.all(np.isfinite(xa))) and bool(np.all(np.isfinite(ya))), "x and y must be finite"
    )
    xm, ym = float(np.mean(xa)), float(np.mean(ya))
    sxx = float(np.sum((xa - xm) ** 2))
    _require(sxx > 0.0, "x is degenerate (zero variance)")
    sxy = float(np.sum((xa - xm) * (ya - ym)))
    slope = sxy / sxx
    intercept = ym - slope * xm
    resid = ya - (intercept + slope * xa)
    sst = float(np.sum((ya - ym) ** 2))
    r2 = 1.0 if sst == 0.0 else 1.0 - float(np.sum(resid**2)) / sst
    return OLSResult(slope, intercept, r2, int(xa.size))


def synthetic_impact_fills(
    n_obs: int = 25, intercept: float = 0.5, slope: float = 8.0, noise_amp: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic synthetic (participation, per-share cost) fills.

    Pinned, RNG-free construction (identical across languages):
        part_i = 0.02 + 0.18 * i / (n_obs - 1),        i = 0..n_obs-1
        cost_i = intercept + slope * part_i + noise_amp * sin(2.5*i + 0.3)
    """
    _require(_is_int(n_obs) and n_obs >= 2, "n_obs must be an integer >= 2")
    i = np.arange(n_obs, dtype=float)
    part = 0.02 + 0.18 * i / (n_obs - 1)
    cost = intercept + slope * part + noise_amp * np.sin(2.5 * i + 0.3)
    return part, cost


def estimate_impact(participations: Sequence[float], costs: Sequence[float]) -> OLSResult:
    """Market-impact regression: per-share cost on participation rate."""
    return ols(participations, costs)
