"""Power-law impact execution: square-root costs, volume curves, alpha-aware
schedules and a pre-trade cost report (API_SPEC Section 8).

Why this module exists
----------------------
Almgren-Chriss with linear temporary impact makes the cost per share grow
linearly with trading speed.  Measured costs grow roughly with the square root
of order size (the "square-root law", Almgren et al. 2005; Toth et al. 2011;
Bouchaud et al. 2018), so the linear model under-prices small orders by
multiples.  This module generalises the temporary impact to a power law,
lets liquidity vary by interval (an intraday volume curve), and adds an
adverse drift so a schedule can account for the information behind the order.

Model (formal statement)
------------------------
Sell ``X`` shares over ``N`` intervals of length ``tau``; holdings
``x_0 = X .. x_N = 0`` with trades ``n_j = x_{j-1} - x_j >= 0``.  Interval
``j`` carries a liquidity weight ``w_j > 0``, its fraction of daily volume
(uniform default ``w_j = tau`` with ``tau`` in trading days).  Then

    eta_j  = eta * w_j**(-alpha)
    c_j(n) = eta_j * n**(1 + alpha) - gamma_p * n**2 / 2
    E[IS]  = gamma_p * X**2 / 2 + sum_j c_j(n_j) + drift * tau * sum_{k=1}^{N-1} x_k
    Var    = sigma**2 * tau * sum_{j=1}^{N} x_j**2

``alpha`` in (0, 1]: ``alpha = 1`` with uniform weights is exactly
Almgren-Chriss (``eta_j = eta/tau`` and ``c_j = (eta_tilde/tau) n**2``);
``alpha = 1/2`` is the square-root law.  ``drift >= 0`` is the expected adverse
price move per share per time unit (a sell order whose price is expected to
keep falling).  The schedule minimises ``E + lambda_ra * Var``.

Plain reading of the calibration (``impact_params_from_market`` in
``etrade.market``): ``eta = Y * sigma / ADV**alpha``.  Trading at a constant
participation ``p`` of volume then costs ``Y * sigma * p**alpha`` per share,
so a one-day TWAP of ``Q`` shares costs ``Y * daily_vol * sqrt(Q/ADV)`` of
price at ``alpha = 1/2``: the square-root law, exactly.

Solver (pinned)
---------------
Newton's method on the interior holdings with the exact tridiagonal Hessian,
solved by the Thomas algorithm, from the linear (TWAP) start; an Armijo
backtracking line search (constant 1e-4, halving to a minimum step of 2**-40)
keeps every trade strictly positive; convergence when the Newton step is
below ``1e-12 * X`` in every coordinate or the Newton decrement ``-g.d`` is
below ``1e-14 * max(1, |U|)``, at most 100 iterations.  When an
adverse drift makes it optimal to finish before ``N``, the longest horizon
``m`` whose interior optimum trades strictly every interval is found by
bisection and verified against the KKT condition ``c_m'(n_m) <= drift * tau``.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

#: Upper bound on intervals for the power-law solver (pinned, API_SPEC s8).
MAX_IMPACT_INTERVALS: int = 10_000

#: Newton iteration cap per solve (pinned).
NEWTON_MAX_ITER: int = 100

#: Convergence: every Newton step component below this times X (pinned).
NEWTON_STEP_TOL: float = 1e-12

#: Convergence: Newton decrement (-g.d) below this times max(1, |U|) (pinned).
#: Needed because near the optimum the step can stay above NEWTON_STEP_TOL*X
#: purely from rounding in the gradient while no further decrease is possible.
NEWTON_DECREMENT_TOL: float = 1e-14

#: Armijo sufficient-decrease constant (pinned).
ARMIJO_C: float = 1e-4

#: Smallest line-search step before a solve is declared infeasible (pinned).
MIN_STEP: float = 2.0**-40

#: Illustrative US-equity intraday volume curve: fractions of daily volume in
#: thirteen 30-minute bins (U-shaped: heavy open and close).  Not fitted to
#: data; the calibration release replaces it.  Sums to 1.
US_EQUITY_VOLUME_CURVE_13: tuple[float, ...] = (
    0.12,
    0.09,
    0.075,
    0.065,
    0.06,
    0.055,
    0.05,
    0.055,
    0.06,
    0.065,
    0.075,
    0.09,
    0.14,
)

Holdings = Sequence[float] | npt.NDArray[np.float64]


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _is_int(v: object) -> bool:
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


@dataclass(frozen=True)
class ImpactParams:
    """Power-law impact execution problem (API_SPEC 8.1).

    ``weights`` defaults to ``tau`` per interval (uniform volume); when given
    it must have length ``n_intervals`` with finite entries ``> 0``.
    """

    total_shares: float
    n_intervals: int
    tau: float
    sigma: float
    eta: float
    alpha: float
    gamma_p: float
    lambda_ra: float
    drift: float = 0.0
    weights: tuple[float, ...] | None = field(default=None)

    def __post_init__(self) -> None:
        _require(
            math.isfinite(self.total_shares) and self.total_shares >= 0.0,
            "total_shares must be finite and >= 0",
        )
        _require(
            _is_int(self.n_intervals) and 1 <= self.n_intervals <= MAX_IMPACT_INTERVALS,
            f"n_intervals must be an integer in [1, {MAX_IMPACT_INTERVALS}]",
        )
        _require(math.isfinite(self.tau) and self.tau > 0.0, "tau must be > 0")
        _require(math.isfinite(self.sigma) and self.sigma >= 0.0, "sigma must be >= 0")
        _require(math.isfinite(self.eta) and self.eta > 0.0, "eta must be > 0")
        _require(math.isfinite(self.alpha) and 0.0 < self.alpha <= 1.0, "alpha must be in (0, 1]")
        _require(math.isfinite(self.gamma_p) and self.gamma_p >= 0.0, "gamma_p must be >= 0")
        _require(math.isfinite(self.lambda_ra) and self.lambda_ra >= 0.0, "lambda_ra must be >= 0")
        _require(math.isfinite(self.drift) and self.drift >= 0.0, "drift must be >= 0")
        if self.weights is None:
            object.__setattr__(self, "weights", (float(self.tau),) * self.n_intervals)
        else:
            w = tuple(float(v) for v in self.weights)
            _require(len(w) == self.n_intervals, "weights must have length n_intervals")
            _require(all(math.isfinite(v) and v > 0.0 for v in w), "weights must be finite and > 0")
            object.__setattr__(self, "weights", w)
        # Convexity of every c_j on (0, X]: c_j''(n) >= c_j''(X) > 0.
        if self.total_shares > 0.0:
            x_pow = self.total_shares ** (self.alpha - 1.0)
            for eta_j in self.etas():
                _require(
                    self.alpha * (1.0 + self.alpha) * eta_j * x_pow > self.gamma_p,
                    "model invalid: temporary impact not convex enough against permanent "
                    "impact (alpha*(1+alpha)*eta_j*X**(alpha-1) <= gamma_p)",
                )

    def etas(self) -> list[float]:
        """Per-interval temporary-impact coefficients ``eta * w_j**(-alpha)``."""
        assert self.weights is not None
        return [self.eta * w ** (-self.alpha) for w in self.weights]


# ------------------------------------------------------------------ costs


def _holdings(p: ImpactParams, holdings: Holdings) -> npt.NDArray[np.float64]:
    x = np.asarray(holdings, dtype=float)
    _require(
        x.ndim == 1 and x.size == p.n_intervals + 1, "holdings must have length n_intervals + 1"
    )
    _require(bool(np.all(np.isfinite(x))), "holdings must be finite")
    _require(
        abs(float(x[-1])) <= 1e-9 * max(1.0, abs(float(x[0]))),
        "holdings path must fully liquidate (x_N = 0)",
    )
    _require(
        bool(np.all(np.diff(x) <= 1e-12 * max(1.0, abs(float(x[0]))))),
        "holdings must be non-increasing (a sell program)",
    )
    return x


def _trades(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.maximum(-np.diff(x), 0.0)


def impact_expected_cost(p: ImpactParams, holdings: Holdings) -> float:
    """``E[IS]`` of a non-increasing, fully liquidating schedule (API_SPEC 8.1).

    ``X`` in the permanent term is the schedule's ``x_0``.
    """
    x = _holdings(p, holdings)
    n = _trades(x)
    etas = np.asarray(p.etas())
    x0 = float(x[0])
    temp = float(np.sum(etas * n ** (1.0 + p.alpha) - 0.5 * p.gamma_p * n**2))
    drift_cost = p.drift * p.tau * float(np.sum(x[1:-1]))
    return 0.5 * p.gamma_p * x0**2 + temp + drift_cost


def impact_variance_cost(p: ImpactParams, holdings: Holdings) -> float:
    """``Var[IS] = sigma^2 * tau * sum_{j>=1} x_j^2``."""
    x = _holdings(p, holdings)
    return p.sigma**2 * p.tau * float(np.sum(x[1:] ** 2))


def vwap_holdings(total_shares: float, weights: Sequence[float]) -> npt.NDArray[np.float64]:
    """Volume-weighted schedule: trade ``n_j = X * w_j / sum(w)``; ``x_N = 0`` exactly."""
    _require(
        math.isfinite(total_shares) and total_shares >= 0.0, "total_shares must be finite and >= 0"
    )
    w = np.asarray(weights, dtype=float)
    _require(w.ndim == 1 and w.size >= 1, "weights must be a non-empty 1-D sequence")
    _require(bool(np.all(np.isfinite(w) & (w > 0.0))), "weights must be finite and > 0")
    frac = np.cumsum(w) / float(np.sum(w))
    x = np.concatenate(([total_shares], total_shares * (1.0 - frac)))
    x[-1] = 0.0
    return x


# ------------------------------------------------------------------ solver


@dataclass(frozen=True)
class ScheduleResult:
    """Optimal schedule: ``holdings`` (length N+1), the ``horizon`` m actually
    used (``x_k = 0`` for ``k >= m``) and the total Newton ``iterations``."""

    holdings: npt.NDArray[np.float64] = field(repr=False)
    horizon: int
    iterations: int


def _thomas(diag: list[float], off: list[float], rhs: list[float]) -> list[float]:
    """Solve a symmetric tridiagonal system (``off[k]`` couples k and k+1)."""
    m = len(diag)
    c = [0.0] * m
    d = [0.0] * m
    c[0] = off[0] / diag[0] if m > 1 else 0.0
    d[0] = rhs[0] / diag[0]
    for k in range(1, m):
        denom = diag[k] - off[k - 1] * c[k - 1]
        c[k] = off[k] / denom if k < m - 1 else 0.0
        d[k] = (rhs[k] - off[k - 1] * d[k - 1]) / denom
    out = [0.0] * m
    out[m - 1] = d[m - 1]
    for k in range(m - 2, -1, -1):
        out[k] = d[k] - c[k] * out[k + 1]
    return out


def _objective(p: ImpactParams, etas: list[float], x: list[float]) -> float:
    big_x = x[0]
    total = 0.5 * p.gamma_p * big_x * big_x
    a1 = 1.0 + p.alpha
    for j in range(1, len(x)):
        n = x[j - 1] - x[j]
        total += etas[j - 1] * n**a1 - 0.5 * p.gamma_p * n * n
    hold = 0.0
    for k in range(1, len(x) - 1):
        hold += x[k]
    sq = 0.0
    for k in range(1, len(x)):
        sq += x[k] * x[k]
    return total + p.drift * p.tau * hold + p.lambda_ra * p.sigma**2 * p.tau * sq


def _solve_horizon(p: ImpactParams, etas: list[float], m: int) -> tuple[list[float], int] | None:
    """Interior optimum on the first ``m`` intervals (x_m = 0), or None when
    no strictly monotone optimum is reached (pinned Newton, API_SPEC 8.2)."""
    big_x = p.total_shares
    if m == 1:
        return [big_x, 0.0], 0
    a, a1 = p.alpha, 1.0 + p.alpha
    g2 = p.gamma_p
    v2 = 2.0 * p.lambda_ra * p.sigma**2 * p.tau
    dt = p.drift * p.tau
    e = etas[:m]
    x = [big_x * (m - k) / m for k in range(m + 1)]
    for it in range(1, NEWTON_MAX_ITER + 1):
        n = [x[j - 1] - x[j] for j in range(1, m + 1)]
        c1 = [a1 * e[j] * n[j] ** a - g2 * n[j] for j in range(m)]
        c2 = [a * a1 * e[j] * n[j] ** (a - 1.0) - g2 for j in range(m)]
        grad = [c1[k] - c1[k - 1] + v2 * x[k] + dt for k in range(1, m)]
        diag = [c2[k - 1] + c2[k] + v2 for k in range(1, m)]
        off = [-c2[k] for k in range(1, m)]
        step = _thomas(diag, off, [-g for g in grad])
        u0 = _objective(p, e, x)
        slope = sum(g * s for g, s in zip(grad, step, strict=True))
        if max(
            abs(s) for s in step
        ) <= NEWTON_STEP_TOL * big_x or -slope <= NEWTON_DECREMENT_TOL * max(1.0, abs(u0)):
            return x, it
        t = 1.0
        while t >= MIN_STEP:
            xn = [x[0]] + [x[k] + t * step[k - 1] for k in range(1, m)] + [0.0]
            if all(xn[j - 1] - xn[j] > 0.0 for j in range(1, m + 1)):
                if _objective(p, e, xn) <= u0 + ARMIJO_C * t * slope:
                    break
            t *= 0.5
        else:
            return None
        x = xn
    return None


def optimal_schedule(p: ImpactParams) -> ScheduleResult:
    """Minimise ``E + lambda_ra*Var`` over non-increasing liquidating schedules
    (API_SPEC 8.2).  ``X = 0`` gives the zero schedule."""
    n_int = p.n_intervals
    if p.total_shares == 0.0:
        return ScheduleResult(np.zeros(n_int + 1), n_int, 0)
    etas = p.etas()
    iters = 0

    def pad(xm: list[float]) -> npt.NDArray[np.float64]:
        out = np.zeros(n_int + 1)
        out[: len(xm)] = xm
        out[len(xm) - 1 :] = 0.0
        return out

    full = _solve_horizon(p, etas, n_int)
    if full is not None:
        return ScheduleResult(pad(full[0]), n_int, full[1])
    _require(p.drift > 0.0, "optimal schedule: Newton solve did not converge")
    lo, hi = 1, n_int
    best = _solve_horizon(p, etas, 1)
    assert best is not None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        r = _solve_horizon(p, etas, mid)
        if r is not None:
            lo, best = mid, r
            iters += r[1]
        else:
            hi = mid
    xm, it = best
    iters += it
    n_last = xm[lo - 1] - xm[lo]
    marginal = (1.0 + p.alpha) * etas[lo - 1] * n_last**p.alpha - p.gamma_p * n_last
    _require(
        marginal <= p.drift * p.tau * (1.0 + 1e-9) + 1e-12,
        "optimal schedule: KKT check failed for the early-finish horizon",
    )
    return ScheduleResult(pad(xm), lo, iters)


# ------------------------------------------------------------------ pre-trade


def sqrt_law_cost_bps(daily_vol: float, order_shares: float, adv: float, y: float = 1.0) -> float:
    """Square-root-law impact estimate ``1e4 * Y * daily_vol * sqrt(Q/ADV)`` bps."""
    _require(math.isfinite(daily_vol) and daily_vol > 0.0, "daily_vol must be finite and > 0")
    _require(math.isfinite(order_shares) and order_shares >= 0.0, "order_shares must be >= 0")
    _require(math.isfinite(adv) and adv > 0.0, "adv must be finite and > 0")
    _require(math.isfinite(y) and y > 0.0, "y must be finite and > 0")
    return 1e4 * y * daily_vol * math.sqrt(order_shares / adv)


@dataclass(frozen=True)
class PretradeRow:
    """One strategy in a pre-trade report; costs in bps of ``X * price``.
    ``impact_bps`` includes permanent impact and drift; ``total_bps`` adds the
    half spread."""

    label: str
    lambda_ra: float
    impact_bps: float
    sd_bps: float
    total_bps: float
    horizon: int


@dataclass(frozen=True)
class PretradeReport:
    """Pre-trade report (API_SPEC 8.4)."""

    sqrt_law_bps: float
    half_spread_bps: float
    rows: tuple[PretradeRow, ...]
