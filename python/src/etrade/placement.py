"""Limit-order placement for execution: queue-position fill probability, the
passive-or-aggressive choice for one child order, and liquidation with limit
and market orders (API_SPEC Section 13).

Why this module exists
----------------------
Sections 1, 8 and 12 decide *how much* to trade per interval and assume every
child order executes by crossing the spread.  In practice most child orders
are first posted passively: a resting limit order at the touch saves the half
spread but may not fill, and an unfilled remainder must still be completed,
usually after the price has moved away.  This module provides the three
pieces needed to reason about that trade-off: the probability that a resting
order fills given its queue position, the expected cost of posting against
crossing for one child order, and a small dynamic-programming liquidation
policy that chooses a posting depth (or a market order) per period, checked
by a seeded Monte Carlo against two simple baselines.

Fill probability by queue position (API_SPEC 13.1)
--------------------------------------------------
A FIFO queue at our price level holds ``q_ahead`` units ahead of our order of
``size`` units.  Market orders of one unit arrive as a Poisson process of
rate ``mu`` (units per time unit) and execute the front of the queue; each
unit ahead of us is cancelled independently at rate ``theta``; our own units
are never cancelled.  Our order is filled completely once all units ahead
have left and ``size`` further market orders have arrived, so the fill time
is a sum of independent exponentials

    T = sum_{k=1}^{q_ahead} Exp(mu + theta k) + sum_{i=1}^{size} Exp(mu)

(the pure-death chain of Cont, Stoikov and Talreja 2010 with proportional
cancellations; Cont and de Larrard 2013 use the same Poisson queue
dynamics).  ``P(T <= h)`` has no closed form that is numerically stable for
deep queues (the hypoexponential partial-fraction sum alternates), so it is
evaluated by uniformisation, pinned in API_SPEC 13.1: an exact
representation of the chain as a discrete chain observed at Poisson(``L``)
times, truncated ten standard deviations either side of the mode.  With
``theta = 0`` the result is the Poisson tail ``P(N(h) >= q_ahead + size)``.
Moallemi and Yuan (2016) show that queue position has first-order value for
a resting order; the decision in 13.2 uses this probability to price it.

Passive or aggressive (API_SPEC 13.2)
-------------------------------------
For one sell child order of ``size`` shares, costs per share relative to the
current mid: crossing now pays the half spread plus the Section 8 temporary
impact ``I = eta * weight**(-alpha) * size**alpha`` (the Section 8 ``E[IS]``
of a one-interval schedule, divided by the size).  Posting at the touch
earns the half spread with probability ``P`` (from 13.1) and otherwise
completes the order by crossing at the end of the horizon after an adverse
mid move ``adverse_move``.  The impact ``I`` is charged to passive fills too:
the square-root law describes the price pressure of a metaorder's volume and
is measured to hold for both order types (Toth et al. 2011), so posting
avoids the spread, not the impact.  Hence

    passive - aggressive = (1 - P) * adverse_move - 2 P * half_spread

and posting wins exactly when ``P > adverse_move / (adverse_move + spread)``.
With a negligible spread, crossing is always (weakly) preferred.

Liquidation with limit and market orders (API_SPEC 13.3)
--------------------------------------------------------
Discrete version of Cartea and Jaimungal (2015): sell ``Q`` units over ``N``
periods of length ``dt``.  Each period with inventory ``q > 0`` we either
post one unit at depth ``delta >= 0`` beyond the touch, filled with
probability ``p_touch * exp(-kappa delta)`` and earning ``half_spread +
delta``, or send a one-unit market order costing ``half_spread + beta``.
Inventory left at the end is swept at ``half_spread + beta q`` per unit; a
running penalty ``phi q**2 dt`` per period expresses urgency.  The one-step
maximisation has the closed form ``delta* = max(0, 1/kappa - g)`` with
``g = half_spread + (C_{j+1}(q) - C_{j+1}(q-1))``: the depth falls as the
marginal cost of holding one more unit rises.  The Monte Carlo compares the
policy with all-aggressive (an even schedule of market orders) and
all-passive-then-sweep (post at the touch every period, sweep the rest).

References
----------
Cont, R., Stoikov, S. and Talreja, R. (2010). A stochastic model for order
book dynamics. Operations Research 58(3), 549-563.
Cont, R. and de Larrard, A. (2013). Price dynamics in a Markovian limit
order market. SIAM Journal on Financial Mathematics 4(1), 1-25.
Moallemi, C. C. and Yuan, K. (2016). A model for queue position valuation in
a limit order book. Working paper, Columbia Business School.
Cartea, A. and Jaimungal, S. (2015). Optimal execution with limit and market
orders. Quantitative Finance 15(8), 1279-1291.
Toth, B., Lemperiere, Y., Deremble, C., de Lataillade, J., Kockelkoren, J.
and Bouchaud, J.-P. (2011). Anomalous price impact and the critical nature
of liquidity in financial markets. Physical Review X 1(2), 021006.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .almgren_chriss import MAX_PATHS

#: Upper bound on ``q_ahead + size`` in :func:`queue_fill_probability` (pinned).
MAX_QUEUE_UNITS: int = 5_000

#: Upper bound on uniformisation steps in :func:`queue_fill_probability` (pinned).
MAX_UNIFORMISATION_STEPS: int = 200_000

#: Upper bound on ``total_units`` and ``n_periods`` of a liquidation (pinned).
MAX_LIQ_UNITS: int = 1_000
MAX_LIQ_PERIODS: int = 1_000


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _is_int(v: object) -> bool:
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


# ------------------------------------------------------------------ fill probability


def uniformisation_window(rate_horizon: float) -> tuple[int, int]:
    """Truncation window ``[k_lo, k_hi]`` of the Poisson(``L``) weights
    (API_SPEC 13.1): ``m = floor(L)``, ``half = ceil(10 sqrt(L) + 10)``."""
    m = math.floor(rate_horizon)
    half = math.ceil(10.0 * math.sqrt(rate_horizon) + 10.0)
    return max(0, m - half), m + half


def queue_fill_probability(
    q_ahead: int, size: int, mu: float, theta: float, horizon: float
) -> float:
    """Probability that a resting order of ``size`` units with ``q_ahead``
    units ahead fills completely within ``horizon`` (API_SPEC 13.1).

    Market orders of one unit arrive at rate ``mu``; each unit ahead cancels
    at rate ``theta``.  Evaluated by pinned uniformisation.
    """
    _require(_is_int(q_ahead) and q_ahead >= 0, "q_ahead must be an integer >= 0")
    _require(_is_int(size) and size >= 1, "size must be an integer >= 1")
    _require(
        q_ahead + size <= MAX_QUEUE_UNITS,
        f"q_ahead + size must be <= {MAX_QUEUE_UNITS}",
    )
    _require(math.isfinite(mu) and mu > 0.0, "mu must be finite and > 0")
    _require(math.isfinite(theta) and theta >= 0.0, "theta must be finite and >= 0")
    _require(math.isfinite(horizon) and horizon >= 0.0, "horizon must be finite and >= 0")
    lam = mu + theta * q_ahead
    big_l = lam * horizon
    _require(math.isfinite(big_l), "horizon too long: (mu + theta*q_ahead)*horizon overflows")
    if big_l == 0.0:
        return 0.0
    k_lo, k_hi = uniformisation_window(big_l)
    _require(
        k_hi <= MAX_UNIFORMISATION_STEPS,
        "horizon too long: (mu + theta*q_ahead)*horizon needs more than "
        f"{MAX_UNIFORMISATION_STEPS} uniformisation steps",
    )
    n_st = q_ahead + size
    rates = np.full(n_st, mu)
    rates[:q_ahead] = mu + theta * np.arange(q_ahead, 0, -1, dtype=float)
    go = rates / lam
    stay = 1.0 - go
    m = math.floor(big_l)
    w = np.zeros(k_hi + 1)
    w[m] = 1.0
    for k in range(m + 1, k_hi + 1):
        w[k] = w[k - 1] * big_l / k
    for k in range(m - 1, k_lo - 1, -1):
        w[k] = w[k + 1] * (k + 1) / big_l
    p = np.zeros(n_st)
    p[0] = 1.0
    absorbed = 0.0
    acc = 0.0
    total_w = 0.0
    for k in range(k_hi + 1):
        if k > 0:
            move = p * go
            p = p * stay
            p[1:] = p[1:] + move[:-1]
            absorbed = absorbed + float(move[-1])
        if k >= k_lo:
            acc += float(w[k]) * absorbed
            total_w += float(w[k])
    return min(1.0, max(0.0, acc / total_w))


# ------------------------------------------------------------------ passive or aggressive


@dataclass(frozen=True)
class PlacementDecision:
    """Expected cost per share (currency, relative to the mid) of posting at
    the touch and of crossing now (API_SPEC 13.2).  ``passive`` is true when
    posting is strictly cheaper; ties go to crossing."""

    passive_cost: float
    aggressive_cost: float
    impact_cost: float
    breakeven_fill_prob: float
    passive: bool


def place_or_cross(
    size: float,
    spread: float,
    fill_prob: float,
    adverse_move: float,
    eta: float,
    alpha: float,
    weight: float,
) -> PlacementDecision:
    """Compare posting one child order at the touch with crossing now
    (API_SPEC 13.2).  ``eta``, ``alpha`` and ``weight`` are the Section 8
    impact parameters of the interval the child order trades in."""
    _require(math.isfinite(size) and size > 0.0, "size must be finite and > 0")
    _require(math.isfinite(spread) and spread >= 0.0, "spread must be finite and >= 0")
    _require(math.isfinite(fill_prob) and 0.0 <= fill_prob <= 1.0, "fill_prob must be in [0, 1]")
    _require(
        math.isfinite(adverse_move) and adverse_move >= 0.0,
        "adverse_move must be finite and >= 0",
    )
    _require(math.isfinite(eta) and eta >= 0.0, "eta must be finite and >= 0")
    _require(math.isfinite(alpha) and 0.0 < alpha <= 1.0, "alpha must be in (0, 1]")
    _require(math.isfinite(weight) and weight > 0.0, "weight must be finite and > 0")
    hs = 0.5 * spread
    impact = eta * weight ** (-alpha) * size**alpha
    aggressive = hs + impact
    passive = fill_prob * (impact - hs) + (1.0 - fill_prob) * (adverse_move + hs + impact)
    denom = adverse_move + spread
    breakeven = adverse_move / denom if denom > 0.0 else 1.0
    return PlacementDecision(
        passive_cost=passive,
        aggressive_cost=aggressive,
        impact_cost=impact,
        breakeven_fill_prob=breakeven,
        passive=passive < aggressive,
    )


# ------------------------------------------------------------------ liquidation


@dataclass(frozen=True)
class LiquidationParams:
    """Sell ``total_units`` over ``n_periods`` with limit and market orders
    (API_SPEC 13.3).

    ``kappa`` is the fill decay per unit of depth (1/currency), ``p_touch``
    the per-period fill probability at the touch, ``beta`` the book walk per
    unit of a market order (currency/unit**2), ``phi`` the running inventory
    penalty (currency/unit**2 per time unit), ``sigma`` the mid volatility
    (currency per sqrt time unit).
    """

    total_units: int
    n_periods: int
    dt: float
    sigma: float
    spread: float
    kappa: float
    p_touch: float
    beta: float
    phi: float = 0.0

    def __post_init__(self) -> None:
        _require(
            _is_int(self.total_units) and 1 <= self.total_units <= MAX_LIQ_UNITS,
            f"total_units must be an integer in [1, {MAX_LIQ_UNITS}]",
        )
        _require(
            _is_int(self.n_periods) and 1 <= self.n_periods <= MAX_LIQ_PERIODS,
            f"n_periods must be an integer in [1, {MAX_LIQ_PERIODS}]",
        )
        _require(math.isfinite(self.dt) and self.dt > 0.0, "dt must be finite and > 0")
        _require(math.isfinite(self.sigma) and self.sigma >= 0.0, "sigma must be finite and >= 0")
        _require(
            math.isfinite(self.spread) and self.spread >= 0.0, "spread must be finite and >= 0"
        )
        _require(math.isfinite(self.kappa) and self.kappa > 0.0, "kappa must be finite and > 0")
        _require(
            math.isfinite(self.p_touch) and 0.0 < self.p_touch <= 1.0,
            "p_touch must be in (0, 1]",
        )
        _require(math.isfinite(self.beta) and self.beta >= 0.0, "beta must be finite and >= 0")
        _require(math.isfinite(self.phi) and self.phi >= 0.0, "phi must be finite and >= 0")


@dataclass(frozen=True)
class LiquidationPolicy:
    """Dynamic-programming policy (API_SPEC 13.3).

    ``cost[j, q]`` (shape ``(N + 1, Q + 1)``) is the optimal expected cost to
    go, including the running penalty; ``depth[j, q]`` and ``market[j, q]``
    (shape ``(N, Q + 1)``) the action at period ``j`` with inventory ``q``
    (``depth`` is 0 where ``market`` is true and for ``q = 0``).
    """

    cost: npt.NDArray[np.float64] = field(repr=False)
    depth: npt.NDArray[np.float64] = field(repr=False)
    market: npt.NDArray[np.bool_] = field(repr=False)

    @property
    def expected_cost(self) -> float:
        """``C_0(Q)``: the policy's expected shortfall plus penalties."""
        return float(self.cost[0, -1])


def liquidation_policy(p: LiquidationParams) -> LiquidationPolicy:
    """Backward induction of API_SPEC 13.3 (loop order pinned: ``j`` from
    ``N - 1`` down to 0, ``q`` from 1 up to ``Q``)."""
    n_per, big_q = p.n_periods, p.total_units
    hs = 0.5 * p.spread
    cost = np.zeros((n_per + 1, big_q + 1))
    depth = np.zeros((n_per, big_q + 1))
    market = np.zeros((n_per, big_q + 1), dtype=bool)
    for q in range(big_q + 1):
        cost[n_per, q] = q * (hs + p.beta * q)
    inv_k = 1.0 / p.kappa
    for j in range(n_per - 1, -1, -1):
        for q in range(1, big_q + 1):
            c_hold = float(cost[j + 1, q])
            c_less = float(cost[j + 1, q - 1])
            g = hs + (c_hold - c_less)
            d = max(0.0, inv_k - g)
            gain = p.p_touch * math.exp(-p.kappa * d) * (d + g)
            post = c_hold - gain
            mkt = hs + p.beta + c_less
            if mkt < post:
                market[j, q] = True
                base = mkt
            else:
                depth[j, q] = d
                base = post
            cost[j, q] = p.phi * q * q * p.dt + base
    return LiquidationPolicy(cost=cost, depth=depth, market=market)


def aggressive_schedule(total_units: int, n_periods: int) -> list[int]:
    """Even market-order schedule ``n_j = floor(Q (j+1)/N) - floor(Q j/N)``."""
    return [
        (total_units * (j + 1)) // n_periods - (total_units * j) // n_periods
        for j in range(n_periods)
    ]


def aggressive_cost(p: LiquidationParams) -> float:
    """Expected shortfall of the all-aggressive baseline (deterministic:
    ``sum_j n_j (half_spread + beta n_j)``)."""
    hs = 0.5 * p.spread
    return float(
        sum(n * (hs + p.beta * n) for n in aggressive_schedule(p.total_units, p.n_periods))
    )


@dataclass(frozen=True)
class LiquidationSimResult:
    """Implementation shortfall (currency, ``S_0 = 0``) of the optimal policy
    and the two baselines; savings are paired per path (baseline - optimal,
    positive = policy cheaper); SEs are sample std (ddof 1) / sqrt(M)."""

    optimal_mean: float
    optimal_se: float
    optimal_std: float
    aggressive_mean: float
    aggressive_se: float
    aggressive_std: float
    passive_mean: float
    passive_se: float
    passive_std: float
    saving_vs_aggressive_mean: float
    saving_vs_aggressive_se: float
    saving_vs_passive_mean: float
    saving_vs_passive_se: float
    n_paths: int


def simulate_liquidation(
    p: LiquidationParams, n_paths: int = 10_000, seed: int = 0
) -> LiquidationSimResult:
    """Seeded Monte Carlo of the policy against all-aggressive and
    all-passive-then-sweep (loop and draw order pinned in API_SPEC 13.3).

    Per period ``j``: draw ``u ~ U(0,1)`` then ``z ~ N(0,1)`` (one each per
    path, shared by the three strategies); trade at the current mid ``S``
    (optimal: the policy's market order at ``S - hs - beta`` or a post at
    depth ``d`` filled iff ``u < p_touch exp(-kappa d)`` at ``S + hs + d``;
    passive: post at the touch, filled iff ``u < p_touch`` at ``S + hs``;
    aggressive: ``n_j`` units at ``S - hs - beta n_j``); then
    ``S += sigma sqrt(dt) z``.  The remainder ``q`` is swept at
    ``S_N - hs - beta q``.  ``IS = -proceeds``.
    """
    _require(
        _is_int(n_paths) and 2 <= n_paths <= MAX_PATHS,
        f"n_paths must be an integer in [2, {MAX_PATHS}]",
    )
    pol = liquidation_policy(p)
    sched = aggressive_schedule(p.total_units, p.n_periods)
    hs = 0.5 * p.spread
    srt = p.sigma * math.sqrt(p.dt)
    fill_at = p.p_touch * np.exp(-p.kappa * pol.depth)
    rng = np.random.default_rng(seed)
    big_q = p.total_units
    s = np.zeros(n_paths)
    q_opt = np.full(n_paths, big_q)
    q_pas = np.full(n_paths, big_q)
    pr_opt = np.zeros(n_paths)
    pr_agg = np.zeros(n_paths)
    pr_pas = np.zeros(n_paths)
    for j in range(p.n_periods):
        u = rng.random(n_paths)
        z = rng.standard_normal(n_paths)
        live = q_opt > 0
        is_mkt = live & pol.market[j, q_opt]
        filled = live & ~pol.market[j, q_opt] & (u < fill_at[j, q_opt])
        pr_opt = np.where(is_mkt, pr_opt + (s - hs - p.beta), pr_opt)
        pr_opt = np.where(filled, pr_opt + (s + hs + pol.depth[j, q_opt]), pr_opt)
        q_opt = q_opt - (is_mkt | filled).astype(q_opt.dtype)
        pas_fill = (q_pas > 0) & (u < p.p_touch)
        pr_pas = np.where(pas_fill, pr_pas + (s + hs), pr_pas)
        q_pas = q_pas - pas_fill.astype(q_pas.dtype)
        n_j = sched[j]
        if n_j > 0:
            pr_agg = pr_agg + n_j * (s - hs - p.beta * n_j)
        s = s + srt * z
    pr_opt = pr_opt + q_opt * (s - hs - p.beta * q_opt)
    pr_pas = pr_pas + q_pas * (s - hs - p.beta * q_pas)
    is_opt = -pr_opt
    is_agg = -pr_agg
    is_pas = -pr_pas
    sav_a = is_agg - is_opt
    sav_p = is_pas - is_opt
    root = math.sqrt(n_paths)

    def sd(v: npt.NDArray[np.float64]) -> float:
        return float(np.std(v, ddof=1))

    return LiquidationSimResult(
        optimal_mean=float(np.mean(is_opt)),
        optimal_se=sd(is_opt) / root,
        optimal_std=sd(is_opt),
        aggressive_mean=float(np.mean(is_agg)),
        aggressive_se=sd(is_agg) / root,
        aggressive_std=sd(is_agg),
        passive_mean=float(np.mean(is_pas)),
        passive_se=sd(is_pas) / root,
        passive_std=sd(is_pas),
        saving_vs_aggressive_mean=float(np.mean(sav_a)),
        saving_vs_aggressive_se=sd(sav_a) / root,
        saving_vs_passive_mean=float(np.mean(sav_p)),
        saving_vs_passive_se=sd(sav_p) / root,
        n_paths=n_paths,
    )
