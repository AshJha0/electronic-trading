"""Execution extensions: transient impact, adaptive re-planning and
internalisation (API_SPEC Section 12).

Why this module exists
----------------------
Section 1 and Section 8 treat temporary impact as instantaneous: a trade
moves the price it is executed at and the move is gone by the next interval.
Real order books refill at a finite speed, so a trade also raises the price
paid by the trades that follow it.  Obizhaeva and Wang (2013) model this with
a limit-order book of constant depth whose displacement decays exponentially
(resilience ``rho``); the optimal schedule then trades a block at the start,
a constant rate in between and a block at the end.  This module provides that
model in discrete time, a Monte Carlo comparison of a static schedule with
one re-planned every interval, and the crossing (internalisation) of opposite
parent orders before any of them reaches the market.

Transient impact (formal statement, API_SPEC 12.1)
--------------------------------------------------
Sell ``X`` over ``N`` intervals of length ``tau``; trade ``n_j`` (signed) at
the start of interval ``j``.  The book displacement ``D`` (price pushed down
by our own past selling) obeys

    D_1 = 0,    D_{j+1} = a * (D_j + kappa * n_j),    a = exp(-rho * tau)

A trade of ``n`` against a book of depth ``1/kappa`` pays the displacement
already present plus half its own: ``n * D_j + kappa * n**2 / 2``.  With
permanent impact ``gamma_p`` booked as in Section 1,

    E[IS] = gamma_p X**2 / 2 + sum_j [ (kappa - gamma_p)/2 n_j**2 + n_j D_j ]
          = gamma_p X**2 / 2 + n' A n / 2,   A = kappa K - gamma_p I,
                                              K_ij = a**|i - j|.

``K`` is the Kac-Murdock-Szego matrix, whose inverse is tridiagonal, so the
static optimum ``n = X A^{-1} 1 / (1' A^{-1} 1)`` is found exactly by one
tridiagonal (Thomas) solve.  With ``gamma_p = 0`` the solution is
``n proportional to [1, 1-a, ..., 1-a, 1]``: equal blocks at both ends and a
constant rate between (Obizhaeva and Wang 2013, Proposition 1 in discrete
time; Alfonsi, Fruth and Schied 2010).  As ``rho -> infinity`` (``a -> 0``)
the cost is ``(kappa - gamma_p)/2 sum n_j**2``, which is Section 1's linear
temporary impact with ``kappa = 2 eta / tau``.

Adaptive re-planning (API_SPEC 12.2)
------------------------------------
Section 1 dynamics plus an observable AR(1) short-term signal ``s`` (the
expected price drift per time unit) and random partial fills.  At every
interval the adaptive trader re-solves the remaining mean-variance problem
from the realised holdings and signal; the static trader sends the child
orders planned at time 0.  Without a signal and with full fills the two are
identical path by path: the Almgren-Chriss schedule is time consistent
(the tail of the optimum is the optimum of the tail), so re-planning cannot
help.  It helps when the re-plan uses information the plan did not have: a
signal (trade more when the price is expected to fall) or fill shortfalls
(spread an unfilled remainder over the remaining intervals rather than
dumping it at the close).  Almgren and Lorenz (2007) show a separate source
of gain that this module does not model: under a *global* mean-variance
criterion a non-time-consistent adaptive rule can beat the static optimum.

Internalisation (API_SPEC 12.3)
-------------------------------
Opposite parent orders in one name over one horizon are crossed at the mid
first; only the net residual goes to the market.  Each side, internalised or
not, is costed with the Section 8 model as one volume-weighted program
(TWAP under the default weights), and the saving is attributed pro rata.
Because that cost is increasing in size and the residual never exceeds the
larger side, crossing never increases total impact cost.

References
----------
Obizhaeva, A. A. and Wang, J. (2013). Optimal trading strategy and
supply/demand dynamics. Journal of Financial Markets 16(1), 1-32.
Alfonsi, A., Fruth, A. and Schied, A. (2010). Optimal execution strategies
in limit order books with general shape functions. Quantitative Finance
10(2), 143-157.
Almgren, R. and Chriss, N. (2000). Optimal execution of portfolio
transactions. Journal of Risk 3(2), 5-39.
Almgren, R. and Lorenz, J. (2007). Adaptive arrival price. Trading 2007(1),
59-66.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .almgren_chriss import MAX_PATHS, ACParams
from .impact import impact_expected_cost, vwap_holdings
from .market import bps, impact_params_from_market

#: Upper bound on intervals for every Section 12 entry point (pinned).
MAX_EXEC_INTERVALS: int = 1_000

#: Upper bound on the number of parent orders per side in ``internalise``.
MAX_ORDERS: int = 10_000

Holdings = Sequence[float] | npt.NDArray[np.float64]


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _is_int(v: object) -> bool:
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


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


# ------------------------------------------------------------------ transient


@dataclass(frozen=True)
class TransientParams:
    """Transient-impact problem (API_SPEC 12.1).

    ``kappa`` is the price displacement per share of a trade (inverse book
    depth, currency/share**2), ``rho`` the resilience rate per time unit.
    """

    total_shares: float
    n_intervals: int
    tau: float
    kappa: float
    rho: float
    gamma_p: float = 0.0

    def __post_init__(self) -> None:
        _require(
            math.isfinite(self.total_shares) and self.total_shares >= 0.0,
            "total_shares must be finite and >= 0",
        )
        _require(
            _is_int(self.n_intervals) and 1 <= self.n_intervals <= MAX_EXEC_INTERVALS,
            f"n_intervals must be an integer in [1, {MAX_EXEC_INTERVALS}]",
        )
        _require(math.isfinite(self.tau) and self.tau > 0.0, "tau must be > 0")
        _require(math.isfinite(self.kappa) and self.kappa > 0.0, "kappa must be > 0")
        _require(math.isfinite(self.rho) and self.rho > 0.0, "rho must be finite and > 0")
        _require(math.isfinite(self.gamma_p) and self.gamma_p >= 0.0, "gamma_p must be >= 0")
        a = self.decay()
        _require(
            self.gamma_p < self.kappa * (1.0 - a) / (1.0 + a),
            "model invalid: gamma_p >= kappa*(1-a)/(1+a) (cost not convex in the trades)",
        )

    def decay(self) -> float:
        """Per-interval displacement decay ``a = exp(-rho * tau)``."""
        return math.exp(-self.rho * self.tau)


def _path(p: TransientParams, holdings: Holdings) -> npt.NDArray[np.float64]:
    x = np.asarray(holdings, dtype=float)
    _require(
        x.ndim == 1 and x.size == p.n_intervals + 1, "holdings must have length n_intervals + 1"
    )
    _require(bool(np.all(np.isfinite(x))), "holdings must be finite")
    _require(
        abs(float(x[-1])) <= 1e-9 * max(1.0, abs(float(x[0]))),
        "holdings path must fully liquidate (x_N = 0)",
    )
    return x


def transient_expected_cost(p: TransientParams, holdings: Holdings) -> float:
    """``E[IS]`` of any fully liquidating schedule by the displacement
    recursion (API_SPEC 12.1).  Trades may have either sign; ``X`` in the
    permanent term is the schedule's ``x_0``."""
    x = _path(p, holdings)
    a = p.decay()
    half = 0.5 * (p.kappa - p.gamma_p)
    d = 0.0
    total = 0.0
    for j in range(1, x.size):
        n = float(x[j - 1] - x[j])
        total += half * n * n + n * d
        d = a * (d + p.kappa * n)
    x0 = float(x[0])
    return 0.5 * p.gamma_p * x0 * x0 + total


def transient_optimal_schedule(p: TransientParams) -> npt.NDArray[np.float64]:
    """Static schedule minimising ``E[IS]`` (holdings, length ``N + 1``).

    Solves ``A n = c 1`` with ``1' n = X`` exactly through the tridiagonal
    system ``((1-a**2) kappa I - gamma_p T) y = T 1`` with ``T = (1-a**2) K^{-1}``
    (API_SPEC 12.1).  ``x_0 = X`` and ``x_N = 0`` exactly.
    """
    n_int = p.n_intervals
    big_x = p.total_shares
    x = np.zeros(n_int + 1)
    x[0] = big_x
    if big_x == 0.0 or n_int == 1:
        return x
    a = p.decay()
    b = -math.expm1(-p.rho * p.tau)  # 1 - a without cancellation
    ka = b * (1.0 + a) * p.kappa  # (1 - a**2) * kappa
    g = p.gamma_p
    diag = [ka - g * (1.0 + a * a)] * n_int
    diag[0] = ka - g
    diag[-1] = ka - g
    off = [g * a] * (n_int - 1)
    rhs = [b * b] * n_int
    rhs[0] = b
    rhs[-1] = b
    y = _thomas(diag, off, rhs)
    total = sum(y)
    for j in range(1, n_int):
        x[j] = x[j - 1] - big_x * y[j - 1] / total
    x[n_int] = 0.0
    return x


# ------------------------------------------------------------------ adaptive


@dataclass(frozen=True)
class SignalParams:
    """Observable AR(1) drift signal and fill risk (API_SPEC 12.2).

    ``s_j = phi * s_{j-1} + sqrt(1 - phi**2) * signal_sd * eps_j``, ``s_0 = 0``;
    the price drifts by ``s_{j-1} * tau`` over interval ``j``.  A child order
    in interval ``j < N`` fills the fraction ``1 - fill_risk * u_j``; the last
    interval always fills completely.
    """

    phi: float = 0.0
    signal_sd: float = 0.0
    fill_risk: float = 0.0

    def __post_init__(self) -> None:
        _require(math.isfinite(self.phi) and 0.0 <= self.phi < 1.0, "phi must be in [0, 1)")
        _require(math.isfinite(self.signal_sd) and self.signal_sd >= 0.0, "signal_sd must be >= 0")
        _require(
            math.isfinite(self.fill_risk) and 0.0 <= self.fill_risk < 1.0,
            "fill_risk must be in [0, 1)",
        )


def _check_ac(p: ACParams) -> None:
    _require(
        p.n_intervals <= MAX_EXEC_INTERVALS,
        f"n_intervals must be an integer in [1, {MAX_EXEC_INTERVALS}]",
    )


def _replan_system(p: ACParams, m: int) -> tuple[list[float], list[float]]:
    e = p.eta_tilde / p.tau
    diag = [2.0 * e + p.lambda_ra * p.sigma**2 * p.tau] * (m - 1)
    off = [-e] * (m - 1)
    return diag, off


def replan(p: ACParams, sig: SignalParams, holding: float, signal: float, k: int) -> list[float]:
    """Optimal remaining holdings ``[x_k, x_{k+1}, .., x_N = 0]`` from interval
    ``k`` (0-based) given the current holding and signal (API_SPEC 12.2)."""
    _check_ac(p)
    _require(_is_int(k) and 0 <= k < p.n_intervals, "k must be an integer in [0, n_intervals)")
    _require(math.isfinite(holding), "holding must be finite")
    _require(math.isfinite(signal), "signal must be finite")
    m = p.n_intervals - k
    if m == 1:
        return [holding, 0.0]
    diag, off = _replan_system(p, m)
    e = p.eta_tilde / p.tau
    rhs = [0.5 * p.tau * sig.phi**i * signal for i in range(m - 1)]
    rhs[0] += e * holding
    return [holding, *_thomas(diag, off, rhs), 0.0]


def replan_coefficients(p: ACParams, sig: SignalParams) -> tuple[list[float], list[float]]:
    """Feedback form of :func:`replan`: with ``m`` intervals left the next
    child order is ``c[m] * holding - d[m] * signal`` (index 0 unused).

    The re-plan is linear in ``(holding, signal)``, so ``c`` and ``d`` are the
    first-step responses to the two right-hand sides (API_SPEC 12.2).
    """
    _check_ac(p)
    n_int = p.n_intervals
    c = [0.0] * (n_int + 1)
    d = [0.0] * (n_int + 1)
    if n_int >= 1:
        c[1] = 1.0
    e = p.eta_tilde / p.tau
    for m in range(2, n_int + 1):
        diag, off = _replan_system(p, m)
        r_hold = [0.0] * (m - 1)
        r_hold[0] = e
        r_sig = [0.5 * p.tau * sig.phi**i for i in range(m - 1)]
        c[m] = 1.0 - _thomas(diag, off, r_hold)[0]
        d[m] = _thomas(diag, off, r_sig)[0]
    return c, d


@dataclass(frozen=True)
class AdaptiveSimResult:
    """Static vs adaptive implementation shortfall over ``n_paths`` paths.

    ``saving`` is the paired per-path difference ``static - adaptive``
    (positive = adaptive cheaper); SEs are sample std / sqrt(M)."""

    static_mean: float
    static_se: float
    static_std: float
    adaptive_mean: float
    adaptive_se: float
    adaptive_std: float
    saving_mean: float
    saving_se: float
    n_paths: int


def simulate_adaptive(
    p: ACParams, sig: SignalParams, n_paths: int = 10_000, seed: int = 0
) -> AdaptiveSimResult:
    """Monte Carlo of static vs adaptive execution (loop and draws pinned in
    API_SPEC 12.2).  Both strategies see the same draws on each path.

    Per interval ``j = 1..N``, for each strategy: child order ``q`` (static:
    the time-0 plan's ``n_j``; adaptive: ``c[m] x - d[m] s`` with
    ``m = N - j + 1``; at ``j = N`` both send the whole remainder); then draw
    ``u``, ``xi``, ``eps`` (in that order, one each per path, shared by the
    two strategies); fill ``e = f q`` with ``f = 1 - fill_risk u`` (``f = 1`` at
    ``j = N``); proceeds ``+= e (S - eta e / tau)``; then
    ``S += s tau + sigma sqrt(tau) xi - gamma_p e`` and
    ``s = phi s + sqrt(1 - phi**2) signal_sd eps``.  ``IS = -proceeds``
    (``S_0 = 0``).
    """
    _check_ac(p)
    _require(
        _is_int(n_paths) and 2 <= n_paths <= MAX_PATHS,
        f"n_paths must be an integer in [2, {MAX_PATHS}]",
    )
    n_int = p.n_intervals
    plan = replan(p, sig, p.total_shares, 0.0, 0)
    c, d = replan_coefficients(p, sig)
    rng = np.random.default_rng(seed)
    srt = p.sigma * math.sqrt(p.tau)
    innov = math.sqrt(1.0 - sig.phi**2) * sig.signal_sd
    x_st = np.full(n_paths, p.total_shares)
    x_ad = np.full(n_paths, p.total_shares)
    px_st = np.zeros(n_paths)
    px_ad = np.zeros(n_paths)
    pr_st = np.zeros(n_paths)
    pr_ad = np.zeros(n_paths)
    sgn = np.zeros(n_paths)
    for j in range(1, n_int + 1):
        m = n_int - j + 1
        if j == n_int:
            q_st = x_st.copy()
            q_ad = x_ad.copy()
        else:
            q_st = np.full(n_paths, plan[j - 1] - plan[j])
            q_ad = c[m] * x_ad - d[m] * sgn
        u = rng.random(n_paths)
        xi = rng.standard_normal(n_paths)
        eps = rng.standard_normal(n_paths)
        f = np.ones(n_paths) if j == n_int else 1.0 - sig.fill_risk * u
        e_st = f * q_st
        e_ad = f * q_ad
        pr_st += e_st * (px_st - p.eta * e_st / p.tau)
        pr_ad += e_ad * (px_ad - p.eta * e_ad / p.tau)
        common = sgn * p.tau + srt * xi
        px_st = px_st + common - p.gamma_p * e_st
        px_ad = px_ad + common - p.gamma_p * e_ad
        sgn = sig.phi * sgn + innov * eps
        x_st = x_st - e_st
        x_ad = x_ad - e_ad
    is_st = -pr_st
    is_ad = -pr_ad
    sav = is_st - is_ad
    sd_st = float(np.std(is_st, ddof=1))
    sd_ad = float(np.std(is_ad, ddof=1))
    sd_sav = float(np.std(sav, ddof=1))
    root = math.sqrt(n_paths)
    return AdaptiveSimResult(
        static_mean=float(np.mean(is_st)),
        static_se=sd_st / root,
        static_std=sd_st,
        adaptive_mean=float(np.mean(is_ad)),
        adaptive_se=sd_ad / root,
        adaptive_std=sd_ad,
        saving_mean=float(np.mean(sav)),
        saving_se=sd_sav / root,
        n_paths=n_paths,
    )


# ------------------------------------------------------------------ internalisation


@dataclass(frozen=True)
class InternalisationResult:
    """Crossing of opposite parent orders (API_SPEC 12.3).  Costs are
    currency; ``*_saved_bps`` are per order in bps of its own notional and
    ``saved_bps`` is in bps of the gross notional ``(B + S) * price``."""

    crossed_shares: float
    residual_shares: float
    residual_side: str
    baseline_cost: float
    internal_cost: float
    saved_cost: float
    saved_bps: float
    buy_saved_bps: tuple[float, ...]
    sell_saved_bps: tuple[float, ...]


def _orders(name: str, q: Sequence[float]) -> list[float]:
    out = [float(v) for v in q]
    _require(len(out) <= MAX_ORDERS, f"{name} must have at most {MAX_ORDERS} orders")
    _require(all(math.isfinite(v) and v > 0.0 for v in out), f"{name} must be finite and > 0")
    return out


def internalise(
    buys: Sequence[float],
    sells: Sequence[float],
    *,
    price: float,
    spread: float,
    adv: float,
    daily_vol: float,
    n_intervals: int,
    tau: float,
    alpha: float = 0.5,
    y: float = 1.0,
    perm_frac: float = 0.10,
    weights: Sequence[float] | None = None,
) -> InternalisationResult:
    """Cross buy and sell parent orders at the mid, send the residual to the
    market and report the Section 8 impact cost saved (API_SPEC 12.3)."""
    b_q = _orders("buys", buys)
    s_q = _orders("sells", sells)
    _require(len(b_q) + len(s_q) >= 1, "at least one order is required")
    _require(
        _is_int(n_intervals) and 1 <= n_intervals <= MAX_EXEC_INTERVALS,
        f"n_intervals must be an integer in [1, {MAX_EXEC_INTERVALS}]",
    )

    def cost(q: float) -> float:
        if q == 0.0:
            return 0.0
        ip = impact_params_from_market(
            price=price,
            spread=spread,
            adv=adv,
            daily_vol=daily_vol,
            total_shares=q,
            n_intervals=n_intervals,
            tau=tau,
            lambda_ra=0.0,
            alpha=alpha,
            y=y,
            perm_frac=perm_frac,
            weights=weights,
        )
        assert ip.weights is not None
        return impact_expected_cost(ip, vwap_holdings(q, ip.weights))

    big_b = sum(b_q)
    big_s = sum(s_q)
    crossed = min(big_b, big_s)
    residual = abs(big_b - big_s)
    side = "buy" if big_b > big_s else "sell" if big_s > big_b else "none"
    c_b = cost(big_b)
    c_s = cost(big_s)
    c_r = cost(residual)
    baseline = c_b + c_s
    saved = baseline - c_r

    def per_order(
        qs: list[float], total: float, c_side: float, majority: bool
    ) -> tuple[float, ...]:
        out = []
        for q in qs:
            base_i = c_side * q / total
            int_i = c_r * q / total if majority else 0.0
            out.append(bps(base_i - int_i, q * price))
        return tuple(out)

    return InternalisationResult(
        crossed_shares=crossed,
        residual_shares=residual,
        residual_side=side,
        baseline_cost=baseline,
        internal_cost=c_r,
        saved_cost=saved,
        saved_bps=bps(saved, (big_b + big_s) * price),
        buy_saved_bps=per_order(b_q, big_b, c_b, side == "buy"),
        sell_saved_bps=per_order(s_q, big_s, c_s, side == "sell"),
    )
