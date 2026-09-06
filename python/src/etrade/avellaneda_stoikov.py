"""Avellaneda-Stoikov market making (2008 paper, finite horizon).

Derivation sketch
-----------------
The mid price is an arithmetic Brownian motion ``dS = sigma dW`` (prices may
go negative under this model - documented, not guarded).  A market maker with
exponential utility ``-exp(-gamma * W_T)`` over terminal wealth quotes bid/ask
around the mid; market orders lift the quotes with Poisson intensity
``lambda(delta) = A * exp(-k * delta)`` where ``delta`` is the quote's
distance from the mid.

*Reservation price.*  Solving the HJB equation for the maker who cannot trade
(frozen inventory) with ansatz ``u(s, q, t) = -exp(-gamma*(x + q*s))
* exp(theta(q, t))`` yields the indifference (reservation) price - the mid
value at which the maker is indifferent to holding one more/fewer share:

    r(s, q, t) = s - q * gamma * sigma^2 * (T - t).

A long maker (q > 0) shades quotes *down* to attract sellers of its
inventory; a short maker shades up.

*Optimal spread.*  A first-order expansion of the HJB optimality conditions
in the fill intensities gives distances ``delta_b, delta_a`` around the
reservation price whose sum (the total quoted spread) is

    spread(t) = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / k),

with the quotes placed symmetrically around ``r``:
``ask = r + spread/2``, ``bid = r - spread/2``.  The first term is inventory-
risk compensation (shrinks to 0 at the terminal time), the second is the
monopolist-dealer markup from the exponential fill intensity (the terminal
floor).  As ``gamma -> 0`` the markup term ``(2/gamma)*ln(1 + gamma/k) ->
2/k`` analytically, but the utility derivation degenerates - this
implementation requires ``gamma > 0`` by validation, per the spec.

*Fills.*  In a discrete step of length ``dt`` each quoted side fills (unit
size) with the thinned-Poisson probability

    P(fill) = 1 - exp(-lambda(delta) * dt),   lambda(delta) = A * exp(-k*delta).

``delta`` is the distance of the quote from the *current mid* (it can be
negative when the inventory skew pushes a quote through the mid; the
probability formula still applies and stays in (0, 1)).  This exact thinned
probability is the pinned choice; the paper's own simulation (as commonly
reproduced) uses the first-order ``lambda*dt``, which over-fills by ~12% at
``lambda*dt ~ 0.28`` — that is why the mean PnL here (~56 on the liquid set)
sits below the paper's Table 1 value (~65 at ``gamma = 0.1``) while the PnL
standard deviations agree.

Simulation loop (order pinned; see API_SPEC.md)
-----------------------------------------------
Per step i (t = i*dt), vectorised across paths:
  1. quote:   compute r and spread from the *current* s, q, remaining time
              T - t; apply the inventory cap: if q >= q_max the bid is not
              quoted, if q <= -q_max the ask is not quoted (one-sided
              quoting at the cap; both suppressed only if q_max = 0).
  2. fills:   draw u_ask then u_bid ~ U(0,1) (always drawn, even for a
              suppressed side, to keep the stream deterministic); side fills
              iff quoted and u < P(fill).
  3. update:  ask fill: q -= 1, cash += ask;  bid fill: q += 1, cash -= bid.
  4. mid:     draw z ~ N(0,1);  s += sigma_t * sqrt(dt) * z, where sigma_t
              = sigma * stress_factor for t >= stress_time_frac * T (the
              *agent* keeps using the base sigma in its formulas - the
              stress is a model-misspecification scenario).
Terminal: remaining inventory is liquidated at the final mid with penalty
``eta_liq * |q|``:  PnL = cash + q * S_T - eta_liq * |q|.

The comparison agent quotes a fixed symmetric spread around the *mid* (no
inventory skew), by default the time-average of the A-S spread,
``gamma*sigma^2*T/2 + (2/gamma)*ln(1 + gamma/k)``, with the same cap, fill
mechanics, RNG stream and terminal liquidation.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

#: Upper bound on the number of simulation steps T/dt (pinned, API_SPEC §0).
MAX_AS_STEPS: int = 10_000_000

#: Upper bound on Monte Carlo path counts (pinned, API_SPEC §0).
MAX_PATHS: int = 10_000_000


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _is_int(v: object) -> bool:
    """True for Python/NumPy integers, False for bool and floats."""
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


@dataclass(frozen=True)
class ASParams:
    """Avellaneda-Stoikov model parameters.

    Attributes
    ----------
    s0      : initial mid price (any real; arithmetic model).
    sigma   : arithmetic mid volatility per sqrt(time unit), >= 0.
    horizon : terminal time T > 0.
    dt      : simulation step, 0 < dt <= T, with T/dt integral (1e-9 tol)
              and T/dt <= MAX_AS_STEPS.
    gamma   : risk aversion, strictly > 0 (gamma = 0 invalid per spec).
    k       : intensity decay, > 0.
    a_rate  : intensity scale A, > 0.
    q_max   : hard inventory cap (units), integer >= 0.
    eta_liq : terminal liquidation penalty per unit inventory, >= 0.

    Units are abstract but must be consistent: ``horizon``/``dt`` in time
    units, ``sigma`` in price/sqrt(time unit), ``gamma`` in 1/price,
    ``k`` in 1/price, ``a_rate`` in fills per time unit, ``eta_liq`` in price
    per unit of inventory.
    """

    s0: float
    sigma: float
    horizon: float
    dt: float
    gamma: float
    k: float
    a_rate: float
    q_max: int
    eta_liq: float

    def __post_init__(self) -> None:
        _require(math.isfinite(self.s0), "s0 must be finite")
        _require(math.isfinite(self.sigma) and self.sigma >= 0.0, "sigma must be >= 0")
        _require(math.isfinite(self.horizon) and self.horizon > 0.0, "horizon must be > 0")
        _require(math.isfinite(self.dt) and 0.0 < self.dt <= self.horizon,
                 "dt must satisfy 0 < dt <= horizon")
        _require(math.isfinite(self.gamma) and self.gamma > 0.0,
                 "gamma must be > 0 (gamma = 0 is outside the A-S model)")
        _require(math.isfinite(self.k) and self.k > 0.0, "k must be > 0")
        _require(math.isfinite(self.a_rate) and self.a_rate > 0.0, "a_rate (A) must be > 0")
        _require(_is_int(self.q_max) and self.q_max >= 0,
                 "q_max must be an integer >= 0")
        _require(math.isfinite(self.eta_liq) and self.eta_liq >= 0.0,
                 "eta_liq must be >= 0")
        n = self.horizon / self.dt
        _require(abs(n - round(n)) < 1e-9, "horizon must be an integer multiple of dt")
        _require(n <= MAX_AS_STEPS, f"horizon/dt must be <= {MAX_AS_STEPS} steps")

    @property
    def n_steps(self) -> int:
        """Number of simulation steps T/dt."""
        return int(round(self.horizon / self.dt))


def reservation_price(params: ASParams, s: float, q: float, t: float) -> float:
    """r(s, q, t) = s - q * gamma * sigma^2 * (T - t).

    Requires finite ``s`` and ``q`` and ``0 <= t <= T``.
    """
    _require(math.isfinite(s) and math.isfinite(q), "s and q must be finite")
    _require(0.0 <= t <= params.horizon, "t must be in [0, horizon]")
    return s - q * params.gamma * params.sigma ** 2 * (params.horizon - t)


def optimal_spread(params: ASParams, t: float) -> float:
    """Total spread gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/k).

    Strictly positive; decreases linearly in t to the terminal floor
    (2/gamma)*ln(1 + gamma/k) at t = T.
    """
    _require(0.0 <= t <= params.horizon, "t must be in [0, horizon]")
    return (params.gamma * params.sigma ** 2 * (params.horizon - t)
            + 2.0 / params.gamma * math.log1p(params.gamma / params.k))


def quotes(params: ASParams, s: float, q: float, t: float) -> tuple[float, float]:
    """(bid, ask) = reservation price -/+ half the optimal spread."""
    r = reservation_price(params, s, q, t)
    half = 0.5 * optimal_spread(params, t)
    return r - half, r + half


def intensity(params: ASParams, delta: float) -> float:
    """Arrival intensity lambda(delta) = A * exp(-k * delta).

    ``delta`` is signed (a quote through the mid has delta < 0 and a higher
    intensity); it must be finite.
    """
    _require(math.isfinite(delta), "delta must be finite")
    return params.a_rate * math.exp(-params.k * delta)


def fill_probability(params: ASParams, delta: float) -> float:
    """Per-step fill probability 1 - exp(-lambda(delta) * dt), in [0, 1)."""
    return -math.expm1(-intensity(params, delta) * params.dt)


def average_as_spread(params: ASParams) -> float:
    """Time average over [0, T] of the A-S spread (the fixed agent's default).

    (1/T) * integral = gamma*sigma^2*T/2 + (2/gamma)*ln(1 + gamma/k).
    """
    return (0.5 * params.gamma * params.sigma ** 2 * params.horizon
            + 2.0 / params.gamma * math.log1p(params.gamma / params.k))


@dataclass(frozen=True)
class ASSimResult:
    """Per-path arrays and aggregate statistics of a market-making simulation.

    Per-path arrays (length n_paths): ``pnl``, ``mean_abs_q_path`` (time
    average of |q| over the path), ``max_abs_q_path`` (path max of |q|),
    ``rms_q_path`` (sqrt of the time average of q^2), ``fills_path`` (total
    number of fills, both sides counted — a step where both the bid and the
    ask fill counts as two fills).  Aggregates are means/stds over paths; ``se_*`` are standard
    errors (SE of a mean = sample std / sqrt(M); SE of std_pnl uses the
    normal approximation std/sqrt(2(M-1))).
    """

    pnl: np.ndarray = field(repr=False)
    mean_abs_q_path: np.ndarray = field(repr=False)
    max_abs_q_path: np.ndarray = field(repr=False)
    rms_q_path: np.ndarray = field(repr=False)
    fills_path: np.ndarray = field(repr=False)
    overall_max_abs_q: float
    cap_violated: bool

    @property
    def n_paths(self) -> int:
        return int(self.pnl.size)

    @property
    def mean_pnl(self) -> float:
        return float(np.mean(self.pnl))

    @property
    def std_pnl(self) -> float:
        return float(np.std(self.pnl, ddof=1))

    @property
    def sharpe(self) -> float:
        """Sharpe-like ratio mean/std of terminal PnL (inf-safe)."""
        s = self.std_pnl
        return math.inf if s == 0.0 else self.mean_pnl / s

    @property
    def mean_abs_q(self) -> float:
        return float(np.mean(self.mean_abs_q_path))

    @property
    def mean_max_abs_q(self) -> float:
        return float(np.mean(self.max_abs_q_path))

    @property
    def rms_q(self) -> float:
        return float(np.mean(self.rms_q_path))

    @property
    def mean_fills(self) -> float:
        return float(np.mean(self.fills_path))

    def se(self, stat: str) -> float:
        """Standard error of an aggregate statistic ('mean_pnl', 'std_pnl',
        'mean_abs_q', 'mean_max_abs_q', 'rms_q', 'mean_fills')."""
        m = self.n_paths
        arrays = {"mean_pnl": self.pnl, "mean_abs_q": self.mean_abs_q_path,
                  "mean_max_abs_q": self.max_abs_q_path, "rms_q": self.rms_q_path,
                  "mean_fills": self.fills_path}
        if stat == "std_pnl":
            return self.std_pnl / math.sqrt(2.0 * (m - 1))
        if stat in arrays:
            return float(np.std(arrays[stat], ddof=1)) / math.sqrt(m)
        raise ValueError(f"unknown statistic '{stat}'")


def simulate(params: ASParams, n_paths: int, seed: int,
             agent: Literal["as", "fixed"] = "as",
             fixed_spread: float | None = None,
             stress_factor: float = 1.0,
             stress_time_frac: float = 0.5) -> ASSimResult:
    """Simulate the market maker over ``n_paths`` seeded paths.

    ``agent='as'`` quotes the A-S reservation price +/- half the optimal
    spread; ``agent='fixed'`` quotes the mid +/- half a fixed spread
    (default: `average_as_spread`).  ``stress_factor > 1`` multiplies the
    *realised* mid volatility for t >= stress_time_frac * T while the agent
    keeps quoting with the base sigma.  Loop order and draw order are pinned
    in the module docstring / API_SPEC.md.  Deterministic per (seed, agent,
    n_paths).
    """
    _require(_is_int(n_paths) and 2 <= n_paths <= MAX_PATHS,
             f"n_paths must be an integer in [2, {MAX_PATHS}]")
    _require(math.isfinite(stress_factor) and stress_factor > 0.0,
             "stress_factor must be > 0")
    _require(math.isfinite(stress_time_frac) and 0.0 <= stress_time_frac <= 1.0,
             "stress_time_frac must be in [0, 1]")
    if agent not in ("as", "fixed"):
        raise ValueError("agent must be 'as' or 'fixed'")
    if fixed_spread is None:
        fixed_spread = average_as_spread(params)
    _require(math.isfinite(fixed_spread) and fixed_spread > 0.0,
             "fixed_spread must be finite and > 0")

    rng = np.random.default_rng(seed)
    n_steps = params.n_steps
    g, sig2, big_t, dt = params.gamma, params.sigma ** 2, params.horizon, params.dt
    markup = 2.0 / g * math.log1p(g / params.k)
    sqdt = math.sqrt(dt)

    s = np.full(n_paths, params.s0)
    q = np.zeros(n_paths)
    cash = np.zeros(n_paths)
    sum_abs_q = np.zeros(n_paths)
    sum_q2 = np.zeros(n_paths)
    max_abs_q = np.zeros(n_paths)
    fills = np.zeros(n_paths)
    cap_violated = False

    for i in range(n_steps):
        t = i * dt
        # 1. quote
        if agent == "as":
            r = s - q * g * sig2 * (big_t - t)
            half = 0.5 * (g * sig2 * (big_t - t) + markup)
        else:
            r = s
            half = 0.5 * fixed_spread
        ask = r + half
        bid = r - half
        quote_ask = q > -params.q_max
        quote_bid = q < params.q_max
        # 2. fill draws (u_ask first, then u_bid; both always drawn)
        u_ask = rng.random(n_paths)
        u_bid = rng.random(n_paths)
        p_ask = -np.expm1(-params.a_rate * np.exp(-params.k * (ask - s)) * dt)
        p_bid = -np.expm1(-params.a_rate * np.exp(-params.k * (s - bid)) * dt)
        fill_ask = quote_ask & (u_ask < p_ask)
        fill_bid = quote_bid & (u_bid < p_bid)
        # 3. inventory / cash update (unit fill size)
        q += fill_bid.astype(float) - fill_ask.astype(float)
        cash += np.where(fill_ask, ask, 0.0) - np.where(fill_bid, bid, 0.0)
        fills += fill_ask.astype(float) + fill_bid.astype(float)   # count both sides
        abs_q = np.abs(q)
        if np.any(abs_q > params.q_max + 1e-12):
            cap_violated = True
        sum_abs_q += abs_q
        sum_q2 += q * q
        np.maximum(max_abs_q, abs_q, out=max_abs_q)
        # 4. mid move (stressed vol only affects the realised path)
        z = rng.standard_normal(n_paths)
        sig_t = params.sigma * (stress_factor if t >= stress_time_frac * big_t else 1.0)
        s = s + sig_t * sqdt * z

    pnl = cash + q * s - params.eta_liq * np.abs(q)
    return ASSimResult(
        pnl=pnl,
        mean_abs_q_path=sum_abs_q / n_steps,
        max_abs_q_path=max_abs_q,
        rms_q_path=np.sqrt(sum_q2 / n_steps),
        fills_path=fills,
        overall_max_abs_q=float(np.max(max_abs_q)),
        cap_violated=cap_violated,
    )
