"""Hedged market making and risk controls (API_SPEC Section 14).

Why this module exists
----------------------
The Section 9 maker (`etrade.mm`) can only shed inventory by skewing its own
quotes and waiting for the other side to trade.  A desk that makes markets in
an ETF also trades the future or the basket: it lays inventory risk off in a
correlated instrument, pays that instrument's half spread and fee to do so,
and keeps the residual (basis) risk.  The same desk runs hard risk controls
that sit outside the quoting model: a loss limit, a position limit, a quote
throttle and a kill switch.  This module adds both to the Section 9 loop and
keeps the Section 9 accounting exact.

Model
-----
Instrument A (the one quoted) moves ``sigma*sqrt(dt)*z`` per step as in
Section 9.  Instrument B moves ``sigma*sqrt(dt)*(rho*z + sqrt(1 - rho^2)*z_b)``:
B is quoted in A-equivalent units (same per-step volatility), so the
minimum-variance hedge ratio is ``rho`` shares of B per share of A
(Ederington 1979).  The maker holds ``q`` of A and ``y`` of B.  The hedging
rule acts on ``u = q + y/hedge_ratio``, the A shares not covered at the
chosen ratio.  The quoting agent skews on the risk-equivalent exposure
``x = q + rho*y``: the per-step PnL variance of the book is
``sigma^2 dt (q^2 + 2 rho q y + y^2)``, whose half-derivative in ``q`` is
``sigma^2 dt x``, so ``x`` takes the place of ``q`` in the reservation price
(the multi-asset reservation price of Gueant 2017, Ch. 6).  A full hedge at
``hedge_ratio = rho`` leaves ``x = (1 - rho^2) q``: once the hedge has taken
the risk, the book need not lean on its quotes to shed it.

Simulation loop (order pinned)
------------------------------
Per path, per step ``i``, six numbers are drawn always: the five Section 9
draws ``z, u_sig, u_ask, u_bid, u_inf`` and then ``z_b ~ N(0,1)``.  Then:
stops (kill switch, loss limit), quotes (throttled), Section 9 fills, the
position limit, the hedge, accounting, and the moves of A and B.  See
API_SPEC 14.4 for every step.

Attribution (exact per path)
----------------------------
    pnl = spread + fees + inventory - liquidation + hedge_pnl - hedge_costs - risk_costs

``hedge_pnl`` is the sum of ``y * (B move)``; ``hedge_costs`` is
``hedge_cost * |dy|`` over every B trade including the terminal unwind;
``risk_costs`` is ``flatten_penalty * |q|`` over every forced flatten of A.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .avellaneda_stoikov import MAX_PATHS, ASParams
from .mm import MM_STATS, MMConfig

_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


# ------------------------------------------------------------ deterministic pieces


def hedge_trade(
    q: float, y: float, hedge_ratio: float, threshold: float, target_frac: float
) -> float:
    """B shares to trade under the pinned hedging rule (API_SPEC 14.2).

    With ``u = q + y/hedge_ratio`` the A shares not covered at the chosen
    ratio, trade ``dy = -hedge_ratio * (1 - target_frac) * u`` when
    ``|u| > threshold`` (leaving ``target_frac * u`` uncovered), else ``0``.  ``hedge_ratio = 0``
    disables hedging and returns ``0``.
    """
    _require(math.isfinite(q) and math.isfinite(y), "q and y must be finite")
    _require(
        math.isfinite(hedge_ratio) and hedge_ratio >= 0.0, "hedge_ratio must be finite and >= 0"
    )
    _require(
        math.isfinite(threshold) and threshold >= 0.0, "hedge_threshold must be finite and >= 0"
    )
    _require(
        math.isfinite(target_frac) and 0.0 <= target_frac < 1.0, "hedge_target must be in [0, 1)"
    )
    if hedge_ratio == 0.0:
        return 0.0
    u = q + y / hedge_ratio
    if abs(u) <= threshold:
        return 0.0
    return -hedge_ratio * (1.0 - target_frac) * u


def quantile(values: list[float] | npt.NDArray[np.float64], p: float) -> float:
    """Pinned sample quantile (API_SPEC 14.6): linear interpolation between
    order statistics, Hyndman and Fan (1996) type 7 (NumPy's default).

    Sort ``v`` ascending, ``h = (n - 1) p``, ``j = floor(h)``;
    ``Q = v[j] + (h - j) (v[j+1] - v[j])`` (``Q = v[n-1]`` when ``j = n - 1``).
    """
    v = np.sort(np.asarray(values, dtype=float))
    _require(v.ndim == 1 and v.size >= 1, "values must be a non-empty 1-D array")
    _require(bool(np.all(np.isfinite(v))), "values must be finite")
    _require(math.isfinite(p) and 0.0 <= p <= 1.0, "p must be in [0, 1]")
    h = (v.size - 1) * p
    j = math.floor(h)
    if j >= v.size - 1:
        return float(v[-1])
    return float(v[j] + (h - j) * (v[j + 1] - v[j]))


# ------------------------------------------------------------ configuration


@dataclass(frozen=True)
class HedgeConfig:
    """Hedge instrument and rule (API_SPEC 14.1); ``hedge_ratio = 0`` is no hedge."""

    rho: float = 0.0
    hedge_ratio: float = 0.0
    hedge_cost: float = 0.0
    hedge_threshold: float = 0.0
    hedge_target: float = 0.0

    def __post_init__(self) -> None:
        _require(math.isfinite(self.rho) and -1.0 <= self.rho <= 1.0, "rho must be in [-1, 1]")
        _require(
            math.isfinite(self.hedge_ratio) and self.hedge_ratio >= 0.0,
            "hedge_ratio must be finite and >= 0",
        )
        _require(
            math.isfinite(self.hedge_cost) and self.hedge_cost >= 0.0,
            "hedge_cost must be finite and >= 0",
        )
        _require(
            math.isfinite(self.hedge_threshold) and self.hedge_threshold >= 0.0,
            "hedge_threshold must be finite and >= 0",
        )
        _require(
            math.isfinite(self.hedge_target) and 0.0 <= self.hedge_target < 1.0,
            "hedge_target must be in [0, 1)",
        )


@dataclass(frozen=True)
class RiskConfig:
    """Risk controls (API_SPEC 14.1); every default is off.

    ``loss_limit`` (0 = off), ``position_limit`` (0 = off),
    ``flatten_penalty`` per share flattened, ``throttle_updates`` quote
    updates per ``throttle_window`` steps (0 = off), ``kill_step`` (-1 = off).
    """

    loss_limit: float = 0.0
    position_limit: int = 0
    flatten_penalty: float = 0.0
    throttle_updates: int = 0
    throttle_window: int = 1
    kill_step: int = -1

    def __post_init__(self) -> None:
        _require(
            math.isfinite(self.loss_limit) and self.loss_limit >= 0.0,
            "loss_limit must be finite and >= 0",
        )
        _require(
            _is_int(self.position_limit) and self.position_limit >= 0,
            "position_limit must be an integer >= 0",
        )
        _require(
            math.isfinite(self.flatten_penalty) and self.flatten_penalty >= 0.0,
            "flatten_penalty must be finite and >= 0",
        )
        _require(
            _is_int(self.throttle_updates) and self.throttle_updates >= 0,
            "throttle_updates must be an integer >= 0",
        )
        _require(
            _is_int(self.throttle_window) and self.throttle_window >= 1,
            "throttle_window must be an integer >= 1",
        )
        _require(
            _is_int(self.kill_step) and self.kill_step >= -1, "kill_step must be an integer >= -1"
        )


#: Statistics reported by :class:`HedgedResult` (API_SPEC 14.5).
HEDGE_STATS: tuple[str, ...] = (
    *MM_STATS,
    "hedge_pnl",
    "hedge_costs",
    "hedge_trades",
    "risk_costs",
    "rms_x",
    "rms_skew",
    "quote_updates",
    "loss_stop",
    "kill_stop",
    "pos_flattens",
    "overshoot",
    "stop_move",
)


@dataclass(frozen=True)
class HedgedResult:
    """Per-path arrays (``paths[name]``, length n_paths) for every name in
    :data:`HEDGE_STATS`, plus the largest attribution residual seen."""

    paths: dict[str, npt.NDArray[np.float64]] = field(repr=False)
    max_identity_error: float

    @property
    def n_paths(self) -> int:
        return int(self.paths["pnl"].size)

    def _arr(self, stat: str) -> npt.NDArray[np.float64]:
        _require(stat in HEDGE_STATS, f"unknown statistic '{stat}'")
        return self.paths[stat]

    def mean(self, stat: str) -> float:
        """Mean over paths of a statistic."""
        return float(np.mean(self._arr(stat)))

    def se(self, stat: str) -> float:
        """Standard error of the mean: sample std (ddof 1) / sqrt(n_paths)."""
        return float(np.std(self._arr(stat), ddof=1)) / math.sqrt(self.n_paths)

    def var(self, stat: str) -> float:
        """Population variance ``m2 = mean((v - mean)^2)`` (API_SPEC 14.5)."""
        v = self._arr(stat)
        return float(np.mean((v - np.mean(v)) ** 2))

    def var_se(self, stat: str) -> float:
        """Delta-method standard error of :meth:`var`: ``sqrt((m4 - m2^2)/n)``."""
        v = self._arr(stat)
        d = v - np.mean(v)
        m2 = float(np.mean(d**2))
        m4 = float(np.mean(d**4))
        return math.sqrt(max(0.0, m4 - m2 * m2) / self.n_paths)

    def quantile(self, stat: str, p: float) -> float:
        """Pinned type-7 quantile (:func:`quantile`) of a statistic over paths."""
        return quantile(self._arr(stat), p)


# ------------------------------------------------------------ simulation


def simulate_hedged(
    params: ASParams,
    n_paths: int,
    seed: int,
    config: MMConfig | None = None,
    hedge: HedgeConfig | None = None,
    risk: RiskConfig | None = None,
) -> HedgedResult:
    """Simulate the Section 9 maker with a hedge and risk controls (API_SPEC 14.4).

    Deterministic per seed.  With the default ``hedge`` and ``risk`` the
    result equals Section 9 in distribution (not sample by sample: the sixth
    draw shifts the stream).
    """
    cfg = MMConfig() if config is None else config
    hc = HedgeConfig() if hedge is None else hedge
    rc = RiskConfig() if risk is None else risk
    _require(
        _is_int(n_paths) and 2 <= n_paths <= MAX_PATHS,
        f"n_paths must be an integer in [2, {MAX_PATHS}]",
    )
    n_steps = params.n_steps
    _require(cfg.latency_steps <= n_steps, "latency_steps must be <= n_steps")
    _require(rc.kill_step <= n_steps, "kill_step must be <= n_steps")
    _require(rc.throttle_window <= n_steps, "throttle_window must be <= n_steps")
    rng = np.random.default_rng(seed)
    g, sig, big_t, dt, k, a_rate, qmax = (
        params.gamma,
        params.sigma,
        params.horizon,
        params.dt,
        params.k,
        params.a_rate,
        params.q_max,
    )
    sig2 = sig * sig
    sqdt = math.sqrt(dt)
    markup = 2.0 / g * math.log1p(g / k)
    fixed_half = 0.5 * (
        cfg.fixed_spread if cfg.fixed_spread is not None else 0.5 * g * sig2 * big_t + markup
    )
    c1 = math.log1p(g / k) / g
    c2 = math.sqrt(sig2 * g / (2.0 * k * a_rate) * (1.0 + g / k) ** (1.0 + k / g))
    shift_scale = (2.0 * cfg.signal_accuracy - 1.0) * sig * sqdt * _SQRT_2_OVER_PI
    h, rho = hc.hedge_ratio, hc.rho
    rho_perp = math.sqrt(max(0.0, 1.0 - rho * rho))
    m, w = rc.throttle_updates, rc.throttle_window
    pen = rc.flatten_penalty

    rows = np.arange(n_paths)
    s = np.full(n_paths, params.s0)
    b = np.full(n_paths, params.s0)
    hist = [s.copy() for _ in range(cfg.latency_steps + 1)]
    q = np.zeros(n_paths)
    y = np.zeros(n_paths)
    cash = np.zeros(n_paths)
    active = np.ones(n_paths, dtype=bool)
    bid_q = np.full(n_paths, params.s0)
    ask_q = np.full(n_paths, params.s0)
    upd = np.full((n_paths, max(m, 1)), -np.inf)
    ptr = np.zeros(n_paths, dtype=np.int64)
    mark_prev = np.zeros(n_paths)
    acc = {name: np.zeros(n_paths) for name in HEDGE_STATS}
    sum_q2 = np.zeros(n_paths)
    sum_x2 = np.zeros(n_paths)
    sum_skew2 = np.zeros(n_paths)

    for i in range(n_steps):
        t = i * dt
        # 1. the six draws, always, in the pinned order.
        z = rng.standard_normal(n_paths)
        u_sig = rng.random(n_paths)
        u_ask = rng.random(n_paths)
        u_bid = rng.random(n_paths)
        u_inf = rng.random(n_paths)
        z_b = rng.standard_normal(n_paths)
        # 2. stops: kill switch first, then the loss limit on the marked PnL.
        mark = cash + q * s + y * b
        kill = active & (i == rc.kill_step)
        loss = active & ~kill & (rc.loss_limit > 0.0) & (mark < -rc.loss_limit)
        stop = kill | loss
        acc["kill_stop"] += kill
        acc["loss_stop"] += loss
        acc["overshoot"] = np.where(loss, np.maximum(0.0, -rc.loss_limit - mark), acc["overshoot"])
        acc["stop_move"] = np.where(loss, np.abs(mark - mark_prev), acc["stop_move"])
        cost_a = np.where(stop, pen * np.abs(q), 0.0)
        cash += np.where(stop, q * s, 0.0) - cost_a
        acc["risk_costs"] += cost_a
        q = np.where(stop, 0.0, q)
        cost_b = np.where(stop, hc.hedge_cost * np.abs(y), 0.0)
        cash += np.where(stop, y * b, 0.0) - cost_b
        acc["hedge_costs"] += cost_b
        acc["hedge_trades"] += stop & (y != 0.0)
        y = np.where(stop, 0.0, y)
        active &= ~stop
        mark_prev = mark
        # 3. desired quotes, skewed on the risk-equivalent exposure x.
        x = q + rho * y
        sz = np.sign(z)
        pred = np.where(u_sig < cfg.signal_accuracy, sz, -sz)
        fair = hist[0] + shift_scale * pred
        if cfg.agent == "as":
            skew = x * g * sig2 * (big_t - t)
            half = 0.5 * (g * sig2 * (big_t - t) + markup)
            bid, ask = fair - skew - half, fair - skew + half
        elif cfg.agent == "glft":
            skew = x * c2
            bid = fair - (c1 + (2.0 * x + 1.0) / 2.0 * c2)
            ask = fair + (c1 - (2.0 * x - 1.0) / 2.0 * c2)
        else:
            skew = np.zeros(n_paths)
            bid, ask = fair - fixed_half, fair + fixed_half
        if cfg.tick > 0.0:
            bid = np.floor(bid / cfg.tick + 1e-9) * cfg.tick
            ask = np.ceil(ask / cfg.tick - 1e-9) * cfg.tick
        sum_skew2 += np.where(active, skew * skew, 0.0)
        # 4. throttle: update only if fewer than m updates in the last w steps.
        if m > 0:
            allow = active & (upd[rows, ptr] <= i - w)
            upd[rows[allow], ptr[allow]] = i
            ptr = np.where(allow, (ptr + 1) % m, ptr)
        else:
            allow = active
        bid_q = np.where(allow, bid, bid_q)
        ask_q = np.where(allow, ask, ask_q)
        acc["quote_updates"] += allow
        # 5. Section 9 fills against the standing quotes.
        can_ask = active & (q > -qmax)
        can_bid = active & (q < qmax)
        with np.errstate(over="ignore"):  # a stale quote deep through the mid fills surely
            p_ask = -np.expm1(-a_rate * np.exp(-k * (ask_q - s)) * dt)
            p_bid = -np.expm1(-a_rate * np.exp(-k * (s - bid_q)) * dt)
        fill_ask = can_ask & (u_ask < p_ask)
        fill_bid = can_bid & (u_bid < p_bid)
        move = sig * sqdt * z
        informed = u_inf < cfg.informed_prob
        inf_ask = informed & (z > 0.0) & can_ask & ~fill_ask & (ask_q < s + move)
        inf_bid = informed & (z < 0.0) & can_bid & ~fill_bid & (bid_q > s + move)
        fa = fill_ask | inf_ask
        fb = fill_bid | inf_bid
        fa_f, fb_f = fa.astype(float), fb.astype(float)
        n_fill = fa_f + fb_f
        q = q + fb_f - fa_f
        cash += np.where(fa, ask_q, 0.0) - np.where(fb, bid_q, 0.0) + cfg.maker_fee * n_fill
        acc["spread"] += np.where(fa, ask_q - s, 0.0) + np.where(fb, s - bid_q, 0.0)
        acc["fees"] += cfg.maker_fee * n_fill
        acc["fills"] += n_fill
        acc["informed_fills"] += inf_ask.astype(float) + inf_bid.astype(float)
        # 6. position limit: flatten A at the mid less the penalty.
        if rc.position_limit > 0:
            hit = active & (np.abs(q) >= rc.position_limit)
            cost = np.where(hit, pen * np.abs(q), 0.0)
            cash += np.where(hit, q * s, 0.0) - cost
            acc["risk_costs"] += cost
            acc["pos_flattens"] += hit
            q = np.where(hit, 0.0, q)
        # 7. hedge in B at its mid plus the hedge cost.
        if h > 0.0:
            u = q + y / h
            do = active & (np.abs(u) > hc.hedge_threshold)
            dy = np.where(do, -h * (1.0 - hc.hedge_target) * u, 0.0)
            cost = hc.hedge_cost * np.abs(dy)
            cash -= dy * b + cost
            acc["hedge_costs"] += cost
            acc["hedge_trades"] += do
            y = y + dy
        # 8. accounting, then the moves of A and B.
        move_b = sig * sqdt * (rho * z + rho_perp * z_b)
        acc["inventory"] += q * move
        acc["markout"] += (fb_f - fa_f) * move
        acc["hedge_pnl"] += y * move_b
        x = q + rho * y
        sum_q2 += q * q
        sum_x2 += x * x
        s = s + move
        b = b + move_b
        hist = [*hist[1:], s.copy()]

    liq = params.eta_liq * np.abs(q)
    unwind = hc.hedge_cost * np.abs(y)
    acc["hedge_costs"] += unwind
    acc["hedge_trades"] += y != 0.0
    pnl = cash + q * s + y * b - liq - unwind
    acc["liquidation"] = liq
    acc["pnl"] = pnl
    acc["rms_q"] = np.sqrt(sum_q2 / n_steps)
    acc["rms_x"] = np.sqrt(sum_x2 / n_steps)
    acc["rms_skew"] = np.sqrt(sum_skew2 / n_steps)
    recon = (
        acc["spread"]
        + acc["fees"]
        + acc["inventory"]
        - liq
        + acc["hedge_pnl"]
        - acc["hedge_costs"]
        - acc["risk_costs"]
    )
    scale = np.maximum(1.0, np.abs(pnl))
    err = float(np.max(np.abs(recon - pnl) / scale))
    return HedgedResult(paths=acc, max_identity_error=err)
