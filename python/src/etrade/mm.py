"""Market making that can lose money: informed flow, ticks, fees, latency,
fair-value signals, PnL attribution and markouts (API_SPEC Section 9).

Why this module exists
----------------------
The Section 2 simulator (`etrade.avellaneda_stoikov.simulate`) is the paper's
world: fills are independent of where the price goes next, quotes may sit at
any real price, trading is free and quotes never go stale.  In that world a
market maker cannot lose money on average.  This module keeps the A-S quoting
logic and adds, one switch at a time, the frictions that decide profit in a
real book, plus the accounting a desk uses to see where the money went.

Simulation loop (order pinned)
------------------------------
Per path, per step ``i`` (``t = i*dt``), five random numbers are drawn in this
order, always, whatever the configuration:

1. ``z ~ N(0,1)``: the coming mid move ``sigma*sqrt(dt)*z``;
2. ``u_sig ~ U(0,1)``: whether the fair-value signal is right;
3. ``u_ask``, then ``u_bid ~ U(0,1)``: uninformed fill draws;
4. ``u_inf ~ U(0,1)``: whether an informed trader arrives.

Then: the signal predicts ``sign(z)`` when ``u_sig < signal_accuracy`` and
``-sign(z)`` otherwise, and shifts fair value by
``(2*accuracy - 1) * pred * sigma*sqrt(dt)*sqrt(2/pi)`` (the expected move given
the signal; accuracy 0.5 is no signal).  Quotes are computed from the mid
``latency_steps`` steps ago plus that shift, rounded away from fair value to
the tick grid when ``tick > 0`` (bid down, ask up).  Uninformed fills use the
thinned-Poisson probability of Section 2 with distances from the *true*
current mid.  An informed trader (``u_inf < informed_prob``) trades in the
direction of ``z`` against the quote, only if that side did not already fill,
the cap allows it and the quote is inside the coming move.  Every fill earns
``maker_fee`` (positive = rebate).  Then the mid moves.

Attribution (exact per path)
----------------------------
    spread     = sum over fills of (ask - mid) or (mid - bid), mid before the move
    fees       = maker_fee * fills
    inventory  = sum over steps of q_after_fills * (mid move)
    liquidation= eta_liq * |q_T|
    pnl        = cash + q_T * S_T - liquidation = spread + fees + inventory - liquidation

``markout`` is the one-step markout of the fills (part of ``inventory``):
the mid move right after each fill, signed so that a negative value is
adverse selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

from .avellaneda_stoikov import MAX_PATHS, ASParams

Agent = Literal["as", "glft", "fixed"]

_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


# ------------------------------------------------------------ deterministic pieces


def round_to_tick(bid: float, ask: float, tick: float) -> tuple[float, float]:
    """Round a quote pair away from fair value to the tick grid (API_SPEC 9.2).

    ``bid -> floor(bid/tick + 1e-9)*tick``, ``ask -> ceil(ask/tick - 1e-9)*tick``;
    ``tick = 0`` leaves the quotes unchanged.
    """
    _require(math.isfinite(tick) and tick >= 0.0, "tick must be finite and >= 0")
    _require(math.isfinite(bid) and math.isfinite(ask), "bid and ask must be finite")
    if tick == 0.0:
        return bid, ask
    return math.floor(bid / tick + 1e-9) * tick, math.ceil(ask / tick - 1e-9) * tick


def glft_half_spreads(params: ASParams, q: float) -> tuple[float, float]:
    """Closed-form asymptotic GLFT distances ``(delta_bid, delta_ask)`` (API_SPEC 9.3).

    Gueant, Lehalle and Fernandez-Tapia (2013) asymptotic approximation for
    the inventory-constrained market maker:

        c1 = (1/gamma) ln(1 + gamma/k)
        c2 = sqrt( sigma^2 gamma / (2 k A) * (1 + gamma/k)^(1 + k/gamma) )
        delta_bid = c1 + (2q + 1)/2 * c2,   delta_ask = c1 - (2q - 1)/2 * c2

    Time-independent; quotes are ``bid = s - delta_bid``, ``ask = s + delta_ask``.
    """
    _require(math.isfinite(q), "q must be finite")
    g, k, a, s2 = params.gamma, params.k, params.a_rate, params.sigma**2
    c1 = math.log1p(g / k) / g
    c2 = math.sqrt(s2 * g / (2.0 * k * a) * (1.0 + g / k) ** (1.0 + k / g))
    return c1 + (2.0 * q + 1.0) / 2.0 * c2, c1 - (2.0 * q - 1.0) / 2.0 * c2


def weighted_mid(bid: float, ask: float, bid_size: float, ask_size: float) -> float:
    """Size-weighted mid ``(bid*ask_size + ask*bid_size)/(bid_size + ask_size)``.

    A simple proxy for the microprice (Stoikov 2018): a large bid queue pulls
    the fair value towards the ask.  Sizes must be ``>= 0`` with a positive sum.
    """
    for name, v in (("bid", bid), ("ask", ask)):
        _require(math.isfinite(v), f"{name} must be finite")
    _require(ask >= bid, "ask must be >= bid")
    _require(
        math.isfinite(bid_size) and bid_size >= 0.0 and math.isfinite(ask_size) and ask_size >= 0.0,
        "sizes must be finite and >= 0",
    )
    _require(bid_size + ask_size > 0.0, "bid_size + ask_size must be > 0")
    return (bid * ask_size + ask * bid_size) / (bid_size + ask_size)


def order_flow_imbalance(
    bid_px: list[float] | npt.NDArray[np.float64],
    bid_sz: list[float] | npt.NDArray[np.float64],
    ask_px: list[float] | npt.NDArray[np.float64],
    ask_sz: list[float] | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Per-update order-flow imbalance of Cont, Kukanov and Stoikov (2014).

    For consecutive top-of-book states ``n-1 -> n``:

        e_n = 1[Pb_n >= Pb_{n-1}] Qb_n - 1[Pb_n <= Pb_{n-1}] Qb_{n-1}
            - 1[Pa_n <= Pa_{n-1}] Qa_n + 1[Pa_n >= Pa_{n-1}] Qa_{n-1}

    Returns the ``len - 1`` values ``e_1..e_{n}``; their sum over a window is
    the OFI, which is linearly related to the mid move over that window.
    """
    arrs = [np.asarray(v, dtype=float) for v in (bid_px, bid_sz, ask_px, ask_sz)]
    n = arrs[0].size
    _require(n >= 2, "need at least 2 book states")
    _require(all(a.ndim == 1 and a.size == n for a in arrs), "book series must be 1-D and equal length")
    _require(all(bool(np.all(np.isfinite(a))) for a in arrs), "book series must be finite")
    pb, qb, pa, qa = arrs
    _require(bool(np.all(qb >= 0.0)) and bool(np.all(qa >= 0.0)), "sizes must be >= 0")
    up_b = (pb[1:] >= pb[:-1]).astype(float)
    dn_b = (pb[1:] <= pb[:-1]).astype(float)
    dn_a = (pa[1:] <= pa[:-1]).astype(float)
    up_a = (pa[1:] >= pa[:-1]).astype(float)
    return up_b * qb[1:] - dn_b * qb[:-1] - dn_a * qa[1:] + up_a * qa[:-1]


# ------------------------------------------------------------ simulation


@dataclass(frozen=True)
class MMConfig:
    """Frictions and agent for :func:`simulate_mm` (API_SPEC 9.1)."""

    agent: Agent = "as"
    tick: float = 0.0
    maker_fee: float = 0.0
    latency_steps: int = 0
    informed_prob: float = 0.0
    signal_accuracy: float = 0.5
    fixed_spread: float | None = None

    def __post_init__(self) -> None:
        _require(self.agent in ("as", "glft", "fixed"), "agent must be 'as', 'glft' or 'fixed'")
        _require(math.isfinite(self.tick) and self.tick >= 0.0, "tick must be finite and >= 0")
        _require(math.isfinite(self.maker_fee), "maker_fee must be finite")
        _require(
            isinstance(self.latency_steps, int)
            and not isinstance(self.latency_steps, bool)
            and self.latency_steps >= 0,
            "latency_steps must be an integer >= 0",
        )
        _require(
            math.isfinite(self.informed_prob) and 0.0 <= self.informed_prob <= 1.0,
            "informed_prob must be in [0, 1]",
        )
        _require(
            math.isfinite(self.signal_accuracy) and 0.5 <= self.signal_accuracy <= 1.0,
            "signal_accuracy must be in [0.5, 1]",
        )
        if self.fixed_spread is not None:
            _require(
                math.isfinite(self.fixed_spread) and self.fixed_spread > 0.0,
                "fixed_spread must be finite and > 0",
            )


#: Statistics reported by :class:`MMResult` (API_SPEC 9.5).
MM_STATS: tuple[str, ...] = (
    "pnl",
    "spread",
    "fees",
    "inventory",
    "liquidation",
    "markout",
    "fills",
    "informed_fills",
    "rms_q",
)


@dataclass(frozen=True)
class MMResult:
    """Per-path arrays (``paths[name]``, length n_paths) for every name in
    :data:`MM_STATS`, plus the largest attribution residual seen."""

    paths: dict[str, npt.NDArray[np.float64]] = field(repr=False)
    max_identity_error: float

    @property
    def n_paths(self) -> int:
        return int(self.paths["pnl"].size)

    def mean(self, stat: str) -> float:
        """Mean over paths of a statistic."""
        _require(stat in MM_STATS, f"unknown statistic '{stat}'")
        return float(np.mean(self.paths[stat]))

    def se(self, stat: str) -> float:
        """Standard error of the mean: sample std (ddof 1) / sqrt(n_paths)."""
        _require(stat in MM_STATS, f"unknown statistic '{stat}'")
        return float(np.std(self.paths[stat], ddof=1)) / math.sqrt(self.n_paths)


def simulate_mm(params: ASParams, n_paths: int, seed: int, config: MMConfig | None = None) -> MMResult:
    """Simulate the market maker with frictions (API_SPEC 9.4); deterministic per seed."""
    cfg = MMConfig() if config is None else config
    _require(
        isinstance(n_paths, int) and not isinstance(n_paths, bool) and 2 <= n_paths <= MAX_PATHS,
        f"n_paths must be an integer in [2, {MAX_PATHS}]",
    )
    n_steps = params.n_steps
    _require(cfg.latency_steps <= n_steps, "latency_steps must be <= n_steps")
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
    if cfg.agent == "glft":
        c1 = math.log1p(g / k) / g
        c2 = math.sqrt(sig2 * g / (2.0 * k * a_rate) * (1.0 + g / k) ** (1.0 + k / g))
    shift_scale = (2.0 * cfg.signal_accuracy - 1.0) * sig * sqdt * _SQRT_2_OVER_PI

    s = np.full(n_paths, params.s0)
    hist = [s.copy() for _ in range(cfg.latency_steps + 1)]
    q = np.zeros(n_paths)
    cash = np.zeros(n_paths)
    acc = {name: np.zeros(n_paths) for name in MM_STATS}
    sum_q2 = np.zeros(n_paths)

    for i in range(n_steps):
        t = i * dt
        z = rng.standard_normal(n_paths)
        u_sig = rng.random(n_paths)
        u_ask = rng.random(n_paths)
        u_bid = rng.random(n_paths)
        u_inf = rng.random(n_paths)
        sz = np.sign(z)
        pred = np.where(u_sig < cfg.signal_accuracy, sz, -sz)
        fair = hist[0] + shift_scale * pred
        if cfg.agent == "as":
            r = fair - q * g * sig2 * (big_t - t)
            half = 0.5 * (g * sig2 * (big_t - t) + markup)
            bid, ask = r - half, r + half
        elif cfg.agent == "glft":
            bid = fair - (c1 + (2.0 * q + 1.0) / 2.0 * c2)
            ask = fair + (c1 - (2.0 * q - 1.0) / 2.0 * c2)
        else:
            bid, ask = fair - fixed_half, fair + fixed_half
        if cfg.tick > 0.0:
            bid = np.floor(bid / cfg.tick + 1e-9) * cfg.tick
            ask = np.ceil(ask / cfg.tick - 1e-9) * cfg.tick
        can_ask = q > -qmax
        can_bid = q < qmax
        p_ask = -np.expm1(-a_rate * np.exp(-k * (ask - s)) * dt)
        p_bid = -np.expm1(-a_rate * np.exp(-k * (s - bid)) * dt)
        fill_ask = can_ask & (u_ask < p_ask)
        fill_bid = can_bid & (u_bid < p_bid)
        move = sig * sqdt * z
        informed = u_inf < cfg.informed_prob
        inf_ask = informed & (z > 0.0) & can_ask & ~fill_ask & (ask < s + move)
        inf_bid = informed & (z < 0.0) & can_bid & ~fill_bid & (bid > s + move)
        fa = fill_ask | inf_ask
        fb = fill_bid | inf_bid
        fa_f, fb_f = fa.astype(float), fb.astype(float)
        n_fill = fa_f + fb_f
        q = q + fb_f - fa_f
        cash = cash + np.where(fa, ask, 0.0) - np.where(fb, bid, 0.0) + cfg.maker_fee * n_fill
        acc["spread"] += np.where(fa, ask - s, 0.0) + np.where(fb, s - bid, 0.0)
        acc["fees"] += cfg.maker_fee * n_fill
        acc["fills"] += n_fill
        acc["informed_fills"] += inf_ask.astype(float) + inf_bid.astype(float)
        acc["inventory"] += q * move
        acc["markout"] += (fb_f - fa_f) * move
        sum_q2 += q * q
        s = s + move
        hist = [*hist[1:], s.copy()]

    liq = params.eta_liq * np.abs(q)
    pnl = cash + q * s - liq
    acc["liquidation"] = liq
    acc["pnl"] = pnl
    acc["rms_q"] = np.sqrt(sum_q2 / n_steps)
    recon = acc["spread"] + acc["fees"] + acc["inventory"] - liq
    scale = np.maximum(1.0, np.abs(pnl))
    err = float(np.max(np.abs(recon - pnl) / scale))
    return MMResult(paths=acc, max_identity_error=err)
