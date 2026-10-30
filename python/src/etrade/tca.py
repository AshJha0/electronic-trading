"""Post-trade transaction-cost analysis (TCA): benchmark slippage, markouts,
participation and an A/B test of two strategies.  Pinned in API_SPEC.md
Section 10.

Benchmark slippage
------------------
Plain idea: compare the average price an order got with a price it could
have had, and express the gap in basis points of that benchmark, positive
when the order did worse.  Formally, with fills ``(b_j, n_j, p_j)`` (bin,
quantity, price), executed quantity ``Q = sum n_j``, average price
``pbar = sum n_j p_j / Q`` and ``sign = +1`` for a sell, ``-1`` for a buy
(the Perold convention of :mod:`etrade.analytics`):

    cost_bps(B) = 1e4 * sign * (B - pbar) / B

for the benchmarks arrival price, close, the full-period market VWAP
``sum_b V_b P_b / sum_b V_b`` and the interval VWAP, the same sum taken only
over the bins in which the order traded (Kissell, *The Science of
Algorithmic Trading and Portfolio Management*, 2013, Ch. 3).

Worked example: a buy filled at an average of 100.10 against arrival 100.00
costs ``1e4 * (100.10 - 100.00) / 100.00 = 10`` bps; the same fills as a sell
would show -10 bps (a gain).

Markouts
--------
Plain idea: after each fill, watch where the mid goes.  A buy followed by a
falling mid was adverse selection.  Formally, with mid series ``m_t`` and
``d = +1`` for a buy, ``-1`` for a sell, the markout of fill ``j`` at
horizon ``h`` steps is ``d * (m_{t_j + h} - m_{t_j})`` (currency per share,
negative = adverse); the result reports the plain mean over fills and the
quantity-weighted mean per share at each horizon.

A/B comparison
--------------
Welch's unequal-variance t-test on two strategies' per-order costs (Welch,
1947), with the Student-t tail from the regularized incomplete beta function
by the modified Lentz continued fraction (Press et al., *Numerical Recipes*,
3rd ed., Sections 6.1 and 6.4) and a pinned Lanczos log-gamma, so all four
languages run one algorithm.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass

#: Iteration cap of the incomplete-beta continued fraction.
MAX_BETACF_ITER: int = 10_000

_BETACF_EPS = 1e-15
_FPMIN = 1e-300
_LANCZOS = (
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _is_int(v: object) -> bool:
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


def _sign(side: str) -> float:
    _require(side in ("buy", "sell"), "side must be 'buy' or 'sell'")
    return 1.0 if side == "sell" else -1.0


# ------------------------------------------------------------------ slippage


@dataclass(frozen=True)
class SlippageReport:
    """Average price, benchmarks and slippage in bps (positive = cost)."""

    executed_qty: float
    avg_price: float
    market_vwap: float
    interval_vwap: float
    arrival_bps: float
    vwap_bps: float
    interval_vwap_bps: float
    close_bps: float


def _check_fills(bins: Sequence[int], qtys: Sequence[float], n_bins: int, name: str) -> None:
    _require(len(qtys) > 0, "fills must be non-empty")
    _require(len(bins) == len(qtys), "fill arrays must have equal length")
    for b in bins:
        _require(_is_int(b) and 0 <= b < n_bins, f"{name} must be integers in [0, {n_bins})")
    for n in qtys:
        _require(math.isfinite(n) and n >= 0.0, "fill_qtys must be finite and >= 0")


def _check_volumes(bin_volumes: Sequence[float]) -> float:
    _require(len(bin_volumes) > 0, "bins must be non-empty")
    total = 0.0
    for v in bin_volumes:
        _require(math.isfinite(v) and v >= 0.0, "bin_volumes must be finite and >= 0")
        total += v
    _require(total > 0.0, "total market volume must be > 0")
    return total


def slippage(
    side: str,
    arrival_price: float,
    close_price: float,
    fill_bins: Sequence[int],
    fill_qtys: Sequence[float],
    fill_prices: Sequence[float],
    bin_volumes: Sequence[float],
    bin_vwaps: Sequence[float],
) -> SlippageReport:
    """Slippage of an executed order against arrival, VWAP, interval VWAP
    and close (API_SPEC 10.1).  ``bin_volumes``/``bin_vwaps`` describe the
    whole market (including the order's own fills) in each time bin."""
    sign = _sign(side)
    _require(
        math.isfinite(arrival_price) and arrival_price > 0.0,
        "arrival_price must be finite and > 0",
    )
    _require(math.isfinite(close_price) and close_price > 0.0, "close_price must be finite and > 0")
    _require(len(bin_vwaps) == len(bin_volumes), "bin arrays must have equal length")
    total_v = _check_volumes(bin_volumes)
    for p in bin_vwaps:
        _require(math.isfinite(p) and p > 0.0, "bin_vwaps must be finite and > 0")
    _check_fills(fill_bins, fill_qtys, len(bin_volumes), "fill_bins")
    _require(len(fill_prices) == len(fill_qtys), "fill arrays must have equal length")
    for p in fill_prices:
        _require(math.isfinite(p) and p > 0.0, "fill_prices must be finite and > 0")

    q = 0.0
    paid = 0.0
    traded = [False] * len(bin_volumes)
    for b, n, p in zip(fill_bins, fill_qtys, fill_prices, strict=True):
        q += n
        paid += n * p
        if n > 0.0:
            traded[b] = True
    _require(q > 0.0, "executed quantity must be > 0")
    avg = paid / q

    vp = 0.0
    iv = 0.0
    ivp = 0.0
    for b in range(len(bin_volumes)):
        vp += bin_volumes[b] * bin_vwaps[b]
        if traded[b]:
            iv += bin_volumes[b]
            ivp += bin_volumes[b] * bin_vwaps[b]
    _require(iv > 0.0, "market volume in traded bins must be > 0")
    vwap = vp / total_v
    ivwap = ivp / iv

    def cost(bench: float) -> float:
        return 1e4 * sign * (bench - avg) / bench

    return SlippageReport(
        q, avg, vwap, ivwap, cost(arrival_price), cost(vwap), cost(ivwap), cost(close_price)
    )


# ------------------------------------------------------------------ markouts


@dataclass(frozen=True)
class MarkoutResult:
    """Mean markout per fill and per share at each horizon (negative = adverse)."""

    horizons: tuple[int, ...]
    per_fill: tuple[float, ...]
    per_share: tuple[float, ...]


def markouts(
    side: str,
    fill_steps: Sequence[int],
    fill_qtys: Sequence[float],
    mids: Sequence[float],
    horizons: Sequence[int],
) -> MarkoutResult:
    """Markouts ``d * (m[t_j + h] - m[t_j])`` with ``d = +1`` buy, ``-1``
    sell, averaged per fill and per share (API_SPEC 10.2)."""
    d = -_sign(side)
    _require(len(mids) > 0, "mids must be non-empty")
    for m in mids:
        _require(math.isfinite(m), "mids must be finite")
    _check_fills(fill_steps, fill_qtys, len(mids), "fill_steps")
    _require(len(horizons) > 0, "horizons must be non-empty")
    for h in horizons:
        _require(_is_int(h) and h >= 0, "horizons must be integers >= 0")
        for t in fill_steps:
            _require(t + h < len(mids), "fill step + horizon exceeds mid series")
    q = 0.0
    for n in fill_qtys:
        q += n
    _require(q > 0.0, "executed quantity must be > 0")
    per_fill = []
    per_share = []
    for h in horizons:
        s = 0.0
        w = 0.0
        for t, n in zip(fill_steps, fill_qtys, strict=True):
            mk = d * (mids[t + h] - mids[t])
            s += mk
            w += n * mk
        per_fill.append(s / len(fill_steps))
        per_share.append(w / q)
    return MarkoutResult(tuple(int(h) for h in horizons), tuple(per_fill), tuple(per_share))


# ------------------------------------------------------------- participation


@dataclass(frozen=True)
class ParticipationResult:
    """Order volume as a fraction of market volume, per bin and overall."""

    per_bin: tuple[float, ...]
    overall: float


def participation(
    fill_bins: Sequence[int], fill_qtys: Sequence[float], bin_volumes: Sequence[float]
) -> ParticipationResult:
    """Per-bin ``q_b / V_b`` (0 for an empty bin) and overall ``Q / sum V``
    (API_SPEC 10.3).  Market volume includes the order's own fills, so
    ``q_b <= V_b`` is required."""
    total_v = _check_volumes(bin_volumes)
    _check_fills(fill_bins, fill_qtys, len(bin_volumes), "fill_bins")
    qb = [0.0] * len(bin_volumes)
    q = 0.0
    for b, n in zip(fill_bins, fill_qtys, strict=True):
        qb[b] += n
        q += n
    rates = []
    for b, v in enumerate(bin_volumes):
        _require(qb[b] <= v, "fill volume exceeds market volume in a bin")
        rates.append(qb[b] / v if v > 0.0 else 0.0)
    return ParticipationResult(tuple(rates), q / total_v)


# --------------------------------------------------------------- statistics


def ln_gamma(x: float) -> float:
    """Natural log of the gamma function, Lanczos ``g = 7``, nine terms
    (API_SPEC 10.4); reflection for ``x < 0.5``."""
    _require(math.isfinite(x) and x > 0.0, "x must be finite and > 0")
    if x < 0.5:
        return math.log(math.pi / abs(math.sin(math.pi * x))) - ln_gamma(1.0 - x)
    z = x - 1.0
    s = _LANCZOS[0]
    for i in range(1, 9):
        s += _LANCZOS[i] / (z + i)
    t = z + 7.5
    return 0.5 * math.log(2.0 * math.pi) + (z + 0.5) * math.log(t) - t + math.log(s)


def _betacf(a: float, b: float, x: float) -> float:
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAX_BETACF_ITER + 1):
        m2 = 2.0 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= _BETACF_EPS:
            return h
    raise ValueError("incomplete beta continued fraction did not converge")


def betainc_reg(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta ``I_x(a, b)`` (API_SPEC 10.4)."""
    _require(math.isfinite(a) and a > 0.0, "a must be finite and > 0")
    _require(math.isfinite(b) and b > 0.0, "b must be finite and > 0")
    _require(math.isfinite(x) and 0.0 <= x <= 1.0, "x must be in [0, 1]")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    ln_bt = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * math.log(x) + b * math.log1p(-x)
    bt = math.exp(ln_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(t: float, df: float) -> float:
    """Student-t CDF with ``df`` degrees of freedom (non-integer allowed)."""
    _require(math.isfinite(t), "t must be finite")
    _require(math.isfinite(df) and df > 0.0, "df must be finite and > 0")
    tail = 0.5 * betainc_reg(0.5 * df, 0.5, df / (df + t * t))
    return 1.0 - tail if t >= 0.0 else tail


@dataclass(frozen=True)
class WelchResult:
    """Welch two-sample t-test of ``mean(a) - mean(b)``."""

    mean_diff: float
    t_stat: float
    df: float
    p_value: float


def _mean_var(xs: Sequence[float], name: str) -> tuple[float, float]:
    _require(len(xs) >= 2, f"{name} must have at least 2 values")
    s = 0.0
    for v in xs:
        _require(math.isfinite(v), f"{name} must be finite")
        s += v
    mean = s / len(xs)
    ss = 0.0
    for v in xs:
        ss += (v - mean) * (v - mean)
    return mean, ss / (len(xs) - 1)


def welch_ttest(a: Sequence[float], b: Sequence[float]) -> WelchResult:
    """Welch's t-test on per-order costs of strategies A and B (API_SPEC 10.4).

    ``t = (mean_a - mean_b) / sqrt(va/na + vb/nb)``, Welch-Satterthwaite
    ``df`` and the two-sided ``p = I_{df/(df+t^2)}(df/2, 1/2)``.
    """
    ma, va = _mean_var(a, "a")
    mb, vb = _mean_var(b, "b")
    ea = va / len(a)
    eb = vb / len(b)
    se2 = ea + eb
    _require(se2 > 0.0, "samples have zero variance")
    diff = ma - mb
    t = diff / math.sqrt(se2)
    df = se2 * se2 / (ea * ea / (len(a) - 1) + eb * eb / (len(b) - 1))
    p = betainc_reg(0.5 * df, 0.5, df / (df + t * t))
    return WelchResult(diff, t, df, p)
