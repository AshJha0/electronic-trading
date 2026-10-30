"""Calibration estimators: market quantities measured from recorded data.

The market-unit layer (API_SPEC Section 7) and the power-law impact model
(Section 8) take inputs a desk quotes: daily volatility, quoted spread, ADV,
an intraday volume curve, the rate at which a resting quote is filled.  This
module measures those quantities from plain arrays of timestamps, prices and
sizes, so it needs no data access and every port computes the same numbers.
The estimators are pinned in API_SPEC.md Section 11; the tooling under
``data/calibration/`` turns one recorded NASDAQ ITCH day into these arrays.

Binning rule (shared by the curves)
-----------------------------------
A session ``[t0, t1]`` is split into ``n`` equal bins.  An observation at
time ``t`` with ``t0 <= t <= t1`` falls in bin
``j = min(floor((t - t0) * n / (t1 - t0)), n - 1)``; observations outside
the session are ignored.  The closing instant belongs to the last bin.

Volume curve
------------
Fraction of session volume traded in each bin: ``w_j = V_j / sum(V)``.
The result is a weight vector for ``vwap_holdings`` and the Section 8
liquidity weights.

Volatility curve
----------------
Realised variance (Andersen, Bollerslev, Diebold and Labys 2001): the sum of
squared log returns of the mid between consecutive observations.  A return
from ``t_{i-1}`` to ``t_i`` counts when both instants are inside the session
and is assigned to the bin of ``t_i``.  ``bin_vol_j = sqrt(RV_j)`` and
``daily_vol = sqrt(sum_j RV_j)`` (a fraction, the session's open-to-close
volatility; the overnight return is not included).

Spread statistics
-----------------
Mean and median of a sample of spreads, in ticks and in basis points of the
contemporaneous mid.  The same estimator serves the quoted spread
(``ask - bid`` sampled on a clock) and the effective spread
(``2 |p - m|`` per trade, the SEC Rule 605 definition).

Fill intensity
--------------
Avellaneda and Stoikov (2008) model the rate at which a quote at distance
``delta`` from the mid is hit as ``lambda(delta) = A exp(-k delta)``.  Given
the distance each market order reached from the prevailing mid, the rate of
orders reaching at least ``delta_b = (b + 1/2) tick`` is counted for
``b = 0 .. n_levels - 1`` and ``ln(rate_b)`` is regressed on ``delta_b`` by
weighted least squares over buckets with a non-zero count ``c_b``, with
weight ``c_b`` (for Poisson counts ``Var(ln c_b) ~ 1/c_b``), so a sparse
tail bucket cannot outweigh the touch.  The fit is
reported in model units (``a_rate`` per day, ``k`` per unit of price) and in
the Section 7.2 market units (``touch_fills_per_day = lambda(tick/2)``,
``halving_ticks = ln 2 / (k tick)``), so ``as_params_from_market`` inverts it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

#: Largest number of bins accepted by the curve estimators.
MAX_CALIBRATION_BINS: int = 10_000

#: Largest number of distance buckets accepted by ``fill_intensity``.
MAX_FILL_LEVELS: int = 1_000

#: Slack, in ticks, when comparing a distance with a bucket edge, so that a
#: touch print at exactly half a tick is not lost to rounding.
FILL_EDGE_SLACK: float = 1e-9

#: Volume curve fitted to the NASDAQ TotalView-ITCH sample day 2019-12-30:
#: thirteen 30-minute fractions of continuous-session NASDAQ volume, the
#: equal-weighted mean of AAPL, MSFT and QQQ, rounded to 1e-4 (API_SPEC 11.6).
FITTED_VOLUME_CURVE_13: tuple[float, ...] = (
    0.1987,
    0.1247,
    0.0742,
    0.0680,
    0.0589,
    0.0604,
    0.0414,
    0.0430,
    0.0484,
    0.0466,
    0.0376,
    0.0535,
    0.1446,
)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _session(session_start: float, session_end: float, n_bins: int) -> None:
    _require(math.isfinite(session_start), "session_start must be finite")
    _require(
        math.isfinite(session_end) and session_end > session_start,
        "session_end must be finite and > session_start",
    )
    _require(
        isinstance(n_bins, int)
        and not isinstance(n_bins, bool)
        and 1 <= n_bins <= MAX_CALIBRATION_BINS,
        f"n_bins must be an integer in [1, {MAX_CALIBRATION_BINS}]",
    )


def _bin(t: float, session_start: float, session_end: float, n_bins: int) -> int:
    j = math.floor((t - session_start) * n_bins / (session_end - session_start))
    return min(int(j), n_bins - 1)


def volume_curve(
    times: Sequence[float],
    sizes: Sequence[float],
    session_start: float,
    session_end: float,
    n_bins: int,
) -> list[float]:
    """Fraction of session volume per bin (API_SPEC 11.2); sums to 1."""
    _session(session_start, session_end, n_bins)
    _require(len(times) == len(sizes), "times and sizes must have the same length")
    vol = [0.0] * n_bins
    for t, s in zip(times, sizes, strict=True):
        _require(math.isfinite(t), "times must be finite")
        _require(math.isfinite(s) and s >= 0.0, "sizes must be finite and >= 0")
        if session_start <= t <= session_end:
            vol[_bin(t, session_start, session_end, n_bins)] += s
    total = 0.0
    for v in vol:
        total += v
    _require(total > 0.0, "no volume inside the session")
    return [v / total for v in vol]


@dataclass(frozen=True)
class VolatilityCurve:
    """Realised volatility of the session and of each bin (fractions)."""

    daily_vol: float
    bin_vol: list[float]
    n_returns: int


def volatility_curve(
    times: Sequence[float],
    mids: Sequence[float],
    session_start: float,
    session_end: float,
    n_bins: int,
) -> VolatilityCurve:
    """Realised volatility per bin and for the session (API_SPEC 11.3)."""
    _session(session_start, session_end, n_bins)
    _require(len(times) == len(mids), "times and mids must have the same length")
    rv = [0.0] * n_bins
    n_returns = 0
    for i, (t, m) in enumerate(zip(times, mids, strict=True)):
        _require(math.isfinite(t), "times must be finite")
        _require(math.isfinite(m) and m > 0.0, "mids must be finite and > 0")
        if i == 0:
            continue
        _require(t >= times[i - 1], "times must be non-decreasing")
        if times[i - 1] >= session_start and t <= session_end:
            r = math.log(m / mids[i - 1])
            rv[_bin(t, session_start, session_end, n_bins)] += r * r
            n_returns += 1
    _require(n_returns > 0, "no return inside the session")
    total = 0.0
    for v in rv:
        total += v
    return VolatilityCurve(
        daily_vol=math.sqrt(total), bin_vol=[math.sqrt(v) for v in rv], n_returns=n_returns
    )


@dataclass(frozen=True)
class SpreadStats:
    """Mean and median of a spread sample, in ticks and in bps of the mid."""

    mean_ticks: float
    median_ticks: float
    mean_bps: float
    median_bps: float
    n: int


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def spread_stats(spreads: Sequence[float], mids: Sequence[float], tick: float) -> SpreadStats:
    """Spread statistics of a sample (API_SPEC 11.4).

    ``spreads`` in currency (``ask - bid`` for the quoted spread,
    ``2 |p - m|`` for the effective spread), ``mids`` the mid of each
    observation.
    """
    _require(math.isfinite(tick) and tick > 0.0, "tick must be finite and > 0")
    _require(len(spreads) == len(mids), "spreads and mids must have the same length")
    _require(len(spreads) > 0, "spreads must not be empty")
    ticks: list[float] = []
    bps: list[float] = []
    for s, m in zip(spreads, mids, strict=True):
        _require(math.isfinite(s) and s >= 0.0, "spreads must be finite and >= 0")
        _require(math.isfinite(m) and m > 0.0, "mids must be finite and > 0")
        ticks.append(s / tick)
        bps.append(1e4 * s / m)
    sum_t = 0.0
    sum_b = 0.0
    for a, b in zip(ticks, bps, strict=True):
        sum_t += a
        sum_b += b
    n = len(ticks)
    return SpreadStats(
        mean_ticks=sum_t / n,
        median_ticks=_median(ticks),
        mean_bps=sum_b / n,
        median_bps=_median(bps),
        n=n,
    )


@dataclass(frozen=True)
class FillIntensity:
    """Fitted ``lambda(delta) = A exp(-k delta)``, in model and market units."""

    a_rate: float
    k: float
    touch_fills_per_day: float
    halving_ticks: float
    n_buckets: int


def fill_intensity(
    distances: Sequence[float],
    tick: float,
    n_levels: int,
    days: float = 1.0,
    sides: float = 2.0,
) -> FillIntensity:
    """Fit the exponential fill intensity (API_SPEC 11.5).

    ``distances`` holds, for each market order, the largest distance from the
    prevailing mid at which it printed (currency, >= 0); ``days`` is the
    length of the sample in trading days and ``sides`` the number of quote
    sides the sample pools (2 for buys and sells together).
    """
    _require(math.isfinite(tick) and tick > 0.0, "tick must be finite and > 0")
    _require(
        isinstance(n_levels, int)
        and not isinstance(n_levels, bool)
        and 2 <= n_levels <= MAX_FILL_LEVELS,
        f"n_levels must be an integer in [2, {MAX_FILL_LEVELS}]",
    )
    _require(math.isfinite(days) and days > 0.0, "days must be finite and > 0")
    _require(math.isfinite(sides) and sides > 0.0, "sides must be finite and > 0")
    counts = [0] * n_levels
    for d in distances:
        _require(math.isfinite(d) and d >= 0.0, "distances must be finite and >= 0")
        x = d / tick
        for b in range(n_levels):
            if x >= b + 0.5 - FILL_EDGE_SLACK:
                counts[b] += 1
            else:
                break
    xs: list[float] = []
    ys: list[float] = []
    ws: list[float] = []
    for b in range(n_levels):
        if counts[b] > 0:
            xs.append((b + 0.5) * tick)
            ys.append(math.log(counts[b] / (sides * days)))
            ws.append(float(counts[b]))
    n = len(xs)
    _require(n >= 2, "fewer than two distance buckets have fills")
    w_sum = 0.0
    x_bar = 0.0
    y_bar = 0.0
    for x, y, w in zip(xs, ys, ws, strict=True):
        w_sum += w
        x_bar += w * x
        y_bar += w * y
    x_bar /= w_sum
    y_bar /= w_sum
    sxx = 0.0
    sxy = 0.0
    for x, y, w in zip(xs, ys, ws, strict=True):
        sxx += w * (x - x_bar) * (x - x_bar)
        sxy += w * (x - x_bar) * (y - y_bar)
    k = -sxy / sxx
    _require(k > 0.0, "fitted fill intensity does not decay with distance")
    a_rate = math.exp(y_bar + k * x_bar)
    return FillIntensity(
        a_rate=a_rate,
        k=k,
        touch_fills_per_day=a_rate * math.exp(-k * tick / 2.0),
        halving_ticks=math.log(2.0) / (k * tick),
        n_buckets=n,
    )
