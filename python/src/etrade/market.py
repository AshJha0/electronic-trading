"""Market-unit layer: build model parameters from quantities a desk quotes.

The core modules take parameters in abstract model units (``eta`` in
currency x time / share^2, ``k`` in 1/price, ...).  Nobody on a desk thinks in
those units.  This module converts from the numbers people actually quote -
price, tick, quoted spread, daily volatility in percent, average daily volume
(ADV), how often a quote at the touch gets filled - and back to basis points
for reporting.  The conversions are pinned in API_SPEC.md Section 7.

Almgren-Chriss impact calibration
---------------------------------
Plain idea: impact is measured in spreads per fraction of daily volume.
Almgren and Chriss (2000, Section 4) calibrate their example this way:
trading 1% of ADV per day costs one quoted spread of temporary impact, and
trading 10% of ADV moves the price permanently by one spread.  Formally, with
quoted spread ``s`` and ADV ``V`` (shares per day, ``tau`` in days):

    sigma   = price * daily_vol
    eta     = s / (temp_frac * V)        temp_frac default 0.01
    gamma_p = s / (perm_frac * V)        perm_frac default 0.10

Worked example (the paper's own numbers): price 50, spread 1/8, ADV 5e6,
annual vol 30% (daily 0.95/50 = 1.9%) gives sigma = 0.95, eta = 2.5e-6,
gamma_p = 2.5e-7 - exactly the repository's ``params_liquid.json`` AC set.
The calibration is linear by construction; the square-root alternative is a
later release.

Avellaneda-Stoikov intensity and risk calibration
-------------------------------------------------
Plain idea: describe flow by how often a quote at the touch is hit, and how
quickly that rate falls as the quote moves away; describe risk aversion by
how many ticks the maker skews per unit of inventory at the start of the day.
Formally, with tick ``d``, touch distance ``d/2`` (one-tick market, quoting
at the best bid and offer), touch fill rate ``R`` per side per day, halving
distance ``h`` ticks and opening skew ``c`` ticks per unit:

    sigma   = price * daily_vol
    k       = ln(2) / (h * d)            intensity halves every h ticks
    A       = R * exp(k * d / 2)         so that lambda(d/2) = R
    gamma   = c * d / (sigma^2 * T)      so that gamma*sigma^2*T = c ticks
    eta_liq = liq_ticks * d
    dt      = dt_seconds / session_seconds   (T in trading days)

Worked example (``us_large_cap`` preset): price 100, tick 0.01, daily vol
1.5%, touch filled every 20 s per side (R = 1170 per day), halving 0.462
ticks, skew 2.25 ticks give k ~ 150 per dollar, A ~ 2477 per day and
gamma = 0.01; the opening A-S spread is then about 3.6 bps, against roughly
130-170 bps for the paper-replication set.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .almgren_chriss import ACParams, twap_holdings
from .avellaneda_stoikov import ASParams
from .impact import (
    Holdings,
    ImpactParams,
    PretradeReport,
    PretradeRow,
    impact_expected_cost,
    impact_variance_cost,
    optimal_schedule,
    sqrt_law_cost_bps,
    vwap_holdings,
)

#: Default fraction of ADV per day whose trading costs one spread of temporary impact.
DEFAULT_TEMP_FRAC: float = 0.01

#: Default fraction of ADV whose trading moves the price permanently by one spread.
DEFAULT_PERM_FRAC: float = 0.10

#: Default length of a trading session in seconds (US equities, 6.5 hours).
DEFAULT_SESSION_SECONDS: float = 23_400.0


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _div(a: float, b: float) -> float:
    """IEEE division (inf for a positive over an underflowed 0), as in the
    other ports, so the parameter constructor reports the bad value instead
    of Python raising ZeroDivisionError."""
    return a / b if b != 0.0 else math.inf


def _exp(v: float) -> float:
    """IEEE exp (inf on overflow instead of OverflowError)."""
    try:
        return math.exp(v)
    except OverflowError:
        return math.inf


def _positive(name: str, v: float) -> None:
    _require(math.isfinite(v) and v > 0.0, f"{name} must be finite and > 0")


def ac_params_from_market(
    *,
    price: float,
    spread: float,
    adv: float,
    daily_vol: float,
    total_shares: float,
    n_intervals: int,
    tau: float,
    lambda_ra: float,
    temp_frac: float = DEFAULT_TEMP_FRAC,
    perm_frac: float = DEFAULT_PERM_FRAC,
) -> ACParams:
    """Almgren-Chriss parameters from market quantities (API_SPEC 7.1).

    ``spread`` is the full quoted spread in currency, ``adv`` shares per day,
    ``daily_vol`` a fraction (0.015 for 1.5%), ``tau`` in trading days.
    """
    _positive("price", price)
    _positive("spread", spread)
    _positive("adv", adv)
    _positive("daily_vol", daily_vol)
    _require(math.isfinite(temp_frac) and 0.0 < temp_frac <= 1.0, "temp_frac must be in (0, 1]")
    _require(math.isfinite(perm_frac) and 0.0 < perm_frac <= 1.0, "perm_frac must be in (0, 1]")
    return ACParams(
        total_shares=total_shares,
        n_intervals=n_intervals,
        tau=tau,
        sigma=price * daily_vol,
        eta=_div(spread, temp_frac * adv),
        gamma_p=_div(spread, perm_frac * adv),
        lambda_ra=lambda_ra,
    )


def as_params_from_market(
    *,
    price: float,
    tick: float,
    daily_vol: float,
    touch_fills_per_day: float,
    halving_ticks: float,
    skew_ticks: float,
    q_max: int,
    liq_ticks: float,
    dt_seconds: float,
    session_seconds: float = DEFAULT_SESSION_SECONDS,
    horizon_days: float = 1.0,
) -> ASParams:
    """Avellaneda-Stoikov parameters from market quantities (API_SPEC 7.2)."""
    _positive("price", price)
    _positive("tick", tick)
    _positive("daily_vol", daily_vol)
    _positive("touch_fills_per_day", touch_fills_per_day)
    _positive("halving_ticks", halving_ticks)
    _positive("skew_ticks", skew_ticks)
    _require(math.isfinite(liq_ticks) and liq_ticks >= 0.0, "liq_ticks must be finite and >= 0")
    _positive("dt_seconds", dt_seconds)
    _positive("session_seconds", session_seconds)
    _positive("horizon_days", horizon_days)
    sigma = price * daily_vol
    k = _div(math.log(2.0), halving_ticks * tick)
    return ASParams(
        s0=price,
        sigma=sigma,
        horizon=horizon_days,
        dt=_div(dt_seconds, session_seconds),
        gamma=_div(skew_ticks * tick, sigma * sigma * horizon_days),
        k=k,
        a_rate=touch_fills_per_day * _exp(k * tick / 2.0),
        q_max=q_max,
        eta_liq=liq_ticks * tick,
    )


def bps(amount: float, notional: float) -> float:
    """``amount`` as basis points of ``notional`` (1e4 * amount / notional)."""
    _require(math.isfinite(amount), "amount must be finite")
    _positive("notional", notional)
    return 1e4 * amount / notional


def half_spread_cost(spread: float, total_shares: float) -> float:
    """Cost of crossing half the quoted spread on every share: ``spread*X/2``.

    This is the paper's fixed cost ``epsilon * X`` with ``epsilon = spread/2``,
    which the core cost functions omit (API_SPEC 1.1); add it to ``E[IS]``
    when reporting total expected cost.
    """
    _positive("spread", spread)
    _require(
        math.isfinite(total_shares) and total_shares >= 0.0, "total_shares must be finite and >= 0"
    )
    return 0.5 * spread * total_shares


# ------------------------------------------------------------- power-law impact


def impact_params_from_market(
    *,
    price: float,
    spread: float,
    adv: float,
    daily_vol: float,
    total_shares: float,
    n_intervals: int,
    tau: float,
    lambda_ra: float,
    alpha: float = 0.5,
    y: float = 1.0,
    perm_frac: float = DEFAULT_PERM_FRAC,
    drift_bps_per_day: float = 0.0,
    weights: Sequence[float] | None = None,
) -> ImpactParams:
    """Power-law impact parameters from market quantities (API_SPEC 8.3).

    ``eta = y * sigma / adv**alpha`` so that trading at participation ``p``
    costs ``y * sigma * p**alpha`` per share; ``gamma_p`` as in 7.1;
    ``drift = price * drift_bps_per_day * 1e-4`` per share per day;
    ``weights`` are fractions of daily volume per interval (default ``tau``).
    """
    _positive("price", price)
    _positive("spread", spread)
    _positive("adv", adv)
    _positive("daily_vol", daily_vol)
    _positive("y", y)
    _require(math.isfinite(perm_frac) and 0.0 < perm_frac <= 1.0, "perm_frac must be in (0, 1]")
    _require(
        math.isfinite(drift_bps_per_day) and drift_bps_per_day >= 0.0,
        "drift_bps_per_day must be finite and >= 0",
    )
    _require(math.isfinite(alpha) and 0.0 < alpha <= 1.0, "alpha must be in (0, 1]")
    sigma = price * daily_vol
    return ImpactParams(
        total_shares=total_shares,
        n_intervals=n_intervals,
        tau=tau,
        sigma=sigma,
        eta=_div(y * sigma, adv**alpha),
        alpha=alpha,
        gamma_p=_div(spread, perm_frac * adv),
        lambda_ra=lambda_ra,
        drift=price * drift_bps_per_day * 1e-4,
        weights=None if weights is None else tuple(weights),
    )


def pretrade_report(
    *,
    price: float,
    spread: float,
    adv: float,
    daily_vol: float,
    total_shares: float,
    n_intervals: int,
    tau: float,
    lambdas: Sequence[float],
    alpha: float = 0.5,
    y: float = 1.0,
    perm_frac: float = DEFAULT_PERM_FRAC,
    drift_bps_per_day: float = 0.0,
    weights: Sequence[float] | None = None,
) -> PretradeReport:
    """Pre-trade cost report (API_SPEC 8.4): TWAP, VWAP and the optimal
    schedule at each ``lambda_ra`` in ``lambdas``, all costed on the same
    power-law model, plus the closed-form square-root-law estimate."""
    _require(len(lambdas) > 0, "lambdas must be non-empty")
    base = {
        "price": price,
        "spread": spread,
        "adv": adv,
        "daily_vol": daily_vol,
        "total_shares": total_shares,
        "n_intervals": n_intervals,
        "tau": tau,
        "alpha": alpha,
        "y": y,
        "perm_frac": perm_frac,
        "drift_bps_per_day": drift_bps_per_day,
        "weights": weights,
    }
    p0 = impact_params_from_market(lambda_ra=0.0, **base)  # type: ignore[arg-type]
    notional = total_shares * price
    _positive("total_shares * price", notional)
    hs = bps(half_spread_cost(spread, total_shares), notional)

    def row(label: str, lam: float, x: Holdings, horizon: int, p: ImpactParams) -> PretradeRow:
        e = bps(impact_expected_cost(p, x), notional)
        sd = bps(math.sqrt(impact_variance_cost(p, x)), notional)
        return PretradeRow(label, lam, e, sd, e + hs, horizon)

    rows = [
        row("twap", 0.0, twap_holdings(total_shares, n_intervals), n_intervals, p0),
        row("vwap", 0.0, vwap_holdings(total_shares, p0.weights or ()), n_intervals, p0),
    ]
    for lam in lambdas:
        _require(math.isfinite(lam) and lam >= 0.0, "lambdas must be finite and >= 0")
        p = impact_params_from_market(lambda_ra=float(lam), **base)  # type: ignore[arg-type]
        s = optimal_schedule(p)
        rows.append(row("optimal", float(lam), s.holdings, s.horizon, p))
    return PretradeReport(sqrt_law_cost_bps(daily_vol, total_shares, adv, y), hs, tuple(rows))
