/// \file market.hpp
/// \brief Market-unit layer: build model parameters from quoted market
/// quantities and report results in basis points (API_SPEC Section 7).
///
/// Almgren-Chriss impact follows the paper's own calibration: trading 1% of
/// ADV per day costs one spread of temporary impact, 10% of ADV moves the
/// price permanently by one spread.  Avellaneda-Stoikov flow is described by
/// the fill rate of a quote at the touch and the distance over which that
/// rate halves; risk aversion by the opening skew per unit of inventory.
#pragma once

#include "etrade/almgren_chriss.hpp"
#include "etrade/avellaneda_stoikov.hpp"
#include "etrade/impact.hpp"

#include <string>
#include <vector>

namespace etrade {

/// Default fraction of ADV per day that costs one spread of temporary impact.
inline constexpr double kDefaultTempFrac = 0.01;
/// Default fraction of ADV that moves the price permanently by one spread.
inline constexpr double kDefaultPermFrac = 0.10;
/// Default trading-session length in seconds (US equities, 6.5 hours).
inline constexpr double kDefaultSessionSeconds = 23400.0;

/// Almgren-Chriss inputs in market units (API_SPEC 7.1).
struct ACMarket {
    double price = 0.0;        ///< Price, > 0.
    double spread = 0.0;       ///< Full quoted spread in currency, > 0.
    double adv = 0.0;          ///< Average daily volume (shares per day), > 0.
    double daily_vol = 0.0;    ///< Daily volatility as a fraction (0.015 = 1.5%), > 0.
    double total_shares = 0.0; ///< Shares to sell, >= 0.
    int n_intervals = 1;       ///< Number of trading intervals.
    double tau = 1.0;          ///< Interval length in trading days.
    double lambda_ra = 0.0;    ///< Risk aversion (1/currency), >= 0.
    double temp_frac = kDefaultTempFrac;  ///< In (0, 1].
    double perm_frac = kDefaultPermFrac;  ///< In (0, 1].
};

/// Avellaneda-Stoikov inputs in market units (API_SPEC 7.2).
struct ASMarket {
    double price = 0.0;               ///< Price (initial mid), > 0.
    double tick = 0.0;                ///< Tick size, > 0.
    double daily_vol = 0.0;           ///< Daily volatility as a fraction, > 0.
    double touch_fills_per_day = 0.0; ///< Fills per side per day of a quote half a tick from mid, > 0.
    double halving_ticks = 0.0;       ///< Ticks over which the intensity halves, > 0.
    double skew_ticks = 0.0;          ///< Opening skew per unit of inventory in ticks, > 0.
    int q_max = 0;                    ///< Hard inventory cap, >= 0.
    double liq_ticks = 0.0;           ///< Terminal liquidation penalty in ticks, >= 0.
    double dt_seconds = 0.0;          ///< Step length in seconds, > 0.
    double session_seconds = kDefaultSessionSeconds;  ///< Session length in seconds, > 0.
    double horizon_days = 1.0;        ///< Horizon in trading days, > 0.
};

/// sigma = price*daily_vol, eta = spread/(temp_frac*adv),
/// gamma_p = spread/(perm_frac*adv).  Throws std::invalid_argument on bad
/// market inputs; the result is validated as any ACParams.
[[nodiscard]] ACParams ac_params_from_market(const ACMarket& m);

/// k = ln2/(halving_ticks*tick), A = touch_fills_per_day*exp(k*tick/2),
/// gamma = skew_ticks*tick/(sigma^2*horizon_days), eta_liq = liq_ticks*tick,
/// dt = dt_seconds/session_seconds.  Throws std::invalid_argument on bad
/// market inputs; the result is validated as any ASParams.
[[nodiscard]] ASParams as_params_from_market(const ASMarket& m);

/// Basis points of a notional: 1e4 * amount / notional (notional > 0).
[[nodiscard]] double bps(double amount, double notional);

/// Cost of crossing half the quoted spread on every share: 0.5*spread*X.
[[nodiscard]] double half_spread_cost(double spread, double total_shares);

/// Power-law impact inputs in market units (API_SPEC 8.3).
struct ImpactMarket {
    double price = 0.0;            ///< Price, > 0.
    double spread = 0.0;           ///< Full quoted spread, > 0.
    double adv = 0.0;              ///< Average daily volume, > 0.
    double daily_vol = 0.0;        ///< Daily volatility as a fraction, > 0.
    double total_shares = 0.0;     ///< Shares to sell, >= 0.
    int n_intervals = 1;           ///< Number of intervals.
    double tau = 1.0;              ///< Interval length in trading days.
    double lambda_ra = 0.0;        ///< Risk aversion, >= 0.
    double alpha = 0.5;            ///< Impact exponent in (0, 1].
    double y = 1.0;                ///< Square-root-law prefactor Y, > 0.
    double perm_frac = kDefaultPermFrac;  ///< In (0, 1].
    double drift_bps_per_day = 0.0;       ///< Expected adverse drift, >= 0.
    std::vector<double> weights;   ///< Volume fractions per interval (empty = tau each).
};

/// eta = y*sigma/adv^alpha, gamma_p = spread/(perm_frac*adv),
/// drift = price*drift_bps_per_day*1e-4 (API_SPEC 8.3).
[[nodiscard]] ImpactParams impact_params_from_market(const ImpactMarket& m);

/// One strategy in a pre-trade report (bps of X*price).
struct PretradeRow {
    std::string label;      ///< "twap", "vwap" or "optimal".
    double lambda_ra = 0.0;
    double impact_bps = 0.0;
    double sd_bps = 0.0;
    double total_bps = 0.0; ///< impact_bps + half_spread_bps.
    int horizon = 0;
};

/// Pre-trade report (API_SPEC 8.4).
struct PretradeReport {
    double sqrt_law_bps = 0.0;
    double half_spread_bps = 0.0;
    std::vector<PretradeRow> rows;
};

/// TWAP, VWAP and the optimal schedule per lambda, costed on the power-law
/// model; `m.lambda_ra` is ignored.
[[nodiscard]] PretradeReport pretrade_report(const ImpactMarket& m,
                                             const std::vector<double>& lambdas);

}  // namespace etrade
