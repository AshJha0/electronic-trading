/// \file analytics.hpp
/// \brief Execution analytics: implementation-shortfall decomposition
/// (Perold convention), participation rate, and a tiny native OLS for
/// market-impact estimation.
///
/// IS decomposition for a parent order of X shares with decision price P_d,
/// arrival price P_0, close price P_T, fills (n_j, p_j), executed
/// `Q = sum n_j`, remainder `R = X - Q`.  All components are currency costs
/// (positive = money lost vs the paper portfolio).  For a sell
/// (`sign = +1`; a buy mirrors with `sign = -1`):
///
///     delay       = Q * sign*(P_d - P_0)
///     trading     = sum_j n_j * sign*(P_0 - p_j)
///     opportunity = R * sign*(P_d - P_T)
///     total       = delay + trading + opportunity
///                 = sign*(X*P_d - sum n_j*p_j - R*P_T)   (exact identity)
///
/// The impact regression fits per-share cost against participation rate with
/// a closed-form OLS (no fitting library).
#pragma once

#include <utility>
#include <vector>

namespace etrade {

/// Side of a parent order.
enum class Side {
    kBuy,   ///< Buy order: cost = paid above the decision price.
    kSell,  ///< Sell order: cost = received below the decision price.
};

/// Implementation-shortfall components (currency costs; positive = loss).
struct ISDecomposition {
    double total;          ///< delay + trading + opportunity (exact identity).
    double delay;          ///< Drift cost before trading started.
    double trading;        ///< Execution slippage of the fills vs arrival.
    double opportunity;    ///< Cost of the unfilled remainder vs decision.
    double executed_qty;   ///< Q = sum of fill quantities.
    double remaining_qty;  ///< R = X - Q.
};

/// Decompose implementation shortfall into delay/trading/opportunity.
/// Validates finiteness, `total_qty >= 0`, equal-length non-negative fills
/// and `Q <= X` (std::invalid_argument on violation).
ISDecomposition decompose_shortfall(Side side, double decision_price,
                                    double arrival_price, double close_price,
                                    double total_qty,
                                    const std::vector<double>& fill_qtys,
                                    const std::vector<double>& fill_prices);

/// Executed volume as a fraction of market volume over the same window.
/// Requires `executed_volume >= 0` and `market_volume > 0`.
double participation_rate(double executed_volume, double market_volume);

/// Simple linear regression y = intercept + slope * x.
struct OLSResult {
    double slope;      ///< Sxy / Sxx.
    double intercept;  ///< mean(y) - slope * mean(x).
    double r_squared;  ///< 1 - SSR/SST (1.0 when SST == 0).
    int n_obs;         ///< Number of observations.
};

/// Tiny native OLS in closed form.  Requires equal lengths, >= 2 finite
/// observations and non-degenerate x (Sxx > 0).
OLSResult ols(const std::vector<double>& x, const std::vector<double>& y);

/// Deterministic synthetic (participation, per-share cost) fills — pinned,
/// RNG-free, identical across languages:
///
///     part_i = 0.02 + 0.18*i/(n_obs - 1),           i = 0..n_obs-1
///     cost_i = intercept + slope*part_i + noise_amp*sin(2.5*i + 0.3)
///
/// Requires `n_obs >= 2`.  Returns (participations, costs).
std::pair<std::vector<double>, std::vector<double>> synthetic_impact_fills(
    int n_obs = 25, double intercept = 0.5, double slope = 8.0,
    double noise_amp = 0.05);

/// Market-impact regression: per-share cost on participation rate.
OLSResult estimate_impact(const std::vector<double>& participations,
                          const std::vector<double>& costs);

}  // namespace etrade
