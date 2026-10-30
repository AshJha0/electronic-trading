/// \file execution.hpp
/// \brief Execution extensions: transient (Obizhaeva-Wang) impact, adaptive
/// re-planning with a short-term signal and fill risk, and internalisation of
/// opposite parent orders (API_SPEC Section 12).
///
/// Transient impact: book displacement `D_{j+1} = a*(D_j + kappa*n_j)` with
/// `a = exp(-rho*tau)`; a trade pays `n*D_j + kappa*n^2/2`, permanent impact
/// is booked as in Section 1.  The static optimum is one tridiagonal solve.
/// Adaptive: Almgren-Chriss dynamics plus an AR(1) drift signal and random
/// partial fills; the adaptive trader re-solves every interval.
/// Internalisation: cross at the mid, send the net residual, cost each side
/// with the Section 8 model.
#pragma once

#include "etrade/almgren_chriss.hpp"
#include "etrade/market.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace etrade {

/// Upper bound on intervals for every Section 12 entry point (pinned).
inline constexpr int kMaxExecIntervals = 1000;
/// Upper bound on the number of parent orders per side in internalise.
inline constexpr int kMaxOrders = 10000;

/// Transient-impact problem (API_SPEC 12.1); validated on construction
/// (std::invalid_argument), including convexity `gamma_p < kappa*(1-a)/(1+a)`.
struct TransientParams {
    double total_shares;  ///< X >= 0.
    int n_intervals;      ///< N in [1, kMaxExecIntervals].
    double tau;           ///< Interval length > 0.
    double kappa;         ///< Displacement per share (inverse depth) > 0.
    double rho;           ///< Resilience rate, finite and > 0.
    double gamma_p;       ///< Permanent impact >= 0.

    TransientParams(double total_shares, int n_intervals, double tau, double kappa, double rho,
                    double gamma_p = 0.0);

    /// Per-interval displacement decay a = exp(-rho*tau).
    [[nodiscard]] double decay() const;
};

/// E[IS] of any fully liquidating schedule (length N+1) by the displacement recursion.
[[nodiscard]] double transient_expected_cost(const TransientParams& p,
                                             const std::vector<double>& holdings);

/// Static schedule minimising E[IS] (holdings, length N+1, x_0 = X, x_N = 0).
[[nodiscard]] std::vector<double> transient_optimal_schedule(const TransientParams& p);

/// Observable AR(1) drift signal and fill risk (API_SPEC 12.2); validated on construction.
struct SignalParams {
    double phi;        ///< AR(1) coefficient in [0, 1).
    double signal_sd;  ///< Stationary sd of the signal >= 0.
    double fill_risk;  ///< Max unfilled fraction in [0, 1).

    explicit SignalParams(double phi = 0.0, double signal_sd = 0.0, double fill_risk = 0.0);
};

/// Optimal remaining holdings [x_k, .., x_N = 0] from interval k (0-based).
[[nodiscard]] std::vector<double> replan(const ACParams& p, const SignalParams& sig, double holding,
                                         double signal, int k);

/// Feedback form of replan: with m intervals left the next child order is
/// `c[m]*holding - d[m]*signal` (index 0 unused).  Returns {c, d}.
[[nodiscard]] std::pair<std::vector<double>, std::vector<double>> replan_coefficients(
    const ACParams& p, const SignalParams& sig);

/// Static vs adaptive implementation shortfall; saving = static - adaptive
/// (paired), SEs are sample std (ddof=1) / sqrt(M).
struct AdaptiveSimResult {
    double static_mean = 0.0;
    double static_se = 0.0;
    double static_std = 0.0;
    double adaptive_mean = 0.0;
    double adaptive_se = 0.0;
    double adaptive_std = 0.0;
    double saving_mean = 0.0;
    double saving_se = 0.0;
    int n_paths = 0;
};

/// Monte Carlo of static vs adaptive execution (API_SPEC 12.2).  Uses
/// std::mt19937_64(seed); paths outer, intervals inner, draws u, xi, eps.
[[nodiscard]] AdaptiveSimResult simulate_adaptive(const ACParams& p, const SignalParams& sig,
                                                  int n_paths = 10000, std::uint64_t seed = 0);

/// Crossing of opposite parent orders (API_SPEC 12.3).
struct InternalisationResult {
    double crossed_shares = 0.0;
    double residual_shares = 0.0;
    std::string residual_side;  ///< "buy", "sell" or "none".
    double baseline_cost = 0.0;
    double internal_cost = 0.0;
    double saved_cost = 0.0;
    double saved_bps = 0.0;  ///< bps of (B + S) * price.
    std::vector<double> buy_saved_bps;
    std::vector<double> sell_saved_bps;
};

/// Cross buys and sells at the mid, send the residual to the market and
/// report the Section 8 impact cost saved.  `weights` empty = uniform.
[[nodiscard]] InternalisationResult internalise(const std::vector<double>& buys,
                                                const std::vector<double>& sells, double price,
                                                double spread, double adv, double daily_vol,
                                                int n_intervals, double tau, double alpha = 0.5,
                                                double y = 1.0,
                                                double perm_frac = kDefaultPermFrac,
                                                const std::vector<double>& weights = {});

}  // namespace etrade
