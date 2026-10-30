/// \file impact.hpp
/// \brief Power-law impact execution: square-root costs, volume curves,
/// alpha-aware (drift) schedules and a pre-trade report (API_SPEC Section 8).
///
/// Temporary cost per interval is `eta_j * n^(1+alpha)` with
/// `eta_j = eta * w_j^(-alpha)` (`w_j` = fraction of daily volume in the
/// interval).  `alpha = 1` with uniform weights is exactly Almgren-Chriss;
/// `alpha = 1/2` is the square-root law.  The optimal schedule is found by a
/// pinned Newton method (tridiagonal Hessian, Armijo line search) with a
/// bisection over horizons when an adverse drift makes it optimal to finish
/// early, verified by a KKT check.
#pragma once

#include <array>
#include <vector>

namespace etrade {

/// Upper bound on intervals for the power-law solver (pinned).
inline constexpr int kMaxImpactIntervals = 10000;
/// Newton iteration cap per solve (pinned).
inline constexpr int kNewtonMaxIter = 100;
/// Convergence: Newton step below this times X in every coordinate (pinned).
inline constexpr double kNewtonStepTol = 1e-12;
/// Convergence: Newton decrement below this times max(1, |U|) (pinned).
inline constexpr double kNewtonDecrementTol = 1e-14;
/// Armijo sufficient-decrease constant (pinned).
inline constexpr double kArmijoC = 1e-4;
/// Smallest line-search step, 2^-40 (pinned).
inline constexpr double kMinStep = 9.094947017729282e-13;

/// Illustrative US-equity intraday volume curve: thirteen 30-minute fractions
/// of daily volume (U-shaped, sums to 1; not fitted to data).
inline constexpr std::array<double, 13> kUsEquityVolumeCurve13 = {
    0.12, 0.09, 0.075, 0.065, 0.06, 0.055, 0.05, 0.055, 0.06, 0.065, 0.075, 0.09, 0.14};

/// Power-law impact execution problem (API_SPEC 8.1); validated on construction.
class ImpactParams {
public:
    /// `weights` empty selects the default `tau` per interval.  Throws
    /// std::invalid_argument on invalid input, including the convexity
    /// condition `alpha*(1+alpha)*eta_j*X^(alpha-1) > gamma_p`.
    ImpactParams(double total_shares, int n_intervals, double tau, double sigma, double eta,
                 double alpha, double gamma_p, double lambda_ra, double drift = 0.0,
                 std::vector<double> weights = {});

    [[nodiscard]] double total_shares() const noexcept { return total_shares_; }
    [[nodiscard]] int n_intervals() const noexcept { return n_intervals_; }
    [[nodiscard]] double tau() const noexcept { return tau_; }
    [[nodiscard]] double sigma() const noexcept { return sigma_; }
    [[nodiscard]] double eta() const noexcept { return eta_; }
    [[nodiscard]] double alpha() const noexcept { return alpha_; }
    [[nodiscard]] double gamma_p() const noexcept { return gamma_p_; }
    [[nodiscard]] double lambda_ra() const noexcept { return lambda_ra_; }
    [[nodiscard]] double drift() const noexcept { return drift_; }
    [[nodiscard]] const std::vector<double>& weights() const noexcept { return weights_; }
    /// Per-interval coefficients `eta * w_j^(-alpha)`.
    [[nodiscard]] const std::vector<double>& etas() const noexcept { return etas_; }

private:
    double total_shares_, tau_, sigma_, eta_, alpha_, gamma_p_, lambda_ra_, drift_;
    int n_intervals_;
    std::vector<double> weights_, etas_;
};

/// E[IS] of a non-increasing, fully liquidating schedule of length N+1.
[[nodiscard]] double impact_expected_cost(const ImpactParams& p, const std::vector<double>& holdings);

/// Var[IS] = sigma^2 * tau * sum_{j>=1} x_j^2 (same path checks).
[[nodiscard]] double impact_variance_cost(const ImpactParams& p, const std::vector<double>& holdings);

/// Volume-weighted schedule: trade `X * w_j / sum(w)` per interval; `x_N = 0`.
[[nodiscard]] std::vector<double> vwap_holdings(double total_shares, const std::vector<double>& weights);

/// Result of optimal_schedule: holdings (N+1), horizon m used, Newton iterations.
struct ScheduleResult {
    std::vector<double> holdings;
    int horizon = 0;
    int iterations = 0;
};

/// Minimise E + lambda_ra*Var over non-increasing liquidating schedules
/// (API_SPEC 8.2).  Throws std::runtime_error if the solver fails.
[[nodiscard]] ScheduleResult optimal_schedule(const ImpactParams& p);

/// Square-root-law estimate `1e4 * y * daily_vol * sqrt(Q / adv)` in bps.
[[nodiscard]] double sqrt_law_cost_bps(double daily_vol, double order_shares, double adv,
                                       double y = 1.0);

}  // namespace etrade
