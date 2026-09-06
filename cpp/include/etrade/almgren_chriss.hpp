/// \file almgren_chriss.hpp
/// \brief Almgren-Chriss discrete-time optimal execution (2000 paper).
///
/// Sell \c X shares over \c N intervals of length \c tau (horizon
/// `T = N*tau`, times `t_j = j*tau`).  Holdings `x_0 = X, ..., x_N = 0`;
/// trade list `n_j = x_{j-1} - x_j`.  The mid price follows an arithmetic
/// random walk with linear permanent impact `g(v) = gamma_p*v`; the trade in
/// interval j executes at `S_{j-1} - eta*n_j/tau` (linear temporary impact
/// `h(v) = eta*v`).  Implementation shortfall `IS = X*S_0 - sum n_j*S~_j`.
///
/// Exact discrete moments for any fully liquidating schedule (pinned sums):
///
///     E[IS]   = 0.5*gamma_p*X^2 + (eta_tilde/tau)*sum n_j^2
///     Var[IS] = sigma^2 * tau * sum_{j=1..N} x_j^2
///
/// with `eta_tilde = eta - gamma_p*tau/2` (must be > 0 — paper footnote).
/// The sums coincide with the paper's hyperbolic closed forms (with the
/// discrete kappa) to machine precision; they are used because they also
/// cover TWAP/POV/arbitrary schedules.  The paper's fixed-cost term
/// `epsilon*sum|n_j|` is omitted (schedule-independent for a one-sided
/// program).  The mean-variance-optimal trajectory is the sinh solution
///
///     x_j = X * sinh(kappa*(T - t_j)) / sinh(kappa*T),
///     kappa = arccosh(1 + kappa_tilde^2*tau^2/2) / tau,
///     kappa_tilde^2 = lambda_ra*sigma^2/eta_tilde,
///
/// evaluated (pinned, API_SPEC 1.4) as `kappa = log1p(x + sqrt(x*(2+x)))/tau`
/// with `x = kappa_tilde^2*tau^2/2` (no cancellation for small x) and
/// `x_j = X*exp(-kappa*t_j)*expm1(-2*kappa*(T-t_j))/expm1(-2*kappa*T)` (no
/// sinh overflow for large kappa*T), degenerating to the exact linear TWAP
/// schedule `x_j = X*(N-j)/N` when `kappa*T < kKappaTLimit` (covers
/// `lambda_ra == 0` and `sigma == 0` with no 0/0).  Prices may go negative
/// under the arithmetic walk — a documented model property, never an error.
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace etrade {

/// Threshold on kappa*T below which the exact linear (TWAP) branch is used.
inline constexpr double kKappaTLimit = 1e-6;

/// Upper bound on the number of AC trading intervals N (pinned, API_SPEC 0).
inline constexpr int kMaxAcIntervals = 10'000'000;

/// Upper bound on the number of POV intervals M = ceil(X/clip) (pinned).
inline constexpr int kMaxPovIntervals = 10'000'000;

/// Upper bound on Monte Carlo path counts (pinned, API_SPEC 0).
inline constexpr int kMaxPaths = 10'000'000;

/// Parameters of the discrete Almgren-Chriss problem (sell \c X shares).
///
/// The constructor validates every field (std::invalid_argument) and the
/// paper's model-validity condition `gamma_p*tau < 2*eta`
/// (std::domain_error on violation, including equality).
struct ACParams {
    double total_shares;  ///< X >= 0, shares to sell (0 = empty trajectory).
    int n_intervals;      ///< N in [1, kMaxAcIntervals], number of trading intervals.
    double tau;           ///< Interval length (> 0).
    double sigma;         ///< Arithmetic vol per sqrt(time unit) (>= 0).
    double eta;           ///< Linear temporary impact coefficient (> 0).
    double gamma_p;       ///< Linear permanent impact coefficient (>= 0).
    double lambda_ra;     ///< Risk aversion (>= 0); 0 gives the TWAP limit.

    ACParams(double total_shares, int n_intervals, double tau, double sigma,
             double eta, double gamma_p, double lambda_ra);

    /// Total horizon T = N * tau.
    double horizon() const { return n_intervals * tau; }

    /// Adjusted temporary impact eta_tilde = eta*(1 - gamma_p*tau/(2*eta)).
    double eta_tilde() const { return eta * (1.0 - gamma_p * tau / (2.0 * eta)); }

    /// Copy of these params with a different risk aversion.
    ACParams with_lambda(double new_lambda_ra) const;
};

/// Discrete urgency parameter kappa: the root of
/// `(2/tau^2)*(cosh(kappa*tau) - 1) = kappa_tilde^2`; 0 when
/// `lambda_ra*sigma^2 == 0`.
double ac_kappa(const ACParams& params);

/// Characteristic decay time theta = 1/kappa (the paper's e-folding time,
/// not ln(2)/kappa); +infinity in the TWAP limit (kappa == 0) — reported as
/// infinity, not an error.
double half_life(const ACParams& params);

/// Alias of half_life() under the paper's name (theta = 1/kappa).
double characteristic_time(const ACParams& params);

/// Optimal holdings path x_0..x_N (length N+1, x_0 = X exactly, x_N = 0
/// exactly, every entry finite for any valid parameters).
std::vector<double> ac_trajectory(const ACParams& params);

/// Optimal trade list n_1..n_N (shares sold per interval, sums to X).
std::vector<double> ac_trade_list(const ACParams& params);

/// Trade list n_j = x_{j-1} - x_j from a finite holdings path (length >= 2).
std::vector<double> trades_from_holdings(const std::vector<double>& holdings);

/// Linear TWAP holdings path x_j = X*(N-j)/N (length N+1).  Requires finite
/// `total_shares >= 0` and `1 <= n_intervals <= kMaxAcIntervals`.
std::vector<double> twap_holdings(double total_shares, int n_intervals);

/// POV (percentage-of-volume) holdings path: sells
/// `participation*market_volume_rate*tau` shares per interval until done
/// (last clip partial, `x_M = 0` exactly); number of intervals
/// `M = ceil(X/clip - 1e-12)`.  Requires finite inputs,
/// `market_volume_rate > 0`, `participation` in the open interval (0, 1)
/// and `X/clip <= kMaxPovIntervals` (a clip too small for the order is an
/// input error, never a huge allocation).
std::vector<double> pov_holdings(double total_shares, double tau,
                                 double market_volume_rate, double participation);

/// E[IS] of a fully liquidating schedule through the exact discrete sum
/// `0.5*gamma_p*X^2 + (eta_tilde/tau)*sum n_j^2`.  The path must be finite,
/// of length >= 2, and satisfy `|x_N| <= 1e-9 * max(1, |x_0|)`.  The order
/// size X in the permanent-impact term is the schedule's `x_0`, not
/// `params.total_shares` (pinned).
double expected_cost(const ACParams& params, const std::vector<double>& holdings);

/// E[IS] of the optimal trajectory.
double expected_cost(const ACParams& params);

/// Var[IS] = sigma^2 * tau * sum_{j=1..N} x_j^2 (exact discrete sum).  The
/// path must be finite and of length >= 2.
double variance_cost(const ACParams& params, const std::vector<double>& holdings);

/// Var[IS] of the optimal trajectory.
double variance_cost(const ACParams& params);

/// (E, Var) of instantaneous liquidation (all X in the first interval):
/// `E = 0.5*gamma_p*X^2 + eta_tilde*X^2/tau`, `Var = 0`.
std::pair<double, double> instantaneous_cost(const ACParams& params);

/// One point on the efficient frontier of execution.
struct FrontierPoint {
    double lambda_ra;  ///< Risk aversion of this point.
    double kappa;      ///< Discrete urgency parameter.
    double e_cost;     ///< E[IS] of the optimal schedule.
    double var_cost;   ///< Var[IS] of the optimal schedule.
    double std_cost;   ///< sqrt(Var[IS]).
};

/// Sweep a risk-aversion grid (non-empty, each entry finite and >= 0; order
/// preserved, duplicates allowed) and return `(lambda, kappa, E, Var,
/// sqrt(Var))` per point.
std::vector<FrontierPoint> efficient_frontier(const ACParams& params,
                                              const std::vector<double>& lambdas);

/// Monte Carlo implementation-shortfall summary (arithmetic-walk sim).
struct ISSimResult {
    double mean;     ///< Sample mean of IS across paths.
    double var;      ///< Sample variance (ddof = 1).
    double se_mean;  ///< sqrt(var / n_paths).
    double se_var;   ///< var * sqrt(2/(n_paths - 1)) (normal approximation).
    int n_paths;     ///< Number of simulated paths.
};

/// Simulate IS of a schedule under the arithmetic random walk (S_0 = 0; IS
/// is independent of S_0).  Per path, loop j = 1..N: execute n_j at
/// `S_{j-1} - eta*n_j/tau`, then `S_j = S_{j-1} + sigma*sqrt(tau)*xi_j -
/// gamma_p*n_j`.  The schedule must fully liquidate (same check as
/// expected_cost) and `2 <= n_paths <= kMaxPaths`.  Uses std::mt19937_64
/// with std::normal_distribution (own stream — matches the Python closed
/// forms statistically, not bit-for-bit; deterministic per seed within one
/// standard-library implementation, since std::normal_distribution's
/// algorithm is implementation-defined).
ISSimResult simulate_is(const ACParams& params, const std::vector<double>& holdings,
                        int n_paths = 10000, std::uint64_t seed = 0);

/// Simulate IS of the optimal trajectory.
ISSimResult simulate_is(const ACParams& params, int n_paths = 10000,
                        std::uint64_t seed = 0);

}  // namespace etrade
