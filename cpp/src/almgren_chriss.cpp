#include "etrade/almgren_chriss.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace etrade {

namespace {

/// Throw std::invalid_argument unless `cond` holds.
void require(bool cond, const char* msg) {
    if (!cond) throw std::invalid_argument(msg);
}

/// Validate a user-supplied holdings path: length >= 2, all finite.
void require_holdings(const std::vector<double>& x) {
    require(x.size() >= 2, "holdings must have length >= 2");
    for (double v : x) require(std::isfinite(v), "holdings must be finite");
}

/// Pinned liquidation check |x_N| <= 1e-9 * max(1, |x_0|).
void require_liquidating(const std::vector<double>& x) {
    require(std::abs(x.back()) <= 1e-9 * std::max(1.0, std::abs(x.front())),
            "holdings path must fully liquidate (x_N = 0)");
}

}  // namespace

ACParams::ACParams(double total_shares_, int n_intervals_, double tau_,
                   double sigma_, double eta_, double gamma_p_, double lambda_ra_)
    : total_shares(total_shares_),
      n_intervals(n_intervals_),
      tau(tau_),
      sigma(sigma_),
      eta(eta_),
      gamma_p(gamma_p_),
      lambda_ra(lambda_ra_) {
    require(std::isfinite(total_shares) && total_shares >= 0.0,
            "total_shares must be finite and >= 0");
    require(n_intervals >= 1 && n_intervals <= kMaxAcIntervals,
            "n_intervals must be an integer in [1, 10000000]");
    require(std::isfinite(tau) && tau > 0.0, "tau must be > 0");
    require(std::isfinite(sigma) && sigma >= 0.0, "sigma must be >= 0");
    require(std::isfinite(eta) && eta > 0.0, "eta must be > 0");
    require(std::isfinite(gamma_p) && gamma_p >= 0.0, "gamma_p must be >= 0");
    require(std::isfinite(lambda_ra) && lambda_ra >= 0.0, "lambda_ra must be >= 0");
    // Paper validity condition: eta_tilde = eta - gamma_p*tau/2 must be > 0
    // (violation, including equality, invalidates the discrete model).
    if (!(gamma_p * tau < 2.0 * eta)) {
        throw std::domain_error(
            "model invalid: gamma_p*tau >= 2*eta (eta_tilde <= 0); "
            "see Almgren-Chriss (2000) footnote");
    }
}

ACParams ACParams::with_lambda(double new_lambda_ra) const {
    return ACParams(total_shares, n_intervals, tau, sigma, eta, gamma_p,
                    new_lambda_ra);
}

double ac_kappa(const ACParams& params) {
    const double kt2 = params.lambda_ra * params.sigma * params.sigma /
                       params.eta_tilde();
    if (kt2 == 0.0) return 0.0;
    // arccosh(1 + x) = log1p(x + sqrt(x*(2 + x))): free of cancellation for small x
    // where acosh(1 + x) cancels (pinned evaluation form, API_SPEC 1.4).
    const double x = 0.5 * kt2 * params.tau * params.tau;
    return std::log1p(x + std::sqrt(x * (2.0 + x))) / params.tau;
}

double half_life(const ACParams& params) {
    const double kappa = ac_kappa(params);
    return kappa == 0.0 ? std::numeric_limits<double>::infinity() : 1.0 / kappa;
}

double characteristic_time(const ACParams& params) { return half_life(params); }

std::vector<double> ac_trajectory(const ACParams& params) {
    const int n = params.n_intervals;
    const double x0 = params.total_shares;
    std::vector<double> x(static_cast<std::size_t>(n) + 1, 0.0);
    if (x0 == 0.0) return x;
    const double kappa = ac_kappa(params);
    const double big_t = params.horizon();
    if (kappa * big_t < kKappaTLimit) {
        // Exact linear TWAP limit — no 0/0 from the sinh ratio.
        for (int j = 0; j <= n; ++j) x[j] = x0 * (n - j) / n;
    } else {
        // sinh(k(T-t))/sinh(kT) = exp(-k t) * expm1(-2k(T-t)) / expm1(-2kT):
        // overflow-free for any kappa*T; ratio formed first so x_0 == X exactly.
        const double denom = std::expm1(-2.0 * kappa * big_t);
        for (int j = 0; j <= n; ++j) {
            const double t_j = j * params.tau;
            const double ratio = std::expm1(-2.0 * kappa * (big_t - t_j)) / denom;
            x[j] = x0 * std::exp(-kappa * t_j) * ratio;
        }
    }
    x[n] = 0.0;  // pinned: liquidate exactly, no residual epsilon
    return x;
}

std::vector<double> ac_trade_list(const ACParams& params) {
    return trades_from_holdings(ac_trajectory(params));
}

std::vector<double> trades_from_holdings(const std::vector<double>& holdings) {
    require_holdings(holdings);
    std::vector<double> n(holdings.size() - 1);
    for (std::size_t j = 0; j + 1 < holdings.size(); ++j) {
        n[j] = holdings[j] - holdings[j + 1];
    }
    return n;
}

std::vector<double> twap_holdings(double total_shares, int n_intervals) {
    require(std::isfinite(total_shares) && total_shares >= 0.0,
            "total_shares must be finite and >= 0");
    require(n_intervals >= 1 && n_intervals <= kMaxAcIntervals,
            "n_intervals must be an integer in [1, 10000000]");
    std::vector<double> x(static_cast<std::size_t>(n_intervals) + 1);
    for (int j = 0; j <= n_intervals; ++j) {
        x[j] = total_shares * (n_intervals - j) / n_intervals;
    }
    return x;
}

std::vector<double> pov_holdings(double total_shares, double tau,
                                 double market_volume_rate, double participation) {
    require(std::isfinite(total_shares) && total_shares >= 0.0,
            "total_shares must be finite and >= 0");
    require(std::isfinite(tau) && tau > 0.0, "tau must be > 0");
    require(std::isfinite(market_volume_rate) && market_volume_rate > 0.0,
            "market_volume_rate must be > 0");
    require(std::isfinite(participation) && participation > 0.0 &&
                participation < 1.0,
            "participation must be in (0, 1)");
    const double clip = participation * market_volume_rate * tau;
    if (total_shares == 0.0) return {0.0, 0.0};
    const double ratio = total_shares / clip;  // bounded BEFORE any cast/alloc
    require(ratio <= static_cast<double>(kMaxPovIntervals),
            "POV clip too small: would need > 10000000 intervals");
    const int m = static_cast<int>(std::ceil(ratio - 1e-12));
    std::vector<double> x(static_cast<std::size_t>(m) + 1);
    for (int i = 0; i <= m; ++i) {
        x[i] = std::max(total_shares - clip * i, 0.0);
    }
    x[m] = 0.0;  // last clip partial, liquidate exactly
    return x;
}

double expected_cost(const ACParams& params, const std::vector<double>& holdings) {
    require_holdings(holdings);
    require_liquidating(holdings);
    const std::vector<double> n = trades_from_holdings(holdings);
    double sum_n2 = 0.0;
    for (double nj : n) sum_n2 += nj * nj;
    const double x0 = holdings.front();
    return 0.5 * params.gamma_p * x0 * x0 +
           params.eta_tilde() / params.tau * sum_n2;
}

double expected_cost(const ACParams& params) {
    return expected_cost(params, ac_trajectory(params));
}

double variance_cost(const ACParams& params, const std::vector<double>& holdings) {
    require_holdings(holdings);
    double sum_x2 = 0.0;
    for (std::size_t j = 1; j < holdings.size(); ++j) {
        sum_x2 += holdings[j] * holdings[j];
    }
    return params.sigma * params.sigma * params.tau * sum_x2;
}

double variance_cost(const ACParams& params) {
    return variance_cost(params, ac_trajectory(params));
}

std::pair<double, double> instantaneous_cost(const ACParams& params) {
    const double x0 = params.total_shares;
    const double e = 0.5 * params.gamma_p * x0 * x0 +
                     params.eta_tilde() * x0 * x0 / params.tau;
    return {e, 0.0};
}

std::vector<FrontierPoint> efficient_frontier(const ACParams& params,
                                              const std::vector<double>& lambdas) {
    require(!lambdas.empty(), "lambdas grid must be non-empty");
    std::vector<FrontierPoint> out;
    out.reserve(lambdas.size());
    for (double lam : lambdas) {
        const ACParams p = params.with_lambda(lam);
        const double e = expected_cost(p);
        const double v = variance_cost(p);
        out.push_back({lam, ac_kappa(p), e, v, std::sqrt(v)});
    }
    return out;
}

ISSimResult simulate_is(const ACParams& params, const std::vector<double>& holdings,
                        int n_paths, std::uint64_t seed) {
    require(n_paths >= 2 && n_paths <= kMaxPaths,
            "n_paths must be an integer in [2, 10000000]");
    require_holdings(holdings);
    require_liquidating(holdings);
    require(static_cast<std::int64_t>(n_paths) *
                    static_cast<std::int64_t>(holdings.size() - 1) <=
                kMaxMcWork,
            "n_paths * n_intervals must be <= 1000000000");
    const std::vector<double> n = trades_from_holdings(holdings);
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    const double srt = params.sigma * std::sqrt(params.tau);

    // Per path: execute n_j at S_{j-1} - eta*n_j/tau, then walk S.
    // IS = X*S_0 - proceeds with S_0 = 0 (IS is independent of S_0).
    std::vector<double> shortfall(static_cast<std::size_t>(n_paths));
    for (int p = 0; p < n_paths; ++p) {
        double s_prev = 0.0;
        double proceeds = 0.0;
        for (double nj : n) {
            proceeds += nj * (s_prev - params.eta * nj / params.tau);
            s_prev += srt * normal(rng) - params.gamma_p * nj;
        }
        shortfall[p] = -proceeds;
    }

    double mean = 0.0;
    for (double v : shortfall) mean += v;
    mean /= n_paths;
    double var = 0.0;
    for (double v : shortfall) var += (v - mean) * (v - mean);
    var /= (n_paths - 1);  // ddof = 1
    return {mean, var, std::sqrt(var / n_paths),
            var * std::sqrt(2.0 / (n_paths - 1)), n_paths};
}

ISSimResult simulate_is(const ACParams& params, int n_paths, std::uint64_t seed) {
    return simulate_is(params, ac_trajectory(params), n_paths, seed);
}

}  // namespace etrade
