#include "etrade/avellaneda_stoikov.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <random>
#include <stdexcept>

namespace etrade {

const double kDefaultFixedSpread = std::numeric_limits<double>::quiet_NaN();

namespace {

/// Throw std::invalid_argument unless `cond` holds.
void require(bool cond, const char* msg) {
    if (!cond) throw std::invalid_argument(msg);
}

/// Sample mean of a per-path array.
double mean_of(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

/// Sample standard deviation (ddof = 1).
double std_of(const std::vector<double>& v) {
    const double m = mean_of(v);
    double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / static_cast<double>(v.size() - 1));
}

}  // namespace

ASParams::ASParams(double s0_, double sigma_, double horizon_, double dt_,
                   double gamma_, double k_, double a_rate_, int q_max_,
                   double eta_liq_)
    : s0(s0_),
      sigma(sigma_),
      horizon(horizon_),
      dt(dt_),
      gamma(gamma_),
      k(k_),
      a_rate(a_rate_),
      q_max(q_max_),
      eta_liq(eta_liq_) {
    require(std::isfinite(s0), "s0 must be finite");
    require(std::isfinite(sigma) && sigma >= 0.0, "sigma must be >= 0");
    require(std::isfinite(horizon) && horizon > 0.0, "horizon must be > 0");
    require(std::isfinite(dt) && dt > 0.0 && dt <= horizon,
            "dt must satisfy 0 < dt <= horizon");
    require(std::isfinite(gamma) && gamma > 0.0,
            "gamma must be > 0 (gamma = 0 is outside the A-S model)");
    require(std::isfinite(k) && k > 0.0, "k must be > 0");
    require(std::isfinite(a_rate) && a_rate > 0.0, "a_rate (A) must be > 0");
    require(q_max >= 0, "q_max must be an integer >= 0");
    require(std::isfinite(eta_liq) && eta_liq >= 0.0, "eta_liq must be >= 0");
    const double n = horizon / dt;  // may overflow to inf: bound it first
    require(n <= static_cast<double>(kMaxAsSteps),
            "horizon/dt must be <= 10000000 steps");
    require(std::abs(n - std::round(n)) < 1e-9,
            "horizon must be an integer multiple of dt");
}

int ASParams::n_steps() const {
    return static_cast<int>(std::lround(horizon / dt));
}

double reservation_price(const ASParams& params, double s, double q, double t) {
    require(std::isfinite(s) && std::isfinite(q), "s and q must be finite");
    require(t >= 0.0 && t <= params.horizon, "t must be in [0, horizon]");
    return s - q * params.gamma * params.sigma * params.sigma *
                   (params.horizon - t);
}

double optimal_spread(const ASParams& params, double t) {
    require(t >= 0.0 && t <= params.horizon, "t must be in [0, horizon]");
    return params.gamma * params.sigma * params.sigma * (params.horizon - t) +
           2.0 / params.gamma * std::log1p(params.gamma / params.k);
}

std::pair<double, double> quotes(const ASParams& params, double s, double q,
                                 double t) {
    const double r = reservation_price(params, s, q, t);
    const double half = 0.5 * optimal_spread(params, t);
    return {r - half, r + half};
}

double intensity(const ASParams& params, double delta) {
    require(std::isfinite(delta), "delta must be finite");
    return params.a_rate * std::exp(-params.k * delta);
}

double fill_probability(const ASParams& params, double delta) {
    return -std::expm1(-intensity(params, delta) * params.dt);
}

double average_as_spread(const ASParams& params) {
    return 0.5 * params.gamma * params.sigma * params.sigma * params.horizon +
           2.0 / params.gamma * std::log1p(params.gamma / params.k);
}

double ASSimResult::mean_pnl() const { return mean_of(pnl); }

double ASSimResult::std_pnl() const { return std_of(pnl); }

double ASSimResult::sharpe() const {
    const double s = std_pnl();
    return s == 0.0 ? std::numeric_limits<double>::infinity() : mean_pnl() / s;
}

double ASSimResult::mean_abs_q() const { return mean_of(mean_abs_q_path); }

double ASSimResult::mean_max_abs_q() const { return mean_of(max_abs_q_path); }

double ASSimResult::rms_q() const { return mean_of(rms_q_path); }

double ASSimResult::mean_fills() const { return mean_of(fills_path); }

double ASSimResult::se(const std::string& stat) const {
    const double m = static_cast<double>(n_paths());
    if (stat == "std_pnl") return std_pnl() / std::sqrt(2.0 * (m - 1.0));
    const std::vector<double>* arr = nullptr;
    if (stat == "mean_pnl") arr = &pnl;
    else if (stat == "mean_abs_q") arr = &mean_abs_q_path;
    else if (stat == "mean_max_abs_q") arr = &max_abs_q_path;
    else if (stat == "rms_q") arr = &rms_q_path;
    else if (stat == "mean_fills") arr = &fills_path;
    if (arr == nullptr) {
        throw std::invalid_argument("unknown statistic '" + stat + "'");
    }
    return std_of(*arr) / std::sqrt(m);
}

ASSimResult simulate(const ASParams& params, int n_paths, std::uint64_t seed,
                     Agent agent, double fixed_spread, double stress_factor,
                     double stress_time_frac) {
    require(n_paths >= 2 && n_paths <= kMaxAsPaths,
            "n_paths must be an integer in [2, 10000000]");
    require(static_cast<std::int64_t>(n_paths) *
                    static_cast<std::int64_t>(params.n_steps()) <=
                kMaxAsWork,
            "n_paths * n_steps must be <= 10000000000");
    require(std::isfinite(stress_factor) && stress_factor > 0.0,
            "stress_factor must be > 0");
    require(std::isfinite(stress_time_frac) && stress_time_frac >= 0.0 &&
                stress_time_frac <= 1.0,
            "stress_time_frac must be in [0, 1]");
    if (std::isnan(fixed_spread)) fixed_spread = average_as_spread(params);
    require(std::isfinite(fixed_spread) && fixed_spread > 0.0,
            "fixed_spread must be finite and > 0");

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::normal_distribution<double> normal(0.0, 1.0);

    const int n_steps = params.n_steps();
    const double g = params.gamma;
    const double sig2 = params.sigma * params.sigma;
    const double big_t = params.horizon;
    const double dt = params.dt;
    const double markup = 2.0 / g * std::log1p(g / params.k);
    const double sqdt = std::sqrt(dt);
    const double stress_t0 = stress_time_frac * big_t;

    ASSimResult res;
    res.pnl.resize(static_cast<std::size_t>(n_paths));
    res.mean_abs_q_path.resize(static_cast<std::size_t>(n_paths));
    res.max_abs_q_path.resize(static_cast<std::size_t>(n_paths));
    res.rms_q_path.resize(static_cast<std::size_t>(n_paths));
    res.fills_path.resize(static_cast<std::size_t>(n_paths));
    res.overall_max_abs_q = 0.0;
    res.cap_violated = false;

    for (int p = 0; p < n_paths; ++p) {
        double s = params.s0;
        int q = 0;
        double cash = 0.0;
        double sum_abs_q = 0.0;
        double sum_q2 = 0.0;
        int max_abs_q = 0;
        int fills = 0;
        for (int i = 0; i < n_steps; ++i) {
            const double t = i * dt;
            // 1. quote from the current s, q, remaining time (cap: one-sided
            //    quoting at |q| = q_max, nothing at q_max = 0).
            double r, half;
            if (agent == Agent::kAS) {
                r = s - q * g * sig2 * (big_t - t);
                half = 0.5 * (g * sig2 * (big_t - t) + markup);
            } else {
                r = s;
                half = 0.5 * fixed_spread;
            }
            const double ask = r + half;
            const double bid = r - half;
            const bool quote_ask = q > -params.q_max;
            const bool quote_bid = q < params.q_max;
            // 2. fill draws — u_ask first, then u_bid, both drawn always so
            //    the stream does not depend on the cap state.
            const double u_ask = uniform(rng);
            const double u_bid = uniform(rng);
            const double p_ask =
                -std::expm1(-params.a_rate * std::exp(-params.k * (ask - s)) * dt);
            const double p_bid =
                -std::expm1(-params.a_rate * std::exp(-params.k * (s - bid)) * dt);
            const bool fill_ask = quote_ask && u_ask < p_ask;
            const bool fill_bid = quote_bid && u_bid < p_bid;
            // 3. inventory / cash update (unit fill size; both sides may fill,
            //    and both count: fills is a total fill count, API_SPEC 2.2).
            if (fill_ask) { q -= 1; cash += ask; ++fills; }
            if (fill_bid) { q += 1; cash -= bid; ++fills; }
            const int abs_q = std::abs(q);
            if (abs_q > params.q_max) res.cap_violated = true;
            sum_abs_q += abs_q;
            sum_q2 += static_cast<double>(q) * q;
            max_abs_q = std::max(max_abs_q, abs_q);
            // 4. mid move — stressed vol affects only the realised path; the
            //    agent keeps quoting with the base sigma.
            const double sig_t =
                params.sigma * (t >= stress_t0 ? stress_factor : 1.0);
            s += sig_t * sqdt * normal(rng);
        }
        // Terminal liquidation at the final mid with penalty eta_liq*|q|.
        res.pnl[p] = cash + q * s - params.eta_liq * std::abs(q);
        res.mean_abs_q_path[p] = sum_abs_q / n_steps;
        res.max_abs_q_path[p] = max_abs_q;
        res.rms_q_path[p] = std::sqrt(sum_q2 / n_steps);
        res.fills_path[p] = fills;
        res.overall_max_abs_q =
            std::max(res.overall_max_abs_q, static_cast<double>(max_abs_q));
    }
    return res;
}

}  // namespace etrade
