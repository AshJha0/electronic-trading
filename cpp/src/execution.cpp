#include "etrade/execution.hpp"
#include "etrade/detail/require.hpp"

#include "etrade/impact.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <string>

namespace etrade {
namespace {

using detail::require;

std::string intervals_msg() {
    return "n_intervals must be an integer in [1, " + std::to_string(kMaxExecIntervals) + "]";
}

/// Symmetric tridiagonal solve (off[k] couples k and k+1).
std::vector<double> thomas(const std::vector<double>& diag, const std::vector<double>& off,
                           const std::vector<double>& rhs) {
    const std::size_t m = diag.size();
    std::vector<double> c(m, 0.0), d(m, 0.0), out(m, 0.0);
    c[0] = m > 1 ? off[0] / diag[0] : 0.0;
    d[0] = rhs[0] / diag[0];
    for (std::size_t k = 1; k < m; ++k) {
        const double denom = diag[k] - off[k - 1] * c[k - 1];
        c[k] = k < m - 1 ? off[k] / denom : 0.0;
        d[k] = (rhs[k] - off[k - 1] * d[k - 1]) / denom;
    }
    out[m - 1] = d[m - 1];
    for (std::size_t k = m - 1; k-- > 0;) out[k] = d[k] - c[k] * out[k + 1];
    return out;
}

void check_ac(const ACParams& p) { require(p.n_intervals <= kMaxExecIntervals, intervals_msg()); }

void replan_system(const ACParams& p, int m, std::vector<double>& diag, std::vector<double>& off) {
    const double e = p.eta_tilde() / p.tau;
    const auto sz = static_cast<std::size_t>(m - 1);
    diag.assign(sz, 2.0 * e + p.lambda_ra * p.sigma * p.sigma * p.tau);
    off.assign(sz, -e);
}

double mean_of(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

double sample_std(const std::vector<double>& v, double mean) {
    double ss = 0.0;
    for (double x : v) ss += (x - mean) * (x - mean);
    return std::sqrt(ss / static_cast<double>(v.size() - 1));
}

void check_orders(const std::string& name, const std::vector<double>& q) {
    require(q.size() <= static_cast<std::size_t>(kMaxOrders),
            name + " must have at most " + std::to_string(kMaxOrders) + " orders");
    for (double v : q) require(std::isfinite(v) && v > 0.0, name + " must be finite and > 0");
}

}  // namespace

// ------------------------------------------------------------------ transient

TransientParams::TransientParams(double total_shares_, int n_intervals_, double tau_,
                                 double kappa_, double rho_, double gamma_p_)
    : total_shares(total_shares_),
      n_intervals(n_intervals_),
      tau(tau_),
      kappa(kappa_),
      rho(rho_),
      gamma_p(gamma_p_) {
    require(std::isfinite(total_shares) && total_shares >= 0.0,
            "total_shares must be finite and >= 0");
    require(1 <= n_intervals && n_intervals <= kMaxExecIntervals, intervals_msg());
    require(std::isfinite(tau) && tau > 0.0, "tau must be > 0");
    require(std::isfinite(kappa) && kappa > 0.0, "kappa must be > 0");
    require(std::isfinite(rho) && rho > 0.0, "rho must be finite and > 0");
    require(std::isfinite(gamma_p) && gamma_p >= 0.0, "gamma_p must be >= 0");
    const double a = decay();
    require(gamma_p < kappa * (1.0 - a) / (1.0 + a),
            "model invalid: gamma_p >= kappa*(1-a)/(1+a) (cost not convex in the trades)");
}

double TransientParams::decay() const { return std::exp(-rho * tau); }

double transient_expected_cost(const TransientParams& p, const std::vector<double>& x) {
    require(x.size() == static_cast<std::size_t>(p.n_intervals) + 1,
            "holdings must have length n_intervals + 1");
    for (double v : x) require(std::isfinite(v), "holdings must be finite");
    require(std::abs(x.back()) <= 1e-9 * std::max(1.0, std::abs(x[0])),
            "holdings path must fully liquidate (x_N = 0)");
    const double a = p.decay();
    const double half = 0.5 * (p.kappa - p.gamma_p);
    double d = 0.0;
    double total = 0.0;
    for (std::size_t j = 1; j < x.size(); ++j) {
        const double n = x[j - 1] - x[j];
        total += half * n * n + n * d;
        d = a * (d + p.kappa * n);
    }
    const double x0 = x[0];
    return 0.5 * p.gamma_p * x0 * x0 + total;
}

std::vector<double> transient_optimal_schedule(const TransientParams& p) {
    const int n_int = p.n_intervals;
    const double big_x = p.total_shares;
    const auto n = static_cast<std::size_t>(n_int);
    std::vector<double> x(n + 1, 0.0);
    x[0] = big_x;
    if (big_x == 0.0 || n_int == 1) return x;
    const double a = p.decay();
    const double b = -std::expm1(-p.rho * p.tau);  // 1 - a without cancellation
    const double ka = b * (1.0 + a) * p.kappa;     // (1 - a^2) * kappa
    const double g = p.gamma_p;
    std::vector<double> diag(n, ka - g * (1.0 + a * a));
    diag[0] = ka - g;
    diag[n - 1] = ka - g;
    const std::vector<double> off(n - 1, g * a);
    std::vector<double> rhs(n, b * b);
    rhs[0] = b;
    rhs[n - 1] = b;
    const std::vector<double> y = thomas(diag, off, rhs);
    double total = 0.0;
    for (double v : y) total += v;
    for (std::size_t j = 1; j < n; ++j) x[j] = x[j - 1] - big_x * y[j - 1] / total;
    x[n] = 0.0;
    return x;
}

// ------------------------------------------------------------------ adaptive

SignalParams::SignalParams(double phi_, double signal_sd_, double fill_risk_)
    : phi(phi_), signal_sd(signal_sd_), fill_risk(fill_risk_) {
    require(std::isfinite(phi) && 0.0 <= phi && phi < 1.0, "phi must be in [0, 1)");
    require(std::isfinite(signal_sd) && signal_sd >= 0.0, "signal_sd must be >= 0");
    require(std::isfinite(fill_risk) && 0.0 <= fill_risk && fill_risk < 1.0,
            "fill_risk must be in [0, 1)");
}

std::vector<double> replan(const ACParams& p, const SignalParams& sig, double holding,
                           double signal, int k) {
    check_ac(p);
    require(0 <= k && k < p.n_intervals, "k must be an integer in [0, n_intervals)");
    require(std::isfinite(holding), "holding must be finite");
    require(std::isfinite(signal), "signal must be finite");
    const int m = p.n_intervals - k;
    if (m == 1) return {holding, 0.0};
    std::vector<double> diag, off;
    replan_system(p, m, diag, off);
    const double e = p.eta_tilde() / p.tau;
    const auto len = static_cast<std::size_t>(m - 1);
    std::vector<double> rhs(len);
    for (std::size_t i = 0; i < len; ++i)
        rhs[i] = 0.5 * p.tau * std::pow(sig.phi, static_cast<double>(i)) * signal;
    rhs[0] += e * holding;
    const std::vector<double> sol = thomas(diag, off, rhs);
    std::vector<double> out;
    out.reserve(sol.size() + 2);
    out.push_back(holding);
    out.insert(out.end(), sol.begin(), sol.end());
    out.push_back(0.0);
    return out;
}

std::pair<std::vector<double>, std::vector<double>> replan_coefficients(const ACParams& p,
                                                                        const SignalParams& sig) {
    check_ac(p);
    const int n_int = p.n_intervals;
    const auto sz = static_cast<std::size_t>(n_int) + 1;
    std::vector<double> c(sz, 0.0), d(sz, 0.0);
    if (n_int >= 1) c[1] = 1.0;
    const double e = p.eta_tilde() / p.tau;
    std::vector<double> diag, off;
    for (int m = 2; m <= n_int; ++m) {
        replan_system(p, m, diag, off);
        const auto len = static_cast<std::size_t>(m - 1);
        std::vector<double> r_hold(len, 0.0);
        r_hold[0] = e;
        std::vector<double> r_sig(len);
        for (std::size_t i = 0; i < len; ++i)
            r_sig[i] = 0.5 * p.tau * std::pow(sig.phi, static_cast<double>(i));
        const auto mi = static_cast<std::size_t>(m);
        c[mi] = 1.0 - thomas(diag, off, r_hold)[0];
        d[mi] = thomas(diag, off, r_sig)[0];
    }
    return {c, d};
}

AdaptiveSimResult simulate_adaptive(const ACParams& p, const SignalParams& sig, int n_paths,
                                    std::uint64_t seed) {
    check_ac(p);
    require(2 <= n_paths && n_paths <= kMaxPaths,
            "n_paths must be an integer in [2, " + std::to_string(kMaxPaths) + "]");
    const int n_int = p.n_intervals;
    const std::vector<double> plan = replan(p, sig, p.total_shares, 0.0, 0);
    const auto cd = replan_coefficients(p, sig);
    const std::vector<double>& c = cd.first;
    const std::vector<double>& d = cd.second;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::normal_distribution<double> normal(0.0, 1.0);
    const double srt = p.sigma * std::sqrt(p.tau);
    const double innov = std::sqrt(1.0 - sig.phi * sig.phi) * sig.signal_sd;
    const auto np = static_cast<std::size_t>(n_paths);
    std::vector<double> is_st(np), is_ad(np), sav(np);
    for (std::size_t i = 0; i < np; ++i) {
        double x_st = p.total_shares;
        double x_ad = p.total_shares;
        double px_st = 0.0, px_ad = 0.0, pr_st = 0.0, pr_ad = 0.0, sgn = 0.0;
        for (int j = 1; j <= n_int; ++j) {
            const auto m = static_cast<std::size_t>(n_int - j + 1);
            double q_st = 0.0;
            double q_ad = 0.0;
            if (j == n_int) {
                q_st = x_st;
                q_ad = x_ad;
            } else {
                const auto ju = static_cast<std::size_t>(j);
                q_st = plan[ju - 1] - plan[ju];
                q_ad = c[m] * x_ad - d[m] * sgn;
            }
            const double u = unif(rng);
            const double xi = normal(rng);
            const double eps = normal(rng);
            const double f = j == n_int ? 1.0 : 1.0 - sig.fill_risk * u;
            const double e_st = f * q_st;
            const double e_ad = f * q_ad;
            pr_st += e_st * (px_st - p.eta * e_st / p.tau);
            pr_ad += e_ad * (px_ad - p.eta * e_ad / p.tau);
            const double common = sgn * p.tau + srt * xi;
            px_st = px_st + common - p.gamma_p * e_st;
            px_ad = px_ad + common - p.gamma_p * e_ad;
            sgn = sig.phi * sgn + innov * eps;
            x_st = x_st - e_st;
            x_ad = x_ad - e_ad;
        }
        is_st[i] = -pr_st;
        is_ad[i] = -pr_ad;
        sav[i] = is_st[i] - is_ad[i];
    }
    AdaptiveSimResult r;
    r.static_mean = mean_of(is_st);
    r.adaptive_mean = mean_of(is_ad);
    r.saving_mean = mean_of(sav);
    r.static_std = sample_std(is_st, r.static_mean);
    r.adaptive_std = sample_std(is_ad, r.adaptive_mean);
    const double sd_sav = sample_std(sav, r.saving_mean);
    const double root = std::sqrt(static_cast<double>(n_paths));
    r.static_se = r.static_std / root;
    r.adaptive_se = r.adaptive_std / root;
    r.saving_se = sd_sav / root;
    r.n_paths = n_paths;
    return r;
}

// ------------------------------------------------------------------ internalisation

InternalisationResult internalise(const std::vector<double>& buys, const std::vector<double>& sells,
                                  double price, double spread, double adv, double daily_vol,
                                  int n_intervals, double tau, double alpha, double y,
                                  double perm_frac, const std::vector<double>& weights) {
    check_orders("buys", buys);
    check_orders("sells", sells);
    require(buys.size() + sells.size() >= 1, "at least one order is required");
    require(1 <= n_intervals && n_intervals <= kMaxExecIntervals, intervals_msg());

    auto cost = [&](double q) {
        if (q == 0.0) return 0.0;
        ImpactMarket m;
        m.price = price;
        m.spread = spread;
        m.adv = adv;
        m.daily_vol = daily_vol;
        m.total_shares = q;
        m.n_intervals = n_intervals;
        m.tau = tau;
        m.lambda_ra = 0.0;
        m.alpha = alpha;
        m.y = y;
        m.perm_frac = perm_frac;
        m.weights = weights;
        const ImpactParams ip = impact_params_from_market(m);
        return impact_expected_cost(ip, vwap_holdings(q, ip.weights()));
    };

    double big_b = 0.0;
    double big_s = 0.0;
    for (double v : buys) big_b += v;
    for (double v : sells) big_s += v;
    InternalisationResult r;
    r.crossed_shares = std::min(big_b, big_s);
    r.residual_shares = std::abs(big_b - big_s);
    r.residual_side = big_b > big_s ? "buy" : big_s > big_b ? "sell" : "none";
    const double c_b = cost(big_b);
    const double c_s = cost(big_s);
    const double c_r = cost(r.residual_shares);
    r.baseline_cost = c_b + c_s;
    r.internal_cost = c_r;
    r.saved_cost = r.baseline_cost - c_r;
    r.saved_bps = bps(r.saved_cost, (big_b + big_s) * price);

    auto per_order = [&](const std::vector<double>& qs, double total, double c_side,
                         bool majority) {
        std::vector<double> out;
        out.reserve(qs.size());
        for (double q : qs) {
            const double base_i = c_side * q / total;
            const double int_i = majority ? c_r * q / total : 0.0;
            out.push_back(bps(base_i - int_i, q * price));
        }
        return out;
    };
    r.buy_saved_bps = per_order(buys, big_b, c_b, r.residual_side == "buy");
    r.sell_saved_bps = per_order(sells, big_s, c_s, r.residual_side == "sell");
    return r;
}

}  // namespace etrade
