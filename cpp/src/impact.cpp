#include "etrade/impact.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace etrade {
namespace {

void require(bool cond, const std::string& msg) {
    if (!cond) throw std::invalid_argument(msg);
}

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

double objective(const ImpactParams& p, const std::vector<double>& etas,
                 const std::vector<double>& x) {
    const double big_x = x[0];
    double total = 0.5 * p.gamma_p() * big_x * big_x;
    const double a1 = 1.0 + p.alpha();
    for (std::size_t j = 1; j < x.size(); ++j) {
        const double n = x[j - 1] - x[j];
        total += etas[j - 1] * std::pow(n, a1) - 0.5 * p.gamma_p() * n * n;
    }
    double hold = 0.0;
    for (std::size_t k = 1; k + 1 < x.size(); ++k) hold += x[k];
    double sq = 0.0;
    for (std::size_t k = 1; k < x.size(); ++k) sq += x[k] * x[k];
    return total + p.drift() * p.tau() * hold +
           p.lambda_ra() * (p.sigma() * p.sigma()) * p.tau() * sq;
}

std::optional<std::pair<std::vector<double>, int>> solve_horizon(const ImpactParams& p, int m) {
    const double big_x = p.total_shares();
    if (m == 1) return std::make_pair(std::vector<double>{big_x, 0.0}, 0);
    const double a = p.alpha(), a1 = 1.0 + p.alpha(), g2 = p.gamma_p();
    const double v2 = 2.0 * p.lambda_ra() * (p.sigma() * p.sigma()) * p.tau();
    const double dt = p.drift() * p.tau();
    const std::vector<double> e(p.etas().begin(), p.etas().begin() + m);
    std::vector<double> x(m + 1);
    for (int k = 0; k <= m; ++k) x[k] = big_x * (m - k) / m;
    std::vector<double> n(m), c1(m), c2(m), grad(m - 1), diag(m - 1), off(m - 1), rhs(m - 1);
    for (int it = 1; it <= kNewtonMaxIter; ++it) {
        for (int j = 0; j < m; ++j) n[j] = x[j] - x[j + 1];
        for (int j = 0; j < m; ++j) {
            c1[j] = a1 * e[j] * std::pow(n[j], a) - g2 * n[j];
            c2[j] = a * a1 * e[j] * std::pow(n[j], a - 1.0) - g2;
        }
        for (int k = 1; k < m; ++k) {
            grad[k - 1] = c1[k] - c1[k - 1] + v2 * x[k] + dt;
            diag[k - 1] = c2[k - 1] + c2[k] + v2;
            off[k - 1] = -c2[k];
            rhs[k - 1] = -grad[k - 1];
        }
        const std::vector<double> step = thomas(diag, off, rhs);
        const double u0 = objective(p, e, x);
        double slope = 0.0, max_step = 0.0;
        for (int k = 0; k < m - 1; ++k) {
            slope += grad[k] * step[k];
            max_step = std::max(max_step, std::abs(step[k]));
        }
        if (max_step <= kNewtonStepTol * big_x ||
            -slope <= kNewtonDecrementTol * std::max(1.0, std::abs(u0))) {
            return std::make_pair(x, it);
        }
        double t = 1.0;
        bool accepted = false;
        std::vector<double> xn(m + 1);
        while (t >= kMinStep) {
            xn[0] = x[0];
            for (int k = 1; k < m; ++k) xn[k] = x[k] + t * step[k - 1];
            xn[m] = 0.0;
            bool positive = true;
            for (int j = 1; j <= m; ++j) positive = positive && (xn[j - 1] - xn[j] > 0.0);
            if (positive && objective(p, e, xn) <= u0 + kArmijoC * t * slope) {
                accepted = true;
                break;
            }
            t *= 0.5;
        }
        if (!accepted) return std::nullopt;
        x = xn;
    }
    return std::nullopt;
}

std::vector<double> check_holdings(const ImpactParams& p, const std::vector<double>& x) {
    require(static_cast<int>(x.size()) == p.n_intervals() + 1,
            "holdings must have length n_intervals + 1");
    for (double v : x) require(std::isfinite(v), "holdings must be finite");
    const double scale = std::max(1.0, std::abs(x[0]));
    require(std::abs(x.back()) <= 1e-9 * scale, "holdings path must fully liquidate (x_N = 0)");
    for (std::size_t j = 1; j < x.size(); ++j) {
        require(x[j] - x[j - 1] <= 1e-12 * scale, "holdings must be non-increasing (a sell program)");
    }
    return x;
}

}  // namespace

ImpactParams::ImpactParams(double total_shares, int n_intervals, double tau, double sigma,
                           double eta, double alpha, double gamma_p, double lambda_ra,
                           double drift, std::vector<double> weights)
    : total_shares_(total_shares), tau_(tau), sigma_(sigma), eta_(eta), alpha_(alpha),
      gamma_p_(gamma_p), lambda_ra_(lambda_ra), drift_(drift), n_intervals_(n_intervals),
      weights_(std::move(weights)) {
    require(std::isfinite(total_shares) && total_shares >= 0.0, "total_shares must be finite and >= 0");
    require(n_intervals >= 1 && n_intervals <= kMaxImpactIntervals,
            "n_intervals must be an integer in [1, 10000]");
    require(std::isfinite(tau) && tau > 0.0, "tau must be > 0");
    require(std::isfinite(sigma) && sigma >= 0.0, "sigma must be >= 0");
    require(std::isfinite(eta) && eta > 0.0, "eta must be > 0");
    require(std::isfinite(alpha) && alpha > 0.0 && alpha <= 1.0, "alpha must be in (0, 1]");
    require(std::isfinite(gamma_p) && gamma_p >= 0.0, "gamma_p must be >= 0");
    require(std::isfinite(lambda_ra) && lambda_ra >= 0.0, "lambda_ra must be >= 0");
    require(std::isfinite(drift) && drift >= 0.0, "drift must be >= 0");
    if (weights_.empty()) {
        weights_.assign(static_cast<std::size_t>(n_intervals), tau);
    } else {
        require(static_cast<int>(weights_.size()) == n_intervals,
                "weights must have length n_intervals");
        for (double w : weights_) require(std::isfinite(w) && w > 0.0, "weights must be finite and > 0");
    }
    etas_.reserve(weights_.size());
    for (double w : weights_) etas_.push_back(eta * std::pow(w, -alpha));
    if (total_shares > 0.0) {
        const double x_pow = std::pow(total_shares, alpha - 1.0);
        for (double eta_j : etas_) {
            require(alpha * (1.0 + alpha) * eta_j * x_pow > gamma_p,
                    "model invalid: temporary impact not convex enough against permanent "
                    "impact (alpha*(1+alpha)*eta_j*X**(alpha-1) <= gamma_p)");
        }
    }
}

double impact_expected_cost(const ImpactParams& p, const std::vector<double>& holdings) {
    const std::vector<double> x = check_holdings(p, holdings);
    const double a1 = 1.0 + p.alpha();
    double temp = 0.0;
    for (std::size_t j = 1; j < x.size(); ++j) {
        const double n = std::max(x[j - 1] - x[j], 0.0);
        temp += p.etas()[j - 1] * std::pow(n, a1) - 0.5 * p.gamma_p() * n * n;
    }
    double hold = 0.0;
    for (std::size_t k = 1; k + 1 < x.size(); ++k) hold += x[k];
    return 0.5 * p.gamma_p() * x[0] * x[0] + temp + p.drift() * p.tau() * hold;
}

double impact_variance_cost(const ImpactParams& p, const std::vector<double>& holdings) {
    const std::vector<double> x = check_holdings(p, holdings);
    double sq = 0.0;
    for (std::size_t k = 1; k < x.size(); ++k) sq += x[k] * x[k];
    return (p.sigma() * p.sigma()) * p.tau() * sq;
}

std::vector<double> vwap_holdings(double total_shares, const std::vector<double>& weights) {
    require(std::isfinite(total_shares) && total_shares >= 0.0, "total_shares must be finite and >= 0");
    require(!weights.empty(), "weights must be a non-empty 1-D sequence");
    double sum = 0.0;
    for (double w : weights) {
        require(std::isfinite(w) && w > 0.0, "weights must be finite and > 0");
        sum += w;
    }
    std::vector<double> x(weights.size() + 1);
    x[0] = total_shares;
    double cum = 0.0;
    for (std::size_t j = 0; j < weights.size(); ++j) {
        cum += weights[j];
        x[j + 1] = total_shares * (1.0 - cum / sum);
    }
    x.back() = 0.0;
    return x;
}

ScheduleResult optimal_schedule(const ImpactParams& p) {
    const int n_int = p.n_intervals();
    ScheduleResult out;
    out.holdings.assign(static_cast<std::size_t>(n_int) + 1, 0.0);
    out.horizon = n_int;
    if (p.total_shares() == 0.0) return out;
    auto pad = [&](const std::vector<double>& xm) {
        std::copy(xm.begin(), xm.end(), out.holdings.begin());
    };
    if (auto full = solve_horizon(p, n_int)) {
        pad(full->first);
        out.iterations = full->second;
        return out;
    }
    if (!(p.drift() > 0.0)) throw std::runtime_error("optimal schedule: Newton solve did not converge");
    int lo = 1, hi = n_int, iters = 0;
    auto best = *solve_horizon(p, 1);
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (auto r = solve_horizon(p, mid)) {
            lo = mid;
            best = *r;
            iters += r->second;
        } else {
            hi = mid;
        }
    }
    iters += best.second;
    const double n_last = best.first[lo - 1] - best.first[lo];
    const double marginal =
        (1.0 + p.alpha()) * p.etas()[lo - 1] * std::pow(n_last, p.alpha()) - p.gamma_p() * n_last;
    if (!(marginal <= p.drift() * p.tau() * (1.0 + 1e-9) + 1e-12)) {
        throw std::runtime_error("optimal schedule: KKT check failed for the early-finish horizon");
    }
    pad(best.first);
    out.horizon = lo;
    out.iterations = iters;
    return out;
}

double sqrt_law_cost_bps(double daily_vol, double order_shares, double adv, double y) {
    require(std::isfinite(daily_vol) && daily_vol > 0.0, "daily_vol must be finite and > 0");
    require(std::isfinite(order_shares) && order_shares >= 0.0, "order_shares must be >= 0");
    require(std::isfinite(adv) && adv > 0.0, "adv must be finite and > 0");
    require(std::isfinite(y) && y > 0.0, "y must be finite and > 0");
    return 1e4 * y * daily_vol * std::sqrt(order_shares / adv);
}

}  // namespace etrade
