#include "etrade/placement.hpp"
#include "etrade/detail/require.hpp"

#include "etrade/almgren_chriss.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <string>

namespace etrade {
namespace {

using detail::require;

struct Stat {
    double mean;
    double sd;
};

Stat stats_of(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    const double mean = s / static_cast<double>(v.size());
    double ss = 0.0;
    for (double x : v) ss += (x - mean) * (x - mean);
    return {mean, std::sqrt(ss / static_cast<double>(v.size() - 1))};
}

}  // namespace

// ------------------------------------------------------------------ fill probability

double queue_fill_probability(int q_ahead, int size, double mu, double theta, double horizon) {
    require(q_ahead >= 0, "q_ahead must be an integer >= 0");
    require(size >= 1, "size must be an integer >= 1");
    require(q_ahead <= kMaxQueueUnits - size,
            "q_ahead + size must be <= " + std::to_string(kMaxQueueUnits));
    require(std::isfinite(mu) && mu > 0.0, "mu must be finite and > 0");
    require(std::isfinite(theta) && theta >= 0.0, "theta must be finite and >= 0");
    require(std::isfinite(horizon) && horizon >= 0.0, "horizon must be finite and >= 0");
    const double lam = mu + theta * static_cast<double>(q_ahead);
    const double big_l = lam * horizon;
    require(std::isfinite(big_l), "horizon too long: (mu + theta*q_ahead)*horizon overflows");
    if (big_l == 0.0) return 0.0;
    const double m_d = std::floor(big_l);
    const double half = std::ceil(10.0 * std::sqrt(big_l) + 10.0);
    require(m_d + half <= static_cast<double>(kMaxUniformisationSteps),
            "horizon too long: (mu + theta*q_ahead)*horizon needs more than " +
                std::to_string(kMaxUniformisationSteps) + " uniformisation steps");
    const auto m = static_cast<long long>(m_d);
    const auto hf = static_cast<long long>(half);
    const long long k_lo = std::max(0LL, m - hf);
    const long long k_hi = m + hf;
    const auto n_st = static_cast<std::size_t>(q_ahead) + static_cast<std::size_t>(size);
    std::vector<double> go(n_st), stay(n_st);
    for (std::size_t s = 0; s < n_st; ++s) {
        const double r = s < static_cast<std::size_t>(q_ahead)
                             ? mu + theta * static_cast<double>(static_cast<std::size_t>(q_ahead) - s)
                             : mu;
        go[s] = r / lam;
        stay[s] = 1.0 - go[s];
    }
    std::vector<double> w(static_cast<std::size_t>(k_hi) + 1, 0.0);
    w[static_cast<std::size_t>(m)] = 1.0;
    for (long long k = m + 1; k <= k_hi; ++k)
        w[static_cast<std::size_t>(k)] =
            w[static_cast<std::size_t>(k - 1)] * big_l / static_cast<double>(k);
    for (long long k = m - 1; k >= k_lo; --k)
        w[static_cast<std::size_t>(k)] =
            w[static_cast<std::size_t>(k + 1)] * static_cast<double>(k + 1) / big_l;
    std::vector<double> p(n_st, 0.0);
    p[0] = 1.0;
    double absorbed = 0.0;
    double acc = 0.0;
    double total_w = 0.0;
    for (long long k = 0; k <= k_hi; ++k) {
        if (k > 0) {
            double prev_move = 0.0;
            for (std::size_t s = 0; s < n_st; ++s) {
                const double mv = p[s] * go[s];
                p[s] = p[s] * stay[s] + prev_move;
                prev_move = mv;
            }
            absorbed = absorbed + prev_move;
        }
        if (k >= k_lo) {
            acc += w[static_cast<std::size_t>(k)] * absorbed;
            total_w += w[static_cast<std::size_t>(k)];
        }
    }
    return std::min(1.0, std::max(0.0, acc / total_w));
}

// ------------------------------------------------------------------ passive or aggressive

PlacementDecision place_or_cross(double size, double spread, double fill_prob, double adverse_move,
                                 double eta, double alpha, double weight) {
    require(std::isfinite(size) && size > 0.0, "size must be finite and > 0");
    require(std::isfinite(spread) && spread >= 0.0, "spread must be finite and >= 0");
    require(std::isfinite(fill_prob) && fill_prob >= 0.0 && fill_prob <= 1.0,
            "fill_prob must be in [0, 1]");
    require(std::isfinite(adverse_move) && adverse_move >= 0.0,
            "adverse_move must be finite and >= 0");
    require(std::isfinite(eta) && eta >= 0.0, "eta must be finite and >= 0");
    require(std::isfinite(alpha) && alpha > 0.0 && alpha <= 1.0, "alpha must be in (0, 1]");
    require(std::isfinite(weight) && weight > 0.0, "weight must be finite and > 0");
    const double hs = 0.5 * spread;
    PlacementDecision d;
    d.impact_cost = eta * std::pow(weight, -alpha) * std::pow(size, alpha);
    d.aggressive_cost = hs + d.impact_cost;
    d.passive_cost = fill_prob * (d.impact_cost - hs) +
                     (1.0 - fill_prob) * (adverse_move + hs + d.impact_cost);
    const double denom = adverse_move + spread;
    d.breakeven_fill_prob = denom > 0.0 ? adverse_move / denom : 1.0;
    d.passive = d.passive_cost < d.aggressive_cost;
    return d;
}

// ------------------------------------------------------------------ liquidation

double LiquidationPolicy::expected_cost() const {
    require(!cost.empty() && !cost.front().empty(), "expected_cost of an empty policy");
    return cost.front().back();
}

LiquidationParams::LiquidationParams(int total_units_, int n_periods_, double dt_, double sigma_,
                                     double spread_, double kappa_, double p_touch_, double beta_,
                                     double phi_)
    : total_units(total_units_),
      n_periods(n_periods_),
      dt(dt_),
      sigma(sigma_),
      spread(spread_),
      kappa(kappa_),
      p_touch(p_touch_),
      beta(beta_),
      phi(phi_) {
    require(1 <= total_units && total_units <= kMaxLiqUnits,
            "total_units must be an integer in [1, " + std::to_string(kMaxLiqUnits) + "]");
    require(1 <= n_periods && n_periods <= kMaxLiqPeriods,
            "n_periods must be an integer in [1, " + std::to_string(kMaxLiqPeriods) + "]");
    require(std::isfinite(dt) && dt > 0.0, "dt must be finite and > 0");
    require(std::isfinite(sigma) && sigma >= 0.0, "sigma must be finite and >= 0");
    require(std::isfinite(spread) && spread >= 0.0, "spread must be finite and >= 0");
    require(std::isfinite(kappa) && kappa > 0.0, "kappa must be finite and > 0");
    require(std::isfinite(p_touch) && p_touch > 0.0 && p_touch <= 1.0,
            "p_touch must be in (0, 1]");
    require(std::isfinite(beta) && beta >= 0.0, "beta must be finite and >= 0");
    require(std::isfinite(phi) && phi >= 0.0, "phi must be finite and >= 0");
}

LiquidationPolicy liquidation_policy(const LiquidationParams& p) {
    const auto n_per = static_cast<std::size_t>(p.n_periods);
    const auto big_q = static_cast<std::size_t>(p.total_units);
    const double hs = 0.5 * p.spread;
    LiquidationPolicy pol;
    pol.cost.assign(n_per + 1, std::vector<double>(big_q + 1, 0.0));
    pol.depth.assign(n_per, std::vector<double>(big_q + 1, 0.0));
    pol.market.assign(n_per, std::vector<bool>(big_q + 1, false));
    for (std::size_t q = 0; q <= big_q; ++q) {
        const auto qd = static_cast<double>(q);
        pol.cost[n_per][q] = qd * (hs + p.beta * qd);
    }
    const double inv_k = 1.0 / p.kappa;
    for (std::size_t j = n_per; j-- > 0;) {
        for (std::size_t q = 1; q <= big_q; ++q) {
            const double c_hold = pol.cost[j + 1][q];
            const double c_less = pol.cost[j + 1][q - 1];
            const double g = hs + (c_hold - c_less);
            const double d = std::max(0.0, inv_k - g);
            const double gain = p.p_touch * std::exp(-p.kappa * d) * (d + g);
            const double post = c_hold - gain;
            const double mkt = hs + p.beta + c_less;
            double base = post;
            if (mkt < post) {
                pol.market[j][q] = true;
                base = mkt;
            } else {
                pol.depth[j][q] = d;
            }
            const auto qd = static_cast<double>(q);
            pol.cost[j][q] = p.phi * qd * qd * p.dt + base;
        }
    }
    return pol;
}

std::vector<int> aggressive_schedule(int total_units, int n_periods) {
    require(1 <= total_units && total_units <= kMaxLiqUnits,
            "total_units must be an integer in [1, " + std::to_string(kMaxLiqUnits) + "]");
    require(1 <= n_periods && n_periods <= kMaxLiqPeriods,
            "n_periods must be an integer in [1, " + std::to_string(kMaxLiqPeriods) + "]");
    std::vector<int> out(static_cast<std::size_t>(n_periods));
    for (int j = 0; j < n_periods; ++j)
        out[static_cast<std::size_t>(j)] =
            (total_units * (j + 1)) / n_periods - (total_units * j) / n_periods;
    return out;
}

double aggressive_cost(const LiquidationParams& p) {
    const double hs = 0.5 * p.spread;
    double total = 0.0;
    for (int n : aggressive_schedule(p.total_units, p.n_periods)) {
        const auto nd = static_cast<double>(n);
        total += nd * (hs + p.beta * nd);
    }
    return total;
}

LiquidationSimResult simulate_liquidation(const LiquidationParams& p, int n_paths,
                                          std::uint64_t seed) {
    require(2 <= n_paths && n_paths <= kMaxPaths,
            "n_paths must be an integer in [2, " + std::to_string(kMaxPaths) + "]");
    const auto pol = liquidation_policy(p);
    const auto sched = aggressive_schedule(p.total_units, p.n_periods);
    const double hs = 0.5 * p.spread;
    const double srt = p.sigma * std::sqrt(p.dt);
    const auto n_per = static_cast<std::size_t>(p.n_periods);
    const auto big_q = static_cast<std::size_t>(p.total_units);
    std::vector<std::vector<double>> fill_at(n_per, std::vector<double>(big_q + 1, 0.0));
    for (std::size_t j = 0; j < n_per; ++j)
        for (std::size_t q = 0; q <= big_q; ++q)
            fill_at[j][q] = p.p_touch * std::exp(-p.kappa * pol.depth[j][q]);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::normal_distribution<double> normal(0.0, 1.0);
    const auto m = static_cast<std::size_t>(n_paths);
    std::vector<double> is_opt(m), is_agg(m), is_pas(m), sav_a(m), sav_p(m);
    for (std::size_t i = 0; i < m; ++i) {
        double s = 0.0;
        std::size_t q_opt = big_q;
        std::size_t q_pas = big_q;
        double pr_opt = 0.0;
        double pr_agg = 0.0;
        double pr_pas = 0.0;
        for (std::size_t j = 0; j < n_per; ++j) {
            const double u = unif(rng);
            const double z = normal(rng);
            if (q_opt > 0) {
                if (pol.market[j][q_opt]) {
                    pr_opt += s - hs - p.beta;
                    --q_opt;
                } else if (u < fill_at[j][q_opt]) {
                    pr_opt += s + hs + pol.depth[j][q_opt];
                    --q_opt;
                }
            }
            if (q_pas > 0 && u < p.p_touch) {
                pr_pas += s + hs;
                --q_pas;
            }
            const auto nj = static_cast<double>(sched[j]);
            if (sched[j] > 0) pr_agg += nj * (s - hs - p.beta * nj);
            s += srt * z;
        }
        const auto qo = static_cast<double>(q_opt);
        const auto qp = static_cast<double>(q_pas);
        pr_opt += qo * (s - hs - p.beta * qo);
        pr_pas += qp * (s - hs - p.beta * qp);
        is_opt[i] = -pr_opt;
        is_agg[i] = -pr_agg;
        is_pas[i] = -pr_pas;
        sav_a[i] = is_agg[i] - is_opt[i];
        sav_p[i] = is_pas[i] - is_opt[i];
    }
    const double root = std::sqrt(static_cast<double>(m));
    const Stat so = stats_of(is_opt);
    const Stat sa = stats_of(is_agg);
    const Stat sp = stats_of(is_pas);
    const Stat va = stats_of(sav_a);
    const Stat vp = stats_of(sav_p);
    LiquidationSimResult r;
    r.optimal_mean = so.mean;
    r.optimal_se = so.sd / root;
    r.optimal_std = so.sd;
    r.aggressive_mean = sa.mean;
    r.aggressive_se = sa.sd / root;
    r.aggressive_std = sa.sd;
    r.passive_mean = sp.mean;
    r.passive_se = sp.sd / root;
    r.passive_std = sp.sd;
    r.saving_vs_aggressive_mean = va.mean;
    r.saving_vs_aggressive_se = va.sd / root;
    r.saving_vs_passive_mean = vp.mean;
    r.saving_vs_passive_se = vp.sd / root;
    r.n_paths = n_paths;
    return r;
}

}  // namespace etrade
