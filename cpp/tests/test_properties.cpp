/// \file test_properties.cpp
/// \brief Seeded randomized property checks: schedule invariants, the IS
/// decomposition identity and price-shift invariance, monotone impact costs,
/// bps linearity, the Section 7 conversions, and "validation never crashes"
/// (every call succeeds or throws std::invalid_argument / std::domain_error).
#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "etrade/almgren_chriss.hpp"
#include "etrade/analytics.hpp"
#include "etrade/avellaneda_stoikov.hpp"
#include "etrade/impact.hpp"
#include "etrade/market.hpp"

namespace {

constexpr int kIters = 300;

struct Rng {
    std::mt19937_64 gen;
    explicit Rng(std::uint64_t seed) : gen(seed) {}
    double uni(double lo, double hi) { return std::uniform_real_distribution<double>(lo, hi)(gen); }
    double log_uni(double lo, double hi) { return std::exp(uni(std::log(lo), std::log(hi))); }
    int integer(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(gen); }
    bool coin() { return integer(0, 1) == 1; }
};

double rel_err(double a, double b) { return std::abs(a - b) / std::max(1.0, std::abs(b)); }

void check_schedule(const std::vector<double>& x, double total) {
    ASSERT_GE(x.size(), 2u);
    EXPECT_LE(std::abs(x.front() - total), 1e-15 * total);  // linear branch rounds X*N/N
    EXPECT_EQ(x.back(), 0.0);
    double sum = 0.0;
    for (std::size_t j = 1; j < x.size(); ++j) {
        EXPECT_LE(x[j], x[j - 1] + 1e-12 * std::max(1.0, total)) << "j=" << j;
        sum += x[j - 1] - x[j];
    }
    EXPECT_LE(std::abs(sum - total), 1e-9 * std::max(1.0, total));
}

etrade::ACParams random_ac(Rng& r, double total) {
    const double tau = r.log_uni(0.01, 2.0);
    const double eta = r.log_uni(1e-7, 1e-4);
    const double gamma_p = r.uni(0.0, 0.95) * 2.0 * eta / tau;
    return etrade::ACParams(total, r.integer(1, 200), tau, r.uni(0.0, 5.0), eta, gamma_p,
                            r.coin() ? 0.0 : r.log_uni(1e-9, 1e-3));
}

/// Random valid impact params (convexity condition holds by construction).
etrade::ImpactParams random_impact(Rng& r, double total, double gamma_frac) {
    const int n = r.integer(1, 20);
    const double tau = 1.0 / n;
    const double alpha = r.uni(0.3, 1.0);
    const double eta = r.log_uni(1e-4, 1e-1);
    std::vector<double> w;
    for (int j = 0; j < n; ++j) w.push_back(r.uni(0.5, 1.5) * tau);
    double bound = std::numeric_limits<double>::infinity();
    for (double wj : w) {
        bound = std::min(bound, alpha * (1.0 + alpha) * eta * std::pow(wj, -alpha) *
                                    std::pow(total, alpha - 1.0));
    }
    return etrade::ImpactParams(total, n, tau, r.uni(0.0, 2.0), eta, alpha, gamma_frac * bound,
                                r.coin() ? 0.0 : r.log_uni(1e-9, 1e-5), 0.0, w);
}

template <class F>
void must_not_crash(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
    } catch (const std::domain_error&) {
    }
}

}  // namespace

TEST(Properties, AcTrajectoryInvariants) {
    Rng r(1);
    for (int it = 0; it < kIters; ++it) {
        const double total = r.coin() ? r.log_uni(1.0, 1e8) : 0.0;
        const etrade::ACParams p = random_ac(r, total);
        check_schedule(etrade::ac_trajectory(p), total);
        check_schedule(etrade::twap_holdings(total, p.n_intervals), total);
    }
}

TEST(Properties, ImpactOptimalScheduleInvariants) {
    Rng r(2);
    for (int it = 0; it < 100; ++it) {
        const double total = r.log_uni(1e2, 1e6);
        const etrade::ImpactParams p = random_impact(r, total, r.uni(0.0, 0.9));
        const std::vector<double> x = etrade::optimal_schedule(p).holdings;
        ASSERT_EQ(static_cast<int>(x.size()), p.n_intervals() + 1);
        EXPECT_LE(rel_err(x.front(), total), 1e-12);
        EXPECT_LE(std::abs(x.back()), 1e-9 * total);
        double sum = 0.0;
        for (std::size_t j = 1; j < x.size(); ++j) {
            EXPECT_LE(x[j], x[j - 1] + 1e-9 * total);
            sum += x[j - 1] - x[j];
        }
        EXPECT_LE(std::abs(sum - total), 1e-9 * total);
    }
}

TEST(Properties, ShortfallDecompositionIdentityAndShiftInvariance) {
    Rng r(3);
    for (int it = 0; it < kIters; ++it) {
        const etrade::Side side = r.coin() ? etrade::Side::kBuy : etrade::Side::kSell;
        const double sign = side == etrade::Side::kSell ? 1.0 : -1.0;
        const double pd = r.uni(10.0, 200.0);
        const double p0 = pd * r.uni(0.98, 1.02);
        const double pt = pd * r.uni(0.95, 1.05);
        const double total = r.log_uni(1.0, 1e6);
        const int nf = r.integer(0, 8);
        std::vector<double> qty, px;
        double left = total;
        for (int j = 0; j < nf; ++j) {
            const double nj = left * r.uni(0.0, 0.5);
            left -= nj;
            qty.push_back(nj);
            px.push_back(p0 * r.uni(0.99, 1.01));
        }
        const etrade::ISDecomposition d =
            etrade::decompose_shortfall(side, pd, p0, pt, total, qty, px);
        double proceeds = 0.0, abs_sum = 0.0;
        for (std::size_t j = 0; j < qty.size(); ++j) proceeds += qty[j] * px[j];
        abs_sum = total * pd;
        const double scale = std::max(1.0, abs_sum);
        EXPECT_LE(std::abs(d.total - (d.delay + d.trading + d.opportunity)), 1e-9 * scale);
        const double closed = sign * (total * pd - proceeds - d.remaining_qty * pt);
        EXPECT_LE(std::abs(d.total - closed), 1e-9 * scale);

        const double c = r.uni(-1000.0, 1000.0);
        std::vector<double> px_c(px);
        for (double& v : px_c) v += c;
        const etrade::ISDecomposition e =
            etrade::decompose_shortfall(side, pd + c, p0 + c, pt + c, total, qty, px_c);
        const double tol = 1e-9 * std::max(1.0, total * (pd + std::abs(c)));
        EXPECT_LE(std::abs(e.delay - d.delay), tol);
        EXPECT_LE(std::abs(e.trading - d.trading), tol);
        EXPECT_LE(std::abs(e.opportunity - d.opportunity), tol);
        EXPECT_LE(std::abs(e.total - d.total), tol);
    }
}

TEST(Properties, ImpactCostsNonDecreasingInSize) {
    Rng r(4);
    for (int it = 0; it < kIters; ++it) {
        const double x1 = r.log_uni(1.0, 1e6);
        const double x2 = x1 * r.uni(1.0, 10.0);
        const double vol = r.uni(0.005, 0.05);
        const double adv = r.log_uni(1e5, 1e8);
        EXPECT_LE(etrade::sqrt_law_cost_bps(vol, x1, adv) * x1,
                  etrade::sqrt_law_cost_bps(vol, x2, adv) * x2);

        const etrade::ACParams base = random_ac(r, x2);
        const etrade::ACParams p1(x1, base.n_intervals, base.tau, base.sigma, base.eta,
                                  base.gamma_p, base.lambda_ra);
        EXPECT_LE(etrade::instantaneous_cost(p1).first, etrade::instantaneous_cost(base).first);
        EXPECT_LE(etrade::expected_cost(p1, etrade::twap_holdings(x1, base.n_intervals)),
                  etrade::expected_cost(base, etrade::twap_holdings(x2, base.n_intervals)));
    }
    for (int it = 0; it < 100; ++it) {
        const double x1 = r.log_uni(1e2, 1e5);
        const double x2 = x1 * r.uni(1.0, 10.0);
        // Valid at X2 implies valid at X1 < X2 for alpha <= 1 (X^(alpha-1) decreases).
        const etrade::ImpactParams p2 = random_impact(r, x2, r.uni(0.0, 0.9));
        const etrade::ImpactParams p1(x1, p2.n_intervals(), p2.tau(), p2.sigma(), p2.eta(),
                                      p2.alpha(), p2.gamma_p(), p2.lambda_ra(), p2.drift(),
                                      p2.weights());
        const int n = p2.n_intervals();
        EXPECT_LE(etrade::impact_expected_cost(p1, etrade::twap_holdings(x1, n)),
                  etrade::impact_expected_cost(p2, etrade::twap_holdings(x2, n)) *
                      (1.0 + 1e-12));
    }
}

TEST(Properties, BpsLinearity) {
    Rng r(5);
    for (int it = 0; it < kIters; ++it) {
        const double a = r.uni(-1e6, 1e6);
        const double b = r.uni(-1e6, 1e6);
        const double k = r.uni(-100.0, 100.0);
        const double n = r.log_uni(1e2, 1e10);
        const double sum = etrade::bps(a, n) + etrade::bps(b, n);
        EXPECT_LE(std::abs(etrade::bps(a + b, n) - sum),
                  1e-12 * std::max({std::abs(etrade::bps(a, n)), std::abs(etrade::bps(b, n)),
                                    1e-300}));
        EXPECT_LE(std::abs(etrade::bps(k * a, n) - k * etrade::bps(a, n)),
                  1e-12 * std::abs(k * etrade::bps(a, n)) + 1e-300);
    }
}

TEST(Properties, MarketConversions) {
    Rng r(6);
    const double dt_choices[] = {1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 180.0, 600.0};
    for (int it = 0; it < kIters; ++it) {
        const double price = r.log_uni(1.0, 1000.0);
        const double spread = price * r.log_uni(1e-5, 1e-2);
        const double adv = r.log_uni(1e4, 1e8);
        const double vol = r.uni(0.002, 0.08);
        const double temp = r.uni(0.005, 0.05);
        const double perm = r.uni(0.05, 1.0);
        const etrade::ACMarket am{price, spread, adv, vol, r.log_uni(1.0, 1e6),
                                  r.integer(1, 50), r.uni(0.01, 1.0), r.uni(0.0, 1e-5),
                                  temp, perm};
        const etrade::ACParams ap = etrade::ac_params_from_market(am);
        EXPECT_LE(rel_err(ap.eta, spread / (temp * adv)), 1e-12);
        EXPECT_LE(rel_err(ap.gamma_p, spread / (perm * adv)), 1e-12);
        EXPECT_LE(std::abs(ap.sigma - price * vol), 1e-12 * price * vol);

        const double tick = price * r.log_uni(1e-5, 1e-3);
        const double touch = r.log_uni(10.0, 1e4);
        const double halving = r.uni(0.5, 20.0);
        const double skew = r.uni(0.1, 5.0);
        const etrade::ASMarket sm{price, tick, vol, touch, halving, skew, r.integer(0, 20),
                                  r.uni(0.0, 3.0), dt_choices[r.integer(0, 7)], 23400.0, 1.0};
        const etrade::ASParams sp = etrade::as_params_from_market(sm);
        const double at_touch = etrade::intensity(sp, tick / 2.0);
        EXPECT_LE(std::abs(at_touch - touch), 1e-9 * touch);
        const double d = r.uni(-5.0, 5.0) * tick;
        const double i0 = etrade::intensity(sp, d);
        const double i1 = etrade::intensity(sp, d + halving * tick);
        EXPECT_LE(std::abs(i1 - 0.5 * i0), 1e-9 * i0);
        const double s = price * r.uni(0.9, 1.1);
        const double shift = s - etrade::reservation_price(sp, s, 1.0, 0.0);
        EXPECT_LE(std::abs(shift - skew * tick), 1e-12 * price);  // s - r cancels
    }
}

TEST(Properties, ValidationNeverCrashes) {
    Rng r(7);
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double specials[] = {nan, inf, -inf, 1e308, -1e308, -1.0, 0.0, 1e-300, 1.0, 0.5};
    auto pick = [&]() {
        return r.integer(0, 2) == 0 ? r.uni(-10.0, 1000.0) : specials[r.integer(0, 9)];
    };
    const int ints[] = {INT_MIN, -5, -1, 0, 1, 3, 13, 10001, 10000001, INT_MAX};
    auto pick_int = [&]() { return ints[r.integer(0, 9)]; };
    // Small ints only where a valid value allocates a path of that length.
    const int small_ints[] = {INT_MIN, -1, 0, 1, 5, 10000001};
    auto pick_small = [&]() { return small_ints[r.integer(0, 5)]; };

    for (int it = 0; it < 1000; ++it) {
        must_not_crash([&] {
            (void)etrade::ACParams(pick(), pick_int(), pick(), pick(), pick(), pick(), pick());
        });
        must_not_crash([&] {
            (void)etrade::ASParams(pick(), pick(), pick(), pick(), pick(), pick(), pick(),
                                   pick_int(), pick());
        });
        must_not_crash([&] { (void)etrade::twap_holdings(pick(), pick_small()); });
        must_not_crash([&] {
            // Keep X small so a valid call allocates at most X/clip <= 1e7 points cheaply.
            const double x = r.coin() ? specials[r.integer(0, 9)] : r.uni(0.0, 50.0);
            (void)etrade::pov_holdings(std::min(x, 50.0), pick(), pick(), pick());
        });
        must_not_crash([&] {
            const int nf = r.integer(0, 3);
            std::vector<double> q, px;
            for (int j = 0; j < nf; ++j) {
                q.push_back(pick());
                px.push_back(pick());
            }
            if (r.integer(0, 4) == 0) px.push_back(pick());  // length mismatch
            (void)etrade::decompose_shortfall(r.coin() ? etrade::Side::kBuy : etrade::Side::kSell,
                                              pick(), pick(), pick(), pick(), q, px);
        });
        must_not_crash([&] { (void)etrade::bps(pick(), pick()); });
        must_not_crash([&] {
            const etrade::ACMarket m{pick(), pick(), pick(), pick(), pick(),
                                     pick_int(), pick(), pick(), pick(), pick()};
            (void)etrade::ac_params_from_market(m);
        });
        must_not_crash([&] {
            const etrade::ASMarket m{pick(), pick(), pick(), pick(), pick(), pick(),
                                     pick_int(), pick(), pick(), pick(), pick()};
            (void)etrade::as_params_from_market(m);
        });
        must_not_crash([&] {
            std::vector<double> w;
            const int n = pick_small();
            if (r.coin() && n > 0 && n <= 5) {
                for (int j = 0; j < n; ++j) w.push_back(pick());
            }
            (void)etrade::ImpactParams(pick(), n, pick(), pick(), pick(), pick(), pick(), pick(),
                                       std::abs(pick()), w);
        });
        must_not_crash([&] {
            const etrade::ImpactMarket m{pick(), pick(), pick(), pick(), pick(), pick_small(),
                                         pick(), pick(), pick(), pick(), pick(), pick(), {}};
            (void)etrade::impact_params_from_market(m);
        });
        must_not_crash([&] { (void)etrade::sqrt_law_cost_bps(pick(), pick(), pick(), pick()); });
    }
}
