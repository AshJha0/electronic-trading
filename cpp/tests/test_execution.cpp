/// \file test_execution.cpp
/// \brief Execution extensions (API_SPEC Section 12): every case of
/// data/golden/execution_golden.json, transient impact against dense algebra,
/// the Obizhaeva-Wang shape and the Almgren-Chriss limit, adaptive
/// re-planning, internalisation, validation.
#ifndef ETRADE_DATA_DIR
#define ETRADE_DATA_DIR "../data"
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "etrade/almgren_chriss.hpp"
#include "etrade/execution.hpp"
#include "etrade/impact.hpp"
#include "etrade/market.hpp"
#include "mini_json.hpp"

namespace {

using mini_json::Value;

const Value& execution_golden() {
    static const Value doc =
        mini_json::parse_file(std::string(ETRADE_DATA_DIR) + "/golden/execution_golden.json");
    return doc;
}

double num(const Value& o, const std::string& k) { return o.at(k).number; }
double num_or(const Value& o, const std::string& k, double d) {
    return o.has(k) ? o.at(k).number : d;
}

std::string two_digits(int j) { return (j < 10 ? "0" : "") + std::to_string(j); }

std::vector<double> curve() {
    return {etrade::kUsEquityVolumeCurve13.begin(), etrade::kUsEquityVolumeCurve13.end()};
}

etrade::ACParams ac10() { return etrade::ACParams(1e6, 10, 0.5, 0.95, 2.5e-6, 2.5e-7, 2e-6); }

etrade::InternalisationResult internalise_mkt(const std::vector<double>& b,
                                              const std::vector<double>& s,
                                              double spread = 0.01) {
    return etrade::internalise(b, s, 100.0, spread, 5e6, 0.015, 13, 1.0 / 13.0);
}

std::vector<std::pair<std::string, double>> exact_actual(const Value& c) {
    const std::string kind = c.at("kind").str;
    const Value& in = c.at("inputs");
    std::vector<std::pair<std::string, double>> out;
    if (kind == "transient_schedule") {
        const etrade::TransientParams p(num(in, "total_shares"),
                                        static_cast<int>(num(in, "n_intervals")), num(in, "tau"),
                                        num(in, "kappa"), num(in, "rho"), num(in, "gamma_p"));
        const auto x = etrade::transient_optimal_schedule(p);
        for (int j = 1; j < p.n_intervals; ++j)
            out.emplace_back("xfrac_" + two_digits(j),
                             x[static_cast<std::size_t>(j)] / p.total_shares);
        out.emplace_back("e_cost", etrade::transient_expected_cost(p, x));
        out.emplace_back("twap_e_cost",
                         etrade::transient_expected_cost(
                             p, etrade::twap_holdings(p.total_shares, p.n_intervals)));
        return out;
    }
    // internalise
    std::vector<double> buys, sells;
    const int nb = static_cast<int>(num(in, "n_buys"));
    const int ns = static_cast<int>(num(in, "n_sells"));
    for (int i = 0; i < nb; ++i) buys.push_back(num(in, "buy_" + std::to_string(i)));
    for (int i = 0; i < ns; ++i) sells.push_back(num(in, "sell_" + std::to_string(i)));
    const std::vector<double> w = num(in, "volume_curve") == 1.0 ? curve() : std::vector<double>{};
    const auto r = etrade::internalise(
        buys, sells, num(in, "price"), num(in, "spread"), num(in, "adv"), num(in, "daily_vol"),
        static_cast<int>(num(in, "n_intervals")), num(in, "tau"), num_or(in, "alpha", 0.5),
        num_or(in, "y", 1.0), num_or(in, "perm_frac", etrade::kDefaultPermFrac), w);
    out.emplace_back("crossed_shares", r.crossed_shares);
    out.emplace_back("residual_shares", r.residual_shares);
    out.emplace_back("baseline_cost", r.baseline_cost);
    out.emplace_back("internal_cost", r.internal_cost);
    out.emplace_back("saved_cost", r.saved_cost);
    out.emplace_back("saved_bps", r.saved_bps);
    for (std::size_t i = 0; i < r.buy_saved_bps.size(); ++i)
        out.emplace_back("buy" + std::to_string(i) + "_saved_bps", r.buy_saved_bps[i]);
    for (std::size_t i = 0; i < r.sell_saved_bps.size(); ++i)
        out.emplace_back("sell" + std::to_string(i) + "_saved_bps", r.sell_saved_bps[i]);
    return out;
}

/// Dense A = kappa*K - gamma_p*I, K_ij = a^|i-j|.
std::vector<std::vector<double>> dense_a(const etrade::TransientParams& p) {
    const auto n = static_cast<std::size_t>(p.n_intervals);
    const double a = p.decay();
    std::vector<std::vector<double>> m(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const double dist = i > j ? static_cast<double>(i - j) : static_cast<double>(j - i);
            m[i][j] = p.kappa * std::pow(a, dist) - (i == j ? p.gamma_p : 0.0);
        }
    return m;
}

/// Gaussian elimination with partial pivoting.
std::vector<double> dense_solve(std::vector<std::vector<double>> m, std::vector<double> b) {
    const std::size_t n = b.size();
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t piv = k;
        for (std::size_t i = k + 1; i < n; ++i)
            if (std::abs(m[i][k]) > std::abs(m[piv][k])) piv = i;
        std::swap(m[k], m[piv]);
        std::swap(b[k], b[piv]);
        for (std::size_t i = k + 1; i < n; ++i) {
            const double f = m[i][k] / m[k][k];
            for (std::size_t j = k; j < n; ++j) m[i][j] -= f * m[k][j];
            b[i] -= f * b[k];
        }
    }
    std::vector<double> x(n, 0.0);
    for (std::size_t k = n; k-- > 0;) {
        double s = b[k];
        for (std::size_t j = k + 1; j < n; ++j) s -= m[k][j] * x[j];
        x[k] = s / m[k][k];
    }
    return x;
}

template <class F>
void expect_invalid(F&& f, const std::string& needle) {
    try {
        f();
        ADD_FAILURE() << "no exception, expected '" << needle << "'";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find(needle), std::string::npos)
            << e.what() << " lacks '" << needle << "'";
    }
}

}  // namespace

// ------------------------------------------------------------------ goldens

TEST(ExecutionGolden, CaseCount) { EXPECT_EQ(execution_golden().at("cases").arr.size(), 8u); }

TEST(ExecutionGolden, EveryCaseEveryKey) {
    for (const auto& c : execution_golden().at("cases").arr) {
        const std::string kind = c.at("kind").str;
        const Value& expect = c.at("expect");
        const double tol = c.at("tol").number;
        const std::string name = c.at("name").str;
        if (kind == "adaptive_sim") {
            const Value& in = c.at("inputs");
            const etrade::ACParams ac(num(in, "ac_total_shares"),
                                      static_cast<int>(num(in, "ac_n_intervals")), num(in, "ac_tau"),
                                      num(in, "ac_sigma"), num(in, "ac_eta"), num(in, "ac_gamma_p"),
                                      num(in, "ac_lambda_ra"));
            const etrade::SignalParams sig(num(in, "phi"), num(in, "signal_sd"), num(in, "fill_risk"));
            const auto r = etrade::simulate_adaptive(ac, sig, static_cast<int>(num(in, "n_paths_ref")),
                                                     static_cast<std::uint64_t>(num(in, "seed")));
            const std::vector<std::pair<std::string, std::pair<double, double>>> got = {
                {"static_mean", {r.static_mean, r.static_se}},
                {"adaptive_mean", {r.adaptive_mean, r.adaptive_se}},
                {"saving_mean", {r.saving_mean, r.saving_se}}};
            int checked = 0;
            for (const auto& kv : expect.obj) {
                const std::string& key = kv.first;
                if (key.size() > 3 && key.compare(key.size() - 3, 3, "_se") == 0) continue;
                bool found = false;
                for (const auto& g : got) {
                    if (g.first != key) continue;
                    found = true;
                    ++checked;
                    const double se_ref = num(expect, key + "_se");
                    const double bound = tol * std::sqrt(se_ref * se_ref + g.second.second * g.second.second);
                    EXPECT_LE(std::abs(g.second.first - kv.second.number), bound)
                        << name << "." << key << ": got " << g.second.first << " want "
                        << kv.second.number;
                }
                EXPECT_TRUE(found) << name << "." << key;
            }
            EXPECT_GE(checked, 2) << name;
            continue;
        }
        const auto actual = exact_actual(c);
        ASSERT_EQ(actual.size(), expect.obj.size()) << name;
        for (const auto& [key, val] : actual) {
            const double want = num(expect, key);
            EXPECT_LE(std::abs(val - want), tol * std::max(1.0, std::abs(want)))
                << name << "." << key << ": got " << val << " want " << want;
        }
    }
}

// ------------------------------------------------------------------ transient

TEST(Transient, CostRecursionEqualsQuadraticForm) {
    const etrade::TransientParams p(1e6, 8, 0.25, 3e-6, 1.5, 4e-7);
    std::mt19937_64 rng(3);
    std::uniform_real_distribution<double> u(-2e5, 4e5);
    std::vector<double> x(9, 0.0);
    x[0] = 1e6;
    for (std::size_t j = 1; j < 8; ++j) x[j] = u(rng);
    std::vector<double> n(8);
    for (std::size_t j = 0; j < 8; ++j) n[j] = x[j] - x[j + 1];
    const auto a = dense_a(p);
    double q = 0.0;
    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t j = 0; j < 8; ++j) q += n[i] * a[i][j] * n[j];
    const double want = 0.5 * p.gamma_p * 1e12 + 0.5 * q;
    EXPECT_NEAR(etrade::transient_expected_cost(p, x) / want, 1.0, 1e-12);
}

TEST(Transient, OptimumMatchesDenseSolve) {
    for (double gp : {0.0, 1e-7, 4e-7}) {
        const etrade::TransientParams p(1e6, 12, 0.25, 3e-6, 1.5, gp);
        const auto z = dense_solve(dense_a(p), std::vector<double>(12, 1.0));
        double sz = 0.0;
        for (double v : z) sz += v;
        const auto x = etrade::transient_optimal_schedule(p);
        double xj = 1e6;
        for (std::size_t j = 0; j < 12; ++j) {
            EXPECT_NEAR(x[j], xj, 1e-9 * 1e6) << gp << " j=" << j;
            xj -= 1e6 * z[j] / sz;
        }
        EXPECT_EQ(x[12], 0.0);
    }
}

TEST(Transient, ObizhaevaWangShape) {
    const etrade::TransientParams p(1e6, 10, 0.1, 2e-6, 5.0, 0.0);
    const auto x = etrade::transient_optimal_schedule(p);
    std::vector<double> n(10);
    for (std::size_t j = 0; j < 10; ++j) n[j] = x[j] - x[j + 1];
    EXPECT_NEAR(n[0], n[9], 1e-9 * 1e6);
    for (std::size_t j = 2; j < 9; ++j) EXPECT_NEAR(n[j], n[1], 1e-9 * 1e6);
    EXPECT_NEAR(n[1] / n[0], 1.0 - p.decay(), 1e-12);
}

TEST(Transient, OptimumBeatsTwapAndPerturbations) {
    const etrade::TransientParams p(2.5e5, 26, 1.0 / 26.0, 4e-6, 2.0, 1e-8);
    const auto x = etrade::transient_optimal_schedule(p);
    const double e_opt = etrade::transient_expected_cost(p, x);
    EXPECT_LT(e_opt, etrade::transient_expected_cost(p, etrade::twap_holdings(2.5e5, 26)));
    std::mt19937_64 rng(4);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (int t = 0; t < 20; ++t) {
        auto z = x;
        for (std::size_t j = 1; j < 26; ++j) z[j] += nd(rng) * 1e3;
        EXPECT_GE(etrade::transient_expected_cost(p, z), e_opt);
    }
}

TEST(Transient, FastResilienceReducesToAlmgrenChriss) {
    const etrade::ACParams ac(1e6, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 0.0);
    const double kappa = 2.0 * ac.eta / ac.tau;
    const etrade::TransientParams fast(ac.total_shares, ac.n_intervals, ac.tau, kappa, 1e3,
                                       ac.gamma_p);
    EXPECT_EQ(fast.decay(), 0.0);
    const auto x_ow = etrade::transient_optimal_schedule(fast);
    const auto x_ac = etrade::ac_trajectory(ac);
    for (std::size_t j = 0; j < x_ac.size(); ++j) EXPECT_NEAR(x_ow[j], x_ac[j], 1e-9 * 1e6);
    const std::vector<std::vector<double>> paths = {
        x_ow, etrade::twap_holdings(1e6, 5), {1e6, 0.0, 0.0, 0.0, 0.0, 0.0}};
    for (const auto& x : paths)
        EXPECT_NEAR(etrade::transient_expected_cost(fast, x) / etrade::expected_cost(ac, x), 1.0,
                    1e-12);
    const auto tw = etrade::twap_holdings(1e6, 5);
    std::vector<double> gaps;
    for (double rho : {0.5, 2.0, 8.0}) {
        const etrade::TransientParams p(1e6, 5, 1.0, kappa, rho, ac.gamma_p);
        gaps.push_back(etrade::transient_expected_cost(p, tw) - etrade::expected_cost(ac, tw));
    }
    EXPECT_GT(gaps[0], gaps[1]);
    EXPECT_GT(gaps[1], gaps[2]);
    EXPECT_GT(gaps[2], 0.0);
    const etrade::ImpactParams ip(1e6, 5, 1.0, 0.95, 2.5e-6, 1.0, 2.5e-7, 0.0);
    EXPECT_NEAR(etrade::transient_expected_cost(fast, tw) / etrade::impact_expected_cost(ip, tw),
                1.0, 1e-12);
}

TEST(Transient, SingleIntervalAndZeroShares) {
    const etrade::TransientParams p(5.0, 1, 1.0, 1.0, 1.0);
    EXPECT_EQ(etrade::transient_optimal_schedule(p), (std::vector<double>{5.0, 0.0}));
    EXPECT_NEAR(etrade::transient_expected_cost(p, {5.0, 0.0}), 12.5, 1e-12);
    const etrade::TransientParams z(0.0, 4, 1.0, 1.0, 1.0);
    for (double v : etrade::transient_optimal_schedule(z)) EXPECT_EQ(v, 0.0);
}

// ------------------------------------------------------------------ adaptive

TEST(Adaptive, StaticPlanIsAlmgrenChrissTrajectory) {
    const auto ac = ac10();
    const auto plan = etrade::replan(ac, etrade::SignalParams(), ac.total_shares, 0.0, 0);
    const auto x = etrade::ac_trajectory(ac);
    ASSERT_EQ(plan.size(), x.size());
    for (std::size_t j = 0; j < x.size(); ++j) EXPECT_NEAR(plan[j], x[j], 1e-9 * 1e6);
}

TEST(Adaptive, ReplanIsTimeConsistentWithoutSignal) {
    const auto ac = ac10();
    const etrade::SignalParams sig;
    const auto plan = etrade::replan(ac, sig, ac.total_shares, 0.0, 0);
    for (int k = 1; k < 10; ++k) {
        const auto tail = etrade::replan(ac, sig, plan[static_cast<std::size_t>(k)], 0.0, k);
        for (std::size_t i = 0; i < tail.size(); ++i)
            EXPECT_NEAR(tail[i], plan[static_cast<std::size_t>(k) + i], 1e-9 * 1e6);
    }
}

TEST(Adaptive, FeedbackCoefficientsMatchReplan) {
    const auto ac = ac10();
    const etrade::SignalParams sig(0.6, 0.4, 0.0);
    const auto cd = etrade::replan_coefficients(ac, sig);
    for (int k = 0; k < 10; ++k) {
        const auto m = static_cast<std::size_t>(10 - k);
        const auto x = etrade::replan(ac, sig, 3e5, 0.7, k);
        const double want = cd.first[m] * 3e5 - cd.second[m] * 0.7;
        EXPECT_NEAR(x[0] - x[1], want, 1e-12 * std::abs(want));
    }
}

TEST(Adaptive, SignalMakesTraderHoldWhenPriceExpectedToRise) {
    const auto ac = ac10();
    const etrade::SignalParams sig(0.7, 0.5, 0.0);
    EXPECT_GT(etrade::replan(ac, sig, 5e5, 1.0, 3)[1], etrade::replan(ac, sig, 5e5, 0.0, 3)[1]);
}

TEST(Adaptive, EqualsStaticWithoutSignalOrFillRisk) {
    const auto r = etrade::simulate_adaptive(ac10(), etrade::SignalParams(), 4000, 7);
    EXPECT_LE(std::abs(r.saving_mean), 1e-9 * r.static_mean);
    EXPECT_NEAR(r.static_std / r.adaptive_std, 1.0, 1e-9);
    EXPECT_LE(std::abs(r.static_mean - etrade::expected_cost(ac10())), 4.0 * r.static_se);
}

TEST(Adaptive, HelpsWithSignalOrFillRisk) {
    for (const auto& sig : {etrade::SignalParams(0.7, 0.5, 0.0), etrade::SignalParams(0.0, 0.0, 0.5)}) {
        const auto r = etrade::simulate_adaptive(ac10(), sig, 4000, 8);
        EXPECT_GT(r.saving_mean, 4.0 * r.saving_se);
    }
}

TEST(Adaptive, Deterministic) {
    const etrade::SignalParams sig(0.5, 0.3, 0.2);
    const auto a = etrade::simulate_adaptive(ac10(), sig, 100, 5);
    const auto b = etrade::simulate_adaptive(ac10(), sig, 100, 5);
    EXPECT_EQ(a.static_mean, b.static_mean);
    EXPECT_EQ(a.static_std, b.static_std);
    EXPECT_EQ(a.adaptive_mean, b.adaptive_mean);
    EXPECT_EQ(a.adaptive_std, b.adaptive_std);
    EXPECT_EQ(a.saving_mean, b.saving_mean);
    EXPECT_EQ(a.saving_se, b.saving_se);
    EXPECT_EQ(a.n_paths, b.n_paths);
}

// ------------------------------------------------------------------ internalisation

TEST(Internalise, CrossingNeverIncreasesTotalCost) {
    std::mt19937_64 rng(11);
    std::uniform_real_distribution<double> q(1e3, 3e5);
    std::uniform_int_distribution<int> nb(0, 3), ns(1, 3);
    for (int t = 0; t < 200; ++t) {
        std::vector<double> buys, sells;
        const int b = nb(rng);
        const int s = ns(rng);
        for (int i = 0; i < b; ++i) buys.push_back(q(rng));
        for (int i = 0; i < s; ++i) sells.push_back(q(rng));
        const auto r = internalise_mkt(buys, sells);
        EXPECT_LE(r.internal_cost, r.baseline_cost);
        EXPECT_GE(r.saved_cost, 0.0);
        for (double v : r.buy_saved_bps) EXPECT_GE(v, 0.0);
        for (double v : r.sell_saved_bps) EXPECT_GE(v, 0.0);
    }
}

TEST(Internalise, Accounting) {
    const auto r = internalise_mkt({3e5, 1e5}, {2.5e5});
    EXPECT_EQ(r.crossed_shares, 2.5e5);
    EXPECT_EQ(r.residual_shares, 1.5e5);
    EXPECT_EQ(r.residual_side, "buy");
    const std::vector<double> q = {3e5, 1e5, 2.5e5};
    std::vector<double> per = r.buy_saved_bps;
    per.insert(per.end(), r.sell_saved_bps.begin(), r.sell_saved_bps.end());
    ASSERT_EQ(per.size(), 3u);
    double total = 0.0;
    for (std::size_t i = 0; i < 3; ++i) total += per[i] * q[i] * 100.0 / 1e4;
    EXPECT_NEAR(total / r.saved_cost, 1.0, 1e-12);
    EXPECT_NEAR(r.saved_bps / (1e4 * r.saved_cost / (6.5e5 * 100.0)), 1.0, 1e-15);
    const auto one = internalise_mkt({}, {1e5, 2e5});
    EXPECT_EQ(one.saved_cost, 0.0);
    EXPECT_EQ(one.residual_side, "sell");
    const auto even = internalise_mkt({2e5}, {1e5, 1e5});
    EXPECT_EQ(even.internal_cost, 0.0);
    EXPECT_EQ(even.residual_side, "none");
}

// ------------------------------------------------------------------ validation

TEST(ExecutionValidation, TransientParams) {
    const double inf = std::numeric_limits<double>::infinity();
    expect_invalid([] { (void)etrade::TransientParams(-1.0, 10, 0.1, 2e-6, 5.0, 0.0); }, "total_shares");
    expect_invalid([] { (void)etrade::TransientParams(1e6, 0, 0.1, 2e-6, 5.0, 0.0); }, "n_intervals");
    expect_invalid([] { (void)etrade::TransientParams(1e6, 1001, 0.1, 2e-6, 5.0, 0.0); }, "n_intervals");
    expect_invalid([] { (void)etrade::TransientParams(1e6, 10, 0.0, 2e-6, 5.0, 0.0); }, "tau");
    expect_invalid([] { (void)etrade::TransientParams(1e6, 10, 0.1, 0.0, 5.0, 0.0); }, "kappa");
    expect_invalid([] { (void)etrade::TransientParams(1e6, 10, 0.1, 2e-6, 0.0, 0.0); }, "rho");
    expect_invalid([inf] { (void)etrade::TransientParams(1e6, 10, 0.1, 2e-6, inf, 0.0); }, "rho");
    expect_invalid([] { (void)etrade::TransientParams(1e6, 10, 0.1, 2e-6, 5.0, -1.0); }, "gamma_p");
    expect_invalid([] { (void)etrade::TransientParams(1e6, 10, 0.1, 2e-6, 5.0, 1e-6); }, "gamma_p");
}

TEST(ExecutionValidation, TransientCostHoldings) {
    const etrade::TransientParams p(1e6, 4, 0.25, 2e-6, 5.0);
    expect_invalid([&] { (void)etrade::transient_expected_cost(p, {1.0, 0.0}); }, "length");
    expect_invalid([&] { (void)etrade::transient_expected_cost(p, {1.0, 0.8, 0.6, 0.4, 0.2}); },
                   "liquidate");
    expect_invalid([&] { (void)etrade::transient_expected_cost(p, {1.0, std::nan(""), 0.6, 0.4, 0.0}); },
                   "finite");
}

TEST(ExecutionValidation, SignalParams) {
    expect_invalid([] { (void)etrade::SignalParams(1.0); }, "phi");
    expect_invalid([] { (void)etrade::SignalParams(-0.1); }, "phi");
    expect_invalid([] { (void)etrade::SignalParams(0.0, -1.0); }, "signal_sd");
    expect_invalid([] { (void)etrade::SignalParams(0.0, 0.0, 1.0); }, "fill_risk");
    expect_invalid([] { (void)etrade::SignalParams(0.0, 0.0, std::nan("")); }, "fill_risk");
}

TEST(ExecutionValidation, Adaptive) {
    const auto ac = ac10();
    const etrade::SignalParams sig;
    expect_invalid([&] { (void)etrade::simulate_adaptive(ac, sig, 1, 0); }, "n_paths");
    expect_invalid([&] { (void)etrade::replan(ac, sig, 1.0, 0.0, 10); }, "k must");
    expect_invalid([&] { (void)etrade::replan(ac, sig, 1.0, 0.0, -1); }, "k must");
    expect_invalid([&] { (void)etrade::replan(ac, sig, std::numeric_limits<double>::infinity(), 0.0, 0); },
                   "holding");
    expect_invalid([&] { (void)etrade::replan(ac, sig, 1.0, std::nan(""), 0); }, "signal");
    const etrade::ACParams big(1.0, 1001, 1.0, 1.0, 1.0, 0.0, 0.0);
    expect_invalid([&] { (void)etrade::simulate_adaptive(big, sig); }, "n_intervals");
    expect_invalid([&] { (void)etrade::replan_coefficients(big, sig); }, "n_intervals");
}

TEST(ExecutionValidation, Internalise) {
    expect_invalid([] { (void)internalise_mkt({0.0}, {1.0}); }, "buys");
    expect_invalid([] { (void)internalise_mkt({1.0}, {std::nan("")}); }, "sells");
    expect_invalid([] { (void)internalise_mkt({}, {}); }, "at least one");
    expect_invalid([] { (void)internalise_mkt({1.0}, {2.0}, 0.0); }, "spread");
    expect_invalid([] { (void)etrade::internalise({1.0}, {2.0}, 100.0, 0.01, 5e6, 0.015, 0, 1.0); },
                   "n_intervals");
    expect_invalid([] { (void)etrade::internalise({1.0}, {2.0}, 100.0, 0.01, 5e6, 0.015, 1001, 1.0); },
                   "n_intervals");
    expect_invalid([] {
        (void)etrade::internalise(std::vector<double>(10001, 1.0), {1.0}, 100.0, 0.01, 5e6, 0.015,
                                  13, 1.0 / 13.0);
    }, "at most");
}
