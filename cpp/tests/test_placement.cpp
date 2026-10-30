/// \file test_placement.cpp
/// \brief Limit-order placement (API_SPEC Section 13): every case of
/// data/golden/placement_golden.json, the queue fill probability against the
/// Poisson tail and its monotonicity, the passive-or-aggressive rule, the
/// liquidation policy against its Monte Carlo, validation.
#ifndef ETRADE_DATA_DIR
#define ETRADE_DATA_DIR "../data"
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "etrade/impact.hpp"
#include "etrade/placement.hpp"
#include "mini_json.hpp"

namespace {

using mini_json::Value;

const Value& placement_golden() {
    static const Value doc =
        mini_json::parse_file(std::string(ETRADE_DATA_DIR) + "/golden/placement_golden.json");
    return doc;
}

double num(const Value& o, const std::string& k) { return o.at(k).number; }
int inum(const Value& o, const std::string& k) { return static_cast<int>(num(o, k)); }

etrade::LiquidationParams liq_from(const Value& in) {
    return etrade::LiquidationParams(inum(in, "total_units"), inum(in, "n_periods"), num(in, "dt"),
                                     num(in, "sigma"), num(in, "spread"), num(in, "kappa"),
                                     num(in, "p_touch"), num(in, "beta"), num(in, "phi"));
}

etrade::LiquidationParams patient(double sigma = 0.5, int q = 10, int n = 50) {
    return etrade::LiquidationParams(q, n, 0.02, sigma, 0.02, 30.0, 0.3, 0.005, 0.0);
}

/// P(Poisson(lam) >= n) by direct summation of the lower tail (small n).
double poisson_sf(int n, double lam) {
    double term = std::exp(-lam);
    double cdf = 0.0;
    for (int k = 0; k < n; ++k) {
        cdf += term;
        term *= lam / static_cast<double>(k + 1);
    }
    return 1.0 - cdf;
}

std::vector<std::pair<std::string, double>> exact_actual(const Value& c) {
    const std::string kind = c.at("kind").str;
    const Value& in = c.at("inputs");
    std::vector<std::pair<std::string, double>> out;
    if (kind == "fill_probability" || kind == "placement_decision") {
        const double fp = etrade::queue_fill_probability(
            inum(in, "q_ahead"), inum(in, "size"), num(in, "mu"), num(in, "theta"),
            num(in, "horizon"));
        out.emplace_back("fill_prob", fp);
        if (kind == "fill_probability") return out;
        const auto d = etrade::place_or_cross(num(in, "child_shares"), num(in, "spread"), fp,
                                              num(in, "adverse_move"), num(in, "eta"),
                                              num(in, "alpha"), num(in, "weight"));
        out.emplace_back("passive_cost", d.passive_cost);
        out.emplace_back("aggressive_cost", d.aggressive_cost);
        out.emplace_back("impact_cost", d.impact_cost);
        out.emplace_back("breakeven_fill_prob", d.breakeven_fill_prob);
        out.emplace_back("passive", d.passive ? 1.0 : 0.0);
        return out;
    }
    // liquidation_policy
    const auto p = liq_from(in);
    const auto pol = etrade::liquidation_policy(p);
    int n_market = 0;
    for (const auto& row : pol.market)
        for (std::size_t q = 1; q < row.size(); ++q) n_market += row[q] ? 1 : 0;
    out.emplace_back("dp_cost", pol.expected_cost());
    out.emplace_back("depth_first", pol.depth.front().back());
    out.emplace_back("depth_last", pol.depth.back()[1]);
    out.emplace_back("n_market", static_cast<double>(n_market));
    out.emplace_back("aggressive_cost", etrade::aggressive_cost(p));
    return out;
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

TEST(PlacementGolden, CaseCount) { EXPECT_EQ(placement_golden().at("cases").arr.size(), 11u); }

TEST(PlacementGolden, EveryCaseEveryKey) {
    for (const auto& c : placement_golden().at("cases").arr) {
        const std::string kind = c.at("kind").str;
        const Value& expect = c.at("expect");
        const double tol = c.at("tol").number;
        const std::string name = c.at("name").str;
        if (kind == "liquidation_sim") {
            const Value& in = c.at("inputs");
            const auto r = etrade::simulate_liquidation(
                liq_from(in), inum(in, "n_paths_ref"), static_cast<std::uint64_t>(num(in, "seed")));
            const std::vector<std::pair<std::string, std::pair<double, double>>> got = {
                {"optimal_mean", {r.optimal_mean, r.optimal_se}},
                {"aggressive_mean", {r.aggressive_mean, r.aggressive_se}},
                {"passive_mean", {r.passive_mean, r.passive_se}},
                {"saving_vs_aggressive_mean",
                 {r.saving_vs_aggressive_mean, r.saving_vs_aggressive_se}},
                {"saving_vs_passive_mean", {r.saving_vs_passive_mean, r.saving_vs_passive_se}}};
            ASSERT_EQ(expect.obj.size(), 2 * got.size()) << name;
            for (const auto& g : got) {
                const double se_ref = num(expect, g.first + "_se");
                const double bound =
                    tol * std::sqrt(se_ref * se_ref + g.second.second * g.second.second);
                EXPECT_LE(std::abs(g.second.first - num(expect, g.first)), bound)
                    << name << "." << g.first << ": got " << g.second.first << " want "
                    << num(expect, g.first);
            }
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

// ------------------------------------------------------------------ fill probability

TEST(QueueFill, NoCancellationsIsPoissonTail) {
    EXPECT_NEAR(etrade::queue_fill_probability(30, 5, 10.0, 0.0, 2.0) / poisson_sf(35, 20.0), 1.0,
                1e-10);
    EXPECT_NEAR(etrade::queue_fill_probability(0, 1, 3.0, 0.0, 0.4), 1.0 - std::exp(-1.2), 1e-14);
}

TEST(QueueFill, DecreasesWithQueuePositionAndSize) {
    double prev = 2.0;
    for (int q = 0; q < 120; q += 5) {
        const double v = etrade::queue_fill_probability(q, 10, 20.0, 0.05, 3.0);
        EXPECT_LT(v, prev) << q;
        prev = v;
    }
    prev = 2.0;
    for (int n = 1; n < 40; n += 3) {
        const double v = etrade::queue_fill_probability(20, n, 20.0, 0.05, 3.0);
        EXPECT_LT(v, prev) << n;
        prev = v;
    }
}

TEST(QueueFill, IncreasesWithHorizonAndCancellations) {
    double prev = -1.0;
    for (int i = 0; i < 30; ++i) {
        const double h = 1.0 + 3.0 * i / 29.0;
        const double v = etrade::queue_fill_probability(50, 10, 20.0, 0.05, h);
        EXPECT_GT(v, prev) << h;
        prev = v;
    }
    prev = -1.0;
    for (double t : {0.0, 0.05, 0.2, 1.0}) {
        const double v = etrade::queue_fill_probability(50, 10, 20.0, t, 2.0);
        EXPECT_GT(v, prev) << t;
        prev = v;
    }
}

TEST(QueueFill, Edges) {
    EXPECT_EQ(etrade::queue_fill_probability(10, 3, 5.0, 0.2, 0.0), 0.0);
    EXPECT_EQ(etrade::queue_fill_probability(0, 1, 1e3, 0.0, 100.0), 1.0);
    EXPECT_LE(etrade::queue_fill_probability(4999, 1, 1.0, 0.0, 1.0), 1e-300);
}

// ------------------------------------------------------------------ passive or aggressive

TEST(PlaceOrCross, MatchesSection8AndClosedFormDifference) {
    const auto d = etrade::place_or_cross(2000.0, 0.02, 0.7, 0.01, 1e-4, 0.5, 1.0 / 13.0);
    const etrade::ImpactParams ip(2000.0, 1, 1.0 / 13.0, 0.0, 1e-4, 0.5, 0.0, 0.0);
    EXPECT_NEAR(d.impact_cost / (etrade::impact_expected_cost(ip, {2000.0, 0.0}) / 2000.0), 1.0,
                1e-12);
    EXPECT_NEAR(d.passive_cost - d.aggressive_cost, 0.3 * 0.01 - 2.0 * 0.7 * 0.01, 1e-15);
    EXPECT_TRUE(d.passive);
}

TEST(PlaceOrCross, AggressiveWhenSpreadNegligible) {
    for (double fp : {0.0, 0.2, 0.5, 0.9, 1.0})
        for (double adv : {0.0, 0.004, 0.05})
            EXPECT_FALSE(etrade::place_or_cross(500.0, 0.0, fp, adv, 1e-4, 0.5, 0.1).passive)
                << fp << " " << adv;
}

TEST(PlaceOrCross, BreakevenSeparatesChoices) {
    const double adv = 0.015;
    const double spread = 0.01;
    for (int i = 0; i <= 40; ++i) {
        const double fp = i / 40.0;
        const auto d = etrade::place_or_cross(100.0, spread, fp, adv, 1e-4, 0.6, 0.2);
        EXPECT_NEAR(d.breakeven_fill_prob, adv / (adv + spread), 1e-15);
        if (std::abs(fp - d.breakeven_fill_prob) > 1e-9)
            EXPECT_EQ(d.passive, fp > d.breakeven_fill_prob) << fp;
    }
    EXPECT_EQ(etrade::place_or_cross(100.0, 0.0, 0.5, 0.0, 1e-4, 0.6, 0.2).breakeven_fill_prob,
              1.0);
}

// ------------------------------------------------------------------ liquidation

TEST(Liquidation, OnePeriodClosedForm) {
    const auto pol = etrade::liquidation_policy(patient(0.5, 1, 1));
    const double g = 0.01 + 0.015;
    const double d = 1.0 / 30.0 - g;
    EXPECT_NEAR(pol.depth[0][1], d, 1e-15);
    EXPECT_NEAR(pol.expected_cost(), 0.015 - 0.3 * std::exp(-30.0 * d) / 30.0, 1e-15);
    EXPECT_FALSE(pol.market[0][1]);
}

TEST(Liquidation, DepthFallsWithInventory) {
    const auto pol = etrade::liquidation_policy(patient());
    for (std::size_t j = 0; j < pol.depth.size(); ++j) {
        double prev = 1e300;
        for (std::size_t q = 1; q < pol.depth[j].size(); ++q) {
            if (pol.market[j][q]) continue;
            EXPECT_LE(pol.depth[j][q], prev + 1e-15) << j << " " << q;
            prev = pol.depth[j][q];
        }
    }
}

TEST(Liquidation, MonteCarloMatchesDpAndBeatsBaselines) {
    const auto p = patient();
    const auto r = etrade::simulate_liquidation(p, 20000, 5);
    EXPECT_LE(std::abs(r.optimal_mean - etrade::liquidation_policy(p).expected_cost()),
              4.0 * r.optimal_se);
    EXPECT_LE(std::abs(r.aggressive_mean - etrade::aggressive_cost(p)), 4.0 * r.aggressive_se);
    EXPECT_GT(r.saving_vs_aggressive_mean, 4.0 * r.saving_vs_aggressive_se);
    EXPECT_GT(r.saving_vs_passive_mean, 4.0 * r.saving_vs_passive_se);
}

TEST(Liquidation, DeterministicWithoutVolatility) {
    const auto p = patient(0.0, 4, 4);
    const auto r = etrade::simulate_liquidation(p, 50, 1);
    EXPECT_LE(r.aggressive_std, 1e-15);
    EXPECT_NEAR(r.aggressive_mean, etrade::aggressive_cost(p), 1e-15);
}

TEST(Liquidation, Deterministic) {
    const auto a = etrade::simulate_liquidation(patient(), 200, 9);
    const auto b = etrade::simulate_liquidation(patient(), 200, 9);
    EXPECT_EQ(a.optimal_mean, b.optimal_mean);
    EXPECT_EQ(a.optimal_std, b.optimal_std);
    EXPECT_EQ(a.aggressive_mean, b.aggressive_mean);
    EXPECT_EQ(a.passive_mean, b.passive_mean);
    EXPECT_EQ(a.saving_vs_aggressive_se, b.saving_vs_aggressive_se);
    EXPECT_EQ(a.saving_vs_passive_se, b.saving_vs_passive_se);
}

TEST(Liquidation, AggressiveSchedule) {
    EXPECT_EQ(etrade::aggressive_schedule(10, 4), (std::vector<int>{2, 3, 2, 3}));
    EXPECT_EQ(etrade::aggressive_schedule(3, 5), (std::vector<int>{0, 1, 0, 1, 1}));
    int total = 0;
    for (int n : etrade::aggressive_schedule(997, 50)) total += n;
    EXPECT_EQ(total, 997);
}

// ------------------------------------------------------------------ validation

TEST(PlacementValidation, QueueFill) {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    expect_invalid([] { (void)etrade::queue_fill_probability(-1, 1, 1.0, 0.0, 1.0); }, "q_ahead");
    expect_invalid([] { (void)etrade::queue_fill_probability(0, 0, 1.0, 0.0, 1.0); }, "size");
    expect_invalid([] { (void)etrade::queue_fill_probability(4999, 2, 1.0, 0.0, 1.0); },
                   "q_ahead + size");
    expect_invalid([] { (void)etrade::queue_fill_probability(0, 1, 0.0, 0.0, 1.0); }, "mu");
    expect_invalid([inf] { (void)etrade::queue_fill_probability(0, 1, inf, 0.0, 1.0); }, "mu");
    expect_invalid([] { (void)etrade::queue_fill_probability(0, 1, 1.0, -0.1, 1.0); }, "theta");
    expect_invalid([nan] { (void)etrade::queue_fill_probability(0, 1, 1.0, nan, 1.0); }, "theta");
    expect_invalid([] { (void)etrade::queue_fill_probability(0, 1, 1.0, 0.0, -1.0); }, "horizon");
    expect_invalid([inf] { (void)etrade::queue_fill_probability(0, 1, 1.0, 0.0, inf); }, "horizon");
    expect_invalid([] { (void)etrade::queue_fill_probability(0, 1, 1e6, 0.0, 1.0); },
                   "horizon too long");
    expect_invalid([] { (void)etrade::queue_fill_probability(0, 1, 1e300, 0.0, 1e300); },
                   "horizon too long");
}

TEST(PlacementValidation, PlaceOrCross) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<std::pair<std::size_t, std::pair<double, std::string>>> bad = {
        {0, {0.0, "size"}},        {1, {-0.01, "spread"}},     {2, {1.1, "fill_prob"}},
        {2, {nan, "fill_prob"}},   {3, {-1.0, "adverse_move"}}, {4, {-1.0, "eta"}},
        {5, {0.0, "alpha"}},       {5, {1.5, "alpha"}},         {6, {0.0, "weight"}}};
    for (const auto& b : bad) {
        std::vector<double> a = {100.0, 0.01, 0.5, 0.01, 1e-4, 0.5, 0.1};
        a[b.first] = b.second.first;
        expect_invalid([&a] { (void)etrade::place_or_cross(a[0], a[1], a[2], a[3], a[4], a[5], a[6]); },
                       b.second.second);
    }
}

TEST(PlacementValidation, LiquidationParams) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    using LP = etrade::LiquidationParams;
    expect_invalid([] { (void)LP(0, 50, 0.02, 0.5, 0.02, 30.0, 0.3, 0.005); }, "total_units");
    expect_invalid([] { (void)LP(1001, 50, 0.02, 0.5, 0.02, 30.0, 0.3, 0.005); }, "total_units");
    expect_invalid([] { (void)LP(10, 0, 0.02, 0.5, 0.02, 30.0, 0.3, 0.005); }, "n_periods");
    expect_invalid([] { (void)LP(10, 1001, 0.02, 0.5, 0.02, 30.0, 0.3, 0.005); }, "n_periods");
    expect_invalid([] { (void)LP(10, 50, 0.0, 0.5, 0.02, 30.0, 0.3, 0.005); }, "dt");
    expect_invalid([] { (void)LP(10, 50, 0.02, -1.0, 0.02, 30.0, 0.3, 0.005); }, "sigma");
    expect_invalid([nan] { (void)LP(10, 50, 0.02, 0.5, nan, 30.0, 0.3, 0.005); }, "spread");
    expect_invalid([] { (void)LP(10, 50, 0.02, 0.5, 0.02, 0.0, 0.3, 0.005); }, "kappa");
    expect_invalid([] { (void)LP(10, 50, 0.02, 0.5, 0.02, 30.0, 0.0, 0.005); }, "p_touch");
    expect_invalid([] { (void)LP(10, 50, 0.02, 0.5, 0.02, 30.0, 1.01, 0.005); }, "p_touch");
    expect_invalid([] { (void)LP(10, 50, 0.02, 0.5, 0.02, 30.0, 0.3, -1.0); }, "beta");
    expect_invalid([inf] { (void)LP(10, 50, 0.02, 0.5, 0.02, 30.0, 0.3, 0.005, inf); }, "phi");
    expect_invalid([] { (void)etrade::simulate_liquidation(patient(), 1); }, "n_paths");
    expect_invalid([] { (void)etrade::aggressive_schedule(0, 5); }, "total_units");
}
