/// \file test_numerics.cpp
/// \brief Standard-error formulas checked numerically (Sections 1, 2 and 9)
/// and the pinned product bounds kMaxMcWork / kMaxAsWork (API_SPEC 0).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "etrade/almgren_chriss.hpp"
#include "etrade/avellaneda_stoikov.hpp"
#include "etrade/mm.hpp"

namespace {

double sample_std(const std::vector<double>& v) {
    double m = 0.0;
    for (double x : v) m += x;
    m /= static_cast<double>(v.size());
    double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / static_cast<double>(v.size() - 1));
}

void expect_rel(double actual, double expected, double rel) {
    EXPECT_LE(std::abs(actual - expected), rel * std::max(1e-300, std::abs(expected)))
        << "actual " << actual << " expected " << expected;
}

etrade::ASParams as_base(int q_max, double sigma = 2.0, double a_rate = 140.0) {
    return etrade::ASParams(100.0, sigma, 1.0, 0.005, 0.1, 1.5, a_rate, q_max, 0.0);
}

const char* const kAsStats[] = {"mean_pnl", "mean_abs_q", "mean_max_abs_q", "rms_q",
                                "mean_fills"};

const std::vector<double>& as_path(const etrade::ASSimResult& r, const std::string& s) {
    if (s == "mean_pnl") return r.pnl;
    if (s == "mean_abs_q") return r.mean_abs_q_path;
    if (s == "mean_max_abs_q") return r.max_abs_q_path;
    if (s == "rms_q") return r.rms_q_path;
    return r.fills_path;
}

}  // namespace

// ---- Section 1 Monte Carlo -------------------------------------------------

TEST(Numerics, AcSeFormulasMatchReturnedMoments) {
    const etrade::ACParams p(1e6, 10, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6);
    const int m = 4000;
    const etrade::ISSimResult r = etrade::simulate_is(p, m, 7);
    EXPECT_EQ(r.n_paths, m);
    expect_rel(r.se_mean, std::sqrt(r.var / m), 1e-15);
    expect_rel(r.se_var, r.var * std::sqrt(2.0 / (m - 1)), 1e-15);
    // The SEs are the right size: the exact moments sit within 5 SE.
    EXPECT_LE(std::abs(r.mean - etrade::expected_cost(p)), 5.0 * r.se_mean);
    EXPECT_LE(std::abs(r.var - etrade::variance_cost(p)), 5.0 * r.se_var);
}

TEST(Numerics, AcZeroSigmaIsDeterministic) {
    const etrade::ACParams p(1e6, 8, 0.5, 0.0, 2.5e-6, 2.5e-7, 2e-6);
    const std::vector<double> x = etrade::twap_holdings(1e6, 8);
    const etrade::ISSimResult r = etrade::simulate_is(p, x, 50, 3);
    const double e = etrade::expected_cost(p, x);
    expect_rel(r.mean, e, 1e-12);
    EXPECT_LE(r.var, 1e-20 * e * e);
    EXPECT_LE(r.se_mean, 1e-10 * e);
    EXPECT_LE(r.se_var, 1e-20 * e * e);
    expect_rel(r.se_mean, std::sqrt(r.var / 50), 1e-15);
}

// ---- Section 2 ASSimResult.se ---------------------------------------------

TEST(Numerics, AsSeRecomputedFromPerPathArrays) {
    const etrade::ASSimResult r = etrade::simulate(as_base(3), 300, 11);
    const int m = r.n_paths();
    for (const char* s : kAsStats) {
        const double expected = sample_std(as_path(r, s)) / std::sqrt(static_cast<double>(m));
        EXPECT_NEAR(r.se(s), expected, 1e-12 * std::max(1.0, expected)) << s;
    }
    EXPECT_NEAR(r.std_pnl(), sample_std(r.pnl), 1e-12 * r.std_pnl());
    EXPECT_EQ(r.se("std_pnl"), r.std_pnl() / std::sqrt(2.0 * (m - 1.0)));
}

TEST(Numerics, AsNoQuotingGivesZeroSe) {
    const etrade::ASSimResult r = etrade::simulate(as_base(0), 20, 5);
    for (double v : r.pnl) EXPECT_EQ(v, 0.0);
    for (const char* s : kAsStats) EXPECT_EQ(r.se(s), 0.0) << s;
    EXPECT_EQ(r.std_pnl(), 0.0);
    EXPECT_EQ(r.se("std_pnl"), 0.0);
}

TEST(Numerics, AsSaturatedFillsAreDeterministic) {
    // sigma = 0 and a huge A: both sides fill every step, q stays 0.
    const etrade::ASParams p = as_base(1000, 0.0, 1e12);
    const etrade::ASSimResult r = etrade::simulate(p, 10, 9);
    const double n_steps = p.n_steps();
    for (double f : r.fills_path) EXPECT_EQ(f, 2.0 * n_steps);
    EXPECT_EQ(r.se("mean_fills"), 0.0);
    EXPECT_EQ(r.se("mean_abs_q"), 0.0);
    for (double v : r.pnl) EXPECT_EQ(v, r.pnl[0]);
    EXPECT_LE(r.se("mean_pnl"), 1e-14 * std::max(1.0, std::abs(r.mean_pnl())));
    const double spread = etrade::optimal_spread(p, 0.0);
    expect_rel(r.mean_pnl(), n_steps * spread, 1e-9);
}

// ---- Section 9 MMResult.se --------------------------------------------------

TEST(Numerics, MmSeRecomputedFromPerPathArrays) {
    etrade::MMConfig cfg;
    cfg.informed_prob = 0.2;
    cfg.tick = 0.01;
    const etrade::MMResult r = etrade::simulate_mm(as_base(4), 200, 13, cfg);
    const double m = r.n_paths();
    for (etrade::MMStat s : etrade::kMMStats) {
        const double expected = sample_std(r.path(s)) / std::sqrt(m);
        EXPECT_NEAR(r.se(s), expected, 1e-12 * std::max(1.0, expected));
    }
}

TEST(Numerics, MmNoQuotingGivesZeroSe) {
    etrade::MMConfig cfg;
    cfg.informed_prob = 0.5;
    const etrade::MMResult r = etrade::simulate_mm(as_base(0), 20, 5, cfg);
    for (etrade::MMStat s : etrade::kMMStats) {
        EXPECT_EQ(r.se(s), 0.0);
        EXPECT_EQ(r.mean(s), 0.0);
    }
}

TEST(Numerics, MmSaturatedFillsSeIsZero) {
    const etrade::ASParams p = as_base(1000, 0.0, 1e12);
    const etrade::MMResult r = etrade::simulate_mm(p, 10, 9);
    for (double f : r.fills) EXPECT_EQ(f, 2.0 * p.n_steps());
    EXPECT_EQ(r.se(etrade::MMStat::kFills), 0.0);
    EXPECT_EQ(r.se(etrade::MMStat::kRmsQ), 0.0);
}

// ---- Product bounds (API_SPEC 0) -------------------------------------------

TEST(Numerics, ProductBoundConstants) {
    EXPECT_EQ(etrade::kMaxMcWork, std::int64_t{1000000000});
    EXPECT_EQ(etrade::kMaxAsWork, std::int64_t{10000000000});
}

TEST(Numerics, McProductBoundRejectsJustOver) {
    // 1e7 paths * 101 intervals = 1.01e9 > 1e9: rejected before any draw.
    const etrade::ACParams p(1.0, 101, 1.0, 1.0, 1e-3, 0.0, 0.0);
    const std::vector<double> x = etrade::twap_holdings(1.0, 101);
    try {
        (void)etrade::simulate_is(p, x, etrade::kMaxPaths, 0);
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        EXPECT_STREQ(e.what(), "n_paths * n_intervals must be <= 1000000000");
    }
    // A small product runs.
    EXPECT_NO_THROW((void)etrade::simulate_is(p, x, 10, 0));
}

TEST(Numerics, AsProductBoundRejectsJustOver) {
    // n_steps = 1001, 1e7 paths: 1.001e10 > 1e10.
    const etrade::ASParams p(100.0, 1.0, 1001.0, 1.0, 0.1, 1.5, 140.0, 5, 0.0);
    ASSERT_EQ(p.n_steps(), 1001);
    try {
        (void)etrade::simulate(p, etrade::kMaxAsPaths, 0);
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        EXPECT_STREQ(e.what(), "n_paths * n_steps must be <= 10000000000");
    }
    EXPECT_NO_THROW((void)etrade::simulate(p, 2, 0));
}
