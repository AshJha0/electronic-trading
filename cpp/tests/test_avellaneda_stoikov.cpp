/// \file test_avellaneda_stoikov.cpp
/// \brief Unit and property tests for the Avellaneda-Stoikov module.
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include "etrade/avellaneda_stoikov.hpp"

namespace {

using etrade::ASParams;

/// Liquid large-cap parameter set (mirrors data/params_liquid.json).
ASParams liquid() {
    return ASParams(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, 20, 0.5);
}

TEST(ASClosedForms, ReservationPriceBelowMidWhenLong) {
    const ASParams p = liquid();
    // Long maker (q > 0) shades quotes down (selling pressure).
    EXPECT_LT(etrade::reservation_price(p, 100.0, 3.0, 0.3), 100.0);
    // Short maker shades up; flat maker quotes the mid.
    EXPECT_GT(etrade::reservation_price(p, 100.0, -3.0, 0.3), 100.0);
    EXPECT_DOUBLE_EQ(etrade::reservation_price(p, 100.0, 0.0, 0.3), 100.0);
    // At t = T the skew vanishes: r = s.
    EXPECT_DOUBLE_EQ(etrade::reservation_price(p, 100.0, 3.0, p.horizon), 100.0);
}

TEST(ASClosedForms, SpreadPositiveDecreasingToTerminalFloor) {
    const ASParams p = liquid();
    const double floor = 2.0 / p.gamma * std::log1p(p.gamma / p.k);
    double prev = std::numeric_limits<double>::infinity();
    for (double t : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        const double s = etrade::optimal_spread(p, t);
        EXPECT_GT(s, 0.0);
        EXPECT_LT(s, prev);
        EXPECT_GE(s, floor - 1e-15);
        prev = s;
    }
    EXPECT_DOUBLE_EQ(etrade::optimal_spread(p, p.horizon), floor);
}

TEST(ASClosedForms, QuotesBracketReservationPrice) {
    const ASParams p = liquid();
    const auto [bid, ask] = etrade::quotes(p, 100.0, 3.0, 0.3);
    const double r = etrade::reservation_price(p, 100.0, 3.0, 0.3);
    EXPECT_NEAR(0.5 * (bid + ask), r, 1e-12);
    EXPECT_NEAR(ask - bid, etrade::optimal_spread(p, 0.3), 1e-12);
}

TEST(ASClosedForms, IntensityAndFillProbability) {
    const ASParams p = liquid();
    // lambda decreasing in delta; a quote through the mid (delta < 0) has a
    // HIGHER intensity — documented behaviour, not an error.
    EXPECT_GT(etrade::intensity(p, -0.1), etrade::intensity(p, 0.0));
    EXPECT_GT(etrade::intensity(p, 0.0), etrade::intensity(p, 0.1));
    EXPECT_DOUBLE_EQ(etrade::intensity(p, 0.0), p.a_rate);
    const double prob = etrade::fill_probability(p, 0.6);
    EXPECT_GT(prob, 0.0);
    EXPECT_LT(prob, 1.0);
    EXPECT_NEAR(prob, 1.0 - std::exp(-etrade::intensity(p, 0.6) * p.dt), 1e-15);
}

TEST(ASClosedForms, AverageSpreadIsTimeAverage) {
    const ASParams p = liquid();
    const double avg = etrade::average_as_spread(p);
    // Midpoint of the linear-in-t spread: (spread(0) + spread(T)) / 2.
    EXPECT_NEAR(avg, 0.5 * (etrade::optimal_spread(p, 0.0) +
                            etrade::optimal_spread(p, p.horizon)),
                1e-12);
}

TEST(ASSimulation, CapRespectedAndDeterministic) {
    const ASParams p = liquid();
    const auto r1 = etrade::simulate(p, 200, 7);
    EXPECT_FALSE(r1.cap_violated);
    EXPECT_LE(r1.overall_max_abs_q, static_cast<double>(p.q_max));
    // Same seed twice -> identical results; different seed -> different.
    const auto r2 = etrade::simulate(p, 200, 7);
    for (int i = 0; i < r1.n_paths(); ++i) {
        ASSERT_DOUBLE_EQ(r1.pnl[i], r2.pnl[i]);
    }
    const auto r3 = etrade::simulate(p, 200, 8);
    EXPECT_NE(r1.mean_pnl(), r3.mean_pnl());
}

TEST(ASSimulation, TightCapBindsAndQmaxZeroQuotesNothing) {
    const ASParams tight(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, 1, 0.5);
    const auto r = etrade::simulate(tight, 100, 3);
    EXPECT_FALSE(r.cap_violated);
    EXPECT_LE(r.overall_max_abs_q, 1.0);
    EXPECT_GT(r.mean_fills(), 0.0);

    const ASParams none(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, 0, 0.5);
    const auto r0 = etrade::simulate(none, 50, 3);
    EXPECT_DOUBLE_EQ(r0.mean_fills(), 0.0);
    EXPECT_DOUBLE_EQ(r0.mean_pnl(), 0.0);  // never trades -> zero PnL
    EXPECT_DOUBLE_EQ(r0.std_pnl(), 0.0);
    EXPECT_TRUE(std::isinf(r0.sharpe()));
}

TEST(ASSimulation, InventoryControlBeatsFixedSpread) {
    // A-S skew keeps inventory much tighter than symmetric fixed-spread
    // quoting at the same average spread (statistical, wide margin).
    const ASParams p = liquid();
    const auto r_as = etrade::simulate(p, 1000, 11, etrade::Agent::kAS);
    const auto r_fx = etrade::simulate(p, 1000, 11, etrade::Agent::kFixed);
    EXPECT_LT(r_as.rms_q() * 1.5, r_fx.rms_q());
}

TEST(ASSimulation, StressRaisesPnlVolatility) {
    const ASParams p = liquid();
    const auto base = etrade::simulate(p, 1000, 5);
    const auto stress = etrade::simulate(p, 1000, 5, etrade::Agent::kAS,
                                         std::nullopt, 2.0, 0.5);
    EXPECT_GT(stress.std_pnl(), base.std_pnl());
}

TEST(ASSimulation, SeStatisticsAvailable) {
    const auto r = etrade::simulate(liquid(), 200, 1);
    for (const etrade::ASStat stat :
         {etrade::ASStat::kMeanPnl, etrade::ASStat::kStdPnl, etrade::ASStat::kMeanAbsQ,
          etrade::ASStat::kMeanMaxAbsQ, etrade::ASStat::kRmsQ, etrade::ASStat::kMeanFills}) {
        EXPECT_GT(r.se(stat), 0.0) << static_cast<int>(stat);
    }
}

// The deprecated string overload and NaN sentinel still work for one release.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
TEST(ASSimulation, DeprecatedStringSeAndNanSentinel) {
    const auto r = etrade::simulate(liquid(), 200, 1);
    EXPECT_EQ(r.se(std::string("rms_q")), r.se(etrade::ASStat::kRmsQ));
    EXPECT_EQ(r.se(std::string("std_pnl")), r.se(etrade::ASStat::kStdPnl));
    EXPECT_THROW((void)r.se(std::string("nope")), std::invalid_argument);
    const auto legacy = etrade::simulate(liquid(), 200, 3, etrade::Agent::kFixed,
                                         etrade::kDefaultFixedSpread);
    const auto modern = etrade::simulate(liquid(), 200, 3, etrade::Agent::kFixed);
    EXPECT_EQ(legacy.pnl, modern.pnl);
}
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

// Validated parameters are immutable: assignment is a compile error.
static_assert(!std::is_copy_assignable_v<ASParams>);
static_assert(!std::is_assignable_v<decltype((std::declval<ASParams&>().sigma)), double>);
static_assert(!std::is_assignable_v<decltype((std::declval<ASParams&>().q_max)), int>);
static_assert(std::is_copy_constructible_v<ASParams>);

TEST(ASSimulation, DefaultResultIsZeroAndAggregatesRejectEmpty) {
    const etrade::ASSimResult r{};
    EXPECT_EQ(r.n_paths(), 0);
    EXPECT_EQ(r.overall_max_abs_q, 0.0);
    EXPECT_FALSE(r.cap_violated);
    EXPECT_THROW((void)r.mean_pnl(), std::invalid_argument);
    EXPECT_THROW((void)r.std_pnl(), std::invalid_argument);
    EXPECT_THROW((void)r.se(etrade::ASStat::kMeanPnl), std::invalid_argument);
}

TEST(ASValidation, RejectsBadParams) {
    EXPECT_THROW((void)ASParams(100, 2, 1, 0.005, 0.0, 1.5, 140, 20, 0.5),
                 std::invalid_argument);  // gamma = 0 outside the model
    EXPECT_THROW((void)ASParams(100, 2, 1, 0.005, -0.1, 1.5, 140, 20, 0.5),
                 std::invalid_argument);
    EXPECT_THROW((void)ASParams(100, 2, 1, 0.005, 0.1, 0.0, 140, 20, 0.5),
                 std::invalid_argument);  // k
    EXPECT_THROW((void)ASParams(100, 2, 1, 0.005, 0.1, 1.5, 0.0, 20, 0.5),
                 std::invalid_argument);  // A
    EXPECT_THROW((void)ASParams(100, -1, 1, 0.005, 0.1, 1.5, 140, 20, 0.5),
                 std::invalid_argument);  // sigma
    EXPECT_THROW((void)ASParams(100, 2, 0, 0.005, 0.1, 1.5, 140, 20, 0.5),
                 std::invalid_argument);  // horizon
    EXPECT_THROW((void)ASParams(100, 2, 1, 0.0, 0.1, 1.5, 140, 20, 0.5),
                 std::invalid_argument);  // dt
    EXPECT_THROW((void)ASParams(100, 2, 1, 2.0, 0.1, 1.5, 140, 20, 0.5),
                 std::invalid_argument);  // dt > horizon
    EXPECT_THROW((void)ASParams(100, 2, 1, 0.3, 0.1, 1.5, 140, 20, 0.5),
                 std::invalid_argument);  // T/dt not integral
    EXPECT_THROW((void)ASParams(100, 2, 1, 0.005, 0.1, 1.5, 140, -1, 0.5),
                 std::invalid_argument);  // q_max
    EXPECT_THROW((void)ASParams(100, 2, 1, 0.005, 0.1, 1.5, 140, 20, -0.5),
                 std::invalid_argument);  // eta_liq
}

TEST(ASValidation, RejectsBadQueriesAndSimArgs) {
    const ASParams p = liquid();
    EXPECT_THROW((void)etrade::reservation_price(p, 100, 0, -0.1),
                 std::invalid_argument);
    EXPECT_THROW((void)etrade::optimal_spread(p, 1.5), std::invalid_argument);
    EXPECT_THROW((void)etrade::intensity(p, std::numeric_limits<double>::infinity()),
                 std::invalid_argument);
    EXPECT_THROW((void)etrade::simulate(p, 1, 0), std::invalid_argument);
    EXPECT_THROW((void)etrade::simulate(p, 100, 0, etrade::Agent::kFixed, -1.0),
                 std::invalid_argument);
    EXPECT_THROW((void)etrade::simulate(p, 100, 0, etrade::Agent::kAS,
                                  std::nullopt, 0.0),
                 std::invalid_argument);
    EXPECT_THROW((void)etrade::simulate(p, 100, 0, etrade::Agent::kAS,
                                  std::nullopt, 1.0, 1.5),
                 std::invalid_argument);
}

}  // namespace

// ---------------------------------------------------------------------------
// Robustness tests: per-side fill counting, finite inputs, stress edges, step and path bounds
// ---------------------------------------------------------------------------

namespace {

TEST(ASRobustness, MeanFillsCountsBothSides) {
    // Saturated book: P(fill) rounds to 1 on both sides every step, so the
    // fill count is exactly 2*n_steps (400), not the number of steps (200).
    const ASParams sat(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 1e6, 10000, 0.0);
    const auto r = etrade::simulate(sat, 50, 1);
    EXPECT_DOUBLE_EQ(r.mean_fills(), 2.0 * sat.n_steps());
    for (double f : r.fills_path) EXPECT_DOUBLE_EQ(f, 400.0);
    // Liquid set: total fills bounded by 2 per step, integral, and at least
    // the peak inventory (every unit of |q| needs a fill).
    const auto l = etrade::simulate(liquid(), 400, 8);
    for (int i = 0; i < l.n_paths(); ++i) {
        EXPECT_LE(l.fills_path[i], 2.0 * liquid().n_steps());
        EXPECT_GE(l.fills_path[i], l.max_abs_q_path[i]);
        EXPECT_DOUBLE_EQ(l.fills_path[i], std::round(l.fills_path[i]));
    }
    EXPECT_GT(l.se(etrade::ASStat::kMeanFills), 0.0);
}

TEST(ASRobustness, FixedSpreadMustBeFiniteAndPositive) {
    const ASParams p = liquid();
    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_THROW((void)etrade::simulate(p, 10, 0, etrade::Agent::kFixed, inf), std::invalid_argument);
    EXPECT_THROW((void)etrade::simulate(p, 10, 0, etrade::Agent::kFixed, 0.0), std::invalid_argument);
    EXPECT_THROW((void)etrade::simulate(p, 10, 0, etrade::Agent::kAS,
                                  std::nullopt, 1.0, std::nan("")),
                 std::invalid_argument);
    EXPECT_THROW((void)etrade::simulate(p, 10, 0, etrade::Agent::kAS,
                                  std::nullopt, inf, 0.5),
                 std::invalid_argument);
    EXPECT_THROW((void)etrade::simulate(p, etrade::kMaxAsPaths + 1, 0), std::invalid_argument);
}

TEST(ASRobustness, StressEdgeFractions) {
    const ASParams p = liquid();
    const auto base = etrade::simulate(p, 300, 9);
    // frac = 1.0: no step has t >= T, so bitwise identical to unstressed.
    const auto late = etrade::simulate(p, 300, 9, etrade::Agent::kAS,
                                       std::nullopt, 5.0, 1.0);
    // factor = 1.0 at any fraction: identical.
    const auto unit = etrade::simulate(p, 300, 9, etrade::Agent::kAS,
                                       std::nullopt, 1.0, 0.0);
    for (int i = 0; i < base.n_paths(); ++i) {
        ASSERT_EQ(base.pnl[i], late.pnl[i]);
        ASSERT_EQ(base.pnl[i], unit.pnl[i]);
    }
    // frac = 0.0 with factor 5: every step stressed, dispersion up.
    const auto early = etrade::simulate(p, 300, 9, etrade::Agent::kAS,
                                        std::nullopt, 5.0, 0.0);
    EXPECT_GT(early.std_pnl(), base.std_pnl());
}

TEST(ASRobustness, QuotesRejectNonFiniteState) {
    const ASParams p = liquid();
    const double nan = std::nan("");
    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_THROW((void)etrade::reservation_price(p, nan, 0.0, 0.0), std::invalid_argument);
    EXPECT_THROW((void)etrade::reservation_price(p, 100.0, inf, 0.0), std::invalid_argument);
    EXPECT_THROW((void)etrade::quotes(p, 100.0, inf, 0.0), std::invalid_argument);
    EXPECT_THROW((void)etrade::quotes(p, -inf, 0.0, 0.0), std::invalid_argument);
    EXPECT_THROW((void)etrade::reservation_price(p, 100.0, 0.0, nan), std::invalid_argument);
    EXPECT_THROW((void)etrade::fill_probability(p, nan), std::invalid_argument);
}

TEST(ASRobustness, NStepsIntegralityFloat) {
    const ASParams p3(100.0, 2.0, 1.0, 1.0 / 3.0, 0.1, 1.5, 140.0, 20, 0.5);
    EXPECT_EQ(p3.n_steps(), 3);
    EXPECT_THROW((void)ASParams(100.0, 2.0, 1.0, 0.3, 0.1, 1.5, 140.0, 20, 0.5),
                 std::invalid_argument);
    const ASParams p1(100.0, 2.0, 1.0, 1.0, 0.1, 1.5, 140.0, 20, 0.5);  // dt = T
    EXPECT_EQ(p1.n_steps(), 1);
    const auto r = etrade::simulate(p1, 200, 4);
    for (int i = 0; i < r.n_paths(); ++i) {
        // one step: the time average of |q| is the single post-update value
        EXPECT_DOUBLE_EQ(r.mean_abs_q_path[i], r.max_abs_q_path[i]);
        EXPECT_DOUBLE_EQ(r.rms_q_path[i], r.max_abs_q_path[i]);
        EXPECT_LE(r.fills_path[i], 2.0);
    }
    EXPECT_DOUBLE_EQ(r.mean_abs_q(), r.mean_max_abs_q());
    // step-count bound
    EXPECT_THROW((void)ASParams(100.0, 2.0, 1.0, 1.0 / (10.0 * etrade::kMaxAsSteps), 0.1, 1.5,
                          140.0, 20, 0.5),
                 std::invalid_argument);
}

}  // namespace
