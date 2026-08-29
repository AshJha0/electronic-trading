/// \file test_avellaneda_stoikov.cpp
/// \brief Unit and property tests for the Avellaneda-Stoikov module.
#include <cmath>
#include <limits>
#include <stdexcept>

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
                                         etrade::kDefaultFixedSpread, 2.0, 0.5);
    EXPECT_GT(stress.std_pnl(), base.std_pnl());
}

TEST(ASSimulation, SeStatisticsAvailable) {
    const auto r = etrade::simulate(liquid(), 200, 1);
    for (const char* stat : {"mean_pnl", "std_pnl", "mean_abs_q",
                             "mean_max_abs_q", "rms_q", "mean_fills"}) {
        EXPECT_GT(r.se(stat), 0.0) << stat;
    }
    EXPECT_THROW(r.se("nope"), std::invalid_argument);
}

TEST(ASValidation, RejectsBadParams) {
    EXPECT_THROW(ASParams(100, 2, 1, 0.005, 0.0, 1.5, 140, 20, 0.5),
                 std::invalid_argument);  // gamma = 0 outside the model
    EXPECT_THROW(ASParams(100, 2, 1, 0.005, -0.1, 1.5, 140, 20, 0.5),
                 std::invalid_argument);
    EXPECT_THROW(ASParams(100, 2, 1, 0.005, 0.1, 0.0, 140, 20, 0.5),
                 std::invalid_argument);  // k
    EXPECT_THROW(ASParams(100, 2, 1, 0.005, 0.1, 1.5, 0.0, 20, 0.5),
                 std::invalid_argument);  // A
    EXPECT_THROW(ASParams(100, -1, 1, 0.005, 0.1, 1.5, 140, 20, 0.5),
                 std::invalid_argument);  // sigma
    EXPECT_THROW(ASParams(100, 2, 0, 0.005, 0.1, 1.5, 140, 20, 0.5),
                 std::invalid_argument);  // horizon
    EXPECT_THROW(ASParams(100, 2, 1, 0.0, 0.1, 1.5, 140, 20, 0.5),
                 std::invalid_argument);  // dt
    EXPECT_THROW(ASParams(100, 2, 1, 2.0, 0.1, 1.5, 140, 20, 0.5),
                 std::invalid_argument);  // dt > horizon
    EXPECT_THROW(ASParams(100, 2, 1, 0.3, 0.1, 1.5, 140, 20, 0.5),
                 std::invalid_argument);  // T/dt not integral
    EXPECT_THROW(ASParams(100, 2, 1, 0.005, 0.1, 1.5, 140, -1, 0.5),
                 std::invalid_argument);  // q_max
    EXPECT_THROW(ASParams(100, 2, 1, 0.005, 0.1, 1.5, 140, 20, -0.5),
                 std::invalid_argument);  // eta_liq
}

TEST(ASValidation, RejectsBadQueriesAndSimArgs) {
    const ASParams p = liquid();
    EXPECT_THROW(etrade::reservation_price(p, 100, 0, -0.1),
                 std::invalid_argument);
    EXPECT_THROW(etrade::optimal_spread(p, 1.5), std::invalid_argument);
    EXPECT_THROW(etrade::intensity(p, std::numeric_limits<double>::infinity()),
                 std::invalid_argument);
    EXPECT_THROW(etrade::simulate(p, 1, 0), std::invalid_argument);
    EXPECT_THROW(etrade::simulate(p, 100, 0, etrade::Agent::kFixed, -1.0),
                 std::invalid_argument);
    EXPECT_THROW(etrade::simulate(p, 100, 0, etrade::Agent::kAS,
                                  etrade::kDefaultFixedSpread, 0.0),
                 std::invalid_argument);
    EXPECT_THROW(etrade::simulate(p, 100, 0, etrade::Agent::kAS,
                                  etrade::kDefaultFixedSpread, 1.0, 1.5),
                 std::invalid_argument);
}

}  // namespace
