/// \file test_analytics.cpp
/// \brief Unit tests for the execution-analytics module.
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "etrade/analytics.hpp"

namespace {

using etrade::Side;

TEST(ISDecomposition, SumsToTotalIdentity) {
    // total == delay + trading + opportunity, and equals the direct
    // sign*(X*P_d - sum n_j p_j - R*P_T) expression, exactly.
    const std::vector<double> qty = {300.0, 400.0, 200.0};
    const std::vector<double> px = {99.4, 99.2, 99.1};
    const auto d = etrade::decompose_shortfall(Side::kSell, 100.0, 99.5, 98.8,
                                               1000.0, qty, px);
    EXPECT_NEAR(d.total, d.delay + d.trading + d.opportunity, 1e-10);
    double paper = 1000.0 * 100.0;
    for (std::size_t j = 0; j < qty.size(); ++j) paper -= qty[j] * px[j];
    paper -= d.remaining_qty * 98.8;
    EXPECT_NEAR(d.total, paper, 1e-10);
    EXPECT_DOUBLE_EQ(d.executed_qty, 900.0);
    EXPECT_DOUBLE_EQ(d.remaining_qty, 100.0);
}

TEST(ISDecomposition, BuyMirrorsSell) {
    // Mirroring every price move around the decision price turns a sell
    // decomposition into the identical buy decomposition.
    const auto sell = etrade::decompose_shortfall(
        Side::kSell, 100.0, 99.5, 98.8, 1000.0, {300, 400, 200},
        {99.4, 99.2, 99.1});
    const auto buy = etrade::decompose_shortfall(
        Side::kBuy, 100.0, 100.5, 101.2, 1000.0, {300, 400, 200},
        {100.6, 100.8, 100.9});
    EXPECT_NEAR(sell.total, buy.total, 1e-10);
    EXPECT_NEAR(sell.delay, buy.delay, 1e-10);
    EXPECT_NEAR(sell.trading, buy.trading, 1e-10);
    EXPECT_NEAR(sell.opportunity, buy.opportunity, 1e-10);
}

TEST(ISDecomposition, FullyExecutedHasNoOpportunityCost) {
    const auto d = etrade::decompose_shortfall(Side::kSell, 100.0, 99.5, 98.8,
                                               700.0, {300, 400}, {99.4, 99.2});
    EXPECT_DOUBLE_EQ(d.opportunity, 0.0);
    EXPECT_DOUBLE_EQ(d.remaining_qty, 0.0);
}

TEST(ISDecomposition, RejectsBadInputs) {
    EXPECT_THROW((void)etrade::decompose_shortfall(Side::kSell, 100, 99.5, 98.8,
                                             1000, {300, 400}, {99.4}),
                 std::invalid_argument);  // length mismatch
    EXPECT_THROW((void)etrade::decompose_shortfall(Side::kSell, 100, 99.5, 98.8, 500,
                                             {300, 400}, {99.4, 99.2}),
                 std::invalid_argument);  // over-execution
    EXPECT_THROW((void)etrade::decompose_shortfall(Side::kSell, 100, 99.5, 98.8,
                                             1000, {-1.0}, {99.4}),
                 std::invalid_argument);  // negative fill
    EXPECT_THROW((void)etrade::decompose_shortfall(
                     Side::kSell, std::numeric_limits<double>::quiet_NaN(),
                     99.5, 98.8, 1000, {300}, {99.4}),
                 std::invalid_argument);  // non-finite price
    EXPECT_THROW((void)etrade::decompose_shortfall(Side::kSell, 100, 99.5, 98.8,
                                             -1.0, {}, {}),
                 std::invalid_argument);  // negative parent qty
}

TEST(Participation, ComputesAndValidates) {
    EXPECT_DOUBLE_EQ(etrade::participation_rate(120000.0, 1500000.0), 0.08);
    EXPECT_DOUBLE_EQ(etrade::participation_rate(0.0, 1.0), 0.0);
    EXPECT_THROW((void)etrade::participation_rate(-1.0, 100.0),
                 std::invalid_argument);
    EXPECT_THROW((void)etrade::participation_rate(1.0, 0.0), std::invalid_argument);
}

TEST(Ols, RecoversExactLine) {
    const std::vector<double> x = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> y;
    for (double xi : x) y.push_back(2.0 + 3.0 * xi);
    const auto fit = etrade::ols(x, y);
    EXPECT_NEAR(fit.slope, 3.0, 1e-12);
    EXPECT_NEAR(fit.intercept, 2.0, 1e-12);
    EXPECT_DOUBLE_EQ(fit.r_squared, 1.0);  // SST > 0, SSR = 0
    EXPECT_EQ(fit.n_obs, 5);
}

TEST(Ols, ConstantYGivesRSquaredOne) {
    // SST == 0 -> R^2 defined as 1.0 (the flat fit is exact).
    const auto fit = etrade::ols({0.0, 1.0, 2.0}, {5.0, 5.0, 5.0});
    EXPECT_DOUBLE_EQ(fit.slope, 0.0);
    EXPECT_DOUBLE_EQ(fit.intercept, 5.0);
    EXPECT_DOUBLE_EQ(fit.r_squared, 1.0);
}

TEST(Ols, RejectsDegenerateInputs) {
    EXPECT_THROW((void)etrade::ols({1.0}, {2.0}), std::invalid_argument);
    EXPECT_THROW((void)etrade::ols({1.0, 2.0}, {1.0}), std::invalid_argument);
    EXPECT_THROW((void)etrade::ols({1.0, 1.0}, {1.0, 2.0}), std::invalid_argument);
    EXPECT_THROW(
        (void)etrade::ols({1.0, std::numeric_limits<double>::infinity()}, {1.0, 2.0}),
        std::invalid_argument);
}

TEST(Impact, SyntheticFillsArePinned) {
    const auto [part, cost] = etrade::synthetic_impact_fills();
    ASSERT_EQ(part.size(), 25u);
    EXPECT_DOUBLE_EQ(part.front(), 0.02);
    EXPECT_NEAR(part.back(), 0.20, 1e-15);
    EXPECT_DOUBLE_EQ(cost[0], 0.5 + 8.0 * 0.02 + 0.05 * std::sin(0.3));
    EXPECT_THROW((void)etrade::synthetic_impact_fills(1), std::invalid_argument);
}

TEST(Impact, RegressionRecoversSlopeApproximately) {
    const auto [part, cost] = etrade::synthetic_impact_fills(25, 0.5, 8.0, 0.05);
    const auto fit = etrade::estimate_impact(part, cost);
    EXPECT_NEAR(fit.slope, 8.0, 0.15);  // sinusoidal noise, small bias
    EXPECT_NEAR(fit.intercept, 0.5, 0.02);
    EXPECT_GT(fit.r_squared, 0.98);
}

}  // namespace

// ---------------------------------------------------------------------------
// Robustness tests: price-shift invariance, empty fills, participation above 1
// ---------------------------------------------------------------------------

namespace {

TEST(ISDecomposition, PriceShiftInvariance) {
    // Adding c to every price leaves every component unchanged (Q + R = X).
    const double c = 37.5;
    const std::vector<double> qty = {300.0, 400.0, 200.0};
    const std::vector<double> px = {99.4, 99.2, 99.1};
    std::vector<double> px_c;
    for (double p : px) px_c.push_back(p + c);
    for (Side side : {Side::kSell, Side::kBuy}) {
        const auto d0 = etrade::decompose_shortfall(side, 100.0, 99.5, 98.8, 1000.0, qty, px);
        const auto d1 = etrade::decompose_shortfall(side, 100.0 + c, 99.5 + c, 98.8 + c,
                                                    1000.0, qty, px_c);
        EXPECT_NEAR(d0.total, d1.total, 1e-9);
        EXPECT_NEAR(d0.delay, d1.delay, 1e-9);
        EXPECT_NEAR(d0.trading, d1.trading, 1e-9);
        EXPECT_NEAR(d0.opportunity, d1.opportunity, 1e-9);
    }
}

TEST(ISDecomposition, EmptyFills) {
    const auto d = etrade::decompose_shortfall(Side::kSell, 100.0, 99.0, 98.0, 500.0, {}, {});
    EXPECT_DOUBLE_EQ(d.delay, 0.0);
    EXPECT_DOUBLE_EQ(d.trading, 0.0);
    EXPECT_NEAR(d.opportunity, 1000.0, 1e-12);  // 500 * (100 - 98)
    EXPECT_NEAR(d.total, 1000.0, 1e-12);
    EXPECT_DOUBLE_EQ(d.executed_qty, 0.0);
    EXPECT_DOUBLE_EQ(d.remaining_qty, 500.0);
    const auto b = etrade::decompose_shortfall(Side::kBuy, 100.0, 99.0, 98.0, 500.0, {}, {});
    EXPECT_NEAR(b.opportunity, -1000.0, 1e-12);  // price fell: a gain for a buyer
}

TEST(Participation, AboveOneIsInputError) {
    EXPECT_THROW((void)etrade::participation_rate(2e6, 1.5e6), std::invalid_argument);
    EXPECT_DOUBLE_EQ(etrade::participation_rate(1.5e6, 1.5e6), 1.0);  // boundary
    EXPECT_THROW((void)etrade::participation_rate(std::nan(""), 1.0), std::invalid_argument);
    EXPECT_THROW((void)etrade::participation_rate(1.0, std::numeric_limits<double>::infinity()),
                 std::invalid_argument);
}

}  // namespace
