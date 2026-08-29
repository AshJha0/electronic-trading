/// \file test_almgren_chriss.cpp
/// \brief Unit and property tests for the Almgren-Chriss module.
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "etrade/almgren_chriss.hpp"

namespace {

using etrade::ACParams;

/// Liquid large-cap parameter set (mirrors data/params_liquid.json).
ACParams liquid(double lambda_ra = 2e-6) {
    return ACParams(1e6, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, lambda_ra);
}

TEST(ACTrajectory, MonotoneDecreasingAndSumsToX) {
    for (double lam : {0.0, 1e-8, 5e-7, 2e-6, 1e-5}) {
        const ACParams p = liquid(lam);
        const std::vector<double> x = etrade::ac_trajectory(p);
        ASSERT_EQ(x.size(), 6u);
        EXPECT_DOUBLE_EQ(x.front(), p.total_shares);
        EXPECT_DOUBLE_EQ(x.back(), 0.0);
        double sum_n = 0.0;
        for (std::size_t j = 1; j < x.size(); ++j) {
            EXPECT_LT(x[j], x[j - 1]) << "lambda " << lam << " j " << j;
            sum_n += x[j - 1] - x[j];
        }
        EXPECT_NEAR(sum_n, p.total_shares, 1e-12 * p.total_shares);
    }
}

TEST(ACTrajectory, HigherRiskAversionFrontLoadsAndTradesRiskForCost) {
    // Property grid: higher lambda_ra -> faster liquidation (larger n_1),
    // lower Var, higher E.
    const std::vector<double> lambdas = {0.0, 1e-8, 1e-7, 5e-7, 2e-6, 1e-5};
    double prev_n1 = -1.0, prev_e = -1.0;
    double prev_var = std::numeric_limits<double>::infinity();
    for (double lam : lambdas) {
        const ACParams p = liquid(lam);
        const double n1 = etrade::ac_trade_list(p)[0];
        const double e = etrade::expected_cost(p);
        const double v = etrade::variance_cost(p);
        EXPECT_GT(n1, prev_n1) << "lambda " << lam;
        EXPECT_GE(e, prev_e) << "lambda " << lam;
        EXPECT_LT(v, prev_var) << "lambda " << lam;
        prev_n1 = n1;
        prev_e = e;
        prev_var = v;
    }
}

TEST(ACTrajectory, KappaLimitContinuityNearZero) {
    // Just below / above the kappa*T = 1e-6 branch threshold the trajectories
    // agree far inside golden tolerance.
    const ACParams p0 = liquid(0.0);
    const double eta_t = p0.eta_tilde();
    const double big_t = p0.horizon();
    // kappa ~ kappa_tilde for small kappa: pick lambda so kappa*T ~ 1e-6.
    const double lam_at =
        (1e-6 / big_t) * (1e-6 / big_t) * eta_t / (p0.sigma * p0.sigma);
    const auto x_below = etrade::ac_trajectory(p0.with_lambda(lam_at * 0.99));
    const auto x_above = etrade::ac_trajectory(p0.with_lambda(lam_at * 1.01));
    for (std::size_t j = 0; j < x_below.size(); ++j) {
        EXPECT_NEAR(x_below[j], x_above[j], 1e-6);
    }
}

TEST(ACTrajectory, LambdaZeroEqualsTwapExactly) {
    const auto x = etrade::ac_trajectory(liquid(0.0));
    const auto twap = etrade::twap_holdings(1e6, 5);
    for (std::size_t j = 0; j < x.size(); ++j) {
        EXPECT_DOUBLE_EQ(x[j], twap[j]);
    }
}

TEST(ACTrajectory, SigmaZeroTakesTwapBranch) {
    const ACParams p(1e6, 5, 1.0, 0.0, 2.5e-6, 2.5e-7, 2e-6);
    EXPECT_DOUBLE_EQ(etrade::ac_kappa(p), 0.0);
    EXPECT_TRUE(std::isinf(etrade::half_life(p)));
    const auto x = etrade::ac_trajectory(p);
    EXPECT_DOUBLE_EQ(x[1], 8e5);
}

TEST(ACTrajectory, TradeListSatisfiesCoshIdentity) {
    // n_j = (2X sinh(kappa tau/2)/sinh(kappa T)) cosh(kappa (T - t_{j-1/2})).
    const ACParams p = liquid();
    const double kappa = etrade::ac_kappa(p);
    const auto n = etrade::ac_trade_list(p);
    const double pref = 2.0 * p.total_shares * std::sinh(0.5 * kappa * p.tau) /
                        std::sinh(kappa * p.horizon());
    for (int j = 1; j <= p.n_intervals; ++j) {
        const double tmid = (j - 0.5) * p.tau;
        EXPECT_NEAR(n[j - 1], pref * std::cosh(kappa * (p.horizon() - tmid)),
                    1e-6);
    }
}

TEST(ACTrajectory, EdgeCasesXZeroAndSingleClip) {
    const ACParams p0(0.0, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6);
    const auto x0 = etrade::ac_trajectory(p0);
    for (double v : x0) EXPECT_DOUBLE_EQ(v, 0.0);
    EXPECT_DOUBLE_EQ(etrade::expected_cost(p0), 0.0);
    EXPECT_DOUBLE_EQ(etrade::variance_cost(p0), 0.0);

    const ACParams p1(1e6, 1, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6);
    const auto x1 = etrade::ac_trajectory(p1);
    ASSERT_EQ(x1.size(), 2u);
    EXPECT_DOUBLE_EQ(x1[0], 1e6);
    EXPECT_DOUBLE_EQ(x1[1], 0.0);
    // Single clip == instantaneous liquidation.
    const auto [e_inst, v_inst] = etrade::instantaneous_cost(p1);
    EXPECT_NEAR(etrade::expected_cost(p1), e_inst, 1e-6);
    EXPECT_DOUBLE_EQ(etrade::variance_cost(p1), v_inst);
}

TEST(ACBaselines, PovScheduleShape) {
    // clip = 0.15 * 2.5e6 * 1.0 = 375k -> M = ceil(1e6/375k) = 3 intervals.
    const auto x = etrade::pov_holdings(1e6, 1.0, 2.5e6, 0.15);
    ASSERT_EQ(x.size(), 4u);
    EXPECT_DOUBLE_EQ(x[0], 1e6);
    EXPECT_DOUBLE_EQ(x[1], 625000.0);
    EXPECT_DOUBLE_EQ(x[2], 250000.0);
    EXPECT_DOUBLE_EQ(x[3], 0.0);  // partial last clip
}

TEST(ACBaselines, FrontierMonotoneInLambda) {
    const auto frontier = etrade::efficient_frontier(
        liquid(), {0.0, 1e-8, 1e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5});
    ASSERT_EQ(frontier.size(), 8u);
    for (std::size_t i = 1; i < frontier.size(); ++i) {
        EXPECT_GE(frontier[i].e_cost, frontier[i - 1].e_cost);
        EXPECT_LE(frontier[i].std_cost, frontier[i - 1].std_cost);
    }
}

TEST(ACMonteCarlo, ClosedFormsWithinFourSe) {
    const ACParams p = liquid();
    const etrade::ISSimResult mc = etrade::simulate_is(p, 20000, 2024);
    EXPECT_LE(std::abs(mc.mean - etrade::expected_cost(p)), 4.0 * mc.se_mean);
    EXPECT_LE(std::abs(mc.var - etrade::variance_cost(p)), 4.0 * mc.se_var);
}

TEST(ACMonteCarlo, DeterministicPerSeed) {
    const ACParams p = liquid();
    const auto a = etrade::simulate_is(p, 500, 99);
    const auto b = etrade::simulate_is(p, 500, 99);
    EXPECT_DOUBLE_EQ(a.mean, b.mean);
    EXPECT_DOUBLE_EQ(a.var, b.var);
    const auto c = etrade::simulate_is(p, 500, 100);
    EXPECT_NE(a.mean, c.mean);
}

TEST(ACValidation, RejectsBadInputs) {
    EXPECT_THROW(ACParams(-1.0, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6),
                 std::invalid_argument);
    EXPECT_THROW(ACParams(1e6, 0, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6),
                 std::invalid_argument);
    EXPECT_THROW(ACParams(1e6, 5, 0.0, 0.95, 2.5e-6, 2.5e-7, 2e-6),
                 std::invalid_argument);
    EXPECT_THROW(ACParams(1e6, 5, 1.0, -0.1, 2.5e-6, 2.5e-7, 2e-6),
                 std::invalid_argument);
    EXPECT_THROW(ACParams(1e6, 5, 1.0, 0.95, 0.0, 2.5e-7, 2e-6),
                 std::invalid_argument);
    EXPECT_THROW(ACParams(1e6, 5, 1.0, 0.95, 2.5e-6, -1e-9, 2e-6),
                 std::invalid_argument);
    EXPECT_THROW(ACParams(1e6, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, -1e-9),
                 std::invalid_argument);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(ACParams(nan, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6),
                 std::invalid_argument);
}

TEST(ACValidation, ModelValidityIsDomainError) {
    // gamma_p*tau >= 2*eta (equality included) -> std::domain_error.
    EXPECT_THROW(ACParams(1e6, 5, 1.0, 0.95, 2.5e-6, 5.0e-6, 2e-6),
                 std::domain_error);
    EXPECT_THROW(ACParams(1e6, 5, 1.0, 0.95, 2.5e-6, 6.0e-6, 2e-6),
                 std::domain_error);
    EXPECT_NO_THROW(ACParams(1e6, 5, 1.0, 0.95, 2.5e-6, 4.9e-6, 2e-6));
}

TEST(ACValidation, RejectsNonLiquidatingAndDegenerateSchedules) {
    const ACParams p = liquid();
    EXPECT_THROW(etrade::expected_cost(p, {1e6, 5e5, 1.0}),
                 std::invalid_argument);
    EXPECT_THROW(etrade::trades_from_holdings({1.0}), std::invalid_argument);
    EXPECT_THROW(etrade::efficient_frontier(p, {}), std::invalid_argument);
    EXPECT_THROW(etrade::pov_holdings(1e6, 1.0, 2.5e6, 1.0),
                 std::invalid_argument);
    EXPECT_THROW(etrade::pov_holdings(1e6, 1.0, 2.5e6, 0.0),
                 std::invalid_argument);
    EXPECT_THROW(etrade::pov_holdings(1e6, 1.0, 0.0, 0.15),
                 std::invalid_argument);
    EXPECT_THROW(etrade::simulate_is(p, 1, 0), std::invalid_argument);
}

}  // namespace
