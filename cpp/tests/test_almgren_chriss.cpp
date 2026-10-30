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

// ---------------------------------------------------------------------------
// Robustness and literature tests: overflow-free trajectory, log1p kappa, POV size bounds, liquidation checks, paper closed forms
// ---------------------------------------------------------------------------

namespace {

/// FX-like parameter set (mirrors data/params_fx.json).
ACParams fx_set(int n_intervals = 10) {
    return ACParams(5e7, n_intervals, 0.1, 0.006, 2e-9, 5e-11, 1e-4);
}

void expect_finite_large_kappa_t(const ACParams& p, int n_int) {
    const double kappa = etrade::ac_kappa(p);
    ASSERT_GT(kappa * p.horizon(), 710.0);  // std::sinh would overflow
    const auto x = etrade::ac_trajectory(p);
    ASSERT_EQ(x.size(), static_cast<std::size_t>(n_int) + 1);
    EXPECT_DOUBLE_EQ(x.front(), p.total_shares);
    EXPECT_DOUBLE_EQ(x.back(), 0.0);
    double sum_n = 0.0;
    int normal = 0;
    for (std::size_t j = 0; j < x.size(); ++j) {
        ASSERT_TRUE(std::isfinite(x[j])) << "j=" << j;
        if (j > 0) {
            EXPECT_LE(x[j], x[j - 1]);
            sum_n += x[j - 1] - x[j];
            // strictly decreasing while in the normal range
            if (x[j] > std::numeric_limits<double>::min() * x[0]) {
                EXPECT_LT(x[j], x[j - 1]) << "j=" << j;
                ++normal;
            }
        }
    }
    EXPECT_GE(normal, 800);
    EXPECT_NEAR(sum_n, p.total_shares, 1e-12 * p.total_shares);
    EXPECT_NEAR(x[1] / x[0], std::exp(-kappa * p.tau), 1e-9 * std::exp(-kappa * p.tau));
    EXPECT_TRUE(std::isfinite(etrade::expected_cost(p)));
    EXPECT_TRUE(std::isfinite(etrade::variance_cost(p)));
    for (const auto& fp : etrade::efficient_frontier(p, {1e-8, 1e-6, 1e-4, 1.0})) {
        EXPECT_TRUE(std::isfinite(fp.e_cost));
        EXPECT_TRUE(std::isfinite(fp.std_cost));
        EXPECT_TRUE(std::isfinite(fp.kappa));
    }
}

TEST(ACRobustness, TrajectoryFiniteForLargeKappaT) {
    expect_finite_large_kappa_t(ACParams(1e6, 1000, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6),
                                1000);  // kappa*T ~ 846
    expect_finite_large_kappa_t(fx_set(20000), 20000);  // kappa*T ~ 2683
}

TEST(ACRobustness, PaperClosedFormsMatchDiscreteSums) {
    // Almgren-Chriss (2000): E and V of the optimal strategy in closed form
    // (with the discrete kappa) equal the exact sums.
    for (const ACParams& p : {liquid(), ACParams(5e5, 10, 0.5, 1.5, 1e-5, 5e-7, 5e-6),
                              fx_set()}) {
        const double k = etrade::ac_kappa(p), T = p.horizon(), tau = p.tau;
        const double X = p.total_shares, sh = std::sinh(k * T);
        const double e_paper =
            0.5 * p.gamma_p * X * X +
            p.eta_tilde() * X * X * std::tanh(0.5 * k * tau) *
                (tau * std::sinh(2 * k * T) + 2 * T * std::sinh(k * tau)) /
                (2 * tau * tau * sh * sh);
        const double v_paper =
            0.5 * p.sigma * p.sigma * X * X *
            (tau * sh * std::cosh(k * (T - tau)) - T * std::sinh(k * tau)) /
            (sh * sh * std::sinh(k * tau));
        EXPECT_NEAR(etrade::expected_cost(p), e_paper, 1e-12 * e_paper);
        EXPECT_NEAR(etrade::variance_cost(p), v_paper, 1e-12 * v_paper);
    }
}

TEST(ACRobustness, KappaSmallTauAccuracy) {
    // kappa = kappa_tilde*(1 - kappa_tilde^2 tau^2/24 + O(tau^4)).
    for (double lam : {1e-6, 1e-12}) {
        const ACParams p(1e6, 5000, 1e-3, 0.95, 2.5e-6, 2.5e-7, lam);
        const double kt2 = lam * p.sigma * p.sigma / p.eta_tilde();
        const double series = std::sqrt(kt2) * (1.0 - kt2 * p.tau * p.tau / 24.0);
        EXPECT_NEAR(etrade::ac_kappa(p), series, 1e-9 * series);
        EXPECT_NEAR(etrade::half_life(p), 1.0 / series, 1e-9 / series);
        EXPECT_DOUBLE_EQ(etrade::characteristic_time(p), etrade::half_life(p));
    }
}

TEST(ACRobustness, PovClipTooSmallIsInputError) {
    EXPECT_THROW(etrade::pov_holdings(1e6, 1.0, 1.0, 1e-9), std::invalid_argument);
    EXPECT_THROW(etrade::pov_holdings(etrade::kMaxPovIntervals + 1.0, 1.0, 1.0, 0.5),
                 std::invalid_argument);
}

TEST(ACRobustness, PovBoundaryIntervalCounts) {
    const double clip = 0.15 * 2.5e6;  // 375,000
    const auto x2 = etrade::pov_holdings(750000.0, 1.0, 2.5e6, 0.15);
    ASSERT_EQ(x2.size(), 3u);  // X/clip = 2 exactly -> 2 intervals
    EXPECT_DOUBLE_EQ(x2[1], 375000.0);
    EXPECT_DOUBLE_EQ(x2[2], 0.0);
    const auto x3 = etrade::pov_holdings(750000.001, 1.0, 2.5e6, 0.15);
    ASSERT_EQ(x3.size(), 4u);  // one share-thousandth more -> 3 intervals
    const auto n3 = etrade::trades_from_holdings(x3);
    EXPECT_DOUBLE_EQ(n3[0], clip);
    EXPECT_DOUBLE_EQ(n3[1], clip);
    EXPECT_NEAR(n3[2], 0.001, 1e-6);
    EXPECT_DOUBLE_EQ(x3.back(), 0.0);
}

TEST(ACRobustness, SimulateIsRejectsNonLiquidatingSchedule) {
    const ACParams p = liquid();
    EXPECT_THROW(etrade::simulate_is(p, {1e6, 5e5, 1e3}, 10, 0), std::invalid_argument);
    EXPECT_THROW(etrade::simulate_is(p, {1e6, std::nan(""), 0.0}, 10, 0),
                 std::invalid_argument);
    EXPECT_THROW(etrade::simulate_is(p, {1e6}, 10, 0), std::invalid_argument);
    EXPECT_THROW(etrade::simulate_is(p, etrade::kMaxPaths + 1, 0), std::invalid_argument);
}

TEST(ACRobustness, SimulateIsTwapMatchesClosedForms) {
    const ACParams p = liquid();
    const auto twap = etrade::twap_holdings(p.total_shares, p.n_intervals);
    const auto mc = etrade::simulate_is(p, twap, 5000, 17);
    EXPECT_LE(std::abs(mc.mean - etrade::expected_cost(p, twap)), 4.0 * mc.se_mean);
    EXPECT_LE(std::abs(mc.var - etrade::variance_cost(p, twap)), 4.0 * mc.se_var);
    EXPECT_EQ(mc.n_paths, 5000);
}

TEST(ACRobustness, FrontierRejectsNonFiniteLambdaAndKeepsOrder) {
    const ACParams p = liquid();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_THROW(etrade::efficient_frontier(p, {1e-7, nan}), std::invalid_argument);
    EXPECT_THROW(etrade::efficient_frontier(p, {inf}), std::invalid_argument);
    EXPECT_THROW(etrade::efficient_frontier(p, {1e-7, -1e-7}), std::invalid_argument);
    const auto f = etrade::efficient_frontier(p, {2e-6, 1e-8, 2e-6});
    ASSERT_EQ(f.size(), 3u);
    EXPECT_DOUBLE_EQ(f[0].lambda_ra, 2e-6);
    EXPECT_DOUBLE_EQ(f[1].lambda_ra, 1e-8);
    EXPECT_DOUBLE_EQ(f[0].e_cost, f[2].e_cost);
    EXPECT_DOUBLE_EQ(f[0].kappa, f[2].kappa);
    EXPECT_LT(f[1].e_cost, f[0].e_cost);
}

TEST(ACRobustness, TwapAndPovRejectNonFinite) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_THROW(etrade::twap_holdings(inf, 5), std::invalid_argument);
    EXPECT_THROW(etrade::twap_holdings(nan, 5), std::invalid_argument);
    EXPECT_THROW(etrade::pov_holdings(nan, 1.0, 1e6, 0.1), std::invalid_argument);
    EXPECT_THROW(etrade::pov_holdings(inf, 1.0, 1e6, 0.1), std::invalid_argument);
    EXPECT_THROW(etrade::pov_holdings(1e6, inf, 1e6, 0.1), std::invalid_argument);
    EXPECT_THROW(etrade::pov_holdings(1e6, 1.0, inf, 0.1), std::invalid_argument);
    EXPECT_THROW(etrade::pov_holdings(1e6, 1.0, 1e6, nan), std::invalid_argument);
}

TEST(ACRobustness, SizeBoundsAreInputErrors) {
    EXPECT_THROW(ACParams(1e6, etrade::kMaxAcIntervals + 1, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6),
                 std::invalid_argument);
    EXPECT_THROW(etrade::twap_holdings(1e6, etrade::kMaxAcIntervals + 1),
                 std::invalid_argument);
    EXPECT_EQ(etrade::twap_holdings(1.0, etrade::kMaxAcIntervals).size(),
              static_cast<std::size_t>(etrade::kMaxAcIntervals) + 1);
}

TEST(ACRobustness, CostFunctionsValidateHoldings) {
    const ACParams p = liquid();
    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_THROW(etrade::variance_cost(p, {1e6}), std::invalid_argument);
    EXPECT_THROW(etrade::variance_cost(p, {1e6, inf, 0.0}), std::invalid_argument);
    EXPECT_THROW(etrade::expected_cost(p, {1e6, std::nan(""), 0.0}), std::invalid_argument);
    // X is taken from the schedule (x_0), not from params (pinned).
    const auto half = etrade::twap_holdings(5e5, 5);
    const double e_half = 0.5 * p.gamma_p * 5e5 * 5e5 + p.eta_tilde() * 5e5 * 5e5 / 5.0;
    EXPECT_NEAR(etrade::expected_cost(p, half), e_half, 1e-12 * e_half);
}

TEST(ACRobustness, TrajectoryX0ExactAndSinhIdentityModerate) {
    for (double lam : {1e-8, 5e-7, 2e-6, 1e-4}) {
        const ACParams p = liquid(lam);
        const double k = etrade::ac_kappa(p), T = p.horizon();
        const auto x = etrade::ac_trajectory(p);
        EXPECT_EQ(x[0], p.total_shares);  // exact, not within an ulp
        for (int j = 1; j < p.n_intervals; ++j) {
            const double ref = p.total_shares * std::sinh(k * (T - j * p.tau)) / std::sinh(k * T);
            EXPECT_NEAR(x[j], ref, 1e-14 * ref);
        }
    }
}

}  // namespace
