/// \file test_golden.cpp
/// \brief Cross-language golden-value suite: loads ../data/golden/golden.json
/// (relative to the cpp/ dir — tests run with that working directory) and
/// asserts every case.
///
/// Exact cases use the pinned relative-tolerance rule
/// `|actual - expected| <= tol * max(1, |expected|)`.  Statistical cases run
/// this port's own std::mt19937_64 simulation (>= 2000 paths) and assert
/// `|stat - ref| <= 4 * sqrt(se_ref^2 + se_own^2)` per API_SPEC section 4.
#ifndef ETRADE_GOLDEN_PATH
#define ETRADE_GOLDEN_PATH "../data/golden/golden.json"
#endif

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "etrade/almgren_chriss.hpp"
#include "etrade/analytics.hpp"
#include "etrade/avellaneda_stoikov.hpp"
#include "mini_json.hpp"

namespace {

using mini_json::Value;

const Value& golden() {
    static const Value doc = mini_json::parse_file(ETRADE_GOLDEN_PATH);
    return doc;
}

/// Find a golden case by name (throws on a missing case, failing the test).
const Value& find_case(const char* name) {
    for (const auto& c : golden().at("cases").arr) {
        if (c.at("name").str == name) return c;
    }
    throw std::runtime_error(std::string("golden case not found: ") + name);
}

double in(const Value& c, const std::string& key) {
    return c.at("inputs").at(key).number;
}

double exp_val(const Value& c, const std::string& key) {
    return c.at("expect").at(key).number;
}

/// Pinned exact-case rule: relative for large magnitudes, absolute near 0.
void check_exact(const Value& c, const std::string& key, double actual) {
    const double expected = exp_val(c, key);
    const double tol = c.at("tol").number;
    EXPECT_LE(std::abs(actual - expected), tol * std::max(1.0, std::abs(expected)))
        << c.at("name").str << ": key '" << key << "' actual " << actual
        << " expected " << expected;
}

/// Pinned statistical rule: |stat - ref| <= 4*sqrt(se_ref^2 + se_own^2).
void check_stat(const Value& c, const std::string& key, double actual,
                double se_own) {
    const double ref = exp_val(c, key);
    const double se_ref = exp_val(c, key + "_se");
    const double band = c.at("tol").number *
                        std::sqrt(se_ref * se_ref + se_own * se_own);
    EXPECT_LE(std::abs(actual - ref), band)
        << c.at("name").str << ": key '" << key << "' actual " << actual
        << " ref " << ref << " band " << band;
}

etrade::ACParams ac_params(const Value& c) {
    return etrade::ACParams(in(c, "ac_total_shares"),
                            static_cast<int>(in(c, "ac_n_intervals")),
                            in(c, "ac_tau"), in(c, "ac_sigma"), in(c, "ac_eta"),
                            in(c, "ac_gamma_p"), in(c, "ac_lambda_ra"));
}

etrade::ASParams as_params(const Value& c) {
    return etrade::ASParams(in(c, "as_s0"), in(c, "as_sigma"),
                            in(c, "as_horizon"), in(c, "as_dt"),
                            in(c, "as_gamma"), in(c, "as_k"),
                            in(c, "as_a_rate"),
                            static_cast<int>(in(c, "as_q_max")),
                            in(c, "as_eta_liq"));
}

void check_ac_solution(const char* name) {
    const Value& c = find_case(name);
    const etrade::ACParams p = ac_params(c);
    const std::vector<double> x = etrade::ac_trajectory(p);
    const std::vector<double> n = etrade::ac_trade_list(p);
    const int j = static_cast<int>(in(c, "x_index"));
    check_exact(c, "kappa", etrade::ac_kappa(p));
    check_exact(c, "half_life", etrade::half_life(p));
    check_exact(c, "x_j", x[j]);
    check_exact(c, "n_1", n[0]);
    check_exact(c, "e_cost", etrade::expected_cost(p));
    check_exact(c, "var_cost", etrade::variance_cost(p));
}

TEST(Golden, AcLiquidSolution) { check_ac_solution("ac_liquid_solution"); }

TEST(Golden, AcIlliquidSolution) { check_ac_solution("ac_illiquid_solution"); }

TEST(Golden, AcTwapLiquid) {
    const Value& c = find_case("ac_twap_liquid");
    const etrade::ACParams p = ac_params(c);
    const auto twap = etrade::twap_holdings(p.total_shares, p.n_intervals);
    check_exact(c, "e_cost", etrade::expected_cost(p, twap));
    check_exact(c, "var_cost", etrade::variance_cost(p, twap));
}

TEST(Golden, AcPovLiquid) {
    const Value& c = find_case("ac_pov_liquid");
    const etrade::ACParams p = ac_params(c);
    const auto pov = etrade::pov_holdings(p.total_shares, p.tau,
                                          in(c, "pov_market_volume_rate"),
                                          in(c, "pov_participation"));
    check_exact(c, "e_cost", etrade::expected_cost(p, pov));
    check_exact(c, "var_cost", etrade::variance_cost(p, pov));
    check_exact(c, "n_intervals_used", static_cast<double>(pov.size() - 1));
}

TEST(Golden, AcFrontierPointLiquid) {
    const Value& c = find_case("ac_frontier_point_liquid");
    const etrade::ACParams p = ac_params(c);
    const auto frontier =
        etrade::efficient_frontier(p, {in(c, "ac_lambda_ra")});
    ASSERT_EQ(frontier.size(), 1u);
    check_exact(c, "e_cost", frontier[0].e_cost);
    check_exact(c, "std_cost", frontier[0].std_cost);
    check_exact(c, "kappa", frontier[0].kappa);
}

TEST(Golden, AcKappaSmallTau) {
    // kappa_tilde^2*tau^2/2 ~ 2e-9: the log1p form matches; acosh(1+x)
    // would miss by ~1e-8 relative.
    const Value& c = find_case("ac_kappa_small_tau");
    const etrade::ACParams p = ac_params(c);
    ASSERT_EQ(p.n_intervals, 5000);
    check_exact(c, "kappa", etrade::ac_kappa(p));
    check_exact(c, "half_life", etrade::half_life(p));
    check_exact(c, "x_1", etrade::ac_trajectory(p)[1]);
}

TEST(Golden, AcLambdaZeroLimitLiquid) {
    const Value& c = find_case("ac_lambda_zero_limit_liquid");
    const etrade::ACParams p = ac_params(c);
    const auto x = etrade::ac_trajectory(p);
    const auto twap = etrade::twap_holdings(p.total_shares, p.n_intervals);
    double max_diff = 0.0;
    for (std::size_t j = 0; j < x.size(); ++j) {
        max_diff = std::max(max_diff, std::abs(x[j] - twap[j]));
    }
    check_exact(c, "max_abs_diff_vs_twap", max_diff);
}

TEST(Golden, AcInstantLiquidationLiquid) {
    const Value& c = find_case("ac_instant_liquidation_liquid");
    const auto [e, v] = etrade::instantaneous_cost(ac_params(c));
    check_exact(c, "e_cost", e);
    check_exact(c, "var_cost", v);
}

TEST(Golden, AcMcLiquid) {
    const Value& c = find_case("ac_mc_liquid");
    const etrade::ACParams p = ac_params(c);
    // Own RNG stream and seed; the reference stores the Python 20k-path
    // values + SEs, we run 20k paths of std::mt19937_64 (N=5, cheap).
    const etrade::ISSimResult mc = etrade::simulate_is(p, 20000, 123);
    check_stat(c, "mean_cost", mc.mean, mc.se_mean);
    check_stat(c, "var_cost", mc.var, mc.se_var);
}

TEST(Golden, AsReservationLiquid) {
    const Value& c = find_case("as_reservation_liquid");
    const etrade::ASParams p = as_params(c);
    check_exact(c, "reservation_price",
                etrade::reservation_price(p, in(c, "s"), in(c, "q"), in(c, "t")));
}

TEST(Golden, AsSpreadLiquid) {
    const Value& c = find_case("as_spread_liquid");
    const etrade::ASParams p = as_params(c);
    check_exact(c, "spread", etrade::optimal_spread(p, in(c, "t")));
    check_exact(c, "spread_terminal", etrade::optimal_spread(p, p.horizon));
}

TEST(Golden, AsFillProbLiquid) {
    const Value& c = find_case("as_fill_prob_liquid");
    const etrade::ASParams p = as_params(c);
    check_exact(c, "intensity", etrade::intensity(p, in(c, "delta")));
    check_exact(c, "fill_probability",
                etrade::fill_probability(p, in(c, "delta")));
}

TEST(Golden, AsSimLiquid) {
    const Value& c = find_case("as_sim_liquid");
    const etrade::ASSimResult r = etrade::simulate(as_params(c), 4000, 42);
    EXPECT_FALSE(r.cap_violated);
    check_stat(c, "mean_pnl", r.mean_pnl(), r.se(etrade::ASStat::kMeanPnl));
    check_stat(c, "std_pnl", r.std_pnl(), r.se(etrade::ASStat::kStdPnl));
    check_stat(c, "mean_abs_q", r.mean_abs_q(), r.se(etrade::ASStat::kMeanAbsQ));
    check_stat(c, "mean_max_abs_q", r.mean_max_abs_q(), r.se(etrade::ASStat::kMeanMaxAbsQ));
    check_stat(c, "mean_fills", r.mean_fills(), r.se(etrade::ASStat::kMeanFills));
}

TEST(Golden, AsSimFx) {
    const Value& c = find_case("as_sim_fx");
    const etrade::ASSimResult r = etrade::simulate(as_params(c), 4000, 7);
    check_stat(c, "mean_pnl", r.mean_pnl(), r.se(etrade::ASStat::kMeanPnl));
    check_stat(c, "std_pnl", r.std_pnl(), r.se(etrade::ASStat::kStdPnl));
    check_stat(c, "mean_abs_q", r.mean_abs_q(), r.se(etrade::ASStat::kMeanAbsQ));
}

TEST(Golden, AsFixedComparisonLiquid) {
    const Value& c = find_case("as_fixed_comparison_liquid");
    const etrade::ASParams p = as_params(c);
    const etrade::ASSimResult r_as =
        etrade::simulate(p, 4000, 42, etrade::Agent::kAS);
    const etrade::ASSimResult r_fx = etrade::simulate(
        p, 4000, 42, etrade::Agent::kFixed, in(c, "fixed_spread"));
    check_stat(c, "as_rms_q", r_as.rms_q(), r_as.se(etrade::ASStat::kRmsQ));
    check_stat(c, "fixed_rms_q", r_fx.rms_q(), r_fx.se(etrade::ASStat::kRmsQ));
}

void check_is_decomposition(const char* name, etrade::Side side) {
    const Value& c = find_case(name);
    const etrade::ISDecomposition d = etrade::decompose_shortfall(
        side, in(c, "decision_price"), in(c, "arrival_price"),
        in(c, "close_price"), in(c, "total_qty"),
        {in(c, "q1"), in(c, "q2"), in(c, "q3")},
        {in(c, "p1"), in(c, "p2"), in(c, "p3")});
    check_exact(c, "total", d.total);
    check_exact(c, "delay", d.delay);
    check_exact(c, "trading", d.trading);
    check_exact(c, "opportunity", d.opportunity);
}

TEST(Golden, IsDecompositionSell) {
    check_is_decomposition("is_decomposition_sell", etrade::Side::kSell);
}

TEST(Golden, IsDecompositionBuy) {
    check_is_decomposition("is_decomposition_buy", etrade::Side::kBuy);
}

TEST(Golden, ParticipationRate) {
    const Value& c = find_case("participation_rate");
    check_exact(c, "participation",
                etrade::participation_rate(in(c, "executed_volume"),
                                           in(c, "market_volume")));
}

TEST(Golden, ImpactRegression) {
    const Value& c = find_case("impact_regression");
    const auto [part, cost] = etrade::synthetic_impact_fills(
        static_cast<int>(in(c, "n_obs")), in(c, "true_intercept"),
        in(c, "true_slope"), in(c, "noise_amp"));
    const etrade::OLSResult reg = etrade::estimate_impact(part, cost);
    check_exact(c, "slope", reg.slope);
    check_exact(c, "intercept", reg.intercept);
    check_exact(c, "r_squared", reg.r_squared);
}

TEST(Golden, AllCasesCovered) {
    // Guard: every case in the file is one of the 19 handled above.
    const std::vector<std::string> known = {
        "ac_liquid_solution", "ac_illiquid_solution", "ac_twap_liquid",
        "ac_pov_liquid", "ac_frontier_point_liquid",
        "ac_lambda_zero_limit_liquid", "ac_instant_liquidation_liquid",
        "ac_kappa_small_tau", "ac_mc_liquid", "as_reservation_liquid",
        "as_spread_liquid",
        "as_fill_prob_liquid", "as_sim_liquid", "as_sim_fx",
        "as_fixed_comparison_liquid", "is_decomposition_sell",
        "is_decomposition_buy", "participation_rate", "impact_regression"};
    const auto& cases = golden().at("cases").arr;
    ASSERT_EQ(known.size(), 19u);
    EXPECT_EQ(cases.size(), known.size());
    for (const auto& c : cases) {
        bool found = false;
        for (const auto& k : known) found = found || c.at("name").str == k;
        EXPECT_TRUE(found) << "unhandled golden case " << c.at("name").str;
    }
}

}  // namespace
