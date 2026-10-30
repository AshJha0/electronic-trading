/// \file test_impact.cpp
/// \brief Power-law impact execution (API_SPEC Section 8): every case of
/// data/golden/impact_golden.json, reduction to Almgren-Chriss, the
/// square-root law, drift and early finish, VWAP, the report, validation.
#ifndef ETRADE_DATA_DIR
#define ETRADE_DATA_DIR "../data"
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "etrade/almgren_chriss.hpp"
#include "etrade/impact.hpp"
#include "etrade/market.hpp"
#include "mini_json.hpp"

namespace {

using mini_json::Value;

const Value& impact_golden() {
    static const Value doc =
        mini_json::parse_file(std::string(ETRADE_DATA_DIR) + "/golden/impact_golden.json");
    return doc;
}

double num(const Value& o, const std::string& k) { return o.at(k).number; }
double num_or(const Value& o, const std::string& k, double d) { return o.has(k) ? o.at(k).number : d; }

std::vector<double> curve() {
    return {etrade::kUsEquityVolumeCurve13.begin(), etrade::kUsEquityVolumeCurve13.end()};
}

etrade::ImpactMarket market_from(const Value& in) {
    etrade::ImpactMarket m;
    m.price = num(in, "price");
    m.spread = num(in, "spread");
    m.adv = num(in, "adv");
    m.daily_vol = num(in, "daily_vol");
    m.total_shares = num(in, "total_shares");
    m.n_intervals = static_cast<int>(num(in, "n_intervals"));
    m.tau = num(in, "tau");
    m.lambda_ra = num_or(in, "lambda_ra", 0.0);
    m.alpha = num_or(in, "alpha", 0.5);
    m.y = num_or(in, "y", 1.0);
    m.perm_frac = num_or(in, "perm_frac", etrade::kDefaultPermFrac);
    m.drift_bps_per_day = num_or(in, "drift_bps_per_day", 0.0);
    if (num(in, "volume_curve") == 1.0) m.weights = curve();
    return m;
}

std::vector<std::pair<std::string, double>> actual_for(const Value& c) {
    const std::string kind = c.at("kind").str;
    const Value& in = c.at("inputs");
    std::vector<std::pair<std::string, double>> out;
    if (kind == "sqrt_law") {
        out.emplace_back("bps", etrade::sqrt_law_cost_bps(num(in, "daily_vol"), num(in, "order_shares"),
                                                          num(in, "adv"), num(in, "y")));
        return out;
    }
    const etrade::ImpactMarket m = market_from(in);
    if (kind == "pretrade") {
        std::vector<double> lambdas;
        const int nl = static_cast<int>(num(in, "n_lambdas"));
        for (int i = 0; i < nl; ++i) lambdas.push_back(num(in, "lambda_" + std::to_string(i)));
        const etrade::PretradeReport rep = etrade::pretrade_report(m, lambdas);
        out.emplace_back("sqrt_law_bps", rep.sqrt_law_bps);
        out.emplace_back("half_spread_bps", rep.half_spread_bps);
        for (std::size_t i = 0; i < rep.rows.size(); ++i) {
            const std::string pre = "row" + std::to_string(i) + "_";
            out.emplace_back(pre + "impact_bps", rep.rows[i].impact_bps);
            out.emplace_back(pre + "sd_bps", rep.rows[i].sd_bps);
            out.emplace_back(pre + "horizon", static_cast<double>(rep.rows[i].horizon));
        }
        return out;
    }
    const etrade::ImpactParams p = etrade::impact_params_from_market(m);
    if (kind == "impact_vwap") {
        const auto x = etrade::vwap_holdings(p.total_shares(), p.weights());
        out.emplace_back("e_cost", etrade::impact_expected_cost(p, x));
        out.emplace_back("var_cost", etrade::impact_variance_cost(p, x));
        return out;
    }
    const etrade::ScheduleResult r = etrade::optimal_schedule(p);
    for (int j = 1; j < p.n_intervals(); ++j) {
        char key[32];
        std::snprintf(key, sizeof key, "xfrac_%02d", j);
        out.emplace_back(key, r.holdings[j] / p.total_shares());
    }
    out.emplace_back("e_cost", etrade::impact_expected_cost(p, r.holdings));
    out.emplace_back("var_cost", etrade::impact_variance_cost(p, r.holdings));
    out.emplace_back("horizon", static_cast<double>(r.horizon));
    return out;
}

etrade::ImpactMarket large_cap() {
    etrade::ImpactMarket m;
    m.price = 100.0;
    m.spread = 0.01;
    m.adv = 5e6;
    m.daily_vol = 0.015;
    m.total_shares = 250000.0;
    m.n_intervals = 13;
    m.tau = 1.0 / 13.0;
    return m;
}

}  // namespace

TEST(ImpactGolden, EveryCaseEveryKey) {
    const auto& cases = impact_golden().at("cases").arr;
    ASSERT_EQ(cases.size(), 8u);
    for (const auto& c : cases) {
        const auto actual = actual_for(c);
        const Value& expect = c.at("expect");
        ASSERT_EQ(actual.size(), expect.obj.size()) << c.at("name").str;
        const double tol = c.at("tol").number;
        for (const auto& [key, got] : actual) {
            const double want = num(expect, key);
            EXPECT_LE(std::abs(got - want), tol * std::max(1.0, std::abs(want)))
                << c.at("name").str << "." << key << ": got " << got << " want " << want;
        }
    }
}

TEST(Impact, AlphaOneReproducesAlmgrenChriss) {
    const etrade::ACParams ac(1e6, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6);
    const etrade::ImpactParams ip(1e6, 5, 1.0, 0.95, 2.5e-6, 1.0, 2.5e-7, 2e-6);
    const auto r = etrade::optimal_schedule(ip);
    const auto x = etrade::ac_trajectory(ac);
    for (std::size_t j = 0; j < x.size(); ++j) EXPECT_NEAR(r.holdings[j], x[j], 1e-10 * 1e6);
    EXPECT_NEAR(etrade::impact_expected_cost(ip, r.holdings) / etrade::expected_cost(ac), 1.0, 1e-12);
    EXPECT_LE(r.iterations, 3);
}

TEST(Impact, OneDayTwapIsTheSquareRootLaw) {
    const etrade::ImpactMarket m = large_cap();
    const etrade::ImpactParams p = etrade::impact_params_from_market(m);
    const auto x = etrade::twap_holdings(m.total_shares, m.n_intervals);
    const double perm = 0.5 * p.gamma_p() * m.total_shares * m.total_shares * (1.0 - 1.0 / 13.0);
    const double law = etrade::bps(etrade::impact_expected_cost(p, x) - perm, m.total_shares * m.price);
    EXPECT_NEAR(law / etrade::sqrt_law_cost_bps(0.015, 250000.0, 5e6), 1.0, 1e-12);
}

TEST(Impact, DriftShortensHorizonAndSatisfiesKkt) {
    int prev = 1 << 30;
    for (double drift : {0.0, 50.0, 200.0, 1000.0, 5000.0}) {
        etrade::ImpactMarket m = large_cap();
        m.drift_bps_per_day = drift;
        const etrade::ImpactParams p = etrade::impact_params_from_market(m);
        const auto r = etrade::optimal_schedule(p);
        EXPECT_LE(r.horizon, prev);
        prev = r.horizon;
        for (int k = r.horizon; k <= 13; ++k) EXPECT_EQ(r.holdings[k], 0.0);
    }
    EXPECT_LT(prev, 13);
}

TEST(Impact, VwapIsTwapUnderUniformWeightsAndBeatsItUnderTheCurve) {
    const auto xu = etrade::vwap_holdings(1e6, std::vector<double>(10, 1.0));
    const auto xt = etrade::twap_holdings(1e6, 10);
    for (std::size_t j = 0; j < xu.size(); ++j) EXPECT_NEAR(xu[j], xt[j], 1e-6);
    etrade::ImpactMarket m = large_cap();
    m.weights = curve();
    const etrade::ImpactParams p = etrade::impact_params_from_market(m);
    EXPECT_LT(etrade::impact_expected_cost(p, etrade::vwap_holdings(m.total_shares, m.weights)),
              etrade::impact_expected_cost(p, etrade::twap_holdings(m.total_shares, 13)));
}

TEST(Impact, PretradeReportStructure) {
    const etrade::PretradeReport rep = etrade::pretrade_report(large_cap(), {0.0, 1e-6, 1e-5});
    ASSERT_EQ(rep.rows.size(), 5u);
    EXPECT_EQ(rep.rows[0].label, "twap");
    EXPECT_EQ(rep.rows[1].label, "vwap");
    EXPECT_EQ(rep.rows[4].label, "optimal");
    for (const auto& r : rep.rows) EXPECT_DOUBLE_EQ(r.total_bps, r.impact_bps + rep.half_spread_bps);
    EXPECT_GT(rep.rows[2].sd_bps, rep.rows[3].sd_bps);
    EXPECT_GT(rep.rows[3].sd_bps, rep.rows[4].sd_bps);
    EXPECT_THROW((void)etrade::pretrade_report(large_cap(), {}), std::invalid_argument);
}

TEST(Impact, Validation) {
    auto bad = [](auto make) { EXPECT_THROW((void)make(), std::invalid_argument); };
    bad([] { return etrade::ImpactParams(1e5, 13, 1.0 / 13, 1.5, 1e-3, 0.0, 2e-8, 0.0); });
    bad([] { return etrade::ImpactParams(1e5, 13, 1.0 / 13, 1.5, 1e-3, 1.5, 2e-8, 0.0); });
    bad([] { return etrade::ImpactParams(1e5, 13, 1.0 / 13, 1.5, 0.0, 0.5, 2e-8, 0.0); });
    bad([] { return etrade::ImpactParams(1e5, 13, 1.0 / 13, 1.5, 1e-3, 0.5, 2e-8, 0.0, -1.0); });
    bad([] { return etrade::ImpactParams(1e5, 10001, 1e-4, 1.5, 1e-3, 0.5, 2e-8, 0.0); });
    bad([] {
        return etrade::ImpactParams(1e5, 13, 1.0 / 13, 1.5, 1e-3, 0.5, 2e-8, 0.0, 0.0,
                                    std::vector<double>(12, 0.1));
    });
    bad([] { return etrade::ImpactParams(1e8, 13, 1.0 / 13, 1.5, 1e-6, 0.5, 1e-3, 0.0); });
    const etrade::ImpactParams p = etrade::impact_params_from_market(large_cap());
    EXPECT_THROW((void)etrade::impact_expected_cost(p, {1.0, 0.0}), std::invalid_argument);
    auto up = etrade::twap_holdings(250000.0, 13);
    up[3] = up[1] + 1.0;
    EXPECT_THROW((void)etrade::impact_expected_cost(p, up), std::invalid_argument);
    EXPECT_THROW((void)etrade::sqrt_law_cost_bps(0.0, 1.0, 1.0), std::invalid_argument);
    EXPECT_THROW((void)etrade::vwap_holdings(1.0, {}), std::invalid_argument);
    etrade::ImpactMarket m = large_cap();
    m.drift_bps_per_day = -1.0;
    EXPECT_THROW((void)etrade::impact_params_from_market(m), std::invalid_argument);
}

TEST(Impact, ZeroSharesIsTheZeroSchedule) {
    etrade::ImpactMarket m = large_cap();
    m.total_shares = 0.0;
    m.lambda_ra = 1e-6;
    const auto r = etrade::optimal_schedule(etrade::impact_params_from_market(m));
    for (double v : r.holdings) EXPECT_EQ(v, 0.0);
    EXPECT_EQ(r.iterations, 0);
}
