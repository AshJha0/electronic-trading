/// \file test_market.cpp
/// \brief Market-unit layer (API_SPEC Section 7): every case of
/// data/golden/market_golden.json, the paper identity, the defining
/// properties of the A-S conversion, and validation.
#ifndef ETRADE_DATA_DIR
#define ETRADE_DATA_DIR "../data"
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>

#include "etrade/market.hpp"
#include "mini_json.hpp"

namespace {

using mini_json::Value;

const Value& market_golden() {
    static const Value doc =
        mini_json::parse_file(std::string(ETRADE_DATA_DIR) + "/golden/market_golden.json");
    return doc;
}

const Value& presets() {
    static const Value doc = mini_json::parse_file(std::string(ETRADE_DATA_DIR) + "/presets.json");
    return doc;
}

double num(const Value& obj, const std::string& key) { return obj.at(key).number; }

double num_or(const Value& obj, const std::string& key, double fallback) {
    return obj.has(key) ? obj.at(key).number : fallback;
}

etrade::ACMarket ac_from(const Value& in) {
    etrade::ACMarket m;
    m.price = num(in, "price");
    m.spread = num(in, "spread");
    m.adv = num(in, "adv");
    m.daily_vol = num(in, "daily_vol");
    m.total_shares = num(in, "total_shares");
    m.n_intervals = static_cast<int>(num(in, "n_intervals"));
    m.tau = num(in, "tau");
    m.lambda_ra = num(in, "lambda_ra");
    m.temp_frac = num_or(in, "temp_frac", etrade::kDefaultTempFrac);
    m.perm_frac = num_or(in, "perm_frac", etrade::kDefaultPermFrac);
    return m;
}

etrade::ASMarket as_from(const Value& in) {
    etrade::ASMarket m;
    m.price = num(in, "price");
    m.tick = num(in, "tick");
    m.daily_vol = num(in, "daily_vol");
    m.touch_fills_per_day = num(in, "touch_fills_per_day");
    m.halving_ticks = num(in, "halving_ticks");
    m.skew_ticks = num(in, "skew_ticks");
    m.q_max = static_cast<int>(num(in, "q_max"));
    m.liq_ticks = num(in, "liq_ticks");
    m.dt_seconds = num(in, "dt_seconds");
    m.session_seconds = num_or(in, "session_seconds", etrade::kDefaultSessionSeconds);
    m.horizon_days = num_or(in, "horizon_days", 1.0);
    return m;
}

std::vector<std::pair<std::string, double>> actual_for(const Value& c) {
    const Value& in = c.at("inputs");
    if (c.at("kind").str == "market_ac") {
        const etrade::ACMarket m = ac_from(in);
        const etrade::ACParams p = etrade::ac_params_from_market(m);
        const double notional = m.total_shares * m.price;
        return {{"sigma", p.sigma},
                {"eta", p.eta},
                {"gamma_p", p.gamma_p},
                {"kappa", etrade::ac_kappa(p)},
                {"e_cost_bps", etrade::bps(etrade::expected_cost(p), notional)},
                {"sd_cost_bps", etrade::bps(std::sqrt(etrade::variance_cost(p)), notional)},
                {"spread_cost_bps",
                 etrade::bps(etrade::half_spread_cost(m.spread, m.total_shares), notional)}};
    }
    const etrade::ASMarket m = as_from(in);
    const etrade::ASParams a = etrade::as_params_from_market(m);
    return {{"sigma", a.sigma},
            {"k", a.k},
            {"a_rate", a.a_rate},
            {"gamma", a.gamma},
            {"eta_liq", a.eta_liq},
            {"dt", a.dt},
            {"n_steps", static_cast<double>(a.n_steps())},
            {"spread_open_bps", etrade::bps(etrade::optimal_spread(a, 0.0), m.price)},
            {"spread_close_bps", etrade::bps(etrade::optimal_spread(a, a.horizon), m.price)}};
}

}  // namespace

TEST(MarketGolden, EveryCaseEveryKey) {
    const auto& cases = market_golden().at("cases").arr;
    ASSERT_EQ(cases.size(), 15u);
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

TEST(MarketGolden, PresetsCovered) {
    std::set<std::string> names;
    for (const auto& c : market_golden().at("cases").arr) names.insert(c.at("name").str);
    for (const auto& [preset, body] : presets().obj) {
        for (const char* side : {"ac", "as"}) {
            if (body.has(side)) {
                EXPECT_TRUE(names.count("market_" + std::string(side) + "_" + preset)) << preset;
            }
        }
    }
}

TEST(Market, PaperCalibrationReproducesLiquidSet) {
    const etrade::ACParams p =
        etrade::ac_params_from_market(ac_from(presets().at("paper_almgren_chriss_2000").at("ac")));
    EXPECT_NEAR(p.eta, 2.5e-6, 1e-21);
    EXPECT_NEAR(p.gamma_p, 2.5e-7, 1e-22);
    EXPECT_NEAR(p.sigma, 0.95, 1e-15);
}

TEST(Market, AsConversionDefiningProperties) {
    const etrade::ASMarket m = as_from(presets().at("us_large_cap").at("as"));
    const etrade::ASParams a = etrade::as_params_from_market(m);
    const double d0 = m.tick / 2.0;
    EXPECT_NEAR(etrade::intensity(a, d0) / m.touch_fills_per_day, 1.0, 1e-12);
    EXPECT_NEAR(etrade::intensity(a, d0 + m.halving_ticks * m.tick) / etrade::intensity(a, d0),
                0.5, 1e-12);
    EXPECT_NEAR((a.s0 - etrade::reservation_price(a, a.s0, 1.0, 0.0)) / (m.skew_ticks * m.tick),
                1.0, 1e-12);
    EXPECT_EQ(a.n_steps(), 4680);
}

TEST(Market, ReportingHelpers) {
    EXPECT_DOUBLE_EQ(etrade::bps(5.0, 1e4), 5.0);
    EXPECT_DOUBLE_EQ(etrade::half_spread_cost(0.01, 1000.0), 5.0);
    EXPECT_THROW((void)etrade::bps(1.0, 0.0), std::invalid_argument);
    EXPECT_THROW((void)etrade::bps(std::nan(""), 1.0), std::invalid_argument);
    EXPECT_THROW((void)etrade::half_spread_cost(0.0, 1.0), std::invalid_argument);
}

TEST(Market, AcValidation) {
    const etrade::ACMarket base = ac_from(presets().at("us_large_cap").at("ac"));
    auto bad = [&](auto mutate) {
        etrade::ACMarket m = base;
        mutate(m);
        EXPECT_THROW((void)etrade::ac_params_from_market(m), std::invalid_argument);
    };
    bad([](etrade::ACMarket& m) { m.price = 0.0; });
    bad([](etrade::ACMarket& m) { m.spread = -0.01; });
    bad([](etrade::ACMarket& m) { m.adv = INFINITY; });
    bad([](etrade::ACMarket& m) { m.daily_vol = std::nan(""); });
    bad([](etrade::ACMarket& m) { m.temp_frac = 0.0; });
    bad([](etrade::ACMarket& m) { m.temp_frac = 1.5; });
    bad([](etrade::ACMarket& m) { m.perm_frac = 0.0; });
}

TEST(Market, AsValidation) {
    const etrade::ASMarket base = as_from(presets().at("us_large_cap").at("as"));
    auto bad = [&](auto mutate) {
        etrade::ASMarket m = base;
        mutate(m);
        EXPECT_THROW((void)etrade::as_params_from_market(m), std::invalid_argument);
    };
    bad([](etrade::ASMarket& m) { m.price = -1.0; });
    bad([](etrade::ASMarket& m) { m.tick = 0.0; });
    bad([](etrade::ASMarket& m) { m.daily_vol = 0.0; });
    bad([](etrade::ASMarket& m) { m.touch_fills_per_day = 0.0; });
    bad([](etrade::ASMarket& m) { m.halving_ticks = INFINITY; });
    bad([](etrade::ASMarket& m) { m.skew_ticks = 0.0; });
    bad([](etrade::ASMarket& m) { m.liq_ticks = -1.0; });
    bad([](etrade::ASMarket& m) { m.dt_seconds = 0.0; });
    bad([](etrade::ASMarket& m) { m.session_seconds = std::nan(""); });
    bad([](etrade::ASMarket& m) { m.horizon_days = 0.0; });
    bad([](etrade::ASMarket& m) { m.dt_seconds = 7.0; });  // does not divide the session
}
