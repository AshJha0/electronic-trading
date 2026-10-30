/// \file test_calibration.cpp
/// \brief Calibration estimators (API_SPEC Section 11): every case of
/// data/golden/calibration_golden.json, defining properties, the round trip
/// into the market-unit layer, and validation.
#ifndef ETRADE_DATA_DIR
#define ETRADE_DATA_DIR "../data"
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "etrade/calibration.hpp"
#include "etrade/market.hpp"
#include "mini_json.hpp"

namespace {

using mini_json::Value;

const Value& cal_golden() {
    static const Value doc = mini_json::parse_file(std::string(ETRADE_DATA_DIR) +
                                                   "/golden/calibration_golden.json");
    return doc;
}

double num(const Value& obj, const std::string& key) { return obj.at(key).number; }

std::vector<double> array_of(const Value& in, const std::string& name) {
    const int n = static_cast<int>(num(in, "n"));
    std::vector<double> out;
    for (int i = 0; i < n; ++i) out.push_back(num(in, name + "_" + std::to_string(i)));
    return out;
}

std::string idx2(std::size_t j) { return (j < 10 ? "0" : "") + std::to_string(j); }

std::vector<std::pair<std::string, double>> actual_for(const Value& c) {
    const Value& in = c.at("inputs");
    const std::string& kind = c.at("kind").str;
    std::vector<std::pair<std::string, double>> out;
    if (kind == "cal_volume_curve") {
        const auto w = etrade::volume_curve(array_of(in, "times"), array_of(in, "sizes"),
                                            num(in, "session_start"), num(in, "session_end"),
                                            static_cast<int>(num(in, "n_bins")));
        for (std::size_t j = 0; j < w.size(); ++j) out.emplace_back("frac_" + idx2(j), w[j]);
        return out;
    }
    if (kind == "cal_volatility") {
        const auto v = etrade::volatility_curve(array_of(in, "times"), array_of(in, "mids"),
                                                num(in, "session_start"),
                                                num(in, "session_end"),
                                                static_cast<int>(num(in, "n_bins")));
        out.emplace_back("daily_vol", v.daily_vol);
        out.emplace_back("n_returns", static_cast<double>(v.n_returns));
        for (std::size_t j = 0; j < v.bin_vol.size(); ++j)
            out.emplace_back("bin_vol_" + idx2(j), v.bin_vol[j]);
        return out;
    }
    if (kind == "cal_spread") {
        const auto s =
            etrade::spread_stats(array_of(in, "spreads"), array_of(in, "mids"), num(in, "tick"));
        return {{"mean_ticks", s.mean_ticks},
                {"median_ticks", s.median_ticks},
                {"mean_bps", s.mean_bps},
                {"median_bps", s.median_bps},
                {"n", static_cast<double>(s.n)}};
    }
    if (kind == "cal_fill_intensity") {
        const auto f = etrade::fill_intensity(array_of(in, "distances"), num(in, "tick"),
                                              static_cast<int>(num(in, "n_levels")),
                                              num(in, "days"), num(in, "sides"));
        return {{"a_rate", f.a_rate},
                {"k", f.k},
                {"touch_fills_per_day", f.touch_fills_per_day},
                {"halving_ticks", f.halving_ticks},
                {"n_buckets", static_cast<double>(f.n_buckets)}};
    }
    EXPECT_EQ(kind, "cal_fitted_curve");
    for (std::size_t j = 0; j < etrade::FITTED_VOLUME_CURVE_13.size(); ++j)
        out.emplace_back("w_" + idx2(j), etrade::FITTED_VOLUME_CURVE_13[j]);
    return out;
}

}  // namespace

TEST(CalibrationGolden, EveryCaseEveryKey) {
    const auto& cases = cal_golden().at("cases").arr;
    ASSERT_EQ(cases.size(), 9u);
    for (const auto& c : cases) {
        const auto actual = actual_for(c);
        const Value& expect = c.at("expect");
        ASSERT_EQ(actual.size(), expect.obj.size()) << c.at("name").str;
        const double tol = c.at("tol").number;
        for (const auto& [key, got] : actual) {
            ASSERT_TRUE(expect.has(key)) << c.at("name").str << "." << key;
            const double want = num(expect, key);
            EXPECT_LE(std::abs(got - want), tol * std::max(1.0, std::abs(want)))
                << c.at("name").str << "." << key << ": got " << got << " want " << want;
        }
    }
}

TEST(Calibration, FittedCurveSumsToOne) {
    double s = 0.0;
    for (double w : etrade::FITTED_VOLUME_CURVE_13) s += w;
    EXPECT_LE(std::abs(s - 1.0), 1e-12);
}

TEST(Calibration, VolumeCurveIsADistribution) {
    const auto w = etrade::volume_curve({1.0, 2.0, 3.0, 7.0}, {10.0, 20.0, 30.0, 40.0}, 0.0,
                                        8.0, 2);
    ASSERT_EQ(w.size(), 2u);
    EXPECT_NEAR(w[0], 0.6, 1e-15);
    EXPECT_NEAR(w[1], 0.4, 1e-15);
}

TEST(Calibration, VolatilityOfAConstantReturn) {
    const double r = 0.001;
    std::vector<double> t;
    std::vector<double> mids;
    for (int i = 0; i < 11; ++i) {
        t.push_back(static_cast<double>(i));
        mids.push_back(100.0 * std::exp(r * i));
    }
    const auto v = etrade::volatility_curve(t, mids, 0.0, 10.0, 1);
    EXPECT_NEAR(v.daily_vol, r * std::sqrt(10.0), 1e-9 * r * std::sqrt(10.0));
    EXPECT_EQ(v.n_returns, 10);
}

namespace {

std::vector<double> exact_exponential_distances(double tick) {
    const std::vector<int> counts = {1024, 512, 256, 128, 64};
    std::vector<double> dist;
    for (std::size_t b = 0; b < counts.size(); ++b) {
        const int nxt = b + 1 < counts.size() ? counts[b + 1] : 0;
        for (int i = 0; i < counts[b] - nxt; ++i)
            dist.push_back((static_cast<double>(b) + 0.5) * tick);
    }
    return dist;
}

}  // namespace

TEST(Calibration, FillIntensityRecoversAnExactExponential) {
    const double tick = 0.01;
    const auto f = etrade::fill_intensity(exact_exponential_distances(tick), tick, 5);
    const double k_want = -std::log(0.5) / tick;
    EXPECT_NEAR(f.k, k_want, 1e-12 * k_want);
    EXPECT_NEAR(f.halving_ticks, 1.0, 1e-12);
    EXPECT_NEAR(f.touch_fills_per_day, 512.0, 1e-12 * 512.0);
}

TEST(Calibration, FillIntensityRoundTripsThroughTheMarketLayer) {
    const Value* fill_case = nullptr;
    for (const auto& c : cal_golden().at("cases").arr)
        if (c.at("kind").str == "cal_fill_intensity") {
            fill_case = &c;
            break;
        }
    ASSERT_NE(fill_case, nullptr);
    const Value& in = fill_case->at("inputs");
    const auto f = etrade::fill_intensity(array_of(in, "distances"), num(in, "tick"),
                                          static_cast<int>(num(in, "n_levels")));
    etrade::ASMarket m;
    m.price = 100.0;
    m.tick = num(in, "tick");
    m.daily_vol = 0.01;
    m.touch_fills_per_day = f.touch_fills_per_day;
    m.halving_ticks = f.halving_ticks;
    m.skew_ticks = 1.0;
    m.q_max = 10;
    m.liq_ticks = 1.0;
    m.dt_seconds = 5.0;
    const auto a = etrade::as_params_from_market(m);
    EXPECT_NEAR(a.k, f.k, 1e-12 * f.k);
    EXPECT_NEAR(a.a_rate, f.a_rate, 1e-12 * f.a_rate);
}

TEST(Calibration, Validation) {
    using V = std::vector<double>;
    const std::vector<std::pair<std::function<void()>, std::string>> cases = {
        {[] { (void)etrade::volume_curve(V{0.0}, V{1.0}, 0.0, 0.0, 1); },
         "session_end must be finite and > session_start"},
        {[] { (void)etrade::volume_curve(V{0.0}, V{1.0}, 0.0, 1.0, 0); },
         "n_bins must be an integer"},
        {[] { (void)etrade::volume_curve(V{0.0}, V{1.0, 2.0}, 0.0, 1.0, 1); },
         "times and sizes must have the same length"},
        {[] { (void)etrade::volume_curve(V{0.0}, V{-1.0}, 0.0, 1.0, 1); },
         "sizes must be finite and >= 0"},
        {[] { (void)etrade::volume_curve(V{5.0}, V{1.0}, 0.0, 1.0, 1); },
         "no volume inside the session"},
        {[] { (void)etrade::volatility_curve(V{0.0, 1.0}, V{1.0, 0.0}, 0.0, 1.0, 1); },
         "mids must be finite and > 0"},
        {[] { (void)etrade::volatility_curve(V{1.0, 0.0}, V{1.0, 1.0}, 0.0, 1.0, 1); },
         "times must be non-decreasing"},
        {[] { (void)etrade::volatility_curve(V{0.0}, V{1.0}, 0.0, 1.0, 1); },
         "no return inside the session"},
        {[] { (void)etrade::spread_stats(V{}, V{}, 0.01); }, "spreads must not be empty"},
        {[] { (void)etrade::spread_stats(V{0.01}, V{100.0}, 0.0); },
         "tick must be finite and > 0"},
        {[] { (void)etrade::spread_stats(V{-0.01}, V{100.0}, 0.01); },
         "spreads must be finite and >= 0"},
        {[] { (void)etrade::fill_intensity(V{0.005}, 0.01, 1); }, "n_levels must be an integer"},
        {[] { (void)etrade::fill_intensity(V{0.005}, 0.01, 3, 0.0); },
         "days must be finite and > 0"},
        {[] { (void)etrade::fill_intensity(V{0.005}, 0.01, 3, 1.0, 0.0); },
         "sides must be finite and > 0"},
        {[] { (void)etrade::fill_intensity(V{-1.0}, 0.01, 3); },
         "distances must be finite and >= 0"},
        {[] { (void)etrade::fill_intensity(V{0.005}, 0.01, 3); },
         "fewer than two distance buckets have fills"},
        {[] { (void)etrade::fill_intensity(V{0.015, 0.015}, 0.01, 3); },
         "fitted fill intensity does not decay with distance"},
    };
    for (const auto& [call, msg] : cases) {
        try {
            call();
            ADD_FAILURE() << "no throw, expected: " << msg;
        } catch (const std::invalid_argument& e) {
            EXPECT_EQ(std::string(e.what()).rfind(msg, 0), 0u)
                << "got '" << e.what() << "' want prefix '" << msg << "'";
        }
    }
}
