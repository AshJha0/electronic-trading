/// \file test_tca.cpp
/// \brief Post-trade TCA (API_SPEC Section 10): every case of
/// data/golden/tca_golden.json, sign conventions, invariances and validation.
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

#include "etrade/tca.hpp"
#include "mini_json.hpp"

namespace {

using mini_json::Value;
using etrade::Side;

const Value& tca_golden() {
    static const Value doc =
        mini_json::parse_file(std::string(ETRADE_DATA_DIR) + "/golden/tca_golden.json");
    return doc;
}

double num(const Value& obj, const std::string& key) { return obj.at(key).number; }

std::vector<double> arr(const Value& in, const std::string& prefix, const std::string& n_key) {
    std::vector<double> out;
    const int n = static_cast<int>(num(in, n_key));
    for (int i = 0; i < n; ++i) out.push_back(num(in, prefix + "_" + std::to_string(i)));
    return out;
}

std::vector<int> ints(const Value& in, const std::string& prefix, const std::string& n_key) {
    std::vector<int> out;
    for (double v : arr(in, prefix, n_key)) out.push_back(static_cast<int>(v));
    return out;
}

Side side_of(const Value& in) { return num(in, "side_buy") == 1.0 ? Side::kBuy : Side::kSell; }

std::vector<std::pair<std::string, double>> indexed(const std::string& prefix,
                                                    const std::vector<double>& xs) {
    std::vector<std::pair<std::string, double>> out;
    for (std::size_t i = 0; i < xs.size(); ++i) out.emplace_back(prefix + "_" + std::to_string(i), xs[i]);
    return out;
}

std::vector<std::pair<std::string, double>> actual_for(const Value& c) {
    const Value& in = c.at("inputs");
    const std::string& kind = c.at("kind").str;
    if (kind == "slippage") {
        const auto r = etrade::slippage(side_of(in), num(in, "arrival_price"),
                                        num(in, "close_price"), ints(in, "bin", "n_fills"),
                                        arr(in, "qty", "n_fills"), arr(in, "px", "n_fills"),
                                        arr(in, "vol", "n_bins"), arr(in, "vwap", "n_bins"));
        return {{"executed_qty", r.executed_qty},   {"avg_price", r.avg_price},
                {"market_vwap", r.market_vwap},     {"interval_vwap", r.interval_vwap},
                {"arrival_bps", r.arrival_bps},     {"vwap_bps", r.vwap_bps},
                {"interval_vwap_bps", r.interval_vwap_bps}, {"close_bps", r.close_bps}};
    }
    if (kind == "markout") {
        const auto m = etrade::markouts(side_of(in), ints(in, "step", "n_fills"),
                                        arr(in, "qty", "n_fills"), arr(in, "mid", "n_mids"),
                                        ints(in, "h", "n_horizons"));
        auto out = indexed("per_fill", m.per_fill);
        for (auto& kv : indexed("per_share", m.per_share)) out.push_back(kv);
        return out;
    }
    if (kind == "participation") {
        const auto p = etrade::participation(ints(in, "bin", "n_fills"), arr(in, "qty", "n_fills"),
                                             arr(in, "vol", "n_bins"));
        auto out = indexed("rate", p.per_bin);
        out.emplace_back("overall", p.overall);
        return out;
    }
    if (kind == "welch") {
        const auto w = etrade::welch_ttest(arr(in, "a", "n_a"), arr(in, "b", "n_b"));
        return {{"mean_diff", w.mean_diff}, {"t_stat", w.t_stat}, {"df", w.df},
                {"p_value", w.p_value}};
    }
    return {{"cdf", etrade::student_t_cdf(num(in, "t"), num(in, "df"))}};
}

const std::vector<double> kVols = {1000.0, 2000.0, 1500.0, 500.0};
const std::vector<double> kVwaps = {100.00, 100.05, 100.12, 100.08};
const std::vector<int> kBins = {0, 1, 1, 2};
const std::vector<double> kQtys = {100.0, 150.0, 50.0, 200.0};
const std::vector<double> kPxs = {100.02, 100.06, 100.07, 100.15};

void expect_invalid(const std::function<void()>& f, const std::string& fragment) {
    try {
        f();
        ADD_FAILURE() << "expected std::invalid_argument containing '" << fragment << "'";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find(fragment), std::string::npos) << e.what();
    }
}

}  // namespace

TEST(TcaGolden, EveryCaseEveryKey) {
    const auto& cases = tca_golden().at("cases").arr;
    ASSERT_EQ(cases.size(), 11u);
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

TEST(Tca, BuySellSymmetry) {
    const auto b = etrade::slippage(Side::kBuy, 100.01, 100.09, kBins, kQtys, kPxs, kVols, kVwaps);
    const auto s = etrade::slippage(Side::kSell, 100.01, 100.09, kBins, kQtys, kPxs, kVols, kVwaps);
    EXPECT_NEAR(b.arrival_bps, -s.arrival_bps, 1e-12);
    EXPECT_NEAR(b.vwap_bps, -s.vwap_bps, 1e-12);
    EXPECT_NEAR(b.interval_vwap_bps, -s.interval_vwap_bps, 1e-12);
    EXPECT_NEAR(b.close_bps, -s.close_bps, 1e-12);
}

TEST(Tca, PriceScaleAndShiftInvariance) {
    const double c = 3.7;
    const double k = 25.0;
    std::vector<double> px_c, vw_c, px_k, vw_k;
    for (double p : kPxs) {
        px_c.push_back(p * c);
        px_k.push_back(p + k);
    }
    for (double p : kVwaps) {
        vw_c.push_back(p * c);
        vw_k.push_back(p + k);
    }
    const auto r1 = etrade::slippage(Side::kBuy, 100.01, 100.09, kBins, kQtys, kPxs, kVols, kVwaps);
    const auto r2 =
        etrade::slippage(Side::kBuy, 100.01 * c, 100.09 * c, kBins, kQtys, px_c, kVols, vw_c);
    EXPECT_NEAR(r2.arrival_bps, r1.arrival_bps, 1e-10 * std::abs(r1.arrival_bps));
    EXPECT_NEAR(r2.vwap_bps, r1.vwap_bps, 1e-10 * std::abs(r1.vwap_bps));
    const auto r3 =
        etrade::slippage(Side::kBuy, 100.01 + k, 100.09 + k, kBins, kQtys, px_k, kVols, vw_k);
    EXPECT_NEAR(r3.arrival_bps * (100.01 + k), r1.arrival_bps * 100.01, 1e-5);
    EXPECT_NEAR(r3.vwap_bps * r3.market_vwap, r1.vwap_bps * r1.market_vwap, 1e-5);
    EXPECT_NEAR(r3.close_bps * (100.09 + k), r1.close_bps * 100.09, 1e-5);
}

TEST(Tca, ArrivalSlippageIsPeroldTradingCost) {
    const auto r = etrade::slippage(Side::kBuy, 100.01, 100.09, kBins, kQtys, kPxs, kVols, kVwaps);
    const auto d = etrade::decompose_shortfall(Side::kBuy, 100.01, 100.01, 100.09, 500.0, kQtys, kPxs);
    EXPECT_NEAR(r.arrival_bps * 100.01 * r.executed_qty / 1e4, d.trading, 1e-9);
}

TEST(Tca, MarkoutSign) {
    const std::vector<double> mids = {10.0, 10.5, 9.0};
    const auto buy = etrade::markouts(Side::kBuy, {0, 1}, {1.0, 3.0}, mids, {0, 1});
    EXPECT_EQ(buy.per_fill[0], 0.0);
    EXPECT_NEAR(buy.per_fill[1], -0.5, 1e-15);
    EXPECT_NEAR(buy.per_share[1], -1.0, 1e-15);
    const auto sell = etrade::markouts(Side::kSell, {0, 1}, {1.0, 3.0}, mids, {0, 1});
    EXPECT_NEAR(sell.per_fill[1], 0.5, 1e-15);
}

TEST(Tca, KnownTValues) {
    EXPECT_NEAR(etrade::student_t_cdf(1.0, 1.0), 0.75, 1e-14);
    EXPECT_EQ(etrade::student_t_cdf(0.0, 4.0), 0.5);
    EXPECT_NEAR(etrade::student_t_cdf(1.5, 2.0), 0.5 + 1.5 / (2.0 * std::sqrt(4.25)), 1e-14);
    EXPECT_NEAR(etrade::ln_gamma(0.5), 0.5 * std::log(3.14159265358979323846), 1e-14);
    EXPECT_NEAR(etrade::ln_gamma(10.0), std::log(362880.0), 1e-13);
}

TEST(Tca, Validation) {
    expect_invalid([] { (void)etrade::slippage(Side::kBuy, 0.0, 1.0, kBins, kQtys, kPxs, kVols, kVwaps); },
                   "arrival_price");
    expect_invalid([] { (void)etrade::slippage(Side::kBuy, 1.0, 1.0, {0, 1, 1, 4}, kQtys, kPxs, kVols, kVwaps); },
                   "fill_bins must be integers in [0, 4)");
    expect_invalid([] { (void)etrade::slippage(Side::kBuy, 1.0, 1.0, {0, 1, 1, -1}, kQtys, kPxs, kVols, kVwaps); },
                   "fill_bins");
    expect_invalid([] { (void)etrade::slippage(Side::kBuy, 1.0, 1.0, kBins, {0.0, 0.0, 0.0, 0.0}, kPxs, kVols, kVwaps); },
                   "executed quantity");
    expect_invalid([] { (void)etrade::slippage(Side::kBuy, 1.0, 1.0, kBins, kQtys, kPxs, {0.0, 0.0, 0.0, 1.0}, kVwaps); },
                   "traded bins");
    expect_invalid([] { (void)etrade::slippage(Side::kBuy, 1.0, 1.0, {}, {}, {}, kVols, kVwaps); },
                   "non-empty");
    expect_invalid([] { (void)etrade::markouts(Side::kBuy, {0, 1}, {1.0, 1.0}, {1.0, 1.0}, {1}); },
                   "exceeds mid series");
    expect_invalid([] { (void)etrade::markouts(Side::kBuy, {0}, {1.0}, {1.0, 1.0}, {-1}); },
                   "horizons");
    expect_invalid([] { (void)etrade::participation({0}, {1001.0}, kVols); }, "exceeds market volume");
    expect_invalid([] { (void)etrade::participation({0}, {0.0}, {0.0}); }, "total market volume");
    expect_invalid([] { (void)etrade::welch_ttest({1.0}, {1.0, 2.0}); }, "a must have at least 2");
    expect_invalid([] { (void)etrade::welch_ttest({1.0, 1.0}, {2.0, 2.0}); }, "zero variance");
    expect_invalid([] { (void)etrade::student_t_cdf(1.0, 0.0); }, "df");
    expect_invalid([] { (void)etrade::betainc_reg(1.0, 1.0, 1.5); }, "x must be in");
    expect_invalid([] { (void)etrade::ln_gamma(-1.0); }, "x must be finite");
}
