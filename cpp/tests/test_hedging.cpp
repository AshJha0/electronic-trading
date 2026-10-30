/// \file test_hedging.cpp
/// \brief Hedged market making and risk controls (API_SPEC Section 14): every
/// case of data/golden/hedging_golden.json, the per-path attribution
/// identity, no hedge = Section 9 in distribution, the loss-limit overshoot
/// bound, the effect of each control, determinism and validation.
#ifndef ETRADE_DATA_DIR
#define ETRADE_DATA_DIR "../data"
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "etrade/avellaneda_stoikov.hpp"
#include "etrade/hedging.hpp"
#include "etrade/market.hpp"
#include "etrade/mm.hpp"
#include "mini_json.hpp"

namespace {

using mini_json::Value;
using etrade::HedgeStat;

constexpr int kOwnPaths = 2000;

const Value& hedging_golden() {
    static const Value doc =
        mini_json::parse_file(std::string(ETRADE_DATA_DIR) + "/golden/hedging_golden.json");
    return doc;
}

double num(const Value& o, const std::string& k) { return o.at(k).number; }

/// The preset in market units (hedge_sim inputs carry the preset tick under
/// "tick" and the MMConfig tick under "quote_tick").
etrade::ASParams params_from(const Value& in) {
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
    m.session_seconds = num(in, "session_seconds");
    m.horizon_days = num(in, "horizon_days");
    return etrade::as_params_from_market(m);
}

/// The us_large_cap preset (4680 steps).
etrade::ASParams large_cap() {
    etrade::ASMarket m;
    m.price = 100.0;
    m.tick = 0.01;
    m.daily_vol = 0.015;
    m.touch_fills_per_day = 1170.0;
    m.halving_ticks = 0.462;
    m.skew_ticks = 2.25;
    m.q_max = 50;
    m.liq_ticks = 1.0;
    m.dt_seconds = 5.0;
    m.session_seconds = 23400.0;
    m.horizon_days = 1.0;
    return etrade::as_params_from_market(m);
}

etrade::MMConfig book() {
    etrade::MMConfig c;
    c.tick = 0.01;
    c.maker_fee = 0.002;
    c.informed_prob = 0.15;
    return c;
}

etrade::MMConfig fixed_book() {
    etrade::MMConfig c = book();
    c.agent = etrade::MMAgent::kFixed;
    return c;
}

etrade::HedgeConfig full_hedge() {
    etrade::HedgeConfig h;
    h.rho = 0.9;
    h.hedge_ratio = 0.9;
    h.hedge_cost = 0.005;
    return h;
}

etrade::HedgedResult sim(const etrade::MMConfig& cfg,
                         const etrade::HedgeConfig& hedge = etrade::HedgeConfig{},
                         const etrade::RiskConfig& risk = etrade::RiskConfig{}, int n = 600,
                         std::uint64_t seed = 11) {
    return etrade::simulate_hedged(large_cap(), n, seed, cfg, hedge, risk);
}

/// The golden statistic keys (pnl_var handled separately).
std::vector<std::pair<std::string, HedgeStat>> golden_stats() {
    return {{"pnl", HedgeStat::kPnl},
            {"spread", HedgeStat::kSpread},
            {"fees", HedgeStat::kFees},
            {"inventory", HedgeStat::kInventory},
            {"hedge_pnl", HedgeStat::kHedgePnl},
            {"hedge_costs", HedgeStat::kHedgeCosts},
            {"risk_costs", HedgeStat::kRiskCosts},
            {"fills", HedgeStat::kFills},
            {"rms_q", HedgeStat::kRmsQ},
            {"rms_x", HedgeStat::kRmsX},
            {"rms_skew", HedgeStat::kRmsSkew},
            {"quote_updates", HedgeStat::kQuoteUpdates},
            {"loss_stop", HedgeStat::kLossStop},
            {"kill_stop", HedgeStat::kKillStop},
            {"pos_flattens", HedgeStat::kPosFlattens}};
}

etrade::MMConfig config_from(const Value& in) {
    etrade::MMConfig c;
    const int code = static_cast<int>(num(in, "agent_code"));
    c.agent = code == 0 ? etrade::MMAgent::kAs
                        : (code == 1 ? etrade::MMAgent::kGlft : etrade::MMAgent::kFixed);
    c.tick = num(in, "quote_tick");
    c.maker_fee = num(in, "maker_fee");
    c.latency_steps = static_cast<int>(num(in, "latency_steps"));
    c.informed_prob = num(in, "informed_prob");
    c.signal_accuracy = num(in, "signal_accuracy");
    return c;
}

etrade::HedgeConfig hedge_from(const Value& in) {
    etrade::HedgeConfig h;
    h.rho = num(in, "rho");
    h.hedge_ratio = num(in, "hedge_ratio");
    h.hedge_cost = num(in, "hedge_cost");
    h.hedge_threshold = num(in, "hedge_threshold");
    h.hedge_target = num(in, "hedge_target");
    return h;
}

etrade::RiskConfig risk_from(const Value& in) {
    etrade::RiskConfig r;
    r.loss_limit = num(in, "loss_limit");
    r.position_limit = static_cast<int>(num(in, "position_limit"));
    r.flatten_penalty = num(in, "flatten_penalty");
    r.throttle_updates = static_cast<int>(num(in, "throttle_updates"));
    r.throttle_window = static_cast<int>(num(in, "throttle_window"));
    r.kill_step = static_cast<int>(num(in, "kill_step"));
    return r;
}

/// Exact kinds: (key, actual) for every expect key.
std::vector<std::pair<std::string, double>> exact_actual(const Value& c) {
    const std::string kind = c.at("kind").str;
    const Value& in = c.at("inputs");
    std::vector<std::pair<std::string, double>> out;
    if (kind == "hedge_trade") {
        out.emplace_back("dy", etrade::hedge_trade(num(in, "q"), num(in, "y"),
                                                   num(in, "hedge_ratio"),
                                                   num(in, "hedge_threshold"),
                                                   num(in, "hedge_target")));
    } else if (kind == "hedge_quantile") {
        std::vector<double> v;
        for (int i = 0; i < 10; ++i) v.push_back(num(in, "v_" + std::to_string(i)));
        out.emplace_back("q", etrade::quantile(v, num(in, "p")));
    }
    return out;
}

void expect_identity(const etrade::HedgedResult& r) {
    EXPECT_LE(r.max_identity_error, 1e-9);
    for (int i = 0; i < r.n_paths(); ++i) {
        const double recon = r.spread[i] + r.fees[i] + r.inventory[i] - r.liquidation[i] +
                             r.hedge_pnl[i] - r.hedge_costs[i] - r.risk_costs[i];
        EXPECT_LE(std::abs(recon - r.pnl[i]), 1e-9 * std::max(1.0, std::abs(r.pnl[i])));
        EXPECT_GE(r.hedge_costs[i], 0.0);
        EXPECT_GE(r.risk_costs[i], 0.0);
    }
}

}  // namespace

TEST(HedgingGolden, EveryCaseEveryKey) {
    const auto& cases = hedging_golden().at("cases").arr;
    ASSERT_EQ(cases.size(), 18u);
    int n_sim = 0;
    int n_exact = 0;
    for (const auto& c : cases) {
        const std::string name = c.at("name").str;
        const Value& expect = c.at("expect");
        const double tol = c.at("tol").number;
        if (c.at("kind").str == "hedge_sim") {
            ++n_sim;
            const Value& in = c.at("inputs");
            const auto res = etrade::simulate_hedged(
                params_from(in), kOwnPaths, static_cast<std::uint64_t>(num(in, "seed")) + 1,
                config_from(in), hedge_from(in), risk_from(in));
            expect_identity(res);
            const auto stats = golden_stats();
            ASSERT_EQ(expect.obj.size(), 2 * (stats.size() + 1)) << name;
            auto check = [&](const std::string& key, double got, double se_own) {
                const double want = num(expect, key);
                const double se_ref = num(expect, key + "_se");
                const double band = tol * std::hypot(se_ref, se_own);
                EXPECT_LE(std::abs(got - want), band + 1e-12)
                    << name << "." << key << ": got " << got << " want " << want;
            };
            for (const auto& [key, stat] : stats) check(key, res.mean(stat), res.se(stat));
            check("pnl_var", res.var(HedgeStat::kPnl), res.var_se(HedgeStat::kPnl));
            continue;
        }
        ++n_exact;
        const auto actual = exact_actual(c);
        ASSERT_EQ(actual.size(), expect.obj.size()) << name;
        for (const auto& [key, got] : actual) {
            const double want = num(expect, key);
            EXPECT_LE(std::abs(got - want), tol * std::max(1.0, std::abs(want)))
                << name << "." << key << ": got " << got << " want " << want;
        }
    }
    EXPECT_EQ(n_exact, 8);
    EXPECT_EQ(n_sim, 10);
}

TEST(Hedging, AttributionIdentityOnEveryPath) {
    etrade::MMConfig glft;
    glft.agent = etrade::MMAgent::kGlft;
    glft.tick = 0.01;
    glft.latency_steps = 2;
    etrade::RiskConfig pos;
    pos.position_limit = 3;
    pos.flatten_penalty = 0.02;
    etrade::HedgeConfig band;
    band.rho = -0.5;
    band.hedge_ratio = 0.4;
    band.hedge_cost = 0.01;
    band.hedge_threshold = 2.0;
    band.hedge_target = 0.5;
    etrade::RiskConfig all;
    all.loss_limit = 2.0;
    all.flatten_penalty = 0.01;
    all.throttle_updates = 2;
    all.throttle_window = 5;
    all.kill_step = 3000;
    expect_identity(sim(etrade::MMConfig{}, {}, {}, 300, 9));
    expect_identity(sim(book(), full_hedge(), {}, 300, 9));
    expect_identity(sim(glft, full_hedge(), pos, 300, 9));
    expect_identity(sim(fixed_book(), band, all, 300, 9));
}

TEST(Hedging, NoHedgeNoControlsIsSection9InDistribution) {
    const etrade::ASParams p = large_cap();
    const std::pair<HedgeStat, etrade::MMStat> pairs[] = {
        {HedgeStat::kPnl, etrade::MMStat::kPnl},
        {HedgeStat::kSpread, etrade::MMStat::kSpread},
        {HedgeStat::kFees, etrade::MMStat::kFees},
        {HedgeStat::kInventory, etrade::MMStat::kInventory},
        {HedgeStat::kLiquidation, etrade::MMStat::kLiquidation},
        {HedgeStat::kMarkout, etrade::MMStat::kMarkout},
        {HedgeStat::kFills, etrade::MMStat::kFills},
        {HedgeStat::kInformedFills, etrade::MMStat::kInformedFills},
        {HedgeStat::kRmsQ, etrade::MMStat::kRmsQ}};
    for (const auto& cfg : {book(), fixed_book()}) {
        const auto a = etrade::simulate_hedged(p, kOwnPaths, 31, cfg);
        const auto b = etrade::simulate_mm(p, kOwnPaths, 32, cfg);
        for (const auto& [hs, ms] : pairs) {
            const double band = 4.0 * std::hypot(a.se(hs), b.se(ms));
            EXPECT_LE(std::abs(a.mean(hs) - b.mean(ms)), band + 1e-12);
        }
        for (auto stat : {HedgeStat::kHedgePnl, HedgeStat::kHedgeCosts, HedgeStat::kRiskCosts,
                          HedgeStat::kLossStop, HedgeStat::kKillStop}) {
            for (double v : a.path(stat)) EXPECT_EQ(v, 0.0);
        }
        EXPECT_EQ(a.rms_x, a.rms_q);
        for (double v : a.quote_updates) EXPECT_EQ(v, p.n_steps());
    }
}

TEST(Hedging, LossLimitOvershootIsAtMostOneStepMove) {
    etrade::RiskConfig r;
    r.loss_limit = 3.0;
    r.flatten_penalty = 0.01;
    const auto res = sim(fixed_book(), {}, r, 800, 5);
    int stopped = 0;
    for (int i = 0; i < res.n_paths(); ++i) {
        EXPECT_LE(res.overshoot[i], res.stop_move[i] + 1e-9);
        if (res.loss_stop[i] == 1.0) {
            ++stopped;
            EXPECT_EQ(res.liquidation[i], 0.0);
            EXPECT_LT(res.quote_updates[i], 4680.0);
        } else {
            EXPECT_EQ(res.overshoot[i], 0.0);
        }
    }
    EXPECT_GT(stopped, 100);
}

TEST(Hedging, HedgingCutsVarianceAndSkewAtACost) {
    const auto fixed_un = sim(fixed_book());
    const auto fixed_h = sim(fixed_book(), full_hedge());
    EXPECT_LT(fixed_h.var(HedgeStat::kPnl), 0.5 * fixed_un.var(HedgeStat::kPnl));
    EXPECT_LT(fixed_h.mean(HedgeStat::kRmsX), 0.5 * fixed_h.mean(HedgeStat::kRmsQ));
    EXPECT_GT(fixed_h.mean(HedgeStat::kHedgeCosts), 0.0);
    EXPECT_GT(fixed_h.mean(HedgeStat::kHedgeTrades), 0.0);
    const auto as_un = sim(book());
    const auto as_h = sim(book(), full_hedge());
    EXPECT_LT(as_h.mean(HedgeStat::kRmsSkew), 0.6 * as_un.mean(HedgeStat::kRmsSkew));
    EXPECT_LT(as_h.var(HedgeStat::kPnl), as_un.var(HedgeStat::kPnl));
}

TEST(Hedging, EachControlTriggersAndCutsTheTail) {
    const auto base = sim(fixed_book());
    etrade::RiskConfig rp;
    rp.position_limit = 10;
    rp.flatten_penalty = 0.01;
    const auto pos = sim(fixed_book(), {}, rp);
    EXPECT_GT(pos.mean(HedgeStat::kPosFlattens), 1.0);
    EXPECT_GT(pos.mean(HedgeStat::kRiskCosts), 0.0);
    EXPECT_GT(pos.quantile(HedgeStat::kPnl, 0.05), base.quantile(HedgeStat::kPnl, 0.05));
    etrade::RiskConfig rl;
    rl.loss_limit = 5.0;
    rl.flatten_penalty = 0.01;
    const auto loss = sim(fixed_book(), {}, rl);
    EXPECT_GT(loss.mean(HedgeStat::kLossStop), 0.2);
    EXPECT_GT(loss.quantile(HedgeStat::kPnl, 0.05), base.quantile(HedgeStat::kPnl, 0.05));
    etrade::RiskConfig rk;
    rk.kill_step = 1000;
    const auto kill = sim(fixed_book(), {}, rk);
    for (int i = 0; i < kill.n_paths(); ++i) {
        EXPECT_EQ(kill.kill_stop[i], 1.0);
        EXPECT_EQ(kill.quote_updates[i], 1000.0);
    }
    etrade::RiskConfig rt;
    rt.throttle_updates = 1;
    rt.throttle_window = 3;
    const auto thr = sim(book(), {}, rt);
    for (double v : thr.quote_updates) EXPECT_EQ(v, 1560.0);  // ceil(4680 / 3)
    // stale quotes are picked off
    EXPECT_LT(thr.mean(HedgeStat::kPnl), sim(book()).mean(HedgeStat::kPnl));
}

TEST(Hedging, DeterministicBySeed) {
    etrade::RiskConfig r;
    r.loss_limit = 4.0;
    r.throttle_updates = 2;
    r.throttle_window = 3;
    const auto a = sim(fixed_book(), full_hedge(), r, 100);
    const auto b = sim(fixed_book(), full_hedge(), r, 100);
    const auto d = sim(fixed_book(), full_hedge(), r, 100, 12);
    for (auto stat : etrade::kHedgeStats) EXPECT_EQ(a.path(stat), b.path(stat));
    EXPECT_NE(a.pnl, d.pnl);
}

TEST(Hedging, HedgeTradeLeavesTargetFractionUncovered) {
    const double q = 9.0;
    const double y = -2.0;
    const double hr = 0.8;
    const double u = q + y / hr;
    const double dy = etrade::hedge_trade(q, y, hr, 1.0, 0.25);
    EXPECT_NEAR(q + (y + dy) / hr, 0.25 * u, 1e-14 * std::abs(u));
    EXPECT_EQ(etrade::hedge_trade(1.0, 0.0, 0.9, 1.0, 0.0), 0.0);  // |u| <= threshold
    EXPECT_EQ(etrade::hedge_trade(5.0, 0.0, 0.0, 0.0, 0.0), 0.0);
}

TEST(Hedging, QuantileIsType7) {
    std::mt19937_64 rng(3);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<double> v(57);
    for (auto& x : v) x = normal(rng);
    std::vector<double> sorted = v;
    std::sort(sorted.begin(), sorted.end());
    EXPECT_EQ(etrade::quantile(v, 0.0), sorted.front());
    EXPECT_EQ(etrade::quantile(v, 1.0), sorted.back());
    EXPECT_EQ(etrade::quantile(v, 0.5), sorted[28]);
    // h = 56 * 0.05 = 2.8: between the 3rd and 4th order statistics.
    EXPECT_NEAR(etrade::quantile(v, 0.05), sorted[2] + 0.8 * (sorted[3] - sorted[2]), 1e-14);
    EXPECT_EQ(etrade::quantile({4.0}, 0.3), 4.0);
    EXPECT_DOUBLE_EQ(etrade::quantile({3.0, 1.0, 2.0, 4.0}, 0.5), 2.5);
    for (double p : {0.0, 0.05, 0.33, 0.5, 0.95, 1.0}) {
        const double qv = etrade::quantile(v, p);
        EXPECT_GE(qv, sorted.front());
        EXPECT_LE(qv, sorted.back());
    }
}

TEST(Hedging, Validation) {
    auto bad = [](auto make) { EXPECT_THROW((void)make(), std::invalid_argument); };
    const etrade::ASParams p = large_cap();
    auto hsim = [&p](etrade::HedgeConfig hc) {
        return etrade::simulate_hedged(p, 2, 1, etrade::MMConfig{}, hc);
    };
    auto rsim = [&p](etrade::RiskConfig rc) {
        return etrade::simulate_hedged(p, 2, 1, etrade::MMConfig{}, etrade::HedgeConfig{}, rc);
    };
    const etrade::HedgeConfig hok;
    const etrade::RiskConfig rok;
    bad([&] { auto c = hok; c.rho = 1.5; return hsim(c); });
    bad([&] { auto c = hok; c.hedge_ratio = -0.1; return hsim(c); });
    bad([&] { auto c = hok; c.hedge_cost = std::nan(""); return hsim(c); });
    bad([&] { auto c = hok; c.hedge_threshold = -1.0; return hsim(c); });
    bad([&] { auto c = hok; c.hedge_target = 1.0; return hsim(c); });
    bad([&] { auto c = rok; c.loss_limit = -1.0; return rsim(c); });
    bad([&] { auto c = rok; c.loss_limit = INFINITY; return rsim(c); });
    bad([&] { auto c = rok; c.position_limit = -1; return rsim(c); });
    bad([&] { auto c = rok; c.flatten_penalty = -0.01; return rsim(c); });
    bad([&] { auto c = rok; c.throttle_updates = -1; return rsim(c); });
    bad([&] { auto c = rok; c.throttle_window = 0; return rsim(c); });
    bad([&] { auto c = rok; c.kill_step = -2; return rsim(c); });
    bad([&] { auto c = rok; c.kill_step = 10000; return rsim(c); });
    bad([&] {
        auto c = rok;
        c.throttle_updates = 1;
        c.throttle_window = 10000;
        return rsim(c);
    });
    bad([&] { auto c = etrade::MMConfig{}; c.tick = -0.01; return etrade::simulate_hedged(p, 2, 1, c); });
    bad([&] { return etrade::simulate_hedged(p, 1, 0); });
    const auto res = etrade::simulate_hedged(p, 10, 0);
    bad([&] { return res.mean(static_cast<HedgeStat>(99)); });
    bad([] { return etrade::quantile({}, 0.5); });
    bad([] { return etrade::quantile({1.0, 2.0}, 1.5); });
    bad([] { return etrade::quantile({1.0, NAN}, 0.5); });
    bad([] { return etrade::hedge_trade(1.0, 0.0, 0.9, 0.0, 1.0); });
    bad([] { return etrade::hedge_trade(NAN, 0.0, 0.9, 0.0, 0.0); });
    // The default configs are valid.
    EXPECT_NO_THROW((void)rsim(rok));
}
