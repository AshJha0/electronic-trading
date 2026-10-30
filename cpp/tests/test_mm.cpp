/// \file test_mm.cpp
/// \brief Market making with frictions (API_SPEC Section 9): every case of
/// data/golden/mm_golden.json, the attribution identity, determinism, the
/// pinned draw order and validation.
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
#include "etrade/market.hpp"
#include "etrade/mm.hpp"
#include "mini_json.hpp"

namespace {

using mini_json::Value;

constexpr int kOwnPaths = 2000;

const Value& mm_golden() {
    static const Value doc =
        mini_json::parse_file(std::string(ETRADE_DATA_DIR) + "/golden/mm_golden.json");
    return doc;
}

double num(const Value& o, const std::string& k) { return o.at(k).number; }

/// The preset in market units.  mm_sim inputs carry the MMConfig tick under
/// "quote_tick" and the preset tick under "tick" (API_SPEC 9.7); simulations pass the
/// preset's calibration tick explicitly.
etrade::ASParams params_from(const Value& in, double preset_tick) {
    etrade::ASMarket m;
    m.price = num(in, "price");
    m.tick = preset_tick;
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

/// The us_large_cap preset; the preset tick (0.01) is the A-S calibration
/// tick, independent of MMConfig::tick.
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

std::vector<std::pair<std::string, etrade::MMStat>> golden_stats() {
    return {{"pnl", etrade::MMStat::kPnl},
            {"spread", etrade::MMStat::kSpread},
            {"fees", etrade::MMStat::kFees},
            {"inventory", etrade::MMStat::kInventory},
            {"markout", etrade::MMStat::kMarkout},
            {"fills", etrade::MMStat::kFills},
            {"informed_fills", etrade::MMStat::kInformedFills}};
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

/// Exact kinds: (key, actual) for every expect key.
std::vector<std::pair<std::string, double>> exact_actual(const Value& c) {
    const std::string kind = c.at("kind").str;
    const Value& in = c.at("inputs");
    std::vector<std::pair<std::string, double>> out;
    if (kind == "mm_tick") {
        const auto [b, a] = etrade::round_to_tick(num(in, "bid"), num(in, "ask"), num(in, "tick"));
        out.emplace_back("bid", b);
        out.emplace_back("ask", a);
    } else if (kind == "mm_glft") {
        const auto [db, da] = etrade::glft_half_spreads(params_from(in, num(in, "tick")), num(in, "q"));
        out.emplace_back("delta_bid", db);
        out.emplace_back("delta_ask", da);
    } else if (kind == "mm_wmid") {
        out.emplace_back("wmid", etrade::weighted_mid(num(in, "bid"), num(in, "ask"),
                                                      num(in, "bid_size"), num(in, "ask_size")));
    } else if (kind == "mm_ofi") {
        std::vector<double> pb, qb, pa, qa;
        for (int i = 0; i < 5; ++i) {
            const std::string s = std::to_string(i);
            pb.push_back(num(in, "bid_px_" + s));
            qb.push_back(num(in, "bid_sz_" + s));
            pa.push_back(num(in, "ask_px_" + s));
            qa.push_back(num(in, "ask_sz_" + s));
        }
        const auto e = etrade::order_flow_imbalance(pb, qb, pa, qa);
        double total = 0.0;
        for (std::size_t j = 0; j < e.size(); ++j) {
            out.emplace_back("e_" + std::to_string(j + 1), e[j]);
            total += e[j];
        }
        out.emplace_back("ofi", total);
    }
    return out;
}

}  // namespace

TEST(MMGolden, EveryCaseEveryKey) {
    const auto& cases = mm_golden().at("cases").arr;
    ASSERT_EQ(cases.size(), 14u);
    int n_sim = 0;
    for (const auto& c : cases) {
        const std::string name = c.at("name").str;
        const Value& expect = c.at("expect");
        const double tol = c.at("tol").number;
        if (c.at("kind").str == "mm_sim") {
            ++n_sim;
            const Value& in = c.at("inputs");
            const auto res = etrade::simulate_mm(
                params_from(in, 0.01), kOwnPaths,
                static_cast<std::uint64_t>(num(in, "seed")), config_from(in));
            EXPECT_LE(res.max_identity_error, 1e-9) << name;
            const auto stats = golden_stats();
            ASSERT_EQ(expect.obj.size(), 2 * stats.size()) << name;
            for (const auto& [key, stat] : stats) {
                const double want = num(expect, key);
                const double se_ref = num(expect, key + "_se");
                const double got = res.mean(stat);
                const double se_own = res.se(stat);
                const double band = tol * std::sqrt(se_ref * se_ref + se_own * se_own);
                // Zero-variance statistics (fees and fills without fees) are
                // compared with a small absolute floor.
                EXPECT_LE(std::abs(got - want), std::max(band, 1e-12))
                    << name << "." << key << ": got " << got << " want " << want;
            }
            continue;
        }
        const auto actual = exact_actual(c);
        ASSERT_EQ(actual.size(), expect.obj.size()) << name;
        for (const auto& [key, got] : actual) {
            const double want = num(expect, key);
            EXPECT_LE(std::abs(got - want), tol * std::max(1.0, std::abs(want)))
                << name << "." << key << ": got " << got << " want " << want;
        }
    }
    EXPECT_EQ(n_sim, 6);
}

TEST(MM, AttributionIdentityOnEveryPathForEveryAgent) {
    const etrade::ASParams p = large_cap();
    for (auto agent : {etrade::MMAgent::kAs, etrade::MMAgent::kGlft, etrade::MMAgent::kFixed}) {
        etrade::MMConfig c;
        c.agent = agent;
        c.tick = 0.01;
        c.maker_fee = -0.001;
        c.latency_steps = 3;
        c.informed_prob = 0.2;
        c.signal_accuracy = 0.7;
        const auto r = etrade::simulate_mm(p, 50, 7, c);
        EXPECT_LE(r.max_identity_error, 1e-9);
        for (int i = 0; i < r.n_paths(); ++i) {
            const double recon = r.spread[i] + r.fees[i] + r.inventory[i] - r.liquidation[i];
            EXPECT_LE(std::abs(recon - r.pnl[i]), 1e-9 * std::max(1.0, std::abs(r.pnl[i])));
            EXPECT_NEAR(r.fees[i], -0.001 * r.fills[i], 1e-9);
            EXPECT_LE(r.informed_fills[i], r.fills[i]);
        }
    }
}

TEST(MM, DeterministicBySeed) {
    const etrade::ASParams p = large_cap();
    etrade::MMConfig c;
    c.informed_prob = 0.1;
    const auto a = etrade::simulate_mm(p, 20, 42, c);
    const auto b = etrade::simulate_mm(p, 20, 42, c);
    const auto d = etrade::simulate_mm(p, 20, 43, c);
    for (auto stat : etrade::kMMStats) EXPECT_EQ(a.path(stat), b.path(stat));
    EXPECT_NE(a.pnl, d.pnl);
}

TEST(MM, FiveDrawsPerStepInPinnedOrder) {
    // One-step horizon, two paths: replay the stream by hand (z, u_sig,
    // u_ask, u_bid, u_inf per path) and recompute fills and the move.
    const etrade::ASParams p(100.0, 2.0, 0.5, 0.5, 0.1, 1.5, 140.0, 5, 0.0);
    etrade::MMConfig c;
    c.informed_prob = 0.5;
    c.signal_accuracy = 0.75;
    const auto r = etrade::simulate_mm(p, 2, 2024, c);
    std::mt19937_64 rng(2024);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::normal_distribution<double> normal(0.0, 1.0);
    const double sqdt = std::sqrt(p.dt);
    const double half = 0.5 * (p.gamma * p.sigma * p.sigma * p.horizon +
                               2.0 / p.gamma * std::log1p(p.gamma / p.k));
    for (int path = 0; path < 2; ++path) {
        const double z = normal(rng);
        const double u_sig = uniform(rng);
        const double u_ask = uniform(rng);
        const double u_bid = uniform(rng);
        const double u_inf = uniform(rng);
        const double sz = z > 0.0 ? 1.0 : (z < 0.0 ? -1.0 : 0.0);
        const double pred = u_sig < c.signal_accuracy ? sz : -sz;
        const double fair = p.s0 + 0.5 * p.sigma * sqdt * std::sqrt(2.0 / 3.14159265358979323846) * pred;
        const double bid = fair - half;
        const double ask = fair + half;
        const bool fa0 = u_ask < etrade::fill_probability(p, ask - p.s0);
        const bool fb0 = u_bid < etrade::fill_probability(p, p.s0 - bid);
        const double move = p.sigma * sqdt * z;
        const bool inf = u_inf < c.informed_prob;
        const bool ia = inf && z > 0.0 && !fa0 && ask < p.s0 + move;
        const bool ib = inf && z < 0.0 && !fb0 && bid > p.s0 + move;
        const int q = static_cast<int>(fb0 || ib) - static_cast<int>(fa0 || ia);
        EXPECT_EQ(r.fills[path], static_cast<int>(fa0 || ia) + static_cast<int>(fb0 || ib));
        EXPECT_EQ(r.informed_fills[path], static_cast<int>(ia) + static_cast<int>(ib));
        EXPECT_NEAR(r.inventory[path], q * move, 1e-12);
        EXPECT_NEAR(r.markout[path], q * move, 1e-12);
    }
}

TEST(MM, NoFrictionsMakeMoneyAndFrictionsCost) {
    const etrade::ASParams p = large_cap();
    etrade::MMConfig informed;
    informed.informed_prob = 0.15;
    const double base = etrade::simulate_mm(p, 200, 1).mean(etrade::MMStat::kPnl);
    const auto r = etrade::simulate_mm(p, 200, 1, informed);
    EXPECT_GT(base, 0.0);
    EXPECT_LT(r.mean(etrade::MMStat::kPnl), base);
    EXPECT_LT(r.mean(etrade::MMStat::kMarkout), 0.0);
    EXPECT_GT(r.mean(etrade::MMStat::kInformedFills), 0.0);
}

TEST(MM, FixedAgentDefaultsToAverageSpread) {
    const etrade::ASParams p = large_cap();
    etrade::MMConfig a;
    a.agent = etrade::MMAgent::kFixed;
    etrade::MMConfig b = a;
    b.fixed_spread = etrade::average_as_spread(p);
    EXPECT_EQ(etrade::simulate_mm(p, 5, 3, a).pnl, etrade::simulate_mm(p, 5, 3, b).pnl);
}

TEST(MM, Helpers) {
    const auto [b, a] = etrade::round_to_tick(-0.0149, 0.0149, 0.01);
    EXPECT_NEAR(b, -0.02, 1e-15);
    EXPECT_NEAR(a, 0.02, 1e-15);
    const auto [db, da] = etrade::glft_half_spreads(large_cap(), 0.0);
    EXPECT_DOUBLE_EQ(db, da);
    EXPECT_DOUBLE_EQ(etrade::weighted_mid(99.0, 101.0, 1.0, 1.0), 100.0);
    EXPECT_DOUBLE_EQ(etrade::weighted_mid(99.0, 101.0, 0.0, 5.0), 99.0);
    EXPECT_EQ(etrade::order_flow_imbalance({1, 1}, {5, 5}, {2, 2}, {3, 3}),
              std::vector<double>({0.0}));
}

TEST(MM, Validation) {
    auto bad = [](auto make) { EXPECT_THROW((void)make(), std::invalid_argument); };
    const etrade::ASParams p = large_cap();
    auto sim = [&p](etrade::MMConfig c, int n = 2) { return etrade::simulate_mm(p, n, 1, c); };
    const etrade::MMConfig ok;
    bad([&] { auto c = ok; c.tick = -0.01; return sim(c); });
    bad([&] { auto c = ok; c.tick = std::nan(""); return sim(c); });
    bad([&] { auto c = ok; c.maker_fee = INFINITY; return sim(c); });
    bad([&] { auto c = ok; c.latency_steps = -1; return sim(c); });
    bad([&] { auto c = ok; c.latency_steps = p.n_steps() + 1; return sim(c); });
    bad([&] { auto c = ok; c.informed_prob = -0.1; return sim(c); });
    bad([&] { auto c = ok; c.informed_prob = 1.1; return sim(c); });
    bad([&] { auto c = ok; c.signal_accuracy = 0.4; return sim(c); });
    bad([&] { auto c = ok; c.signal_accuracy = 1.01; return sim(c); });
    bad([&] { auto c = ok; c.fixed_spread = 0.0; return sim(c); });
    bad([&] { auto c = ok; c.fixed_spread = INFINITY; return sim(c); });
    bad([&] { return sim(ok, 1); });
    bad([&] { return sim(ok, etrade::kMaxAsPaths + 1); });
    bad([] { return etrade::round_to_tick(1.0, 2.0, -1.0); });
    bad([] { return etrade::round_to_tick(NAN, 2.0, 0.01); });
    bad([&] { return etrade::glft_half_spreads(p, INFINITY); });
    bad([] { return etrade::weighted_mid(101.0, 99.0, 1.0, 1.0); });
    bad([] { return etrade::weighted_mid(99.0, 101.0, -1.0, 1.0); });
    bad([] { return etrade::weighted_mid(99.0, 101.0, 0.0, 0.0); });
    bad([] { return etrade::weighted_mid(NAN, 101.0, 1.0, 1.0); });
    bad([] { return etrade::order_flow_imbalance({1}, {1}, {2}, {1}); });
    bad([] { return etrade::order_flow_imbalance({1, 1}, {1}, {2, 2}, {1, 1}); });
    bad([] { return etrade::order_flow_imbalance({1, NAN}, {1, 1}, {2, 2}, {1, 1}); });
    bad([] { return etrade::order_flow_imbalance({1, 1}, {1, -1}, {2, 2}, {1, 1}); });
    // The default config is valid.
    EXPECT_NO_THROW((void)sim(ok));
}
