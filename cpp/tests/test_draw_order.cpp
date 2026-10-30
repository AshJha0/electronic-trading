/// \file test_draw_order.cpp
/// \brief Draw-order guards: in-test reference loops of Section 2.2 and
/// Section 9.4, using the library's RNG (std::mt19937_64 seeded once per
/// call, one uniform and one normal distribution, paths outer, steps inner).
/// The pinned order reproduces the library bitwise; deliberately wrong orders
/// diverge on the same seed.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <random>
#include <vector>

#include "etrade/avellaneda_stoikov.hpp"
#include "etrade/mm.hpp"

namespace {

enum class AsOrder { kPinned, kBidFirst, kSkipSuppressed };

/// Section 2 loop, kAS agent, no stress; returns per-path PnL.
std::vector<double> as_reference(const etrade::ASParams& p, int n_paths, std::uint64_t seed,
                                 AsOrder order) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::normal_distribution<double> normal(0.0, 1.0);
    const int n_steps = p.n_steps();
    const double g = p.gamma;
    const double sig2 = p.sigma * p.sigma;
    const double big_t = p.horizon;
    const double dt = p.dt;
    const double markup = 2.0 / g * std::log1p(g / p.k);
    const double sqdt = std::sqrt(dt);
    std::vector<double> pnl(static_cast<std::size_t>(n_paths));
    for (int path = 0; path < n_paths; ++path) {
        double s = p.s0;
        int q = 0;
        double cash = 0.0;
        for (int i = 0; i < n_steps; ++i) {
            const double t = i * dt;
            const double r = s - q * g * sig2 * (big_t - t);
            const double half = 0.5 * (g * sig2 * (big_t - t) + markup);
            const double ask = r + half;
            const double bid = r - half;
            const bool quote_ask = q > -p.q_max;
            const bool quote_bid = q < p.q_max;
            double u_ask = 2.0;
            double u_bid = 2.0;
            if (order == AsOrder::kPinned) {
                u_ask = uniform(rng);
                u_bid = uniform(rng);
            } else if (order == AsOrder::kBidFirst) {
                u_bid = uniform(rng);
                u_ask = uniform(rng);
            } else {
                if (quote_ask) u_ask = uniform(rng);
                if (quote_bid) u_bid = uniform(rng);
            }
            const double p_ask = -std::expm1(-p.a_rate * std::exp(-p.k * (ask - s)) * dt);
            const double p_bid = -std::expm1(-p.a_rate * std::exp(-p.k * (s - bid)) * dt);
            const bool fill_ask = quote_ask && u_ask < p_ask;
            const bool fill_bid = quote_bid && u_bid < p_bid;
            if (fill_ask) { q -= 1; cash += ask; }
            if (fill_bid) { q += 1; cash -= bid; }
            s += p.sigma * sqdt * normal(rng);
        }
        pnl[static_cast<std::size_t>(path)] = cash + q * s - p.eta_liq * std::abs(q);
    }
    return pnl;
}

enum class MmOrder { kPinned, kSwapAskBid, kZLast };

double sign_of(double v) { return v > 0.0 ? 1.0 : (v < 0.0 ? -1.0 : 0.0); }

/// Section 9 loop (kAs agent, tick rounding, latency, signal, informed flow).
std::vector<double> mm_reference(const etrade::ASParams& p, int n_paths, std::uint64_t seed,
                                 const etrade::MMConfig& cfg, MmOrder order) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::normal_distribution<double> normal(0.0, 1.0);
    const int n_steps = p.n_steps();
    const double g = p.gamma;
    const double sig = p.sigma;
    const double sig2 = sig * sig;
    const double big_t = p.horizon;
    const double dt = p.dt;
    const double sqdt = std::sqrt(dt);
    const double markup = 2.0 / g * std::log1p(g / p.k);
    const double shift_scale = (2.0 * cfg.signal_accuracy - 1.0) * sig * sqdt *
                               std::sqrt(2.0 / 3.14159265358979323846);
    std::vector<double> pnl(static_cast<std::size_t>(n_paths));
    std::deque<double> hist;
    for (int path = 0; path < n_paths; ++path) {
        double s = p.s0;
        hist.assign(static_cast<std::size_t>(cfg.latency_steps) + 1, s);
        int q = 0;
        double cash = 0.0;
        for (int i = 0; i < n_steps; ++i) {
            const double t = i * dt;
            double z = 0.0, u_sig = 0.0, u_ask = 0.0, u_bid = 0.0, u_inf = 0.0;
            if (order == MmOrder::kPinned) {
                z = normal(rng);
                u_sig = uniform(rng);
                u_ask = uniform(rng);
                u_bid = uniform(rng);
                u_inf = uniform(rng);
            } else if (order == MmOrder::kSwapAskBid) {
                z = normal(rng);
                u_sig = uniform(rng);
                u_bid = uniform(rng);
                u_ask = uniform(rng);
                u_inf = uniform(rng);
            } else {
                u_sig = uniform(rng);
                u_ask = uniform(rng);
                u_bid = uniform(rng);
                u_inf = uniform(rng);
                z = normal(rng);
            }
            const double sz = sign_of(z);
            const double pred = u_sig < cfg.signal_accuracy ? sz : -sz;
            const double fair = hist.front() + shift_scale * pred;
            const double r = fair - q * g * sig2 * (big_t - t);
            const double half = 0.5 * (g * sig2 * (big_t - t) + markup);
            double bid = r - half;
            double ask = r + half;
            if (cfg.tick > 0.0) {
                bid = std::floor(bid / cfg.tick + 1e-9) * cfg.tick;
                ask = std::ceil(ask / cfg.tick - 1e-9) * cfg.tick;
            }
            const bool can_ask = q > -p.q_max;
            const bool can_bid = q < p.q_max;
            const double p_ask = -std::expm1(-p.a_rate * std::exp(-p.k * (ask - s)) * dt);
            const double p_bid = -std::expm1(-p.a_rate * std::exp(-p.k * (s - bid)) * dt);
            const bool fill_ask = can_ask && u_ask < p_ask;
            const bool fill_bid = can_bid && u_bid < p_bid;
            const double move = sig * sqdt * z;
            const bool informed = u_inf < cfg.informed_prob;
            const bool inf_ask = informed && z > 0.0 && can_ask && !fill_ask && ask < s + move;
            const bool inf_bid = informed && z < 0.0 && can_bid && !fill_bid && bid > s + move;
            const bool fa = fill_ask || inf_ask;
            const bool fb = fill_bid || inf_bid;
            const int n_fill = static_cast<int>(fa) + static_cast<int>(fb);
            q += static_cast<int>(fb) - static_cast<int>(fa);
            cash = cash + (fa ? ask : 0.0) - (fb ? bid : 0.0) + cfg.maker_fee * n_fill;
            s += move;
            hist.pop_front();
            hist.push_back(s);
        }
        pnl[static_cast<std::size_t>(path)] = cash + q * s - p.eta_liq * std::abs(q);
    }
    return pnl;
}

// ~50 steps, q_max = 1 so the cap binds often.
etrade::ASParams small_case() {
    return etrade::ASParams(100.0, 2.0, 1.0, 0.02, 0.1, 1.5, 140.0, 1, 0.01);
}

etrade::MMConfig mm_config() {
    etrade::MMConfig cfg;
    cfg.tick = 0.01;
    cfg.maker_fee = 0.002;
    cfg.latency_steps = 2;
    cfg.informed_prob = 0.3;
    cfg.signal_accuracy = 0.7;
    return cfg;
}

constexpr int kPaths = 6;
constexpr std::uint64_t kSeed = 20240607;

}  // namespace

TEST(DrawOrder, AsPinnedOrderReproducesLibraryBitwise) {
    const etrade::ASParams p = small_case();
    ASSERT_EQ(p.n_steps(), 50);
    const etrade::ASSimResult lib = etrade::simulate(p, kPaths, kSeed);
    const std::vector<double> ref = as_reference(p, kPaths, kSeed, AsOrder::kPinned);
    ASSERT_EQ(lib.pnl.size(), ref.size());
    for (std::size_t i = 0; i < ref.size(); ++i) EXPECT_EQ(lib.pnl[i], ref[i]) << "path " << i;
    EXPECT_GT(lib.overall_max_abs_q, 0.0);  // the cap is reached
}

TEST(DrawOrder, AsWrongOrdersDiverge) {
    const etrade::ASParams p = small_case();
    const etrade::ASSimResult lib = etrade::simulate(p, kPaths, kSeed);
    EXPECT_NE(as_reference(p, kPaths, kSeed, AsOrder::kBidFirst), lib.pnl);
    EXPECT_NE(as_reference(p, kPaths, kSeed, AsOrder::kSkipSuppressed), lib.pnl);
}

TEST(DrawOrder, MmPinnedOrderReproducesLibraryBitwise) {
    const etrade::ASParams p = small_case();
    const etrade::MMResult lib = etrade::simulate_mm(p, kPaths, kSeed, mm_config());
    const std::vector<double> ref = mm_reference(p, kPaths, kSeed, mm_config(), MmOrder::kPinned);
    ASSERT_EQ(lib.pnl.size(), ref.size());
    for (std::size_t i = 0; i < ref.size(); ++i) EXPECT_EQ(lib.pnl[i], ref[i]) << "path " << i;
}

TEST(DrawOrder, MmWrongOrdersDiverge) {
    const etrade::ASParams p = small_case();
    const etrade::MMResult lib = etrade::simulate_mm(p, kPaths, kSeed, mm_config());
    EXPECT_NE(mm_reference(p, kPaths, kSeed, mm_config(), MmOrder::kSwapAskBid), lib.pnl);
    EXPECT_NE(mm_reference(p, kPaths, kSeed, mm_config(), MmOrder::kZLast), lib.pnl);
}
