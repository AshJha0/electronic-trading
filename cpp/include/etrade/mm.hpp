/// \file mm.hpp
/// \brief Market making that can lose money (API_SPEC Section 9): informed
/// flow, tick grid, fees, latency, a fair-value signal, GLFT quotes and an
/// exact per-path PnL attribution.
///
/// The Section 2 simulator is the paper's world: fills are independent of
/// where the price goes next, quotes sit at any real price, trading is free
/// and quotes never go stale.  This module keeps the A-S quoting logic and
/// adds those frictions one switch at a time.
///
/// Simulation loop order is pinned (API_SPEC 9.4): per path, per step
/// `t = i*dt`, five numbers are drawn always, in this order: `z ~ N(0,1)`,
/// `u_sig`, `u_ask`, `u_bid`, `u_inf ~ U(0,1)`.  The signal predicts
/// `sign(z)` when `u_sig < signal_accuracy` (else `-sign(z)`) and shifts fair
/// value by `(2*accuracy - 1)*pred*sigma*sqrt(dt)*sqrt(2/pi)`.  Quotes are
/// built from the mid `latency_steps` steps ago plus that shift and rounded
/// away from fair value to the tick grid.  Uninformed fills use the thinned
/// Poisson probability of Section 2 with distances from the true mid; an
/// informed trader (`u_inf < informed_prob`) trades in the direction of `z`
/// against a quote inside the coming move.  Every fill earns `maker_fee`.
///
/// Attribution (exact per path):
///
///     pnl = cash + q_T*S_T - eta_liq*|q_T| = spread + fees + inventory - liquidation
///
/// `markout` is the one-step markout of the fills (part of `inventory`);
/// negative means adverse selection.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "etrade/avellaneda_stoikov.hpp"

namespace etrade {

/// Quoting agent of the market-making simulator.
enum class MMAgent {
    kAs,     ///< A-S reservation price +/- half the optimal spread.
    kGlft,   ///< GLFT asymptotic closed form (time-independent).
    kFixed,  ///< Fair value +/- half a fixed spread, no inventory skew.
};

/// Frictions and agent for simulate_mm (API_SPEC 9.1).  Validated by
/// simulate_mm with std::invalid_argument.
struct MMConfig {
    MMAgent agent = MMAgent::kAs;
    double tick = 0.0;              ///< Price grid, >= 0 (0 = continuous).
    double maker_fee = 0.0;         ///< Per share per fill; positive = rebate.
    int latency_steps = 0;          ///< Quote staleness in steps, in [0, n_steps].
    double informed_prob = 0.0;     ///< Per-step informed arrival probability, in [0, 1].
    double signal_accuracy = 0.5;   ///< In [0.5, 1]; 0.5 = no signal.
    /// Spread of the kFixed agent, > 0; empty = average A-S spread.
    std::optional<double> fixed_spread;
};

/// Statistics reported by MMResult (API_SPEC 9.5).
enum class MMStat {
    kPnl,
    kSpread,
    kFees,
    kInventory,
    kLiquidation,
    kMarkout,
    kFills,
    kInformedFills,
    kRmsQ,
};

/// Every MMStat, in API_SPEC order.
inline constexpr std::array<MMStat, 9> kMMStats = {
    MMStat::kPnl,     MMStat::kSpread, MMStat::kFees,
    MMStat::kInventory, MMStat::kLiquidation, MMStat::kMarkout,
    MMStat::kFills,   MMStat::kInformedFills, MMStat::kRmsQ};

/// Per-path arrays (length n_paths) of every statistic, plus the largest
/// attribution residual seen.
struct MMResult {
    std::vector<double> pnl;
    std::vector<double> spread;
    std::vector<double> fees;
    std::vector<double> inventory;
    std::vector<double> liquidation;
    std::vector<double> markout;
    std::vector<double> fills;
    std::vector<double> informed_fills;
    std::vector<double> rms_q;
    /// max over paths of |spread + fees + inventory - liquidation - pnl| / max(1, |pnl|).
    double max_identity_error = 0.0;

    /// Number of simulated paths.
    int n_paths() const { return static_cast<int>(pnl.size()); }

    /// The per-path array of a statistic.
    const std::vector<double>& path(MMStat stat) const;

    /// Mean over paths of a statistic.
    double mean(MMStat stat) const;

    /// Standard error of the mean: sample std (ddof 1) / sqrt(n_paths).
    double se(MMStat stat) const;
};

/// Round a quote pair away from fair value to the tick grid (API_SPEC 9.2):
/// `bid -> floor(bid/tick + 1e-9)*tick`, `ask -> ceil(ask/tick - 1e-9)*tick`;
/// `tick = 0` returns the inputs.  Returns (bid, ask).
std::pair<double, double> round_to_tick(double bid, double ask, double tick);

/// GLFT asymptotic distances (delta_bid, delta_ask) at inventory q
/// (API_SPEC 9.3): c1 = log1p(gamma/k)/gamma,
/// c2 = sqrt(sigma^2 gamma/(2kA) (1 + gamma/k)^(1 + k/gamma)),
/// delta_bid = c1 + (2q + 1)/2 c2, delta_ask = c1 - (2q - 1)/2 c2.
std::pair<double, double> glft_half_spreads(const ASParams& params, double q);

/// Size-weighted mid (bid*ask_size + ask*bid_size)/(bid_size + ask_size), a
/// proxy for the microprice (Stoikov 2018).  ask >= bid, sizes >= 0 with a
/// positive sum.
double weighted_mid(double bid, double ask, double bid_size, double ask_size);

/// Per-update order-flow imbalance of Cont, Kukanov and Stoikov (2014) for
/// consecutive top-of-book states; returns len - 1 values.  At least two
/// states, equal lengths, finite, sizes >= 0.
std::vector<double> order_flow_imbalance(const std::vector<double>& bid_px,
                                         const std::vector<double>& bid_sz,
                                         const std::vector<double>& ask_px,
                                         const std::vector<double>& ask_sz);

/// Simulate the market maker with frictions (API_SPEC 9.4) over `n_paths`
/// seeded paths (std::mt19937_64 with std::normal_distribution /
/// std::uniform_real_distribution; statistical agreement with the Python
/// reference, deterministic per seed within one standard library).
/// \param n_paths number of paths, in [2, kMaxAsPaths].
MMResult simulate_mm(const ASParams& params, int n_paths, std::uint64_t seed,
                     const MMConfig& config = MMConfig{});

}  // namespace etrade
