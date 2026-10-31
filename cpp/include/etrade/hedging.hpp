/// \file hedging.hpp
/// \brief Hedged market making and risk controls (API_SPEC Section 14): the
/// Section 9 maker with a correlated hedge instrument B, a loss limit, a
/// position limit, a quote throttle and a kill switch.
///
/// Instrument A (the one quoted) moves `sigma*sqrt(dt)*z` per step as in
/// Section 9.  Instrument B moves `sigma*sqrt(dt)*(rho*z + sqrt(1-rho^2)*z_b)`
/// (A-equivalent units), so the minimum-variance hedge ratio is `rho`
/// (Ederington 1979).  The maker holds `q` of A and `y` of B; the quoting
/// agents skew on the risk-equivalent exposure `x = q + rho*y` (Gueant 2017).
///
/// Simulation loop order is pinned (API_SPEC 14.4): per path, per step
/// `t = i*dt`, six numbers are drawn always, in this order: `z ~ N(0,1)`,
/// `u_sig`, `u_ask`, `u_bid`, `u_inf ~ U(0,1)`, `z_b ~ N(0,1)`.  Then stops
/// (kill switch, loss limit), quotes (throttled), Section 9 fills, the
/// position limit, the hedge, accounting, and the moves of A and B.
///
/// Attribution (exact per path):
///
///     pnl = spread + fees + inventory - liquidation + hedge_pnl - hedge_costs - risk_costs
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "etrade/avellaneda_stoikov.hpp"
#include "etrade/mm.hpp"

namespace etrade {

/// Hedge instrument and rule (API_SPEC 14.1); `hedge_ratio = 0` is no hedge.
/// Validated by simulate_hedged with std::invalid_argument.
struct HedgeConfig {
    double rho = 0.0;              ///< Correlation of B's move with A's, in [-1, 1].
    double hedge_ratio = 0.0;      ///< B shares per A share, >= 0 (0 = no hedging).
    double hedge_cost = 0.0;       ///< Per B share traded (half spread plus fee), >= 0.
    double hedge_threshold = 0.0;  ///< Uncovered A shares tolerated, >= 0.
    double hedge_target = 0.0;     ///< Fraction of u left uncovered, in [0, 1).
};

/// Risk controls (API_SPEC 14.1); every default is off.  Validated by
/// simulate_hedged with std::invalid_argument.
struct RiskConfig {
    double loss_limit = 0.0;       ///< Stop when the mark < -loss_limit; >= 0 (0 = off).
    int position_limit = 0;        ///< Flatten A when |q| >= limit; >= 0 (0 = off).
    double flatten_penalty = 0.0;  ///< Per A share flattened by a control, >= 0.
    int throttle_updates = 0;      ///< Quote updates per throttle_window steps, >= 0 (0 = off).
    int throttle_window = 1;       ///< In [1, n_steps].
    int kill_step = -1;            ///< Step of the kill switch, in [-1, n_steps] (-1 = off).
};

/// Statistics reported by HedgedResult (API_SPEC 14.5).
enum class HedgeStat {
    kPnl,
    kSpread,
    kFees,
    kInventory,
    kLiquidation,
    kMarkout,
    kFills,
    kInformedFills,
    kRmsQ,
    kHedgePnl,
    kHedgeCosts,
    kHedgeTrades,
    kRiskCosts,
    kRmsX,
    kRmsSkew,
    kQuoteUpdates,
    kLossStop,
    kKillStop,
    kPosFlattens,
    kOvershoot,
    kStopMove,
};

/// Every HedgeStat, in API_SPEC order.
inline constexpr std::array<HedgeStat, 21> kHedgeStats = {
    HedgeStat::kPnl,         HedgeStat::kSpread,       HedgeStat::kFees,
    HedgeStat::kInventory,   HedgeStat::kLiquidation,  HedgeStat::kMarkout,
    HedgeStat::kFills,       HedgeStat::kInformedFills, HedgeStat::kRmsQ,
    HedgeStat::kHedgePnl,    HedgeStat::kHedgeCosts,   HedgeStat::kHedgeTrades,
    HedgeStat::kRiskCosts,   HedgeStat::kRmsX,         HedgeStat::kRmsSkew,
    HedgeStat::kQuoteUpdates, HedgeStat::kLossStop,    HedgeStat::kKillStop,
    HedgeStat::kPosFlattens, HedgeStat::kOvershoot,    HedgeStat::kStopMove};

/// Per-path arrays (length n_paths) of every statistic, plus the largest
/// attribution residual seen.
struct HedgedResult {
    std::vector<double> pnl;
    std::vector<double> spread;
    std::vector<double> fees;
    std::vector<double> inventory;
    std::vector<double> liquidation;
    std::vector<double> markout;
    std::vector<double> fills;
    std::vector<double> informed_fills;
    std::vector<double> rms_q;
    std::vector<double> hedge_pnl;
    std::vector<double> hedge_costs;
    std::vector<double> hedge_trades;
    std::vector<double> risk_costs;
    std::vector<double> rms_x;
    std::vector<double> rms_skew;
    std::vector<double> quote_updates;
    std::vector<double> loss_stop;
    std::vector<double> kill_stop;
    std::vector<double> pos_flattens;
    std::vector<double> overshoot;
    std::vector<double> stop_move;
    /// max over paths of |spread + fees + inventory - liquidation + hedge_pnl
    /// - hedge_costs - risk_costs - pnl| / max(1, |pnl|).
    double max_identity_error = 0.0;

    /// Number of simulated paths.
    int n_paths() const { return static_cast<int>(pnl.size()); }

    /// The per-path array of a statistic.
    const std::vector<double>& path(HedgeStat stat) const;

    /// Mean over paths of a statistic.
    double mean(HedgeStat stat) const;

    /// Standard error of the mean: sample std (ddof 1) / sqrt(n_paths).
    double se(HedgeStat stat) const;

    /// Population variance m2 = mean((v - mean)^2) (API_SPEC 14.5).
    double var(HedgeStat stat) const;

    /// Delta-method standard error of var: sqrt(max(0, m4 - m2^2)/n_paths).
    double var_se(HedgeStat stat) const;

    /// Pinned type-7 quantile (etrade::quantile) of a statistic over paths.
    double quantile(HedgeStat stat, double p) const;
};

/// B shares to trade under the pinned hedging rule (API_SPEC 14.2): with
/// `u = q + y/hedge_ratio`, `dy = -hedge_ratio*(1 - target_frac)*u` when
/// `|u| > threshold`, else 0; `hedge_ratio = 0` returns 0.
double hedge_trade(double q, double y, double hedge_ratio, double threshold,
                   double target_frac);

/// Pinned sample quantile (API_SPEC 14.6), Hyndman and Fan (1996) type 7:
/// sort to `v`, `h = (n - 1) p`, `j = floor(h)`,
/// `Q = v[j] + (h - j)(v[j+1] - v[j])` (`v[n-1]` when `j >= n - 1`).
/// Non-empty finite values, `p` in [0, 1].
double quantile(std::vector<double> values, double p);

/// Simulate the Section 9 maker with a hedge and risk controls (API_SPEC
/// 14.4) over `n_paths` seeded paths (std::mt19937_64 with
/// std::normal_distribution / std::uniform_real_distribution; statistical
/// agreement with the Python reference, deterministic per seed within one
/// standard library).  With the default hedge and risk configs the result
/// equals simulate_mm in distribution.
/// \param n_paths number of paths, in [2, kMaxAsPaths].
HedgedResult simulate_hedged(const ASParams& params, int n_paths, std::uint64_t seed,
                             const MMConfig& config = MMConfig{},
                             const HedgeConfig& hedge = HedgeConfig{},
                             const RiskConfig& risk = RiskConfig{});

}  // namespace etrade
