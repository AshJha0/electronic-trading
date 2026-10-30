/// \file placement.hpp
/// \brief Limit-order placement for execution: queue-position fill
/// probability, the passive-or-aggressive choice for one child order, and
/// liquidation with limit and market orders (API_SPEC Section 13).
///
/// Fill probability: FIFO queue, unit market orders Poisson(mu), each unit
/// ahead cancels at rate theta (Cont, Stoikov and Talreja 2010; Cont and de
/// Larrard 2013); evaluated by pinned uniformisation.  Decision: posting at
/// the touch saves the half spread with probability P and otherwise crosses
/// later after an adverse move; the Section 8 impact is paid either way.
/// Liquidation: discrete Cartea-Jaimungal (2015) posting-depth policy by
/// backward induction and a seeded Monte Carlo against two baselines.
#pragma once

#include <cstdint>
#include <vector>

namespace etrade {

/// Upper bound on q_ahead + size in queue_fill_probability (pinned).
inline constexpr int kMaxQueueUnits = 5000;
/// Upper bound on uniformisation steps in queue_fill_probability (pinned).
inline constexpr int kMaxUniformisationSteps = 200000;
/// Upper bounds on total_units and n_periods of a liquidation (pinned).
inline constexpr int kMaxLiqUnits = 1000;
inline constexpr int kMaxLiqPeriods = 1000;

/// Probability that a resting order of `size` units with `q_ahead` units
/// ahead fills completely within `horizon` (API_SPEC 13.1).
[[nodiscard]] double queue_fill_probability(int q_ahead, int size, double mu, double theta,
                                            double horizon);

/// Expected cost per share relative to the mid of posting and of crossing
/// (API_SPEC 13.2); `passive` is true when posting is strictly cheaper.
struct PlacementDecision {
    double passive_cost = 0.0;
    double aggressive_cost = 0.0;
    double impact_cost = 0.0;
    double breakeven_fill_prob = 0.0;
    bool passive = false;
};

/// Compare posting one sell child order at the touch with crossing now.
[[nodiscard]] PlacementDecision place_or_cross(double size, double spread, double fill_prob,
                                               double adverse_move, double eta, double alpha,
                                               double weight);

/// Liquidation with limit and market orders (API_SPEC 13.3); validated on
/// construction (std::invalid_argument).
struct LiquidationParams {
    int total_units;  ///< Q in [1, kMaxLiqUnits].
    int n_periods;    ///< N in [1, kMaxLiqPeriods].
    double dt;        ///< Period length > 0.
    double sigma;     ///< Mid volatility >= 0.
    double spread;    ///< Quoted spread >= 0.
    double kappa;     ///< Fill decay per unit of depth > 0.
    double p_touch;   ///< Per-period fill probability at the touch, (0, 1].
    double beta;      ///< Book walk per unit of a market order >= 0.
    double phi;       ///< Running inventory penalty >= 0.

    LiquidationParams(int total_units, int n_periods, double dt, double sigma, double spread,
                      double kappa, double p_touch, double beta, double phi = 0.0);
};

/// Backward-induction policy: cost is (N+1) x (Q+1), depth and market N x (Q+1).
struct LiquidationPolicy {
    std::vector<std::vector<double>> cost;
    std::vector<std::vector<double>> depth;
    std::vector<std::vector<bool>> market;

    /// C_0(Q): expected shortfall plus penalties.
    [[nodiscard]] double expected_cost() const { return cost.front().back(); }
};

[[nodiscard]] LiquidationPolicy liquidation_policy(const LiquidationParams& p);

/// Even market-order schedule n_j = floor(Q(j+1)/N) - floor(Qj/N).
[[nodiscard]] std::vector<int> aggressive_schedule(int total_units, int n_periods);

/// Expected shortfall of the all-aggressive baseline.
[[nodiscard]] double aggressive_cost(const LiquidationParams& p);

/// Shortfall of the policy and the two baselines; savings are paired
/// (baseline - optimal); SEs are sample std (ddof=1) / sqrt(M).
struct LiquidationSimResult {
    double optimal_mean = 0.0;
    double optimal_se = 0.0;
    double optimal_std = 0.0;
    double aggressive_mean = 0.0;
    double aggressive_se = 0.0;
    double aggressive_std = 0.0;
    double passive_mean = 0.0;
    double passive_se = 0.0;
    double passive_std = 0.0;
    double saving_vs_aggressive_mean = 0.0;
    double saving_vs_aggressive_se = 0.0;
    double saving_vs_passive_mean = 0.0;
    double saving_vs_passive_se = 0.0;
    int n_paths = 0;
};

/// Seeded Monte Carlo (API_SPEC 13.3).  Uses std::mt19937_64(seed); paths
/// outer, periods inner, draws u then z.
[[nodiscard]] LiquidationSimResult simulate_liquidation(const LiquidationParams& p,
                                                        int n_paths = 10000,
                                                        std::uint64_t seed = 0);

}  // namespace etrade
