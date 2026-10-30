/// \file tca.hpp
/// \brief Post-trade transaction-cost analysis (API_SPEC Section 10):
/// benchmark slippage, markouts, participation and a Welch A/B test.
///
/// Slippage is `1e4 * sign * (B - avg_price) / B` bps of each benchmark B
/// (arrival, market VWAP, interval VWAP over the bins the order traded in,
/// close) with `sign = +1` for a sell and `-1` for a buy, the Perold
/// convention of analytics.hpp: positive = cost.  Markouts are
/// `d * (mid[t + h] - mid[t])` with `d = +1` for a buy, negative = adverse.
/// The Student-t tail uses a pinned Lanczos log-gamma and the modified Lentz
/// continued fraction for the incomplete beta (Numerical Recipes, 3rd ed.),
/// so every port runs the same algorithm.
#pragma once

#include "etrade/analytics.hpp"

#include <vector>

namespace etrade {

/// Iteration cap of the incomplete-beta continued fraction.
inline constexpr int kMaxBetacfIter = 10000;

/// Average price, benchmarks and slippage in bps (positive = cost).
struct SlippageReport {
    double executed_qty = 0.0;
    double avg_price = 0.0;
    double market_vwap = 0.0;
    double interval_vwap = 0.0;
    double arrival_bps = 0.0;
    double vwap_bps = 0.0;
    double interval_vwap_bps = 0.0;
    double close_bps = 0.0;
};

/// Slippage against arrival, VWAP, interval VWAP and close (API_SPEC 10.1).
/// `bin_volumes`/`bin_vwaps` describe the whole market per bin, including
/// the order's own fills.  Throws std::invalid_argument on bad inputs.
[[nodiscard]] SlippageReport slippage(Side side, double arrival_price, double close_price,
                                      const std::vector<int>& fill_bins,
                                      const std::vector<double>& fill_qtys,
                                      const std::vector<double>& fill_prices,
                                      const std::vector<double>& bin_volumes,
                                      const std::vector<double>& bin_vwaps);

/// Mean markout per fill and per share at each horizon (negative = adverse).
struct MarkoutResult {
    std::vector<int> horizons;
    std::vector<double> per_fill;
    std::vector<double> per_share;
};

/// Markouts `d * (mids[t_j + h] - mids[t_j])` (API_SPEC 10.2).
[[nodiscard]] MarkoutResult markouts(Side side, const std::vector<int>& fill_steps,
                                     const std::vector<double>& fill_qtys,
                                     const std::vector<double>& mids,
                                     const std::vector<int>& horizons);

/// Order volume as a fraction of market volume, per bin and overall.
struct ParticipationResult {
    std::vector<double> per_bin;
    double overall = 0.0;
};

/// Per-bin `q_b / V_b` (0 for an empty bin) and `Q / sum V` (API_SPEC 10.3).
[[nodiscard]] ParticipationResult participation(const std::vector<int>& fill_bins,
                                                const std::vector<double>& fill_qtys,
                                                const std::vector<double>& bin_volumes);

/// Natural log of the gamma function, pinned Lanczos (g = 7, nine terms).
[[nodiscard]] double ln_gamma(double x);

/// Regularized incomplete beta I_x(a, b).  Throws std::runtime_error if the
/// continued fraction does not converge in kMaxBetacfIter iterations.
[[nodiscard]] double betainc_reg(double a, double b, double x);

/// Student-t CDF with `df > 0` degrees of freedom (non-integer allowed).
[[nodiscard]] double student_t_cdf(double t, double df);

/// Welch two-sample t-test of mean(a) - mean(b).
struct WelchResult {
    double mean_diff = 0.0;
    double t_stat = 0.0;
    double df = 0.0;
    double p_value = 0.0;  ///< Two-sided.
};

/// Welch's t-test on per-order costs of strategies A and B (API_SPEC 10.4).
[[nodiscard]] WelchResult welch_ttest(const std::vector<double>& a, const std::vector<double>& b);

}  // namespace etrade
