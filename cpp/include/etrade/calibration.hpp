/// \file calibration.hpp
/// \brief Calibration estimators: market quantities measured from recorded
/// data (API_SPEC Section 11).
///
/// A session [t0, t1] is split into n equal bins; an observation at t with
/// t0 <= t <= t1 falls in bin min(floor((t - t0) * n / (t1 - t0)), n - 1).
/// Observations outside the session are ignored.
#pragma once

#include <array>
#include <vector>

namespace etrade {

/// Largest number of bins accepted by the curve estimators.
inline constexpr int MAX_CALIBRATION_BINS = 10000;
/// Largest number of distance buckets accepted by fill_intensity.
inline constexpr int MAX_FILL_LEVELS = 1000;
/// Slack, in ticks, when comparing a distance with a bucket edge.
inline constexpr double FILL_EDGE_SLACK = 1e-9;

/// Volume curve fitted to the NASDAQ TotalView-ITCH sample day 2019-12-30:
/// thirteen 30-minute fractions (API_SPEC 11.6).
inline constexpr std::array<double, 13> FITTED_VOLUME_CURVE_13 = {
    0.1987, 0.1247, 0.0742, 0.0680, 0.0589, 0.0604, 0.0414,
    0.0430, 0.0484, 0.0466, 0.0376, 0.0535, 0.1446};

/// Fraction of session volume per bin (API_SPEC 11.2); sums to 1.
/// Throws std::invalid_argument on bad inputs.
[[nodiscard]] std::vector<double> volume_curve(const std::vector<double>& times,
                                               const std::vector<double>& sizes,
                                               double session_start, double session_end,
                                               int n_bins);

/// Realised volatility of the session and of each bin (fractions).
struct VolatilityCurve {
    double daily_vol = 0.0;
    std::vector<double> bin_vol;
    int n_returns = 0;
};

/// Realised volatility per bin and for the session (API_SPEC 11.3).
[[nodiscard]] VolatilityCurve volatility_curve(const std::vector<double>& times,
                                               const std::vector<double>& mids,
                                               double session_start, double session_end,
                                               int n_bins);

/// Mean and median of a spread sample, in ticks and in bps of the mid.
struct SpreadStats {
    double mean_ticks = 0.0;
    double median_ticks = 0.0;
    double mean_bps = 0.0;
    double median_bps = 0.0;
    int n = 0;
};

/// Spread statistics of a sample (API_SPEC 11.4).
[[nodiscard]] SpreadStats spread_stats(const std::vector<double>& spreads,
                                       const std::vector<double>& mids, double tick);

/// Fitted lambda(delta) = A exp(-k delta), in model and market units.
struct FillIntensity {
    double a_rate = 0.0;
    double k = 0.0;
    double touch_fills_per_day = 0.0;
    double halving_ticks = 0.0;
    int n_buckets = 0;
};

/// Fit the exponential fill intensity by weighted least squares
/// (API_SPEC 11.5).
[[nodiscard]] FillIntensity fill_intensity(const std::vector<double>& distances, double tick,
                                           int n_levels, double days = 1.0,
                                           double sides = 2.0);

}  // namespace etrade
