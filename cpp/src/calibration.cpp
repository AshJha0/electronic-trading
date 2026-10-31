#include "etrade/calibration.hpp"
#include "etrade/detail/require.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace etrade {

namespace {

using detail::require;

void check_session(double session_start, double session_end, int n_bins) {
    require(std::isfinite(session_start), "session_start must be finite");
    require(std::isfinite(session_end) && session_end > session_start,
            "session_end must be finite and > session_start");
    require(n_bins >= 1 && n_bins <= MAX_CALIBRATION_BINS,
            "n_bins must be an integer in [1, " + std::to_string(MAX_CALIBRATION_BINS) + "]");
}

std::size_t bin_of(double t, double session_start, double session_end, int n_bins) {
    const double j = std::floor((t - session_start) * n_bins / (session_end - session_start));
    const double last = static_cast<double>(n_bins - 1);
    return static_cast<std::size_t>(std::min(j, last));
}

double median(std::vector<double> s) {
    std::sort(s.begin(), s.end());
    const std::size_t n = s.size();
    if (n % 2 == 1) return s[n / 2];
    return 0.5 * (s[n / 2 - 1] + s[n / 2]);
}

}  // namespace

std::vector<double> volume_curve(const std::vector<double>& times,
                                 const std::vector<double>& sizes, double session_start,
                                 double session_end, int n_bins) {
    check_session(session_start, session_end, n_bins);
    require(times.size() == sizes.size(), "times and sizes must have the same length");
    std::vector<double> vol(static_cast<std::size_t>(n_bins), 0.0);
    for (std::size_t i = 0; i < times.size(); ++i) {
        const double t = times[i];
        const double s = sizes[i];
        require(std::isfinite(t), "times must be finite");
        require(std::isfinite(s) && s >= 0.0, "sizes must be finite and >= 0");
        if (session_start <= t && t <= session_end)
            vol[bin_of(t, session_start, session_end, n_bins)] += s;
    }
    double total = 0.0;
    for (double v : vol) total += v;
    require(total > 0.0, "no volume inside the session");
    std::vector<double> out;
    out.reserve(vol.size());
    for (double v : vol) out.push_back(v / total);
    return out;
}

VolatilityCurve volatility_curve(const std::vector<double>& times,
                                 const std::vector<double>& mids, double session_start,
                                 double session_end, int n_bins) {
    check_session(session_start, session_end, n_bins);
    require(times.size() == mids.size(), "times and mids must have the same length");
    std::vector<double> rv(static_cast<std::size_t>(n_bins), 0.0);
    int n_returns = 0;
    for (std::size_t i = 0; i < times.size(); ++i) {
        const double t = times[i];
        const double m = mids[i];
        require(std::isfinite(t), "times must be finite");
        require(std::isfinite(m) && m > 0.0, "mids must be finite and > 0");
        if (i == 0) continue;
        require(t >= times[i - 1], "times must be non-decreasing");
        if (times[i - 1] >= session_start && t <= session_end) {
            const double r = std::log(m / mids[i - 1]);
            rv[bin_of(t, session_start, session_end, n_bins)] += r * r;
            ++n_returns;
        }
    }
    require(n_returns > 0, "no return inside the session");
    double total = 0.0;
    for (double v : rv) total += v;
    VolatilityCurve out;
    out.daily_vol = std::sqrt(total);
    out.bin_vol.reserve(rv.size());
    for (double v : rv) out.bin_vol.push_back(std::sqrt(v));
    out.n_returns = n_returns;
    return out;
}

SpreadStats spread_stats(const std::vector<double>& spreads, const std::vector<double>& mids,
                         double tick) {
    require(std::isfinite(tick) && tick > 0.0, "tick must be finite and > 0");
    require(spreads.size() == mids.size(), "spreads and mids must have the same length");
    require(!spreads.empty(), "spreads must not be empty");
    std::vector<double> ticks;
    std::vector<double> bps;
    ticks.reserve(spreads.size());
    bps.reserve(spreads.size());
    for (std::size_t i = 0; i < spreads.size(); ++i) {
        const double s = spreads[i];
        const double m = mids[i];
        require(std::isfinite(s) && s >= 0.0, "spreads must be finite and >= 0");
        require(std::isfinite(m) && m > 0.0, "mids must be finite and > 0");
        ticks.push_back(s / tick);
        bps.push_back(1e4 * s / m);
    }
    double sum_t = 0.0;
    double sum_b = 0.0;
    for (std::size_t i = 0; i < ticks.size(); ++i) {
        sum_t += ticks[i];
        sum_b += bps[i];
    }
    const double n = static_cast<double>(ticks.size());
    SpreadStats out;
    out.mean_ticks = sum_t / n;
    out.median_ticks = median(ticks);
    out.mean_bps = sum_b / n;
    out.median_bps = median(bps);
    out.n = static_cast<int>(ticks.size());
    return out;
}

FillIntensity fill_intensity(const std::vector<double>& distances, double tick, int n_levels,
                             double days, double sides) {
    require(std::isfinite(tick) && tick > 0.0, "tick must be finite and > 0");
    require(n_levels >= 2 && n_levels <= MAX_FILL_LEVELS,
            "n_levels must be an integer in [2, " + std::to_string(MAX_FILL_LEVELS) + "]");
    require(std::isfinite(days) && days > 0.0, "days must be finite and > 0");
    require(std::isfinite(sides) && sides > 0.0, "sides must be finite and > 0");
    std::vector<long long> counts(static_cast<std::size_t>(n_levels), 0);
    for (double d : distances) {
        require(std::isfinite(d) && d >= 0.0, "distances must be finite and >= 0");
        const double x = d / tick;
        for (int b = 0; b < n_levels; ++b) {
            if (x >= b + 0.5 - FILL_EDGE_SLACK)
                ++counts[static_cast<std::size_t>(b)];
            else
                break;
        }
    }
    std::vector<double> xs;
    std::vector<double> ys;
    std::vector<double> ws;
    for (int b = 0; b < n_levels; ++b) {
        const long long c = counts[static_cast<std::size_t>(b)];
        if (c > 0) {
            xs.push_back((b + 0.5) * tick);
            ys.push_back(std::log(static_cast<double>(c) / (sides * days)));
            ws.push_back(static_cast<double>(c));
        }
    }
    const std::size_t n = xs.size();
    require(n >= 2, "fewer than two distance buckets have fills");
    double w_sum = 0.0;
    double x_bar = 0.0;
    double y_bar = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        w_sum += ws[i];
        x_bar += ws[i] * xs[i];
        y_bar += ws[i] * ys[i];
    }
    x_bar /= w_sum;
    y_bar /= w_sum;
    double sxx = 0.0;
    double sxy = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sxx += ws[i] * (xs[i] - x_bar) * (xs[i] - x_bar);
        sxy += ws[i] * (xs[i] - x_bar) * (ys[i] - y_bar);
    }
    const double k = -sxy / sxx;
    require(k > 0.0, "fitted fill intensity does not decay with distance");
    const double a_rate = std::exp(y_bar + k * x_bar);
    FillIntensity out;
    out.a_rate = a_rate;
    out.k = k;
    out.touch_fills_per_day = a_rate * std::exp(-k * tick / 2.0);
    out.halving_ticks = std::log(2.0) / (k * tick);
    out.n_buckets = static_cast<int>(n);
    return out;
}

}  // namespace etrade
