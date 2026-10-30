#include "etrade/tca.hpp"
#include "etrade/detail/require.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace etrade {
namespace {

constexpr double kBetacfEps = 1e-15;
constexpr double kFpMin = 1e-300;
constexpr double kPi = 3.14159265358979323846;
constexpr double kLanczos[9] = {0.99999999999980993,  676.5203681218851,   -1259.1392167224028,
                                771.32342877765313,   -176.61502916214059, 12.507343278686905,
                                -0.13857109526572012, 9.9843695780195716e-6,
                                1.5056327351493116e-7};

using detail::require;

double sell_sign(Side side) { return side == Side::kSell ? 1.0 : -1.0; }

void check_fills(const std::vector<int>& bins, const std::vector<double>& qtys, std::size_t n_bins,
                 const char* name) {
    require(!qtys.empty(), "fills must be non-empty");
    require(bins.size() == qtys.size(), "fill arrays must have equal length");
    for (int b : bins) {
        require(b >= 0 && static_cast<std::size_t>(b) < n_bins,
                std::string(name) + " must be integers in [0, " + std::to_string(n_bins) + ")");
    }
    for (double n : qtys) require(std::isfinite(n) && n >= 0.0, "fill_qtys must be finite and >= 0");
}

double check_volumes(const std::vector<double>& v) {
    require(!v.empty(), "bins must be non-empty");
    double total = 0.0;
    for (double x : v) {
        require(std::isfinite(x) && x >= 0.0, "bin_volumes must be finite and >= 0");
        total += x;
    }
    require(total > 0.0, "total market volume must be > 0");
    return total;
}

double fix(double v) { return std::abs(v) < kFpMin ? kFpMin : v; }

double betacf(double a, double b, double x) {
    const double qab = a + b;
    const double qap = a + 1.0;
    const double qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 / fix(1.0 - qab * x / qap);
    double h = d;
    for (int m = 1; m <= kMaxBetacfIter; ++m) {
        const double md = static_cast<double>(m);
        const double m2 = 2.0 * md;
        double aa = md * (b - md) * x / ((qam + m2) * (a + m2));
        d = 1.0 / fix(1.0 + aa * d);
        c = fix(1.0 + aa / c);
        h *= d * c;
        aa = -(a + md) * (qab + md) * x / ((a + m2) * (qap + m2));
        d = 1.0 / fix(1.0 + aa * d);
        c = fix(1.0 + aa / c);
        const double del = d * c;
        h *= del;
        if (std::abs(del - 1.0) <= kBetacfEps) return h;
    }
    throw std::runtime_error("incomplete beta continued fraction did not converge");
}

void mean_var(const std::vector<double>& xs, const char* name, double& mean, double& var) {
    require(xs.size() >= 2, std::string(name) + " must have at least 2 values");
    double s = 0.0;
    for (double v : xs) {
        require(std::isfinite(v), std::string(name) + " must be finite");
        s += v;
    }
    const double n = static_cast<double>(xs.size());
    mean = s / n;
    double ss = 0.0;
    for (double v : xs) ss += (v - mean) * (v - mean);
    var = ss / (n - 1.0);
}

}  // namespace

SlippageReport slippage(Side side, double arrival_price, double close_price,
                        const std::vector<int>& fill_bins, const std::vector<double>& fill_qtys,
                        const std::vector<double>& fill_prices,
                        const std::vector<double>& bin_volumes,
                        const std::vector<double>& bin_vwaps) {
    const double sign = sell_sign(side);
    require(std::isfinite(arrival_price) && arrival_price > 0.0,
            "arrival_price must be finite and > 0");
    require(std::isfinite(close_price) && close_price > 0.0, "close_price must be finite and > 0");
    require(bin_vwaps.size() == bin_volumes.size(), "bin arrays must have equal length");
    const double total_v = check_volumes(bin_volumes);
    for (double p : bin_vwaps) require(std::isfinite(p) && p > 0.0, "bin_vwaps must be finite and > 0");
    check_fills(fill_bins, fill_qtys, bin_volumes.size(), "fill_bins");
    require(fill_prices.size() == fill_qtys.size(), "fill arrays must have equal length");
    for (double p : fill_prices) {
        require(std::isfinite(p) && p > 0.0, "fill_prices must be finite and > 0");
    }

    double q = 0.0;
    double paid = 0.0;
    std::vector<bool> traded(bin_volumes.size(), false);
    for (std::size_t j = 0; j < fill_qtys.size(); ++j) {
        q += fill_qtys[j];
        paid += fill_qtys[j] * fill_prices[j];
        if (fill_qtys[j] > 0.0) traded[static_cast<std::size_t>(fill_bins[j])] = true;
    }
    require(q > 0.0, "executed quantity must be > 0");
    const double avg = paid / q;

    double vp = 0.0;
    double iv = 0.0;
    double ivp = 0.0;
    for (std::size_t b = 0; b < bin_volumes.size(); ++b) {
        vp += bin_volumes[b] * bin_vwaps[b];
        if (traded[b]) {
            iv += bin_volumes[b];
            ivp += bin_volumes[b] * bin_vwaps[b];
        }
    }
    require(iv > 0.0, "market volume in traded bins must be > 0");
    const double vwap = vp / total_v;
    const double ivwap = ivp / iv;
    auto cost = [&](double bench) { return 1e4 * sign * (bench - avg) / bench; };

    SlippageReport r;
    r.executed_qty = q;
    r.avg_price = avg;
    r.market_vwap = vwap;
    r.interval_vwap = ivwap;
    r.arrival_bps = cost(arrival_price);
    r.vwap_bps = cost(vwap);
    r.interval_vwap_bps = cost(ivwap);
    r.close_bps = cost(close_price);
    return r;
}

MarkoutResult markouts(Side side, const std::vector<int>& fill_steps,
                       const std::vector<double>& fill_qtys, const std::vector<double>& mids,
                       const std::vector<int>& horizons) {
    const double d = -sell_sign(side);
    require(!mids.empty(), "mids must be non-empty");
    for (double m : mids) require(std::isfinite(m), "mids must be finite");
    check_fills(fill_steps, fill_qtys, mids.size(), "fill_steps");
    require(!horizons.empty(), "horizons must be non-empty");
    for (int h : horizons) {
        require(h >= 0, "horizons must be integers >= 0");
        for (int t : fill_steps) {
            require(static_cast<std::size_t>(t) + static_cast<std::size_t>(h) < mids.size(),
                    "fill step + horizon exceeds mid series");
        }
    }
    double q = 0.0;
    for (double n : fill_qtys) q += n;
    require(q > 0.0, "executed quantity must be > 0");

    MarkoutResult r;
    r.horizons = horizons;
    for (int h : horizons) {
        double s = 0.0;
        double w = 0.0;
        for (std::size_t j = 0; j < fill_steps.size(); ++j) {
            const auto t = static_cast<std::size_t>(fill_steps[j]);
            const double mk = d * (mids[t + static_cast<std::size_t>(h)] - mids[t]);
            s += mk;
            w += fill_qtys[j] * mk;
        }
        r.per_fill.push_back(s / static_cast<double>(fill_steps.size()));
        r.per_share.push_back(w / q);
    }
    return r;
}

ParticipationResult participation(const std::vector<int>& fill_bins,
                                  const std::vector<double>& fill_qtys,
                                  const std::vector<double>& bin_volumes) {
    const double total_v = check_volumes(bin_volumes);
    check_fills(fill_bins, fill_qtys, bin_volumes.size(), "fill_bins");
    std::vector<double> qb(bin_volumes.size(), 0.0);
    double q = 0.0;
    for (std::size_t j = 0; j < fill_qtys.size(); ++j) {
        qb[static_cast<std::size_t>(fill_bins[j])] += fill_qtys[j];
        q += fill_qtys[j];
    }
    ParticipationResult r;
    for (std::size_t b = 0; b < bin_volumes.size(); ++b) {
        require(qb[b] <= bin_volumes[b], "fill volume exceeds market volume in a bin");
        r.per_bin.push_back(bin_volumes[b] > 0.0 ? qb[b] / bin_volumes[b] : 0.0);
    }
    r.overall = q / total_v;
    return r;
}

double ln_gamma(double x) {
    require(std::isfinite(x) && x > 0.0, "x must be finite and > 0");
    if (x < 0.5) return std::log(kPi / std::abs(std::sin(kPi * x))) - ln_gamma(1.0 - x);
    const double z = x - 1.0;
    double s = kLanczos[0];
    for (int i = 1; i < 9; ++i) s += kLanczos[i] / (z + static_cast<double>(i));
    const double t = z + 7.5;
    return 0.5 * std::log(2.0 * kPi) + (z + 0.5) * std::log(t) - t + std::log(s);
}

double betainc_reg(double a, double b, double x) {
    require(std::isfinite(a) && a > 0.0, "a must be finite and > 0");
    require(std::isfinite(b) && b > 0.0, "b must be finite and > 0");
    require(std::isfinite(x) && x >= 0.0 && x <= 1.0, "x must be in [0, 1]");
    if (x == 0.0) return 0.0;
    if (x == 1.0) return 1.0;
    const double ln_bt = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * std::log(x) +
                         b * std::log1p(-x);
    const double bt = std::exp(ln_bt);
    if (x < (a + 1.0) / (a + b + 2.0)) return bt * betacf(a, b, x) / a;
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b;
}

double student_t_cdf(double t, double df) {
    require(std::isfinite(t), "t must be finite");
    require(std::isfinite(df) && df > 0.0, "df must be finite and > 0");
    const double tail = 0.5 * betainc_reg(0.5 * df, 0.5, df / (df + t * t));
    return t >= 0.0 ? 1.0 - tail : tail;
}

WelchResult welch_ttest(const std::vector<double>& a, const std::vector<double>& b) {
    double ma = 0.0;
    double va = 0.0;
    double mb = 0.0;
    double vb = 0.0;
    mean_var(a, "a", ma, va);
    mean_var(b, "b", mb, vb);
    const double na = static_cast<double>(a.size());
    const double nb = static_cast<double>(b.size());
    const double ea = va / na;
    const double eb = vb / nb;
    const double se2 = ea + eb;
    require(se2 > 0.0, "samples have zero variance");
    WelchResult r;
    r.mean_diff = ma - mb;
    r.t_stat = r.mean_diff / std::sqrt(se2);
    r.df = se2 * se2 / (ea * ea / (na - 1.0) + eb * eb / (nb - 1.0));
    r.p_value = betainc_reg(0.5 * r.df, 0.5, r.df / (r.df + r.t_stat * r.t_stat));
    return r;
}

}  // namespace etrade
