#include "etrade/mm.hpp"
#include "etrade/detail/require.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdlib>
#include <deque>
#include <random>
#include <stdexcept>

namespace etrade {

namespace {

using detail::require;

/// Sample mean of a per-path array.
double mean_of(const std::vector<double>& v) {
    require(!v.empty(), "statistic undefined for an empty result (no paths)");
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

/// Sample standard deviation (ddof = 1).
double std_of(const std::vector<double>& v) {
    require(v.size() >= 2, "sample std needs at least two paths");
    const double m = mean_of(v);
    double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / static_cast<double>(v.size() - 1));
}

/// sign(x) with sign(0) = 0.
double sign_of(double x) { return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0); }

void validate(const MMConfig& c) {
    require(std::isfinite(c.tick) && c.tick >= 0.0, "tick must be finite and >= 0");
    require(std::isfinite(c.maker_fee), "maker_fee must be finite");
    require(c.latency_steps >= 0, "latency_steps must be an integer >= 0");
    require(std::isfinite(c.informed_prob) && c.informed_prob >= 0.0 &&
                c.informed_prob <= 1.0,
            "informed_prob must be in [0, 1]");
    require(std::isfinite(c.signal_accuracy) && c.signal_accuracy >= 0.5 &&
                c.signal_accuracy <= 1.0,
            "signal_accuracy must be in [0.5, 1]");
    if (c.fixed_spread) {
        require(std::isfinite(*c.fixed_spread) && *c.fixed_spread > 0.0,
                "fixed_spread must be finite and > 0");
    }
}

}  // namespace

const std::vector<double>& MMResult::path(MMStat stat) const {
    switch (stat) {
        case MMStat::kPnl: return pnl;
        case MMStat::kSpread: return spread;
        case MMStat::kFees: return fees;
        case MMStat::kInventory: return inventory;
        case MMStat::kLiquidation: return liquidation;
        case MMStat::kMarkout: return markout;
        case MMStat::kFills: return fills;
        case MMStat::kInformedFills: return informed_fills;
        case MMStat::kRmsQ: return rms_q;
    }
    throw std::invalid_argument("unknown statistic");
}

double MMResult::mean(MMStat stat) const { return mean_of(path(stat)); }

double MMResult::se(MMStat stat) const {
    return std_of(path(stat)) / std::sqrt(static_cast<double>(n_paths()));
}

std::pair<double, double> round_to_tick(double bid, double ask, double tick) {
    require(std::isfinite(tick) && tick >= 0.0, "tick must be finite and >= 0");
    require(std::isfinite(bid) && std::isfinite(ask), "bid and ask must be finite");
    if (tick == 0.0) return {bid, ask};
    return {std::floor(bid / tick + 1e-9) * tick, std::ceil(ask / tick - 1e-9) * tick};
}

std::pair<double, double> glft_half_spreads(const ASParams& params, double q) {
    require(std::isfinite(q), "q must be finite");
    const double g = params.gamma;
    const double k = params.k;
    const double c1 = std::log1p(g / k) / g;
    const double c2 = std::sqrt(params.sigma * params.sigma * g / (2.0 * k * params.a_rate) *
                                std::pow(1.0 + g / k, 1.0 + k / g));
    return {c1 + (2.0 * q + 1.0) / 2.0 * c2, c1 - (2.0 * q - 1.0) / 2.0 * c2};
}

double weighted_mid(double bid, double ask, double bid_size, double ask_size) {
    require(std::isfinite(bid), "bid must be finite");
    require(std::isfinite(ask), "ask must be finite");
    require(ask >= bid, "ask must be >= bid");
    require(std::isfinite(bid_size) && bid_size >= 0.0 && std::isfinite(ask_size) &&
                ask_size >= 0.0,
            "sizes must be finite and >= 0");
    require(bid_size + ask_size > 0.0, "bid_size + ask_size must be > 0");
    return (bid * ask_size + ask * bid_size) / (bid_size + ask_size);
}

std::vector<double> order_flow_imbalance(const std::vector<double>& bid_px,
                                         const std::vector<double>& bid_sz,
                                         const std::vector<double>& ask_px,
                                         const std::vector<double>& ask_sz) {
    const std::size_t n = bid_px.size();
    require(n >= 2, "need at least 2 book states");
    require(bid_sz.size() == n && ask_px.size() == n && ask_sz.size() == n,
            "book series must be 1-D and equal length");
    for (const auto* v : {&bid_px, &bid_sz, &ask_px, &ask_sz}) {
        for (double x : *v) require(std::isfinite(x), "book series must be finite");
    }
    for (std::size_t j = 0; j < n; ++j) {
        require(bid_sz[j] >= 0.0 && ask_sz[j] >= 0.0, "sizes must be >= 0");
    }
    std::vector<double> e(n - 1);
    for (std::size_t j = 1; j < n; ++j) {
        const double up_b = bid_px[j] >= bid_px[j - 1] ? 1.0 : 0.0;
        const double dn_b = bid_px[j] <= bid_px[j - 1] ? 1.0 : 0.0;
        const double dn_a = ask_px[j] <= ask_px[j - 1] ? 1.0 : 0.0;
        const double up_a = ask_px[j] >= ask_px[j - 1] ? 1.0 : 0.0;
        e[j - 1] = up_b * bid_sz[j] - dn_b * bid_sz[j - 1] - dn_a * ask_sz[j] +
                   up_a * ask_sz[j - 1];
    }
    return e;
}

MMResult simulate_mm(const ASParams& params, int n_paths, std::uint64_t seed,
                     const MMConfig& config) {
    validate(config);
    require(n_paths >= 2 && n_paths <= kMaxAsPaths,
            "n_paths must be an integer in [2, 10000000]");
    const int n_steps = params.n_steps();
    require(config.latency_steps <= n_steps, "latency_steps must be <= n_steps");

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::normal_distribution<double> normal(0.0, 1.0);

    const double g = params.gamma;
    const double sig = params.sigma;
    const double sig2 = sig * sig;
    const double big_t = params.horizon;
    const double dt = params.dt;
    const double k = params.k;
    const double a_rate = params.a_rate;
    const double sqdt = std::sqrt(dt);
    const double markup = 2.0 / g * std::log1p(g / k);
    const double fixed_half =
        0.5 * (config.fixed_spread ? *config.fixed_spread : 0.5 * g * sig2 * big_t + markup);
    const double c1 = std::log1p(g / k) / g;
    const double c2 = std::sqrt(sig2 * g / (2.0 * k * a_rate) * std::pow(1.0 + g / k, 1.0 + k / g));
    const double shift_scale = (2.0 * config.signal_accuracy - 1.0) * sig * sqdt *
                               std::sqrt(2.0 / 3.14159265358979323846);
    const double tick = config.tick;
    const double fee = config.maker_fee;

    const auto np = static_cast<std::size_t>(n_paths);
    MMResult res;
    for (auto* v : {&res.pnl, &res.spread, &res.fees, &res.inventory, &res.liquidation,
                    &res.markout, &res.fills, &res.informed_fills, &res.rms_q}) {
        v->assign(np, 0.0);
    }

    std::deque<double> hist;
    for (std::size_t p = 0; p < np; ++p) {
        double s = params.s0;
        hist.assign(static_cast<std::size_t>(config.latency_steps) + 1, s);
        int q = 0;
        double cash = 0.0;
        double spread = 0.0;
        double fees = 0.0;
        double inventory = 0.0;
        double markout = 0.0;
        double sum_q2 = 0.0;
        int fills = 0;
        int informed_fills = 0;
        for (int i = 0; i < n_steps; ++i) {
            const double t = i * dt;
            // 1. the five draws, always, in the pinned order.
            const double z = normal(rng);
            const double u_sig = uniform(rng);
            const double u_ask = uniform(rng);
            const double u_bid = uniform(rng);
            const double u_inf = uniform(rng);
            // 2. fair value from the stale mid plus the signal's expected move.
            const double sz = sign_of(z);
            const double pred = u_sig < config.signal_accuracy ? sz : -sz;
            const double fair = hist.front() + shift_scale * pred;
            // 3. quotes, then tick rounding away from fair value.
            double bid, ask;
            if (config.agent == MMAgent::kAs) {
                const double r = fair - q * g * sig2 * (big_t - t);
                const double half = 0.5 * (g * sig2 * (big_t - t) + markup);
                bid = r - half;
                ask = r + half;
            } else if (config.agent == MMAgent::kGlft) {
                bid = fair - (c1 + (2.0 * q + 1.0) / 2.0 * c2);
                ask = fair + (c1 - (2.0 * q - 1.0) / 2.0 * c2);
            } else {
                bid = fair - fixed_half;
                ask = fair + fixed_half;
            }
            if (tick > 0.0) {
                bid = std::floor(bid / tick + 1e-9) * tick;
                ask = std::ceil(ask / tick - 1e-9) * tick;
            }
            // 4. uninformed fills, distances from the true current mid.
            const bool can_ask = q > -params.q_max;
            const bool can_bid = q < params.q_max;
            const double p_ask = -std::expm1(-a_rate * std::exp(-k * (ask - s)) * dt);
            const double p_bid = -std::expm1(-a_rate * std::exp(-k * (s - bid)) * dt);
            const bool fill_ask = can_ask && u_ask < p_ask;
            const bool fill_bid = can_bid && u_bid < p_bid;
            // 5. informed fills against a quote inside the coming move.
            const double move = sig * sqdt * z;
            const bool informed = u_inf < config.informed_prob;
            const bool inf_ask = informed && z > 0.0 && can_ask && !fill_ask && ask < s + move;
            const bool inf_bid = informed && z < 0.0 && can_bid && !fill_bid && bid > s + move;
            // 6. updates and attribution.
            const bool fa = fill_ask || inf_ask;
            const bool fb = fill_bid || inf_bid;
            const int n_fill = static_cast<int>(fa) + static_cast<int>(fb);
            const int dq = static_cast<int>(fb) - static_cast<int>(fa);
            q += dq;
            cash = cash + (fa ? ask : 0.0) - (fb ? bid : 0.0) + fee * n_fill;
            spread += (fa ? ask - s : 0.0) + (fb ? s - bid : 0.0);
            fees += fee * n_fill;
            fills += n_fill;
            informed_fills += static_cast<int>(inf_ask) + static_cast<int>(inf_bid);
            inventory += q * move;
            markout += dq * move;
            sum_q2 += static_cast<double>(q) * q;
            // 7. mid move; the history keeps the last latency_steps + 1 mids.
            s += move;
            hist.pop_front();
            hist.push_back(s);
        }
        const double liq = params.eta_liq * std::abs(q);
        const double pnl = cash + q * s - liq;
        res.pnl[p] = pnl;
        res.spread[p] = spread;
        res.fees[p] = fees;
        res.inventory[p] = inventory;
        res.liquidation[p] = liq;
        res.markout[p] = markout;
        res.fills[p] = fills;
        res.informed_fills[p] = informed_fills;
        res.rms_q[p] = std::sqrt(sum_q2 / n_steps);
        const double recon = spread + fees + inventory - liq;
        res.max_identity_error = std::max(
            res.max_identity_error, std::abs(recon - pnl) / std::max(1.0, std::abs(pnl)));
    }
    return res;
}

}  // namespace etrade
