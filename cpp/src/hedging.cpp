#include "etrade/hedging.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace etrade {

namespace {

/// Throw std::invalid_argument unless `cond` holds.
void require(bool cond, const char* msg) {
    if (!cond) throw std::invalid_argument(msg);
}

/// Sample mean of a per-path array.
double mean_of(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

/// Sample standard deviation (ddof = 1).
double std_of(const std::vector<double>& v) {
    const double m = mean_of(v);
    double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / static_cast<double>(v.size() - 1));
}

/// Central moments (m2, m4) with divisor n.
std::pair<double, double> central_moments(const std::vector<double>& v) {
    const double m = mean_of(v);
    double s2 = 0.0;
    double s4 = 0.0;
    for (double x : v) {
        const double d2 = (x - m) * (x - m);
        s2 += d2;
        s4 += d2 * d2;
    }
    const auto n = static_cast<double>(v.size());
    return {s2 / n, s4 / n};
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

void validate_rule(double hedge_ratio, double threshold, double target_frac) {
    require(std::isfinite(hedge_ratio) && hedge_ratio >= 0.0,
            "hedge_ratio must be finite and >= 0");
    require(std::isfinite(threshold) && threshold >= 0.0,
            "hedge_threshold must be finite and >= 0");
    require(std::isfinite(target_frac) && target_frac >= 0.0 && target_frac < 1.0,
            "hedge_target must be in [0, 1)");
}

void validate(const HedgeConfig& h) {
    require(std::isfinite(h.rho) && h.rho >= -1.0 && h.rho <= 1.0, "rho must be in [-1, 1]");
    require(std::isfinite(h.hedge_cost) && h.hedge_cost >= 0.0,
            "hedge_cost must be finite and >= 0");
    validate_rule(h.hedge_ratio, h.hedge_threshold, h.hedge_target);
}

void validate(const RiskConfig& r) {
    require(std::isfinite(r.loss_limit) && r.loss_limit >= 0.0,
            "loss_limit must be finite and >= 0");
    require(r.position_limit >= 0, "position_limit must be an integer >= 0");
    require(std::isfinite(r.flatten_penalty) && r.flatten_penalty >= 0.0,
            "flatten_penalty must be finite and >= 0");
    require(r.throttle_updates >= 0, "throttle_updates must be an integer >= 0");
    require(r.throttle_window >= 1, "throttle_window must be an integer >= 1");
    require(r.kill_step >= -1, "kill_step must be an integer >= -1");
}

}  // namespace

const std::vector<double>& HedgedResult::path(HedgeStat stat) const {
    switch (stat) {
        case HedgeStat::kPnl: return pnl;
        case HedgeStat::kSpread: return spread;
        case HedgeStat::kFees: return fees;
        case HedgeStat::kInventory: return inventory;
        case HedgeStat::kLiquidation: return liquidation;
        case HedgeStat::kMarkout: return markout;
        case HedgeStat::kFills: return fills;
        case HedgeStat::kInformedFills: return informed_fills;
        case HedgeStat::kRmsQ: return rms_q;
        case HedgeStat::kHedgePnl: return hedge_pnl;
        case HedgeStat::kHedgeCosts: return hedge_costs;
        case HedgeStat::kHedgeTrades: return hedge_trades;
        case HedgeStat::kRiskCosts: return risk_costs;
        case HedgeStat::kRmsX: return rms_x;
        case HedgeStat::kRmsSkew: return rms_skew;
        case HedgeStat::kQuoteUpdates: return quote_updates;
        case HedgeStat::kLossStop: return loss_stop;
        case HedgeStat::kKillStop: return kill_stop;
        case HedgeStat::kPosFlattens: return pos_flattens;
        case HedgeStat::kOvershoot: return overshoot;
        case HedgeStat::kStopMove: return stop_move;
    }
    throw std::invalid_argument("unknown statistic");
}

double HedgedResult::mean(HedgeStat stat) const { return mean_of(path(stat)); }

double HedgedResult::se(HedgeStat stat) const {
    return std_of(path(stat)) / std::sqrt(static_cast<double>(n_paths()));
}

double HedgedResult::var(HedgeStat stat) const { return central_moments(path(stat)).first; }

double HedgedResult::var_se(HedgeStat stat) const {
    const auto [m2, m4] = central_moments(path(stat));
    return std::sqrt(std::max(0.0, m4 - m2 * m2) / static_cast<double>(n_paths()));
}

double HedgedResult::quantile(HedgeStat stat, double p) const {
    return etrade::quantile(path(stat), p);
}

double hedge_trade(double q, double y, double hedge_ratio, double threshold,
                   double target_frac) {
    require(std::isfinite(q) && std::isfinite(y), "q and y must be finite");
    validate_rule(hedge_ratio, threshold, target_frac);
    if (hedge_ratio == 0.0) return 0.0;
    const double u = q + y / hedge_ratio;
    if (std::abs(u) <= threshold) return 0.0;
    return -hedge_ratio * (1.0 - target_frac) * u;
}

double quantile(std::vector<double> values, double p) {
    require(!values.empty(), "values must be non-empty");
    for (double v : values) require(std::isfinite(v), "values must be finite");
    require(std::isfinite(p) && p >= 0.0 && p <= 1.0, "p must be in [0, 1]");
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    const double h = static_cast<double>(n - 1) * p;
    const auto j = static_cast<std::size_t>(std::floor(h));
    if (j >= n - 1) return values[n - 1];
    return values[j] + (h - static_cast<double>(j)) * (values[j + 1] - values[j]);
}

HedgedResult simulate_hedged(const ASParams& params, int n_paths, std::uint64_t seed,
                             const MMConfig& config, const HedgeConfig& hedge,
                             const RiskConfig& risk) {
    validate(config);
    validate(hedge);
    validate(risk);
    require(n_paths >= 2 && n_paths <= kMaxAsPaths,
            "n_paths must be an integer in [2, 10000000]");
    const int n_steps = params.n_steps();
    require(config.latency_steps <= n_steps, "latency_steps must be <= n_steps");
    require(risk.kill_step <= n_steps, "kill_step must be <= n_steps");
    require(risk.throttle_window <= n_steps, "throttle_window must be <= n_steps");

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
    const double h = hedge.hedge_ratio;
    const double rho = hedge.rho;
    const double rho_perp = std::sqrt(std::max(0.0, 1.0 - rho * rho));
    const double hcost = hedge.hedge_cost;
    const int m = risk.throttle_updates;
    const int w = risk.throttle_window;
    const double pen = risk.flatten_penalty;
    const double loss_limit = risk.loss_limit;
    const int pos_limit = risk.position_limit;

    const auto np = static_cast<std::size_t>(n_paths);
    HedgedResult res;
    for (auto* v : {&res.pnl, &res.spread, &res.fees, &res.inventory, &res.liquidation,
                    &res.markout, &res.fills, &res.informed_fills, &res.rms_q,
                    &res.hedge_pnl, &res.hedge_costs, &res.hedge_trades, &res.risk_costs,
                    &res.rms_x, &res.rms_skew, &res.quote_updates, &res.loss_stop,
                    &res.kill_stop, &res.pos_flattens, &res.overshoot, &res.stop_move}) {
        v->assign(np, 0.0);
    }

    // A very negative step: "never updated" in the throttle ring.
    constexpr int kNever = std::numeric_limits<int>::min() / 2;
    std::deque<double> hist;
    std::vector<int> upd(static_cast<std::size_t>(std::max(m, 1)));
    for (std::size_t p = 0; p < np; ++p) {
        double s = params.s0;
        double b = params.s0;
        hist.assign(static_cast<std::size_t>(config.latency_steps) + 1, s);
        std::fill(upd.begin(), upd.end(), kNever);
        std::size_t ptr = 0;
        double q = 0.0;
        double y = 0.0;
        double cash = 0.0;
        bool active = true;
        double bid_q = params.s0;
        double ask_q = params.s0;
        double mark_prev = 0.0;
        double spread = 0.0;
        double fees = 0.0;
        double inventory = 0.0;
        double markout = 0.0;
        double hedge_pnl = 0.0;
        double hedge_costs = 0.0;
        double risk_costs = 0.0;
        double sum_q2 = 0.0;
        double sum_x2 = 0.0;
        double sum_skew2 = 0.0;
        int fills = 0;
        int informed_fills = 0;
        int hedge_trades = 0;
        int quote_updates = 0;
        int pos_flattens = 0;
        double loss_stop = 0.0;
        double kill_stop = 0.0;
        double overshoot = 0.0;
        double stop_move = 0.0;
        for (int i = 0; i < n_steps; ++i) {
            const double t = i * dt;
            // 1. the six draws, always, in the pinned order.
            const double z = normal(rng);
            const double u_sig = uniform(rng);
            const double u_ask = uniform(rng);
            const double u_bid = uniform(rng);
            const double u_inf = uniform(rng);
            const double z_b = normal(rng);
            // 2. stops: kill switch first, then the loss limit on the marked PnL.
            const double mark = cash + q * s + y * b;
            const bool kill = active && i == risk.kill_step;
            const bool loss = active && !kill && loss_limit > 0.0 && mark < -loss_limit;
            if (kill || loss) {
                if (kill) kill_stop = 1.0;
                if (loss) {
                    loss_stop = 1.0;
                    overshoot = std::max(0.0, -loss_limit - mark);
                    stop_move = std::abs(mark - mark_prev);
                }
                const double cost_a = pen * std::abs(q);
                cash += q * s - cost_a;
                risk_costs += cost_a;
                q = 0.0;
                const double cost_b = hcost * std::abs(y);
                cash += y * b - cost_b;
                hedge_costs += cost_b;
                if (y != 0.0) ++hedge_trades;
                y = 0.0;
                active = false;
            }
            mark_prev = mark;
            // 3. desired quotes, skewed on the risk-equivalent exposure x.
            const double x = q + rho * y;
            const double sz = sign_of(z);
            const double pred = u_sig < config.signal_accuracy ? sz : -sz;
            const double fair = hist.front() + shift_scale * pred;
            double bid, ask, skew;
            if (config.agent == MMAgent::kAs) {
                skew = x * g * sig2 * (big_t - t);
                const double half = 0.5 * (g * sig2 * (big_t - t) + markup);
                bid = fair - skew - half;
                ask = fair - skew + half;
            } else if (config.agent == MMAgent::kGlft) {
                skew = x * c2;
                bid = fair - (c1 + (2.0 * x + 1.0) / 2.0 * c2);
                ask = fair + (c1 - (2.0 * x - 1.0) / 2.0 * c2);
            } else {
                skew = 0.0;
                bid = fair - fixed_half;
                ask = fair + fixed_half;
            }
            if (tick > 0.0) {
                bid = std::floor(bid / tick + 1e-9) * tick;
                ask = std::ceil(ask / tick - 1e-9) * tick;
            }
            if (active) sum_skew2 += skew * skew;
            // 4. throttle: update only if fewer than m updates in the last w steps.
            bool allow = active;
            if (m > 0 && allow) {
                allow = upd[ptr] <= i - w;
                if (allow) {
                    upd[ptr] = i;
                    ptr = (ptr + 1) % static_cast<std::size_t>(m);
                }
            }
            if (allow) {
                bid_q = bid;
                ask_q = ask;
                ++quote_updates;
            }
            // 5. Section 9 fills against the standing quotes.
            const bool can_ask = active && q > -params.q_max;
            const bool can_bid = active && q < params.q_max;
            const double p_ask = -std::expm1(-a_rate * std::exp(-k * (ask_q - s)) * dt);
            const double p_bid = -std::expm1(-a_rate * std::exp(-k * (s - bid_q)) * dt);
            const bool fill_ask = can_ask && u_ask < p_ask;
            const bool fill_bid = can_bid && u_bid < p_bid;
            const double move = sig * sqdt * z;
            const bool informed = u_inf < config.informed_prob;
            const bool inf_ask = informed && z > 0.0 && can_ask && !fill_ask && ask_q < s + move;
            const bool inf_bid = informed && z < 0.0 && can_bid && !fill_bid && bid_q > s + move;
            const bool fa = fill_ask || inf_ask;
            const bool fb = fill_bid || inf_bid;
            const int n_fill = static_cast<int>(fa) + static_cast<int>(fb);
            const int dq = static_cast<int>(fb) - static_cast<int>(fa);
            q += dq;
            cash = cash + (fa ? ask_q : 0.0) - (fb ? bid_q : 0.0) + fee * n_fill;
            spread += (fa ? ask_q - s : 0.0) + (fb ? s - bid_q : 0.0);
            fees += fee * n_fill;
            fills += n_fill;
            informed_fills += static_cast<int>(inf_ask) + static_cast<int>(inf_bid);
            // 6. position limit: flatten A at the mid less the penalty.
            if (active && pos_limit > 0 && std::abs(q) >= pos_limit) {
                const double cost = pen * std::abs(q);
                cash += q * s - cost;
                risk_costs += cost;
                ++pos_flattens;
                q = 0.0;
            }
            // 7. hedge in B at its mid plus the hedge cost.
            if (active && h > 0.0) {
                const double u = q + y / h;
                if (std::abs(u) > hedge.hedge_threshold) {
                    const double dy = -h * (1.0 - hedge.hedge_target) * u;
                    const double cost = hcost * std::abs(dy);
                    cash -= dy * b + cost;
                    hedge_costs += cost;
                    ++hedge_trades;
                    y += dy;
                }
            }
            // 8. accounting, then the moves of A and B.
            const double move_b = sig * sqdt * (rho * z + rho_perp * z_b);
            inventory += q * move;
            markout += dq * move;
            hedge_pnl += y * move_b;
            const double x_end = q + rho * y;
            sum_q2 += q * q;
            sum_x2 += x_end * x_end;
            s += move;
            b += move_b;
            hist.pop_front();
            hist.push_back(s);
        }
        const double liq = params.eta_liq * std::abs(q);
        const double unwind = hcost * std::abs(y);
        hedge_costs += unwind;
        if (y != 0.0) ++hedge_trades;
        const double pnl = cash + q * s + y * b - liq - unwind;
        res.pnl[p] = pnl;
        res.spread[p] = spread;
        res.fees[p] = fees;
        res.inventory[p] = inventory;
        res.liquidation[p] = liq;
        res.markout[p] = markout;
        res.fills[p] = fills;
        res.informed_fills[p] = informed_fills;
        res.rms_q[p] = std::sqrt(sum_q2 / n_steps);
        res.hedge_pnl[p] = hedge_pnl;
        res.hedge_costs[p] = hedge_costs;
        res.hedge_trades[p] = hedge_trades;
        res.risk_costs[p] = risk_costs;
        res.rms_x[p] = std::sqrt(sum_x2 / n_steps);
        res.rms_skew[p] = std::sqrt(sum_skew2 / n_steps);
        res.quote_updates[p] = quote_updates;
        res.loss_stop[p] = loss_stop;
        res.kill_stop[p] = kill_stop;
        res.pos_flattens[p] = pos_flattens;
        res.overshoot[p] = overshoot;
        res.stop_move[p] = stop_move;
        const double recon =
            spread + fees + inventory - liq + hedge_pnl - hedge_costs - risk_costs;
        res.max_identity_error = std::max(
            res.max_identity_error, std::abs(recon - pnl) / std::max(1.0, std::abs(pnl)));
    }
    return res;
}

}  // namespace etrade
