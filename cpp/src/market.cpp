#include "etrade/market.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace etrade {
namespace {

void require(bool cond, const std::string& msg) {
    if (!cond) throw std::invalid_argument(msg);
}

void positive(const char* name, double v) {
    require(std::isfinite(v) && v > 0.0, std::string(name) + " must be finite and > 0");
}

}  // namespace

ACParams ac_params_from_market(const ACMarket& m) {
    positive("price", m.price);
    positive("spread", m.spread);
    positive("adv", m.adv);
    positive("daily_vol", m.daily_vol);
    require(std::isfinite(m.temp_frac) && m.temp_frac > 0.0 && m.temp_frac <= 1.0,
            "temp_frac must be in (0, 1]");
    require(std::isfinite(m.perm_frac) && m.perm_frac > 0.0 && m.perm_frac <= 1.0,
            "perm_frac must be in (0, 1]");
    return ACParams(m.total_shares, m.n_intervals, m.tau, m.price * m.daily_vol,
                    m.spread / (m.temp_frac * m.adv), m.spread / (m.perm_frac * m.adv),
                    m.lambda_ra);
}

ASParams as_params_from_market(const ASMarket& m) {
    positive("price", m.price);
    positive("tick", m.tick);
    positive("daily_vol", m.daily_vol);
    positive("touch_fills_per_day", m.touch_fills_per_day);
    positive("halving_ticks", m.halving_ticks);
    positive("skew_ticks", m.skew_ticks);
    require(std::isfinite(m.liq_ticks) && m.liq_ticks >= 0.0, "liq_ticks must be finite and >= 0");
    positive("dt_seconds", m.dt_seconds);
    positive("session_seconds", m.session_seconds);
    positive("horizon_days", m.horizon_days);
    const double sigma = m.price * m.daily_vol;
    const double k = std::log(2.0) / (m.halving_ticks * m.tick);
    return ASParams(m.price, sigma, m.horizon_days, m.dt_seconds / m.session_seconds,
                    m.skew_ticks * m.tick / (sigma * sigma * m.horizon_days), k,
                    m.touch_fills_per_day * std::exp(k * m.tick / 2.0), m.q_max,
                    m.liq_ticks * m.tick);
}

double bps(double amount, double notional) {
    require(std::isfinite(amount), "amount must be finite");
    positive("notional", notional);
    return 1e4 * amount / notional;
}

double half_spread_cost(double spread, double total_shares) {
    positive("spread", spread);
    require(std::isfinite(total_shares) && total_shares >= 0.0,
            "total_shares must be finite and >= 0");
    return 0.5 * spread * total_shares;
}

}  // namespace etrade
