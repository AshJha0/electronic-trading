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

ImpactParams impact_params_from_market(const ImpactMarket& m) {
    positive("price", m.price);
    positive("spread", m.spread);
    positive("adv", m.adv);
    positive("daily_vol", m.daily_vol);
    positive("y", m.y);
    require(std::isfinite(m.perm_frac) && m.perm_frac > 0.0 && m.perm_frac <= 1.0,
            "perm_frac must be in (0, 1]");
    require(std::isfinite(m.drift_bps_per_day) && m.drift_bps_per_day >= 0.0,
            "drift_bps_per_day must be finite and >= 0");
    require(std::isfinite(m.alpha) && m.alpha > 0.0 && m.alpha <= 1.0, "alpha must be in (0, 1]");
    const double sigma = m.price * m.daily_vol;
    return ImpactParams(m.total_shares, m.n_intervals, m.tau, sigma,
                        m.y * sigma / std::pow(m.adv, m.alpha), m.alpha,
                        m.spread / (m.perm_frac * m.adv), m.lambda_ra,
                        m.price * m.drift_bps_per_day * 1e-4, m.weights);
}

PretradeReport pretrade_report(const ImpactMarket& m, const std::vector<double>& lambdas) {
    require(!lambdas.empty(), "lambdas must be non-empty");
    ImpactMarket base = m;
    base.lambda_ra = 0.0;
    const ImpactParams p0 = impact_params_from_market(base);
    const double notional = m.total_shares * m.price;
    positive("total_shares * price", notional);
    PretradeReport rep;
    rep.half_spread_bps = bps(half_spread_cost(m.spread, m.total_shares), notional);
    auto row = [&](const char* label, double lam, const std::vector<double>& x, int horizon,
                   const ImpactParams& p) {
        PretradeRow r;
        r.label = label;
        r.lambda_ra = lam;
        r.impact_bps = bps(impact_expected_cost(p, x), notional);
        r.sd_bps = bps(std::sqrt(impact_variance_cost(p, x)), notional);
        r.total_bps = r.impact_bps + rep.half_spread_bps;
        r.horizon = horizon;
        return r;
    };
    rep.rows.push_back(row("twap", 0.0, twap_holdings(m.total_shares, m.n_intervals), m.n_intervals, p0));
    rep.rows.push_back(row("vwap", 0.0, vwap_holdings(m.total_shares, p0.weights()), m.n_intervals, p0));
    for (double lam : lambdas) {
        require(std::isfinite(lam) && lam >= 0.0, "lambdas must be finite and >= 0");
        ImpactMarket ml = m;
        ml.lambda_ra = lam;
        const ImpactParams p = impact_params_from_market(ml);
        const ScheduleResult s = optimal_schedule(p);
        rep.rows.push_back(row("optimal", lam, s.holdings, s.horizon, p));
    }
    rep.sqrt_law_bps = sqrt_law_cost_bps(m.daily_vol, m.total_shares, m.adv, m.y);
    return rep;
}

}  // namespace etrade
