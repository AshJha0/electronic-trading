/// \file demo.cpp
/// \brief End-to-end demo of the etrade C++ library.
///
/// Prints: Almgren-Chriss optimal trajectories across risk aversions, the
/// efficient frontier of execution, TWAP/POV/instantaneous baselines with an
/// MC cross-check, Avellaneda-Stoikov market-making simulations (A-S vs
/// fixed-spread agent, liquid and FX-like parameter sets), and an
/// implementation-shortfall decomposition with a market-impact regression.
///
/// Parameter values mirror data/params_liquid.json, data/params_fx.json and
/// data/parent_order.json (hardcoded so the demo runs from any directory).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <optional>
#include <vector>

#include "etrade/almgren_chriss.hpp"
#include "etrade/analytics.hpp"
#include "etrade/avellaneda_stoikov.hpp"
#include "etrade/market.hpp"

namespace {

void hdr(const char* title) {
    std::printf("\n=== %s ", title);
    for (int i = static_cast<int>(std::strlen(title)); i < 66; ++i) {
        std::printf("=");
    }
    std::printf("\n");
}

void ac_section() {
    // data/params_liquid.json
    const etrade::ACParams p(1e6, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6);
    hdr("Almgren-Chriss optimal trajectories (sell 1.0e6 shares, N=5)");
    std::printf("%10s %9s %10s | holdings x_j (thousands)\n", "lambda_ra",
                "kappa", "half-life");
    for (double lam : {0.0, 5e-7, 2e-6, 1e-5}) {
        const etrade::ACParams pl = p.with_lambda(lam);
        const auto x = etrade::ac_trajectory(pl);
        const double hl = etrade::half_life(pl);
        if (std::isfinite(hl)) {
            std::printf("%10.1e %9.4f %10.3f |", lam, etrade::ac_kappa(pl), hl);
        } else {
            std::printf("%10.1e %9.4f %10s |", lam, etrade::ac_kappa(pl), "inf");
        }
        for (double v : x) std::printf(" %8.1f", v / 1e3);
        std::printf("\n");
    }

    hdr("Efficient frontier of execution (E[cost] vs sqrt(Var[cost]))");
    const std::vector<double> grid = {0.0,  1e-8, 1e-7, 5e-7,
                                      1e-6, 2e-6, 5e-6, 1e-5};
    std::printf("%10s %14s %14s %9s\n", "lambda_ra", "E[cost] $",
                "sqrt(Var) $", "kappa");
    for (const auto& fp : etrade::efficient_frontier(p, grid)) {
        std::printf("%10.1e %14.0f %14.0f %9.4f\n", fp.lambda_ra, fp.e_cost,
                    fp.std_cost, fp.kappa);
    }

    hdr("Baselines: optimal vs TWAP vs POV vs instantaneous");
    const auto twap = etrade::twap_holdings(p.total_shares, p.n_intervals);
    const auto pov = etrade::pov_holdings(p.total_shares, p.tau, 2.5e6, 0.15);
    const auto [e_i, v_i] = etrade::instantaneous_cost(p);
    std::printf("%12s %14s %14s\n", "strategy", "E[cost] $", "sqrt(Var) $");
    std::printf("%12s %14.0f %14.0f\n", "AC optimal", etrade::expected_cost(p),
                std::sqrt(etrade::variance_cost(p)));
    std::printf("%12s %14.0f %14.0f\n", "TWAP",
                etrade::expected_cost(p, twap),
                std::sqrt(etrade::variance_cost(p, twap)));
    std::printf("%12s %14.0f %14.0f\n", "POV 15%",
                etrade::expected_cost(p, pov),
                std::sqrt(etrade::variance_cost(p, pov)));
    std::printf("%12s %14.0f %14.0f\n", "Instant", e_i, std::sqrt(v_i));
    const etrade::ISSimResult mc = etrade::simulate_is(p, 20000, 123);
    std::printf("MC check (20k paths): E=%.0f (closed %.0f, se %.0f); "
                "Var/1e9=%.1f (closed %.1f)\n",
                mc.mean, etrade::expected_cost(p), mc.se_mean, mc.var / 1e9,
                etrade::variance_cost(p) / 1e9);
}

void as_section(const etrade::ASParams& ap, const char* label,
                std::uint64_t seed) {
    hdr(label);
    std::printf("params: s0=%g sigma=%g gamma=%g k=%g A=%g T=%g dt=%g "
                "q_max=%d\n",
                ap.s0, ap.sigma, ap.gamma, ap.k, ap.a_rate, ap.horizon, ap.dt,
                ap.q_max);
    const auto [bid, ask] = etrade::quotes(ap, ap.s0, 3.0, 0.0);
    std::printf("example quote at t=0, q=+3: bid=%.4f ask=%.4f (mid %g, "
                "reservation %.4f)\n",
                bid, ask, ap.s0, etrade::reservation_price(ap, ap.s0, 3.0, 0.0));
    const int n_paths = 2000;
    const auto r_as = etrade::simulate(ap, n_paths, seed, etrade::Agent::kAS);
    const auto r_fx =
        etrade::simulate(ap, n_paths, seed, etrade::Agent::kFixed);
    std::printf("%14s %10s %9s %7s %8s %11s %7s %7s\n", "agent", "mean PnL",
                "std PnL", "Sharpe", "mean|q|", "mean max|q|", "rms q",
                "fills");
    struct Row { const char* name; const etrade::ASSimResult* r; };
    for (const Row& row : {Row{"A-S", &r_as}, Row{"fixed-spread", &r_fx}}) {
        std::printf("%14s %10.4f %9.4f %7.2f %8.3f %11.3f %7.3f %7.1f\n",
                    row.name, row.r->mean_pnl(), row.r->std_pnl(),
                    row.r->sharpe(), row.r->mean_abs_q(),
                    row.r->mean_max_abs_q(), row.r->rms_q(),
                    row.r->mean_fills());
    }
    std::printf("inventory control: A-S rms q = %.3f vs fixed %.3f "
                "(similar PnL, much tighter inventory)\n",
                r_as.rms_q(), r_fx.rms_q());
    const auto stress =
        etrade::simulate(ap, n_paths, seed, etrade::Agent::kAS,
                         std::nullopt, 2.0, 0.5);
    std::printf("stress (vol x2 after T/2): std PnL %.4f -> %.4f\n",
                r_as.std_pnl(), stress.std_pnl());
}

void analytics_section() {
    hdr("Implementation-shortfall decomposition (sell parent order)");
    // data/parent_order.json
    const auto d = etrade::decompose_shortfall(
        etrade::Side::kSell, 100.0, 99.5, 98.8, 1000.0, {300.0, 400.0, 200.0},
        {99.4, 99.2, 99.1});
    std::printf("decision 100.00, arrival 99.50, close 98.80; "
                "executed %.0f/1000\n",
                d.executed_qty);
    std::printf("  %12s: %10.2f $\n", "delay", d.delay);
    std::printf("  %12s: %10.2f $\n", "trading", d.trading);
    std::printf("  %12s: %10.2f $\n", "opportunity", d.opportunity);
    std::printf("  %12s: %10.2f $\n", "TOTAL", d.total);
    std::printf("participation rate example: 120k / 1.5m = %.2f%%\n",
                100.0 * etrade::participation_rate(120000.0, 1.5e6));

    hdr("Market-impact regression (cost/share vs participation, native OLS)");
    const auto [part, cost] = etrade::synthetic_impact_fills(25, 0.5, 8.0, 0.05);
    const auto reg = etrade::estimate_impact(part, cost);
    std::printf("fitted: cost = %.4f + %.4f * participation (true 0.50 + "
                "8.00), R^2 = %.4f, n = %d\n",
                reg.intercept, reg.slope, reg.r_squared, reg.n_obs);
}

/// Presets in market units, results in bps (mirrors data/presets.json; the
/// market golden cases pin the same numbers).
void market_section() {
    hdr("Market presets in bps (execution: sell order; market making: A-S spread)");
    struct Row {
        const char* name;
        etrade::ACMarket ac;
        bool has_as;
        etrade::ASMarket as;
    };
    auto acm = [](double price, double spread, double adv, double vol, double x, int n,
                  double tau, double lam) {
        etrade::ACMarket m;
        m.price = price;
        m.spread = spread;
        m.adv = adv;
        m.daily_vol = vol;
        m.total_shares = x;
        m.n_intervals = n;
        m.tau = tau;
        m.lambda_ra = lam;
        return m;
    };
    auto asm_ = [](double price, double tick, double vol, double touch, double halving,
                   double skew, int qmax, double liq, double dt, double session) {
        etrade::ASMarket m;
        m.price = price;
        m.tick = tick;
        m.daily_vol = vol;
        m.touch_fills_per_day = touch;
        m.halving_ticks = halving;
        m.skew_ticks = skew;
        m.q_max = qmax;
        m.liq_ticks = liq;
        m.dt_seconds = dt;
        m.session_seconds = session;
        return m;
    };
    const Row rows[] = {
        {"paper_almgren_chriss_2000", acm(50.0, 0.125, 5e6, 0.019, 1e6, 5, 1.0, 2e-6), false,
         etrade::ASMarket{}},
        {"us_large_cap", acm(100.0, 0.01, 5e6, 0.015, 250000.0, 13, 1.0 / 13.0, 1e-7), true,
         asm_(100.0, 0.01, 0.015, 1170.0, 0.462, 2.25, 50, 1.0, 5.0, 23400.0)},
        {"us_small_cap", acm(20.0, 0.05, 3e5, 0.035, 30000.0, 13, 1.0 / 13.0, 3e-5), true,
         asm_(20.0, 0.01, 0.035, 195.0, 2.0, 5.0, 10, 5.0, 5.0, 23400.0)},
        {"us_etf", acm(500.0, 0.01, 6e7, 0.01, 600000.0, 13, 1.0 / 13.0, 7e-10), true,
         asm_(500.0, 0.01, 0.01, 11700.0, 0.5, 1.0, 200, 1.0, 1.0, 23400.0)},
        {"eurusd", acm(1.10, 5e-5, 2e9, 0.005, 1e8, 24, 1.0 / 24.0, 8e-8), true,
         asm_(1.10, 1e-5, 0.005, 8640.0, 2.0, 3.0, 50, 2.0, 5.0, 86400.0)},
    };
    std::printf("%26s %10s %8s %8s %9s %22s\n", "preset", "order %ADV", "E[IS]", "sd",
                "half-sprd", "A-S spread open/close");
    for (const Row& r : rows) {
        const etrade::ACParams p = etrade::ac_params_from_market(r.ac);
        const double notional = r.ac.total_shares * r.ac.price;
        const double e = etrade::bps(etrade::expected_cost(p), notional);
        const double sd = etrade::bps(std::sqrt(etrade::variance_cost(p)), notional);
        const double hs =
            etrade::bps(etrade::half_spread_cost(r.ac.spread, r.ac.total_shares), notional);
        char mm[64] = "-";
        if (r.has_as) {
            const etrade::ASParams a = etrade::as_params_from_market(r.as);
            std::snprintf(mm, sizeof mm, "%8.2f / %6.2f bps",
                          etrade::bps(etrade::optimal_spread(a, 0.0), a.s0),
                          etrade::bps(etrade::optimal_spread(a, a.horizon), a.s0));
        }
        std::printf("%26s %9.1f%% %6.2fbp %6.2fbp %7.2fbp %22s\n", r.name,
                    r.ac.total_shares / r.ac.adv * 100.0, e, sd, hs, mm);
    }
}

/// Pre-trade report on the power-law model (API_SPEC Section 8).
void pretrade_section() {
    hdr("Pre-trade report: US large cap, sell 5% ADV in a day (sqrt impact, U-curve)");
    etrade::ImpactMarket m;
    m.price = 100.0;
    m.spread = 0.01;
    m.adv = 5e6;
    m.daily_vol = 0.015;
    m.total_shares = 250000.0;
    m.n_intervals = 13;
    m.tau = 1.0 / 13.0;
    m.weights.assign(etrade::kUsEquityVolumeCurve13.begin(), etrade::kUsEquityVolumeCurve13.end());
    const etrade::PretradeReport rep = etrade::pretrade_report(m, {0.0, 1e-7, 1e-6, 1e-5});
    std::printf("square-root-law estimate %.2f bps; half spread %.2f bps\n", rep.sqrt_law_bps,
                rep.half_spread_bps);
    std::printf("%10s %9s %9s %9s %9s %8s\n", "strategy", "lambda", "impact", "sd", "total",
                "horizon");
    for (const auto& r : rep.rows) {
        std::printf("%10s %9.1e %7.2fbp %7.2fbp %7.2fbp %8d\n", r.label.c_str(), r.lambda_ra,
                    r.impact_bps, r.sd_bps, r.total_bps, r.horizon);
    }
    etrade::ImpactMarket md = m;
    md.drift_bps_per_day = 50.0;
    const etrade::ImpactParams p = etrade::impact_params_from_market(md);
    const double notional = m.total_shares * m.price;
    const auto vwap = etrade::vwap_holdings(m.total_shares, m.weights);
    const auto s = etrade::optimal_schedule(p);
    const double e_v = etrade::bps(etrade::impact_expected_cost(p, vwap), notional);
    const double e_o = etrade::bps(etrade::impact_expected_cost(p, s.holdings), notional);
    std::printf("alpha-aware: with 50 bps/day adverse drift, VWAP costs %.2f bps and the "
                "drift-aware schedule %.2f bps (saves %.2f bps)\n", e_v, e_o, e_v - e_o);
}

}  // namespace

int main() {
    std::printf("etrade demo — Almgren-Chriss execution & Avellaneda-Stoikov "
                "market making (C++)\n");
    ac_section();
    // data/params_liquid.json / data/params_fx.json
    const etrade::ASParams liquid(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, 20,
                                  0.5);
    const etrade::ASParams fx(1.1, 0.004, 1.0, 0.005, 100.0, 2000.0, 500.0, 50,
                              0.0002);
    as_section(liquid, "Avellaneda-Stoikov market making (liquid large-cap)",
               42);
    as_section(fx, "Avellaneda-Stoikov market making (FX-like: tight spread)",
               7);
    analytics_section();
    market_section();
    pretrade_section();
    std::printf("\ndone.\n");
    return 0;
}
