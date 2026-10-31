/// \file demo.cpp
/// \brief End-to-end demo of the etrade C++ library.
///
/// Prints: Almgren-Chriss optimal trajectories across risk aversions, the
/// efficient frontier of execution, TWAP/POV/instantaneous baselines with an
/// MC cross-check, Avellaneda-Stoikov market-making simulations (A-S vs
/// fixed-spread agent, liquid and FX-like parameter sets), and an
/// implementation-shortfall decomposition with a market-impact regression,
/// presets in bps, a pre-trade report, post-trade TCA, calibration with the
/// fitted ITCH presets, execution extensions, the market-making friction
/// ledger, limit-order placement, and hedged market making with risk
/// controls.
///
/// Parameter values mirror data/params_liquid.json, data/params_fx.json and
/// data/parent_order.json (hardcoded so the demo runs from any directory).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <vector>

#include "etrade/almgren_chriss.hpp"
#include "etrade/analytics.hpp"
#include "etrade/avellaneda_stoikov.hpp"
#include "etrade/calibration.hpp"
#include "etrade/execution.hpp"
#include "etrade/hedging.hpp"
#include "etrade/impact.hpp"
#include "etrade/market.hpp"
#include "etrade/mm.hpp"
#include "etrade/placement.hpp"
#include "etrade/tca.hpp"

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

/// Post-trade TCA (API_SPEC Section 10); deterministic.
void tca_section() {
    hdr("Post-trade TCA: benchmarks, markouts, participation, A/B test");
    const std::vector<int> fill_bins{0, 1, 1, 2};
    const std::vector<double> fill_qtys{20000.0, 15000.0, 10000.0, 25000.0};
    const std::vector<double> fill_prices{99.96, 99.91, 99.89, 99.84};
    const std::vector<double> bin_volumes{400000.0, 300000.0, 300000.0, 500000.0};
    const std::vector<double> bin_vwaps{99.95, 99.90, 99.85, 99.82};
    const etrade::SlippageReport r =
        etrade::slippage(etrade::Side::kSell, 100.0, 99.80, fill_bins, fill_qtys, fill_prices,
                         bin_volumes, bin_vwaps);
    std::printf("sell 70,000 at avg %.4f: arrival %.4f bps, full-day VWAP %.4f bps,\n",
                r.avg_price, r.arrival_bps, r.vwap_bps);
    std::printf("  interval VWAP %.4f bps, close %.4f bps (positive = cost)\n",
                r.interval_vwap_bps, r.close_bps);
    const etrade::ParticipationResult part =
        etrade::participation(fill_bins, fill_qtys, bin_volumes);
    const std::vector<double> mids{100.00, 99.98, 99.95, 99.93, 99.90,
                                   99.91,  99.88, 99.86, 99.85, 99.84};
    const etrade::MarkoutResult mo =
        etrade::markouts(etrade::Side::kSell, {0, 2, 4, 6}, fill_qtys, mids, {1, 3});
    std::printf("participation %.4f%%; markout per share after 1 step %.6f $, after 3 steps "
                "%.6f $\n", 100.0 * part.overall, mo.per_share[0], mo.per_share[1]);
    const etrade::WelchResult w = etrade::welch_ttest({1.2, 3.4, 2.2, 5.1, 0.7, 2.9, 4.0, 3.3},
                                                      {0.4, 1.1, -0.2, 0.9, 1.6, 0.3});
    std::printf("algo A vs B (8 vs 6 orders): diff %.4f bps, t = %.4f, df = %.4f, p = %.6f\n",
                w.mean_diff, w.t_stat, w.df, w.p_value);
}

etrade::ACMarket make_ac(double price, double spread, double adv, double vol, double x,
                         double lam) {
    etrade::ACMarket m;
    m.price = price;
    m.spread = spread;
    m.adv = adv;
    m.daily_vol = vol;
    m.total_shares = x;
    m.n_intervals = 13;
    m.tau = 1.0 / 13.0;
    m.lambda_ra = lam;
    return m;
}

etrade::ASMarket make_as(double price, double vol, double touch, double halving, double skew,
                         int qmax, double dt) {
    etrade::ASMarket m;
    m.price = price;
    m.tick = 0.01;
    m.daily_vol = vol;
    m.touch_fills_per_day = touch;
    m.halving_ticks = halving;
    m.skew_ticks = skew;
    m.q_max = qmax;
    m.liq_ticks = 1.0;
    m.dt_seconds = dt;
    return m;
}

/// Calibration estimators and the fitted ITCH presets (API_SPEC Section 11).
void calibration_section() {
    hdr("Calibration: fill intensity from 100 market orders; fitted ITCH presets");
    std::vector<double> dist;
    dist.insert(dist.end(), 60, 0.005);
    dist.insert(dist.end(), 25, 0.015);
    dist.insert(dist.end(), 10, 0.025);
    dist.insert(dist.end(), 5, 0.035);
    const etrade::FillIntensity fi = etrade::fill_intensity(dist, 0.01, 4, 1.0, 2.0);
    std::printf("toy fit: A = %.4f/day, k = %.4f per $, touch fills %.4f/day, halving %.4f "
                "ticks\n", fi.a_rate, fi.k, fi.touch_fills_per_day, fi.halving_ticks);
    // data/presets.json, itch_*_20191230
    struct Fitted {
        const char* name;
        etrade::ACMarket ac;
        etrade::ASMarket as;
    };
    const Fitted rows[] = {
        {"itch_aapl_20191230", make_ac(289.72, 0.026401, 5678176.0, 0.01157, 283900.0, 4.66e-8),
         make_as(289.72, 0.01157, 22070.0, 0.46, 2.25, 50, 5.0)},
        {"itch_msft_20191230", make_ac(157.67, 0.012285, 3625411.0, 0.007432, 181300.0, 2.78e-7),
         make_as(157.67, 0.007432, 10507.0, 0.251, 2.25, 50, 5.0)},
        {"itch_qqq_20191230", make_ac(212.16, 0.011602, 3713405.0, 0.005745, 185700.0, 2.21e-7),
         make_as(212.16, 0.005745, 7532.0, 0.214, 1.0, 200, 1.0)},
    };
    std::printf("fitted to one NASDAQ ITCH day, 2019-12-30 (one venue; caveats in API_SPEC "
                "11.6):\n");
    std::printf("%20s %7s %8s %7s %10s %8s %9s %12s\n", "preset", "price", "spread", "vol",
                "A /day", "k /$", "A-S open", "E[IS] 5% ADV");
    for (const Fitted& f : rows) {
        const etrade::ACParams p = etrade::ac_params_from_market(f.ac);
        const etrade::ASParams a = etrade::as_params_from_market(f.as);
        const double e =
            etrade::bps(etrade::expected_cost(p), f.ac.total_shares * f.ac.price);
        const double o = etrade::bps(etrade::optimal_spread(a, 0.0), a.s0);
        std::printf("%20s %7.2f %6.3fbp %6.3f%% %10.1f %8.3f %7.4fbp %10.4fbp\n", f.name,
                    f.ac.price, etrade::bps(f.ac.spread, f.ac.price), 100.0 * f.ac.daily_vol,
                    a.a_rate, a.k, o, e);
    }
}

/// Transient impact, adaptive re-planning, internalisation (API_SPEC Section 12).
void execution_ext_section() {
    hdr("Execution extensions: transient impact, adaptive vs static, internalisation");
    const etrade::TransientParams tp(1e6, 10, 0.1, 1e-6, 5.0);
    const std::vector<double> x = etrade::transient_optimal_schedule(tp);
    const double e_opt = etrade::transient_expected_cost(tp, x);
    const double e_twap = etrade::transient_expected_cost(tp, etrade::twap_holdings(1e6, 10));
    std::printf("Obizhaeva-Wang, rho*tau = 0.5: first / middle / last trade %.2f / %.2f / "
                "%.2f shares\n", x[0] - x[1], x[4] - x[5], x[9] - x[10]);
    std::printf("  E[IS] optimal %.2f $ vs TWAP %.2f $ (saves %.4f%%)\n", e_opt, e_twap,
                100.0 * (1.0 - e_opt / e_twap));
    const etrade::ACParams p(1'000'000.0, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6);
    std::printf("adaptive re-planning, liquid set, 20k paths (each language samples its own "
                "paths):\n");
    std::printf("%24s %13s %11s %9s %6s\n", "case", "static E[IS]", "adaptive", "saving", "se");
    const struct {
        const char* label;
        etrade::SignalParams sig;
    } cases[] = {
        {"no signal, full fills", etrade::SignalParams{}},
        {"AR(1) signal 0.7, 0.5", etrade::SignalParams{0.7, 0.5}},
        {"fill risk 0.5", etrade::SignalParams{0.0, 0.0, 0.5}},
    };
    for (const auto& c : cases) {
        const etrade::AdaptiveSimResult s = etrade::simulate_adaptive(p, c.sig, 20'000, 1);
        std::printf("%24s %13.0f %11.0f %9.0f %6.0f\n", c.label, s.static_mean,
                    s.adaptive_mean, s.saving_mean, s.saving_se);
    }
    const std::vector<double> w(etrade::kUsEquityVolumeCurve13.begin(),
                                etrade::kUsEquityVolumeCurve13.end());
    const etrade::InternalisationResult r = etrade::internalise(
        {200'000.0, 50'000.0}, {150'000.0}, 100.0, 0.01, 5e6, 0.015, 13, 1.0 / 13.0, 0.5, 1.0,
        0.10, w);
    std::printf("internalise buys 200k + 50k against a 150k sell: cross %.0f, send %.0f (%s)\n",
                r.crossed_shares, r.residual_shares, r.residual_side.c_str());
    std::printf("  impact %.2f $ -> %.2f $, saves %.4f bps of gross notional\n",
                r.baseline_cost, r.internal_cost, r.saved_bps);
}

/// Market making with frictions: the friction ledger (API_SPEC Section 9).
void mm_frictions_section() {
    hdr("Market making with frictions: US large-cap A-S preset, one-share quotes");
    // data/presets.json, us_large_cap "as"
    etrade::ASMarket am = make_as(100.0, 0.015, 1170.0, 0.462, 2.25, 50, 5.0);
    const etrade::ASParams a = etrade::as_params_from_market(am);
    std::printf("A = %.4f/day, k = %.4f per $, gamma = %.6f, %d steps of 5 s\n", a.a_rate, a.k,
                a.gamma, a.n_steps());
    for (double q : {0.0, 3.0}) {
        const auto [db, da] = etrade::glft_half_spreads(a, q);
        std::printf("GLFT half spreads at q = %+.0f: bid %.8f $, ask %.8f $\n", q, db, da);
    }
    const auto [rb, ra] = etrade::round_to_tick(99.9934, 100.0066, 0.01);
    std::printf("tick rounding away from fair: 99.9934 / 100.0066 -> %.2f / %.2f; weighted mid "
                "%.3f\n", rb, ra, etrade::weighted_mid(99.99, 100.01, 900.0, 100.0));
    std::printf("friction ledger, $ per day, 5k paths (each language samples its own paths):\n");
    std::printf("%20s %7s %5s | %7s %6s %7s %8s | %6s %8s\n", "case", "PnL", "se", "spread",
                "fees", "invent", "markout", "fills", "informed");
    auto cfg = [](etrade::MMAgent agent, double tick, double fee, int lat, double inf,
                  double acc) {
        etrade::MMConfig c;
        c.agent = agent;
        c.tick = tick;
        c.maker_fee = fee;
        c.latency_steps = lat;
        c.informed_prob = inf;
        c.signal_accuracy = acc;
        return c;
    };
    using etrade::MMAgent;
    // Seeds 101-106 are the mm_sim golden cases of API_SPEC 9.7; the fee row is extra.
    const struct {
        const char* label;
        etrade::MMConfig c;
        std::uint64_t seed;
    } rows[] = {
        {"frictionless", cfg(MMAgent::kAs, 0.0, 0.0, 0, 0.0, 0.5), 101},
        {"tick + 0.2c rebate", cfg(MMAgent::kAs, 0.01, 0.002, 0, 0.0, 0.5), 102},
        {"tick + 0.2c fee", cfg(MMAgent::kAs, 0.01, -0.002, 0, 0.0, 0.5), 107},
        {"+ 15% informed flow", cfg(MMAgent::kAs, 0.01, 0.002, 0, 0.15, 0.5), 103},
        {"+ 60% signal", cfg(MMAgent::kAs, 0.01, 0.002, 0, 0.15, 0.6), 104},
        {"quotes 5 s stale", cfg(MMAgent::kAs, 0.01, 0.002, 1, 0.0, 0.5), 105},
        {"GLFT, tick + rebate", cfg(MMAgent::kGlft, 0.01, 0.002, 0, 0.0, 0.5), 106},
    };
    using etrade::MMStat;
    for (const auto& row : rows) {
        const etrade::MMResult r = etrade::simulate_mm(a, 5'000, row.seed, row.c);
        std::printf("%20s %7.2f %5.2f | %7.2f %6.2f %7.2f %8.2f | %6.1f %8.1f\n", row.label,
                    r.mean(MMStat::kPnl), r.se(MMStat::kPnl), r.mean(MMStat::kSpread),
                    r.mean(MMStat::kFees), r.mean(MMStat::kInventory),
                    r.mean(MMStat::kMarkout), r.mean(MMStat::kFills),
                    r.mean(MMStat::kInformedFills));
    }
    std::printf("PnL = spread + fees + inventory - liquidation on every path; the markout is "
                "part\nof inventory (negative = adverse selection).\n");
}

/// Queue position, passive or aggressive, liquidation (API_SPEC Section 13).
void placement_section() {
    hdr("Limit-order placement: queue fill probability, post or cross, liquidation");
    std::printf("%8s %5s %6s %6s %8s %14s\n", "q_ahead", "size", "mu", "theta", "horizon",
                "P(full fill)");
    const struct {
        int qa;
        int sz;
        double mu, th, h;
    } queues[] = {{0, 5, 10.0, 0.1, 0.5},
                  {30, 5, 10.0, 0.0, 2.0},
                  {50, 10, 20.0, 0.05, 3.0},
                  {400, 50, 100.0, 0.2, 4.0}};
    for (const auto& c : queues) {
        std::printf("%8d %5d %6.1f %6.2f %8.1f %14.10f\n", c.qa, c.sz, c.mu, c.th, c.h,
                    etrade::queue_fill_probability(c.qa, c.sz, c.mu, c.th, c.h));
    }
    const struct {
        const char* label;
        int qa;
        double theta, horizon, adverse;
    } decisions[] = {{"near the front", 5, 0.1, 1.0, 0.01}, {"deep in queue", 60, 0.05, 3.0, 0.03}};
    for (const auto& c : decisions) {
        const double pf = etrade::queue_fill_probability(c.qa, 2, 10.0, c.theta, c.horizon);
        const etrade::PlacementDecision d =
            etrade::place_or_cross(2000.0, 0.02, pf, c.adverse, 1e-4, 0.5, 1.0 / 13.0);
        std::printf("sell 2,000 %s: P(fill) %.6f, post %.6f vs cross %.6f $/sh, breakeven "
                    "%.4f: %s\n", c.label, pf, d.passive_cost, d.aggressive_cost,
                    d.breakeven_fill_prob, d.passive ? "post" : "cross");
    }
    std::printf("liquidate 10 units over 50 periods (patient) or 20 with penalty phi = 0.05 "
                "(urgent):\n");
    std::printf("%8s %9s %8s %4s %7s | %8s %8s %8s %17s\n", "case", "DP cost", "depth", "mkt",
                "all-mkt", "optimal", "all-mkt", "touch", "saving vs all-mkt");
    const struct {
        const char* label;
        etrade::LiquidationParams lp;
        std::uint64_t seed;
    } liqs[] = {
        {"patient", etrade::LiquidationParams(10, 50, 0.02, 0.5, 0.02, 30.0, 0.3, 0.005), 201},
        {"urgent", etrade::LiquidationParams(20, 50, 0.02, 0.5, 0.02, 50.0, 0.4, 0.005, 0.05),
         202},
    };
    for (const auto& c : liqs) {
        const etrade::LiquidationPolicy pol = etrade::liquidation_policy(c.lp);
        int n_mkt = 0;
        for (const auto& row : pol.market) {
            for (std::size_t q = 1; q < row.size(); ++q) {
                n_mkt += row[q] ? 1 : 0;
            }
        }
        const etrade::LiquidationSimResult s = etrade::simulate_liquidation(c.lp, 20'000, c.seed);
        std::printf("%8s %9.5f %8.5f %4d %7.3f | %8.4f %8.4f %8.4f %9.4f +- %.4f\n", c.label,
                    pol.expected_cost(),
                    pol.depth[0][static_cast<std::size_t>(c.lp.total_units)], n_mkt,
                    etrade::aggressive_cost(c.lp), s.optimal_mean, s.aggressive_mean,
                    s.passive_mean, s.saving_vs_aggressive_mean, s.saving_vs_aggressive_se);
    }
    std::printf("costs in $ per position against the starting mid (negative = sold above it); "
                "the\nDP cost adds the urgency penalty, so it exceeds the simulated shortfall "
                "when phi > 0;\ndepth = first posting depth in $; mkt = states where the policy "
                "sends a market order;\nsimulations 20k paths, each language samples its own "
                "paths.\n");
}

}  // namespace

/// Hedged market making and risk controls (API_SPEC Section 14).
void hedging_section() {
    hdr("Hedged market making and risk controls: US large-cap preset, B at rho = 0.9");
    const struct {
        double q, y, ratio, thr, tgt;
    } trades[] = {
        {7.0, 0.0, 0.9, 0.0, 0.0}, {7.0, -4.5, 0.9, 2.5, 0.0}, {-6.0, 1.8, 0.9, 2.5, 0.25}};
    for (const auto& t : trades) {
        // 0.0 + dy keeps a zero trade printing as +0.0000, not -0.0000.
        const double dy = 0.0 + etrade::hedge_trade(t.q, t.y, t.ratio, t.thr, t.tgt);
        std::printf("hedge_trade(q %+.1f, y %+.2f, ratio %.1f, band %.1f, target %.2f): "
                    "dy = %+.4f\n", t.q, t.y, t.ratio, t.thr, t.tgt, dy);
    }
    const std::vector<double> vals = {3.5, -1.25, 0.0, 7.75, -4.5, 2.0, 2.0, -0.5, 10.25, 1.0};
    std::printf("type-7 quantile of the 10 golden values: 5%% %.4f, median %.4f\n",
                etrade::quantile(vals, 0.05), etrade::quantile(vals, 0.5));
    etrade::ASMarket am = make_as(100.0, 0.015, 1170.0, 0.462, 2.25, 50, 5.0);
    const etrade::ASParams a = etrade::as_params_from_market(am);
    std::printf("$ per day, 2k paths (each language samples its own paths); q05 = 5%% quantile of "
                "PnL:\n");
    std::printf("%20s %6s %5s %7s | %6s %6s %6s %6s | %5s %5s %7s\n", "case", "PnL", "se", "q05",
                "hedge", "h.cost", "r.cost", "skew", "stops", "flat", "updates");
    using etrade::MMAgent;
    auto book = [](MMAgent agent) {
        etrade::MMConfig c;
        c.agent = agent;
        c.tick = 0.01;
        c.maker_fee = 0.002;
        c.informed_prob = 0.15;
        return c;
    };
    etrade::HedgeConfig full;
    full.rho = 0.9;
    full.hedge_ratio = 0.9;
    full.hedge_cost = 0.005;
    const etrade::HedgeConfig none;
    auto risk = [](double loss, int pos, double pen, int upd, int win, int kill) {
        etrade::RiskConfig r;
        r.loss_limit = loss;
        r.position_limit = pos;
        r.flatten_penalty = pen;
        r.throttle_updates = upd;
        r.throttle_window = win;
        r.kill_step = kill;
        return r;
    };
    const etrade::RiskConfig off;
    // Seeds are the hedge_sim golden cases of API_SPEC 14.7.
    const struct {
        const char* label;
        etrade::MMConfig c;
        etrade::HedgeConfig h;
        etrade::RiskConfig r;
        std::uint64_t seed;
    } rows[] = {
        {"A-S, unhedged", book(MMAgent::kAs), none, off, 201},
        {"A-S, full hedge", book(MMAgent::kAs), full, off, 202},
        {"fixed, unhedged", book(MMAgent::kFixed), none, off, 204},
        {"fixed, full hedge", book(MMAgent::kFixed), full, off, 205},
        {"fixed, loss limit 5", book(MMAgent::kFixed), none, risk(5.0, 0, 0.01, 0, 1, -1), 206},
        {"fixed, pos limit 10", book(MMAgent::kFixed), none, risk(0.0, 10, 0.01, 0, 1, -1), 207},
        {"A-S, 1 quote/3 steps", book(MMAgent::kAs), none, risk(0.0, 0, 0.0, 1, 3, -1), 208},
        {"fixed, kill at 3/4", book(MMAgent::kFixed), none, risk(0.0, 0, 0.01, 0, 1, 3510), 209},
    };
    using etrade::HedgeStat;
    double worst = 0.0;
    for (const auto& row : rows) {
        const etrade::HedgedResult r = etrade::simulate_hedged(a, 2'000, row.seed, row.c, row.h,
                                                               row.r);
        worst = std::max(worst, r.max_identity_error);
        std::printf("%20s %6.2f %5.2f %7.2f | %6.2f %6.2f %6.2f %6.4f | %5.2f %5.1f %7.0f\n",
                    row.label, r.mean(HedgeStat::kPnl), r.se(HedgeStat::kPnl),
                    r.quantile(HedgeStat::kPnl, 0.05), r.mean(HedgeStat::kHedgePnl),
                    r.mean(HedgeStat::kHedgeCosts), r.mean(HedgeStat::kRiskCosts),
                    r.mean(HedgeStat::kRmsSkew),
                    r.mean(HedgeStat::kLossStop) + r.mean(HedgeStat::kKillStop),
                    r.mean(HedgeStat::kPosFlattens), r.mean(HedgeStat::kQuoteUpdates));
    }
    std::printf("hedge = hedge PnL, h.cost/r.cost = hedge and risk-control costs, skew = rms skew "
                "$;\nstops = share of paths stopped (loss limit or kill switch), flat = "
                "position-limit\nflattens per path. Attribution identity holds on every path: "
                "max residual %.0e.\n", worst);
}

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
    tca_section();
    calibration_section();
    execution_ext_section();
    mm_frictions_section();
    placement_section();
    hedging_section();
    std::printf("\ndone.\n");
    return 0;
}
