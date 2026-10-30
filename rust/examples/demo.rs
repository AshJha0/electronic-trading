//! End-to-end demo of the `etrade` crate.
//!
//! Prints: Almgren-Chriss optimal trajectories across risk aversions, the
//! efficient frontier of execution, TWAP/POV/instantaneous baselines with a
//! Monte Carlo cross-check, Avellaneda-Stoikov market-making simulations
//! (A-S vs fixed-spread agent, liquid and FX parameter sets), and an
//! implementation-shortfall decomposition with a market-impact regression,
//! presets in bps, a pre-trade report, post-trade TCA, calibration with the
//! fitted ITCH presets, execution extensions, the market-making friction
//! ledger and limit-order placement.
//!
//! Run: `cd rust && cargo run --release --example demo`

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use serde_json::Value;

use etrade::{ASStat, Agent, SimConfig};

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../data")
}

fn load(name: &str) -> Result<Value, Box<dyn Error>> {
    let path = data_dir().join(name);
    Ok(serde_json::from_str(&fs::read_to_string(&path)?)?)
}

fn f(obj: &Value, key: &str) -> f64 {
    obj[key]
        .as_f64()
        .unwrap_or_else(|| panic!("missing key '{key}'"))
}

fn hdr(title: &str) {
    println!(
        "\n=== {title} {}",
        "=".repeat(66_usize.saturating_sub(title.len()))
    );
}

fn ac_section(cfg: &Value, order: &Value) -> Result<(), Box<dyn Error>> {
    let p = etrade::ACParams::new(
        f(cfg, "ac_total_shares"),
        f(cfg, "ac_n_intervals") as usize,
        f(cfg, "ac_tau"),
        f(cfg, "ac_sigma"),
        f(cfg, "ac_eta"),
        f(cfg, "ac_gamma_p"),
        f(cfg, "ac_lambda_ra"),
    )?;

    hdr("Almgren-Chriss optimal trajectories (sell 1.0e6 shares, N=5)");
    println!(
        "{:>10} {:>9} {:>10} | holdings x_j (thousands)",
        "lambda_ra", "kappa", "half-life"
    );
    for &lam in &[0.0, 5e-7, 2e-6, 1e-5] {
        let pl = p.with_lambda(lam)?;
        let x = etrade::ac_trajectory(&pl);
        let hl = etrade::half_life(&pl);
        let hl_s = if hl.is_finite() {
            format!("{hl:10.3}")
        } else {
            format!("{:>10}", "inf")
        };
        let xs: Vec<String> = x.iter().map(|v| format!("{:8.1}", v / 1e3)).collect();
        println!(
            "{lam:10.1e} {:9.4} {hl_s} | {}",
            etrade::ac_kappa(&pl),
            xs.join(" ")
        );
    }

    hdr("Efficient frontier of execution (E[cost] vs sqrt(Var[cost]))");
    println!(
        "{:>10} {:>14} {:>14} {:>9}",
        "lambda_ra", "E[cost] $", "sqrt(Var) $", "kappa"
    );
    let grid: Vec<f64> = order["lambda_grid"]
        .as_array()
        .expect("lambda_grid")
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    for fp in etrade::efficient_frontier(&p, &grid)? {
        println!(
            "{:10.1e} {:14.0} {:14.0} {:9.4}",
            fp.lambda_ra, fp.e_cost, fp.std_cost, fp.kappa
        );
    }

    hdr("Baselines: optimal vs TWAP vs POV vs instantaneous");
    let twap = etrade::twap_holdings(p.total_shares(), p.n_intervals())?;
    let pov = etrade::pov_holdings(
        p.total_shares(),
        p.tau(),
        f(cfg, "pov_market_volume_rate"),
        f(cfg, "pov_participation"),
    )?;
    let (e_i, v_i) = etrade::instantaneous_cost(&p);
    let rows = [
        (
            "AC optimal",
            etrade::expected_cost(&p, None)?,
            etrade::variance_cost(&p, None)?,
        ),
        (
            "TWAP",
            etrade::expected_cost(&p, Some(&twap))?,
            etrade::variance_cost(&p, Some(&twap))?,
        ),
        (
            "POV 15%",
            etrade::expected_cost(&p, Some(&pov))?,
            etrade::variance_cost(&p, Some(&pov))?,
        ),
        ("Instant", e_i, v_i),
    ];
    println!(
        "{:>12} {:>14} {:>14}",
        "strategy", "E[cost] $", "sqrt(Var) $"
    );
    for (name, e, v) in rows {
        println!("{name:>12} {e:14.0} {:14.0}", v.sqrt());
    }
    let mc = etrade::simulate_is(&p, None, 20_000, 123)?;
    println!(
        "MC check (20k paths): E={:.0} (closed {:.0}, se {:.0}); Var/1e9={:.1} (closed {:.1})",
        mc.mean,
        etrade::expected_cost(&p, None)?,
        mc.se_mean,
        mc.var / 1e9,
        etrade::variance_cost(&p, None)? / 1e9
    );
    Ok(())
}

fn as_section(cfg: &Value, label: &str, seed: u64) -> Result<(), Box<dyn Error>> {
    let ap = etrade::ASParams::new(
        f(cfg, "as_s0"),
        f(cfg, "as_sigma"),
        f(cfg, "as_horizon"),
        f(cfg, "as_dt"),
        f(cfg, "as_gamma"),
        f(cfg, "as_k"),
        f(cfg, "as_a_rate"),
        f(cfg, "as_q_max") as u32,
        f(cfg, "as_eta_liq"),
    )?;
    hdr(&format!("Avellaneda-Stoikov market making ({label})"));
    let (bid, ask) = etrade::quotes(&ap, ap.s0(), 3.0, 0.0)?;
    println!(
        "params: s0={} sigma={} gamma={} k={} A={} T={} dt={} q_max={}",
        ap.s0(),
        ap.sigma(),
        ap.gamma(),
        ap.k(),
        ap.a_rate(),
        ap.horizon(),
        ap.dt(),
        ap.q_max()
    );
    println!(
        "example quote at t=0, q=+3: bid={bid:.4} ask={ask:.4} (mid {}, reservation {:.4})",
        ap.s0(),
        etrade::reservation_price(&ap, ap.s0(), 3.0, 0.0)?
    );
    let n_paths = 2000;
    let res_as = etrade::simulate(&ap, n_paths, seed, &SimConfig::default())?;
    let cfg_fixed = SimConfig {
        agent: Agent::Fixed,
        ..SimConfig::default()
    };
    let res_fx = etrade::simulate(&ap, n_paths, seed, &cfg_fixed)?;
    println!(
        "{:>14} {:>10} {:>9} {:>7} {:>8} {:>11} {:>7} {:>7}",
        "agent", "mean PnL", "std PnL", "Sharpe", "mean|q|", "mean max|q|", "rms q", "fills"
    );
    for (name, r) in [("A-S", &res_as), ("fixed-spread", &res_fx)] {
        println!(
            "{name:>14} {:10.4} {:9.4} {:7.2} {:8.3} {:11.3} {:7.3} {:7.1}",
            r.mean_pnl(),
            r.std_pnl(),
            r.sharpe(),
            r.mean_abs_q(),
            r.mean_max_abs_q(),
            r.rms_q(),
            r.mean_fills()
        );
    }
    println!(
        "inventory control: A-S rms q = {:.3} (se {:.3}) vs fixed {:.3} (similar PnL, much tighter inventory)",
        res_as.rms_q(),
        res_as.se(ASStat::RmsQ),
        res_fx.rms_q()
    );
    let stress_cfg = SimConfig {
        stress_factor: 2.0,
        ..SimConfig::default()
    };
    let stress = etrade::simulate(&ap, n_paths, seed, &stress_cfg)?;
    println!(
        "stress (vol x2 after T/2): std PnL {:.4} -> {:.4}",
        res_as.std_pnl(),
        stress.std_pnl()
    );
    Ok(())
}

fn analytics_section(order: &Value) -> Result<(), Box<dyn Error>> {
    hdr("Implementation-shortfall decomposition (sell parent order)");
    let qtys: Vec<f64> = order["is_fill_qtys"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let prices: Vec<f64> = order["is_fill_prices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let d = etrade::decompose_shortfall(
        etrade::Side::from_str(order["side"].as_str().unwrap())?,
        f(order, "decision_price"),
        f(order, "arrival_price"),
        f(order, "close_price"),
        f(order, "is_total_qty"),
        &qtys,
        &prices,
    )?;
    println!(
        "decision {:.2}, arrival {:.2}, close {:.2}; executed {:.0}/{:.0}",
        f(order, "decision_price"),
        f(order, "arrival_price"),
        f(order, "close_price"),
        d.executed_qty,
        f(order, "is_total_qty")
    );
    for (name, v) in [
        ("delay", d.delay),
        ("trading", d.trading),
        ("opportunity", d.opportunity),
        ("TOTAL", d.total),
    ] {
        println!("  {name:>12}: {v:10.2} $");
    }
    let part = etrade::participation_rate(120_000.0, 1.5e6)?;
    println!(
        "participation rate example: 120k / 1.5m = {:.2}%",
        part * 100.0
    );

    hdr("Market-impact regression (cost/share vs participation, native OLS)");
    let (parts, costs) = etrade::synthetic_impact_fills(25, 0.5, 8.0, 0.05)?;
    let reg = etrade::estimate_impact(&parts, &costs)?;
    println!(
        "fitted: cost = {:.4} + {:.4} * participation (true 0.50 + 8.00), R^2 = {:.4}, n = {}",
        reg.intercept, reg.slope, reg.r_squared, reg.n_obs
    );
    Ok(())
}

/// Presets in market units, results in bps (API_SPEC Section 7).
fn market_section() -> Result<(), Box<dyn Error>> {
    use etrade::market::{
        ac_params_from_market, as_params_from_market, bps, half_spread_cost, ACMarket, ASMarket,
    };
    hdr("Market presets in bps (execution: sell order; market making: A-S spread)");
    let presets = load("presets.json")?;
    println!(
        "{:>26} {:>10} {:>8} {:>8} {:>9} {:>22}",
        "preset", "order %ADV", "E[IS]", "sd", "half-sprd", "A-S spread open/close"
    );
    for (name, body) in presets.as_object().ok_or("presets must be an object")? {
        let ac = &body["ac"];
        let m = ACMarket {
            price: f(ac, "price"),
            spread: f(ac, "spread"),
            adv: f(ac, "adv"),
            daily_vol: f(ac, "daily_vol"),
            total_shares: f(ac, "total_shares"),
            n_intervals: f(ac, "n_intervals") as usize,
            tau: f(ac, "tau"),
            lambda_ra: f(ac, "lambda_ra"),
            ..ACMarket::default()
        };
        let p = ac_params_from_market(&m)?;
        let notional = m.total_shares * m.price;
        let e = bps(etrade::expected_cost(&p, None)?, notional)?;
        let sd = bps(etrade::variance_cost(&p, None)?.sqrt(), notional)?;
        let hs = bps(half_spread_cost(m.spread, m.total_shares)?, notional)?;
        let mut mm = String::from("-");
        if let Some(asb) = body.get("as") {
            let am = ASMarket {
                price: f(asb, "price"),
                tick: f(asb, "tick"),
                daily_vol: f(asb, "daily_vol"),
                touch_fills_per_day: f(asb, "touch_fills_per_day"),
                halving_ticks: f(asb, "halving_ticks"),
                skew_ticks: f(asb, "skew_ticks"),
                q_max: f(asb, "q_max") as u32,
                liq_ticks: f(asb, "liq_ticks"),
                dt_seconds: f(asb, "dt_seconds"),
                session_seconds: f(asb, "session_seconds"),
                horizon_days: f(asb, "horizon_days"),
            };
            let a = as_params_from_market(&am)?;
            mm = format!(
                "{:8.2} / {:6.2} bps",
                bps(etrade::optimal_spread(&a, 0.0)?, a.s0())?,
                bps(etrade::optimal_spread(&a, a.horizon())?, a.s0())?
            );
        }
        println!(
            "{name:>26} {:9.1}% {e:6.2}bp {sd:6.2}bp {hs:7.2}bp {mm:>22}",
            m.total_shares / m.adv * 100.0
        );
    }
    Ok(())
}

/// Pre-trade report on the power-law model (API_SPEC Section 8).
fn pretrade_section() -> Result<(), Box<dyn Error>> {
    use etrade::impact::{
        impact_expected_cost, optimal_schedule, vwap_holdings, US_EQUITY_VOLUME_CURVE_13,
    };
    use etrade::market::{bps, impact_params_from_market, pretrade_report, ImpactMarket};
    hdr("Pre-trade report: US large cap, sell 5% ADV in a day (sqrt impact, U-curve)");
    let m = ImpactMarket {
        price: 100.0,
        spread: 0.01,
        adv: 5e6,
        daily_vol: 0.015,
        total_shares: 250_000.0,
        n_intervals: 13,
        tau: 1.0 / 13.0,
        weights: Some(US_EQUITY_VOLUME_CURVE_13.to_vec()),
        ..ImpactMarket::default()
    };
    let rep = pretrade_report(&m, &[0.0, 1e-7, 1e-6, 1e-5])?;
    println!(
        "square-root-law estimate {:.2} bps; half spread {:.2} bps",
        rep.sqrt_law_bps, rep.half_spread_bps
    );
    println!(
        "{:>10} {:>9} {:>9} {:>9} {:>9} {:>8}",
        "strategy", "lambda", "impact", "sd", "total", "horizon"
    );
    for r in rep.rows() {
        println!(
            "{:>10} {:>9} {:7.2}bp {:7.2}bp {:7.2}bp {:8}",
            r.label,
            sci(r.lambda_ra),
            r.impact_bps,
            r.sd_bps,
            r.total_bps,
            r.horizon
        );
    }
    let p = impact_params_from_market(&ImpactMarket {
        drift_bps_per_day: 50.0,
        ..m.clone()
    })?;
    let notional = m.total_shares * m.price;
    let vwap = vwap_holdings(m.total_shares, &US_EQUITY_VOLUME_CURVE_13)?;
    let s = optimal_schedule(&p)?;
    let e_v = bps(impact_expected_cost(&p, &vwap)?, notional)?;
    let e_o = bps(impact_expected_cost(&p, s.holdings())?, notional)?;
    println!(
        "alpha-aware: with 50 bps/day adverse drift, VWAP costs {e_v:.2} bps and the \
         drift-aware schedule {e_o:.2} bps (saves {:.2} bps)",
        e_v - e_o
    );
    Ok(())
}

/// Post-trade TCA (API_SPEC Section 10); deterministic.
fn tca_section() -> Result<(), Box<dyn Error>> {
    use etrade::tca::{markouts, participation, slippage, welch_ttest};
    use etrade::Side;
    hdr("Post-trade TCA: benchmarks, markouts, participation, A/B test");
    let fill_bins = [0, 1, 1, 2];
    let fill_qtys = [20_000.0, 15_000.0, 10_000.0, 25_000.0];
    let fill_prices = [99.96, 99.91, 99.89, 99.84];
    let bin_volumes = [400_000.0, 300_000.0, 300_000.0, 500_000.0];
    let bin_vwaps = [99.95, 99.90, 99.85, 99.82];
    let r = slippage(
        Side::Sell,
        100.0,
        99.80,
        &fill_bins,
        &fill_qtys,
        &fill_prices,
        &bin_volumes,
        &bin_vwaps,
    )?;
    println!(
        "sell 70,000 at avg {:.4}: arrival {:.4} bps, full-day VWAP {:.4} bps,",
        r.avg_price, r.arrival_bps, r.vwap_bps
    );
    println!(
        "  interval VWAP {:.4} bps, close {:.4} bps (positive = cost)",
        r.interval_vwap_bps, r.close_bps
    );
    let part = participation(&fill_bins, &fill_qtys, &bin_volumes)?;
    let mids = [
        100.00, 99.98, 99.95, 99.93, 99.90, 99.91, 99.88, 99.86, 99.85, 99.84,
    ];
    let mo = markouts(Side::Sell, &[0, 2, 4, 6], &fill_qtys, &mids, &[1, 3])?;
    println!(
        "participation {:.4}%; markout per share after 1 step {:.6} $, after 3 steps {:.6} $",
        100.0 * part.overall,
        mo.per_share[0],
        mo.per_share[1]
    );
    let w = welch_ttest(
        &[1.2, 3.4, 2.2, 5.1, 0.7, 2.9, 4.0, 3.3],
        &[0.4, 1.1, -0.2, 0.9, 1.6, 0.3],
    )?;
    println!(
        "algo A vs B (8 vs 6 orders): diff {:.4} bps, t = {:.4}, df = {:.4}, p = {:.6}",
        w.mean_diff, w.t_stat, w.df, w.p_value
    );
    Ok(())
}

fn ac_market(b: &Value) -> etrade::market::ACMarket {
    etrade::market::ACMarket {
        price: f(b, "price"),
        spread: f(b, "spread"),
        adv: f(b, "adv"),
        daily_vol: f(b, "daily_vol"),
        total_shares: f(b, "total_shares"),
        n_intervals: f(b, "n_intervals") as usize,
        tau: f(b, "tau"),
        lambda_ra: f(b, "lambda_ra"),
        ..etrade::market::ACMarket::default()
    }
}

fn as_market(b: &Value) -> etrade::market::ASMarket {
    etrade::market::ASMarket {
        price: f(b, "price"),
        tick: f(b, "tick"),
        daily_vol: f(b, "daily_vol"),
        touch_fills_per_day: f(b, "touch_fills_per_day"),
        halving_ticks: f(b, "halving_ticks"),
        skew_ticks: f(b, "skew_ticks"),
        q_max: f(b, "q_max") as u32,
        liq_ticks: f(b, "liq_ticks"),
        dt_seconds: f(b, "dt_seconds"),
        session_seconds: f(b, "session_seconds"),
        horizon_days: f(b, "horizon_days"),
    }
}

/// Calibration estimators and the fitted ITCH presets (API_SPEC Section 11).
fn calibration_section() -> Result<(), Box<dyn Error>> {
    use etrade::calibration::fill_intensity;
    use etrade::market::{ac_params_from_market, as_params_from_market, bps};
    hdr("Calibration: fill intensity from 100 market orders; fitted ITCH presets");
    let mut dist = vec![0.005; 60];
    dist.extend([0.015; 25]);
    dist.extend([0.025; 10]);
    dist.extend([0.035; 5]);
    let fi = fill_intensity(&dist, 0.01, 4, 1.0, 2.0)?;
    println!(
        "toy fit: A = {:.4}/day, k = {:.4} per $, touch fills {:.4}/day, halving {:.4} ticks",
        fi.a_rate, fi.k, fi.touch_fills_per_day, fi.halving_ticks
    );
    let presets = load("presets.json")?;
    println!("fitted to one NASDAQ ITCH day, 2019-12-30 (one venue; caveats in API_SPEC 11.6):");
    println!(
        "{:>20} {:>7} {:>8} {:>7} {:>10} {:>8} {:>9} {:>12}",
        "preset", "price", "spread", "vol", "A /day", "k /$", "A-S open", "E[IS] 5% ADV"
    );
    for name in [
        "itch_aapl_20191230",
        "itch_msft_20191230",
        "itch_qqq_20191230",
    ] {
        let m = ac_market(&presets[name]["ac"]);
        let p = ac_params_from_market(&m)?;
        let a = as_params_from_market(&as_market(&presets[name]["as"]))?;
        let e = bps(etrade::expected_cost(&p, None)?, m.total_shares * m.price)?;
        let o = bps(etrade::optimal_spread(&a, 0.0)?, a.s0())?;
        println!(
            "{name:>20} {:7.2} {:6.3}bp {:6.3}% {:10.1} {:8.3} {o:7.4}bp {e:10.4}bp",
            m.price,
            bps(m.spread, m.price)?,
            100.0 * m.daily_vol,
            a.a_rate(),
            a.k()
        );
    }
    Ok(())
}

/// Transient impact, adaptive re-planning, internalisation (API_SPEC Section 12).
fn execution_ext_section() -> Result<(), Box<dyn Error>> {
    use etrade::execution::{
        internalise, simulate_adaptive, transient_expected_cost, transient_optimal_schedule,
        SignalParams, TransientParams,
    };
    use etrade::impact::US_EQUITY_VOLUME_CURVE_13;
    use etrade::market::ImpactMarket;
    hdr("Execution extensions: transient impact, adaptive vs static, internalisation");
    let tp = TransientParams::new(1e6, 10, 0.1, 1e-6, 5.0, 0.0)?;
    let x = transient_optimal_schedule(&tp);
    let e_opt = transient_expected_cost(&tp, &x)?;
    let e_twap = transient_expected_cost(&tp, &etrade::twap_holdings(1e6, 10)?)?;
    println!(
        "Obizhaeva-Wang, rho*tau = 0.5: first / middle / last trade {:.2} / {:.2} / {:.2} shares",
        x[0] - x[1],
        x[4] - x[5],
        x[9] - x[10]
    );
    println!(
        "  E[IS] optimal {e_opt:.2} $ vs TWAP {e_twap:.2} $ (saves {:.4}%)",
        100.0 * (1.0 - e_opt / e_twap)
    );
    let p = etrade::ACParams::new(1_000_000.0, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6)?;
    println!("adaptive re-planning, liquid set, 20k paths (each language samples its own paths):");
    println!(
        "{:>24} {:>13} {:>11} {:>9} {:>6}",
        "case", "static E[IS]", "adaptive", "saving", "se"
    );
    for (label, sig) in [
        ("no signal, full fills", SignalParams::new(0.0, 0.0, 0.0)?),
        ("AR(1) signal 0.7, 0.5", SignalParams::new(0.7, 0.5, 0.0)?),
        ("fill risk 0.5", SignalParams::new(0.0, 0.0, 0.5)?),
    ] {
        let s = simulate_adaptive(&p, &sig, 20_000, 1)?;
        println!(
            "{label:>24} {:13.0} {:11.0} {:9.0} {:6.0}",
            s.static_mean, s.adaptive_mean, s.saving_mean, s.saving_se
        );
    }
    let m = ImpactMarket {
        price: 100.0,
        spread: 0.01,
        adv: 5e6,
        daily_vol: 0.015,
        n_intervals: 13,
        tau: 1.0 / 13.0,
        weights: Some(US_EQUITY_VOLUME_CURVE_13.to_vec()),
        ..ImpactMarket::default()
    };
    let r = internalise(&[200_000.0, 50_000.0], &[150_000.0], &m)?;
    println!(
        "internalise buys 200k + 50k against a 150k sell: cross {:.0}, send {:.0} ({})",
        r.crossed_shares, r.residual_shares, r.residual_side
    );
    println!(
        "  impact {:.2} $ -> {:.2} $, saves {:.4} bps of gross notional",
        r.baseline_cost, r.internal_cost, r.saved_bps
    );
    Ok(())
}

/// Market making with frictions: the friction ledger (API_SPEC Section 9).
fn mm_frictions_section() -> Result<(), Box<dyn Error>> {
    use etrade::market::as_params_from_market;
    use etrade::mm::{
        glft_half_spreads, round_to_tick, simulate_mm, weighted_mid, MMAgent, MMConfig, MMStat,
    };
    hdr("Market making with frictions: US large-cap A-S preset, one-share quotes");
    let a = as_params_from_market(&as_market(&load("presets.json")?["us_large_cap"]["as"]))?;
    println!(
        "A = {:.4}/day, k = {:.4} per $, gamma = {:.6}, {} steps of 5 s",
        a.a_rate(),
        a.k(),
        a.gamma(),
        a.n_steps()
    );
    for q in [0.0, 3.0] {
        let (db, da) = glft_half_spreads(&a, q)?;
        println!("GLFT half spreads at q = {q:+.0}: bid {db:.8} $, ask {da:.8} $");
    }
    let (rb, ra) = round_to_tick(99.9934, 100.0066, 0.01)?;
    println!(
        "tick rounding away from fair: 99.9934 / 100.0066 -> {rb:.2} / {ra:.2}; weighted mid {:.3}",
        weighted_mid(99.99, 100.01, 900.0, 100.0)?
    );
    println!("friction ledger, $ per day, 5k paths (each language samples its own paths):");
    println!(
        "{:>20} {:>7} {:>5} | {:>7} {:>6} {:>7} {:>8} | {:>6} {:>8}",
        "case", "PnL", "se", "spread", "fees", "invent", "markout", "fills", "informed"
    );
    let cfg = |agent, tick, maker_fee, latency_steps, informed_prob, signal_accuracy| MMConfig {
        agent,
        tick,
        maker_fee,
        latency_steps,
        informed_prob,
        signal_accuracy,
        fixed_spread: None,
    };
    // Seeds 101-106 are the mm_sim golden cases of API_SPEC 9.7; the fee row is extra.
    let rows = [
        ("frictionless", cfg(MMAgent::As, 0.0, 0.0, 0, 0.0, 0.5), 101),
        (
            "tick + 0.2c rebate",
            cfg(MMAgent::As, 0.01, 0.002, 0, 0.0, 0.5),
            102,
        ),
        (
            "tick + 0.2c fee",
            cfg(MMAgent::As, 0.01, -0.002, 0, 0.0, 0.5),
            107,
        ),
        (
            "+ 15% informed flow",
            cfg(MMAgent::As, 0.01, 0.002, 0, 0.15, 0.5),
            103,
        ),
        (
            "+ 60% signal",
            cfg(MMAgent::As, 0.01, 0.002, 0, 0.15, 0.6),
            104,
        ),
        (
            "quotes 5 s stale",
            cfg(MMAgent::As, 0.01, 0.002, 1, 0.0, 0.5),
            105,
        ),
        (
            "GLFT, tick + rebate",
            cfg(MMAgent::Glft, 0.01, 0.002, 0, 0.0, 0.5),
            106,
        ),
    ];
    for (label, c, seed) in rows {
        let r = simulate_mm(&a, 5_000, seed, &c)?;
        println!(
            "{label:>20} {:7.2} {:5.2} | {:7.2} {:6.2} {:7.2} {:8.2} | {:6.1} {:8.1}",
            r.mean(MMStat::Pnl),
            r.se(MMStat::Pnl),
            r.mean(MMStat::Spread),
            r.mean(MMStat::Fees),
            r.mean(MMStat::Inventory),
            r.mean(MMStat::Markout),
            r.mean(MMStat::Fills),
            r.mean(MMStat::InformedFills)
        );
    }
    println!("PnL = spread + fees + inventory - liquidation on every path; the markout is part");
    println!("of inventory (negative = adverse selection).");
    Ok(())
}

/// Queue position, passive or aggressive, liquidation (API_SPEC Section 13).
fn placement_section() -> Result<(), Box<dyn Error>> {
    use etrade::placement::{
        aggressive_cost, liquidation_policy, place_or_cross, queue_fill_probability,
        simulate_liquidation, LiquidationParams,
    };
    hdr("Limit-order placement: queue fill probability, post or cross, liquidation");
    println!(
        "{:>8} {:>5} {:>6} {:>6} {:>8} {:>14}",
        "q_ahead", "size", "mu", "theta", "horizon", "P(full fill)"
    );
    for (qa, sz, mu, th, h) in [
        (0, 5, 10.0, 0.1, 0.5),
        (30, 5, 10.0, 0.0, 2.0),
        (50, 10, 20.0, 0.05, 3.0),
        (400, 50, 100.0, 0.2, 4.0),
    ] {
        let pf = queue_fill_probability(qa, sz, mu, th, h)?;
        println!("{qa:8} {sz:5} {mu:6.1} {th:6.2} {h:8.1} {pf:14.10}");
    }
    for (label, qa, theta, horizon, adverse) in [
        ("near the front", 5, 0.1, 1.0, 0.01),
        ("deep in queue", 60, 0.05, 3.0, 0.03),
    ] {
        let pf = queue_fill_probability(qa, 2, 10.0, theta, horizon)?;
        let d = place_or_cross(2000.0, 0.02, pf, adverse, 1e-4, 0.5, 1.0 / 13.0)?;
        println!(
            "sell 2,000 {label}: P(fill) {pf:.6}, post {:.6} vs cross {:.6} $/sh, breakeven {:.4}: {}",
            d.passive_cost,
            d.aggressive_cost,
            d.breakeven_fill_prob,
            if d.passive { "post" } else { "cross" }
        );
    }
    println!(
        "liquidate 10 units over 50 periods (patient) or 20 with penalty phi = 0.05 (urgent):"
    );
    println!(
        "{:>8} {:>9} {:>8} {:>4} {:>7} | {:>8} {:>8} {:>8} {:>17}",
        "case",
        "DP cost",
        "depth",
        "mkt",
        "all-mkt",
        "optimal",
        "all-mkt",
        "touch",
        "saving vs all-mkt"
    );
    for (label, lp, seed) in [
        (
            "patient",
            LiquidationParams::new(10, 50, 0.02, 0.5, 0.02, 30.0, 0.3, 0.005, 0.0)?,
            201,
        ),
        (
            "urgent",
            LiquidationParams::new(20, 50, 0.02, 0.5, 0.02, 50.0, 0.4, 0.005, 0.05)?,
            202,
        ),
    ] {
        let pol = liquidation_policy(&lp);
        let n_mkt: usize = pol
            .market
            .iter()
            .map(|row| row.iter().skip(1).filter(|&&m| m).count())
            .sum();
        let s = simulate_liquidation(&lp, 20_000, seed)?;
        println!(
            "{label:>8} {:9.5} {:8.5} {n_mkt:4} {:7.3} | {:8.4} {:8.4} {:8.4} {:9.4} +- {:.4}",
            pol.expected_cost(),
            pol.depth[0][lp.total_units()],
            aggressive_cost(&lp),
            s.optimal_mean,
            s.aggressive_mean,
            s.passive_mean,
            s.saving_vs_aggressive_mean,
            s.saving_vs_aggressive_se
        );
    }
    println!("costs in $ per position against the starting mid (negative = sold above it); the");
    println!(
        "DP cost adds the urgency penalty, so it exceeds the simulated shortfall when phi > 0;"
    );
    println!(
        "depth = first posting depth in $; mkt = states where the policy sends a market order;"
    );
    println!("simulations 20k paths, each language samples its own paths.");
    Ok(())
}

/// C-style `%.1e` formatting (Rust's `{:e}` omits the exponent sign and padding).
fn sci(v: f64) -> String {
    if v == 0.0 {
        return "0.0e+00".to_string();
    }
    let exp = v.abs().log10().floor() as i32;
    let mant = v / 10f64.powi(exp);
    let sign = if exp < 0 { '-' } else { '+' };
    format!("{mant:.1}e{sign}{:02}", exp.abs())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("etrade demo — Almgren-Chriss execution & Avellaneda-Stoikov market making (Rust)");
    let liq = load("params_liquid.json")?;
    let fx = load("params_fx.json")?;
    let order = load("parent_order.json")?;
    ac_section(&liq, &order)?;
    as_section(&liq, "liquid large-cap", 42)?;
    as_section(&fx, "FX-like: tight spread, high k", 7)?;
    analytics_section(&order)?;
    market_section()?;
    pretrade_section()?;
    tca_section()?;
    calibration_section()?;
    execution_ext_section()?;
    mm_frictions_section()?;
    placement_section()?;
    println!("\ndone.");
    Ok(())
}
