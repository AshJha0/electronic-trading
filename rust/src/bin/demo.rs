//! End-to-end demo of the `etrade` crate.
//!
//! Prints: Almgren-Chriss optimal trajectories across risk aversions, the
//! efficient frontier of execution, TWAP/POV/instantaneous baselines with a
//! Monte Carlo cross-check, Avellaneda-Stoikov market-making simulations
//! (A-S vs fixed-spread agent, liquid and FX parameter sets), and an
//! implementation-shortfall decomposition with a market-impact regression.
//!
//! Run: `cd rust && cargo run --bin demo`

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
    obj[key].as_f64().unwrap_or_else(|| panic!("missing key '{key}'"))
}

fn hdr(title: &str) {
    println!("\n=== {title} {}", "=".repeat(66_usize.saturating_sub(title.len())));
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
    println!("{:>10} {:>9} {:>10} | holdings x_j (thousands)", "lambda_ra", "kappa", "half-life");
    for &lam in &[0.0, 5e-7, 2e-6, 1e-5] {
        let pl = p.with_lambda(lam)?;
        let x = etrade::ac_trajectory(&pl);
        let hl = etrade::half_life(&pl);
        let hl_s = if hl.is_finite() { format!("{hl:10.3}") } else { format!("{:>10}", "inf") };
        let xs: Vec<String> = x.iter().map(|v| format!("{:8.1}", v / 1e3)).collect();
        println!("{lam:10.1e} {:9.4} {hl_s} | {}", etrade::ac_kappa(&pl), xs.join(" "));
    }

    hdr("Efficient frontier of execution (E[cost] vs sqrt(Var[cost]))");
    println!("{:>10} {:>14} {:>14} {:>9}", "lambda_ra", "E[cost] $", "sqrt(Var) $", "kappa");
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
        ("AC optimal", etrade::expected_cost(&p, None)?, etrade::variance_cost(&p, None)?),
        ("TWAP", etrade::expected_cost(&p, Some(&twap))?, etrade::variance_cost(&p, Some(&twap))?),
        ("POV 15%", etrade::expected_cost(&p, Some(&pov))?, etrade::variance_cost(&p, Some(&pov))?),
        ("Instant", e_i, v_i),
    ];
    println!("{:>12} {:>14} {:>14}", "strategy", "E[cost] $", "sqrt(Var) $");
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
        ap.s0(), ap.sigma(), ap.gamma(), ap.k(), ap.a_rate(), ap.horizon(), ap.dt(), ap.q_max()
    );
    println!(
        "example quote at t=0, q=+3: bid={bid:.4} ask={ask:.4} (mid {}, reservation {:.4})",
        ap.s0(),
        etrade::reservation_price(&ap, ap.s0(), 3.0, 0.0)?
    );
    let n_paths = 2000;
    let res_as = etrade::simulate(&ap, n_paths, seed, &SimConfig::default())?;
    let cfg_fixed = SimConfig { agent: Agent::Fixed, ..SimConfig::default() };
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
    let stress_cfg = SimConfig { stress_factor: 2.0, ..SimConfig::default() };
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
    let qtys: Vec<f64> = order["is_fill_qtys"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let prices: Vec<f64> = order["is_fill_prices"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
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
    println!("participation rate example: 120k / 1.5m = {:.2}%", part * 100.0);

    hdr("Market-impact regression (cost/share vs participation, native OLS)");
    let (parts, costs) = etrade::synthetic_impact_fills(25, 0.5, 8.0, 0.05)?;
    let reg = etrade::estimate_impact(&parts, &costs)?;
    println!(
        "fitted: cost = {:.4} + {:.4} * participation (true 0.50 + 8.00), R^2 = {:.4}, n = {}",
        reg.intercept, reg.slope, reg.r_squared, reg.n_obs
    );
    Ok(())
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
    println!("\ndone.");
    Ok(())
}
