//! Golden-value suite: loads `../data/golden/golden.json` (relative to the
//! rust/ directory) and asserts every case within its tolerance.
//!
//! Exact cases: `|actual - expected| <= tol * max(1, |expected|)` (relative
//! for large magnitudes, absolute near zero).  Statistical cases: this port
//! runs its own seeded RNG with `M >= 2000` paths and asserts
//! `|stat - ref| <= 4*sqrt(se_ref^2 + se_own^2)` per API_SPEC.md Section 4.

use std::fs;
use std::path::Path;
use std::str::FromStr;

use serde_json::Value;

use etrade::{ASStat, Agent, SimConfig};

fn load_golden() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/golden/golden.json");
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&text).expect("golden.json must be valid JSON")
}

/// Fetch a numeric field from a JSON object.
fn f(obj: &Value, key: &str) -> f64 {
    obj.get(key)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("missing/non-numeric key '{key}'"))
}

/// Exact-case check with the pinned relative-tolerance semantics.
fn check_exact(case: &str, key: &str, actual: f64, expected: f64, tol: f64) {
    let bound = tol * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= bound,
        "{case}.{key}: actual={actual:.15e} expected={expected:.15e} \
         |diff|={:.3e} > bound={bound:.3e}",
        (actual - expected).abs()
    );
}

/// Statistical-case check: 4-SE band combining reference and own SE.
fn check_stat(case: &str, key: &str, actual: f64, reference: f64, se_ref: f64, se_own: f64) {
    let band = 4.0 * (se_ref * se_ref + se_own * se_own).sqrt();
    assert!(
        (actual - reference).abs() <= band,
        "{case}.{key}: actual={actual} ref={reference} |diff|={} > 4SE band={band}",
        (actual - reference).abs()
    );
}

fn ac_params(inp: &Value) -> etrade::ACParams {
    etrade::ACParams::new(
        f(inp, "ac_total_shares"),
        f(inp, "ac_n_intervals") as usize,
        f(inp, "ac_tau"),
        f(inp, "ac_sigma"),
        f(inp, "ac_eta"),
        f(inp, "ac_gamma_p"),
        f(inp, "ac_lambda_ra"),
    )
    .expect("golden AC params must be valid")
}

fn as_params(inp: &Value) -> etrade::ASParams {
    etrade::ASParams::new(
        f(inp, "as_s0"),
        f(inp, "as_sigma"),
        f(inp, "as_horizon"),
        f(inp, "as_dt"),
        f(inp, "as_gamma"),
        f(inp, "as_k"),
        f(inp, "as_a_rate"),
        f(inp, "as_q_max") as u32,
        f(inp, "as_eta_liq"),
    )
    .expect("golden AS params must be valid")
}

/// Number of paths this port uses for AS statistical cases (>= 2000 required).
const AS_PATHS: usize = 4000;

#[test]
fn golden_cases_all_pass() {
    let golden = load_golden();
    let cases = golden["cases"].as_array().expect("cases array");
    assert_eq!(cases.len(), 18, "expected the pinned 18 golden cases");

    for case in cases {
        let name = case["name"].as_str().expect("case name");
        let inp = &case["inputs"];
        let exp = &case["expect"];
        let tol = f(case, "tol");

        match name {
            "ac_liquid_solution" | "ac_illiquid_solution" => {
                let p = ac_params(inp);
                let x = etrade::ac_trajectory(&p);
                let n = etrade::ac_trade_list(&p);
                let j = f(inp, "x_index") as usize;
                check_exact(name, "kappa", etrade::ac_kappa(&p), f(exp, "kappa"), tol);
                check_exact(name, "half_life", etrade::half_life(&p), f(exp, "half_life"), tol);
                check_exact(name, "x_j", x[j], f(exp, "x_j"), tol);
                check_exact(name, "n_1", n[0], f(exp, "n_1"), tol);
                check_exact(
                    name,
                    "e_cost",
                    etrade::expected_cost(&p, None).unwrap(),
                    f(exp, "e_cost"),
                    tol,
                );
                check_exact(
                    name,
                    "var_cost",
                    etrade::variance_cost(&p, None).unwrap(),
                    f(exp, "var_cost"),
                    tol,
                );
            }
            "ac_twap_liquid" => {
                let p = ac_params(inp);
                let twap = etrade::twap_holdings(p.total_shares(), p.n_intervals()).unwrap();
                check_exact(
                    name,
                    "e_cost",
                    etrade::expected_cost(&p, Some(&twap)).unwrap(),
                    f(exp, "e_cost"),
                    tol,
                );
                check_exact(
                    name,
                    "var_cost",
                    etrade::variance_cost(&p, Some(&twap)).unwrap(),
                    f(exp, "var_cost"),
                    tol,
                );
            }
            "ac_pov_liquid" => {
                let p = ac_params(inp);
                let pov = etrade::pov_holdings(
                    p.total_shares(),
                    p.tau(),
                    f(inp, "pov_market_volume_rate"),
                    f(inp, "pov_participation"),
                )
                .unwrap();
                check_exact(
                    name,
                    "e_cost",
                    etrade::expected_cost(&p, Some(&pov)).unwrap(),
                    f(exp, "e_cost"),
                    tol,
                );
                check_exact(
                    name,
                    "var_cost",
                    etrade::variance_cost(&p, Some(&pov)).unwrap(),
                    f(exp, "var_cost"),
                    tol,
                );
                check_exact(
                    name,
                    "n_intervals_used",
                    (pov.len() - 1) as f64,
                    f(exp, "n_intervals_used"),
                    tol,
                );
            }
            "ac_frontier_point_liquid" => {
                let p = ac_params(inp);
                let frontier = etrade::efficient_frontier(&p, &[f(inp, "ac_lambda_ra")]).unwrap();
                let fp = &frontier[0];
                check_exact(name, "e_cost", fp.e_cost, f(exp, "e_cost"), tol);
                check_exact(name, "std_cost", fp.std_cost, f(exp, "std_cost"), tol);
                check_exact(name, "kappa", fp.kappa, f(exp, "kappa"), tol);
            }
            "ac_lambda_zero_limit_liquid" => {
                let p = ac_params(inp);
                let x = etrade::ac_trajectory(&p);
                let twap = etrade::twap_holdings(p.total_shares(), p.n_intervals()).unwrap();
                let max_diff = x
                    .iter()
                    .zip(&twap)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                check_exact(name, "max_abs_diff_vs_twap", max_diff, 0.0, tol);
            }
            "ac_instant_liquidation_liquid" => {
                let p = ac_params(inp);
                let (e, v) = etrade::instantaneous_cost(&p);
                check_exact(name, "e_cost", e, f(exp, "e_cost"), tol);
                check_exact(name, "var_cost", v, f(exp, "var_cost"), tol);
            }
            "ac_mc_liquid" => {
                let p = ac_params(inp);
                let n_paths = f(inp, "n_paths_ref") as usize;
                let mc = etrade::simulate_is(&p, None, n_paths, f(inp, "seed") as u64).unwrap();
                check_stat(
                    name,
                    "mean_cost",
                    mc.mean,
                    f(exp, "mean_cost"),
                    f(exp, "mean_cost_se"),
                    mc.se_mean,
                );
                check_stat(
                    name,
                    "var_cost",
                    mc.var,
                    f(exp, "var_cost"),
                    f(exp, "var_cost_se"),
                    mc.se_var,
                );
            }
            "as_reservation_liquid" => {
                let p = as_params(inp);
                let r = etrade::reservation_price(&p, f(inp, "s"), f(inp, "q"), f(inp, "t"))
                    .unwrap();
                check_exact(name, "reservation_price", r, f(exp, "reservation_price"), tol);
            }
            "as_spread_liquid" => {
                let p = as_params(inp);
                check_exact(
                    name,
                    "spread",
                    etrade::optimal_spread(&p, f(inp, "t")).unwrap(),
                    f(exp, "spread"),
                    tol,
                );
                check_exact(
                    name,
                    "spread_terminal",
                    etrade::optimal_spread(&p, p.horizon()).unwrap(),
                    f(exp, "spread_terminal"),
                    tol,
                );
            }
            "as_fill_prob_liquid" => {
                let p = as_params(inp);
                let delta = f(inp, "delta");
                check_exact(
                    name,
                    "intensity",
                    etrade::intensity(&p, delta).unwrap(),
                    f(exp, "intensity"),
                    tol,
                );
                check_exact(
                    name,
                    "fill_probability",
                    etrade::fill_probability(&p, delta).unwrap(),
                    f(exp, "fill_probability"),
                    tol,
                );
            }
            "as_sim_liquid" | "as_sim_fx" => {
                let p = as_params(inp);
                let res =
                    etrade::simulate(&p, AS_PATHS, f(inp, "seed") as u64, &SimConfig::default())
                        .unwrap();
                assert!(!res.cap_violated, "{name}: inventory cap violated");
                let stats: &[(&str, f64, ASStat)] = &[
                    ("mean_pnl", res.mean_pnl(), ASStat::MeanPnl),
                    ("std_pnl", res.std_pnl(), ASStat::StdPnl),
                    ("mean_abs_q", res.mean_abs_q(), ASStat::MeanAbsQ),
                    ("mean_max_abs_q", res.mean_max_abs_q(), ASStat::MeanMaxAbsQ),
                    ("mean_fills", res.mean_fills(), ASStat::MeanFills),
                ];
                for (key, actual, stat) in stats {
                    if exp.get(*key).is_some() {
                        check_stat(
                            name,
                            key,
                            *actual,
                            f(exp, key),
                            f(exp, &format!("{key}_se")),
                            res.se(*stat),
                        );
                    }
                }
            }
            "as_fixed_comparison_liquid" => {
                let p = as_params(inp);
                let seed = f(inp, "seed") as u64;
                let res_as =
                    etrade::simulate(&p, AS_PATHS, seed, &SimConfig::default()).unwrap();
                let cfg_fixed = SimConfig {
                    agent: Agent::Fixed,
                    fixed_spread: Some(f(inp, "fixed_spread")),
                    ..SimConfig::default()
                };
                let res_fx = etrade::simulate(&p, AS_PATHS, seed, &cfg_fixed).unwrap();
                check_stat(
                    name,
                    "as_rms_q",
                    res_as.rms_q(),
                    f(exp, "as_rms_q"),
                    f(exp, "as_rms_q_se"),
                    res_as.se(ASStat::RmsQ),
                );
                check_stat(
                    name,
                    "fixed_rms_q",
                    res_fx.rms_q(),
                    f(exp, "fixed_rms_q"),
                    f(exp, "fixed_rms_q_se"),
                    res_fx.se(ASStat::RmsQ),
                );
            }
            "is_decomposition_sell" | "is_decomposition_buy" => {
                let side = if name.ends_with("sell") { "sell" } else { "buy" };
                let d = etrade::decompose_shortfall(
                    etrade::Side::from_str(side).unwrap(),
                    f(inp, "decision_price"),
                    f(inp, "arrival_price"),
                    f(inp, "close_price"),
                    f(inp, "total_qty"),
                    &[f(inp, "q1"), f(inp, "q2"), f(inp, "q3")],
                    &[f(inp, "p1"), f(inp, "p2"), f(inp, "p3")],
                )
                .unwrap();
                check_exact(name, "total", d.total, f(exp, "total"), tol);
                check_exact(name, "delay", d.delay, f(exp, "delay"), tol);
                check_exact(name, "trading", d.trading, f(exp, "trading"), tol);
                check_exact(name, "opportunity", d.opportunity, f(exp, "opportunity"), tol);
            }
            "participation_rate" => {
                let part = etrade::participation_rate(
                    f(inp, "executed_volume"),
                    f(inp, "market_volume"),
                )
                .unwrap();
                check_exact(name, "participation", part, f(exp, "participation"), tol);
            }
            "impact_regression" => {
                let (part, cost) = etrade::synthetic_impact_fills(
                    f(inp, "n_obs") as usize,
                    f(inp, "true_intercept"),
                    f(inp, "true_slope"),
                    f(inp, "noise_amp"),
                )
                .unwrap();
                let reg = etrade::estimate_impact(&part, &cost).unwrap();
                check_exact(name, "slope", reg.slope, f(exp, "slope"), tol);
                check_exact(name, "intercept", reg.intercept, f(exp, "intercept"), tol);
                check_exact(name, "r_squared", reg.r_squared, f(exp, "r_squared"), tol);
            }
            other => panic!("unhandled golden case '{other}' — port must cover every case"),
        }
    }
}
