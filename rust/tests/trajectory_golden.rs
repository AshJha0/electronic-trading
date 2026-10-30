//! Whole-trajectory golden cases (API_SPEC.md Section 5.1): every indexed
//! key of every case in `data/golden/trajectory_golden.json`, plus `len`.

use std::fs;
use std::path::Path;

use serde_json::Value;

use etrade::impact::{optimal_schedule, US_EQUITY_VOLUME_CURVE_13};
use etrade::market::{impact_params_from_market, ImpactMarket};
use etrade::{ac_trajectory, efficient_frontier, pov_holdings, twap_holdings, ACParams};

fn load() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/golden/trajectory_golden.json");
    let text =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&text).expect("trajectory_golden.json must be valid JSON")
}

fn f(obj: &Value, key: &str) -> f64 {
    obj.get(key)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("missing/non-numeric key '{key}'"))
}

fn check(case: &str, key: &str, actual: f64, expected: f64, tol: f64) {
    let bound = tol * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= bound,
        "{case}.{key}: actual={actual:.17e} expected={expected:.17e} bound={bound:.3e}"
    );
}

fn ac_params(inp: &Value) -> ACParams {
    ACParams::new(
        f(inp, "ac_total_shares"),
        f(inp, "ac_n_intervals") as usize,
        f(inp, "ac_tau"),
        f(inp, "ac_sigma"),
        f(inp, "ac_eta"),
        f(inp, "ac_gamma_p"),
        f(inp, "ac_lambda_ra"),
    )
    .unwrap()
}

/// Check `len` and `prefix_j` for every point; returns the number of keys.
fn check_vector(case: &str, exp: &Value, prefix: &str, actual: &[f64], tol: f64) -> usize {
    let len = f(exp, "len");
    assert_eq!(len, actual.len() as f64, "{case}.len");
    for (j, &a) in actual.iter().enumerate() {
        let key = format!("{prefix}_{j}");
        check(case, &key, a, f(exp, &key), tol);
    }
    actual.len()
}

#[test]
fn trajectory_golden_all_cases() {
    let root = load();
    assert_eq!(root["schema"].as_i64(), Some(2));
    let cases = root["cases"].as_array().expect("cases array");
    assert_eq!(cases.len(), 7);
    for c in cases {
        let name = c["name"].as_str().unwrap();
        let kind = c["kind"].as_str().unwrap();
        let inp = &c["inputs"];
        let exp = &c["expect"];
        let tol = f(c, "tol");
        // every expect key: len plus the indexed keys counted below
        let n_keys = exp.as_object().unwrap().len();
        let checked = match kind {
            "ac_trajectory" => check_vector(name, exp, "x", &ac_trajectory(&ac_params(inp)), tol),
            "twap" => {
                let x = twap_holdings(f(inp, "ac_total_shares"), f(inp, "ac_n_intervals") as usize)
                    .unwrap();
                check_vector(name, exp, "x", &x, tol)
            }
            "pov" => {
                let x = pov_holdings(
                    f(inp, "ac_total_shares"),
                    f(inp, "ac_tau"),
                    f(inp, "pov_market_volume_rate"),
                    f(inp, "pov_participation"),
                )
                .unwrap();
                check_vector(name, exp, "x", &x, tol)
            }
            "frontier" => {
                let n = f(inp, "n_lambdas") as usize;
                let grid: Vec<f64> = (0..n).map(|i| f(inp, &format!("lambda_{i}"))).collect();
                let fr = efficient_frontier(&ac_params(inp), &grid).unwrap();
                assert_eq!(f(exp, "len"), fr.len() as f64, "{name}.len");
                for (i, pt) in fr.iter().enumerate() {
                    check(
                        name,
                        &format!("kappa_{i}"),
                        pt.kappa,
                        f(exp, &format!("kappa_{i}")),
                        tol,
                    );
                    check(
                        name,
                        &format!("e_cost_{i}"),
                        pt.e_cost,
                        f(exp, &format!("e_cost_{i}")),
                        tol,
                    );
                    check(
                        name,
                        &format!("std_cost_{i}"),
                        pt.std_cost,
                        f(exp, &format!("std_cost_{i}")),
                        tol,
                    );
                }
                3 * fr.len()
            }
            "impact_schedule" => {
                let weights = if f(inp, "volume_curve") == 1.0 {
                    Some(US_EQUITY_VOLUME_CURVE_13.to_vec())
                } else {
                    None
                };
                let m = ImpactMarket {
                    price: f(inp, "price"),
                    spread: f(inp, "spread"),
                    adv: f(inp, "adv"),
                    daily_vol: f(inp, "daily_vol"),
                    total_shares: f(inp, "total_shares"),
                    n_intervals: f(inp, "n_intervals") as usize,
                    tau: f(inp, "tau"),
                    lambda_ra: f(inp, "lambda_ra"),
                    alpha: f(inp, "alpha"),
                    y: f(inp, "y"),
                    drift_bps_per_day: f(inp, "drift_bps_per_day"),
                    weights,
                    ..ImpactMarket::default()
                };
                let p = impact_params_from_market(&m).unwrap();
                let r = optimal_schedule(&p).unwrap();
                let fracs: Vec<f64> = r.holdings.iter().map(|v| v / p.total_shares()).collect();
                check_vector(name, exp, "xfrac", &fracs, tol)
            }
            other => panic!("unknown kind {other}"),
        };
        assert_eq!(checked + 1, n_keys, "{name}: unchecked expect keys");
    }
}
