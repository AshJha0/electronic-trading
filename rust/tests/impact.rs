//! Power-law impact execution (API_SPEC.md Section 8): every case of
//! `data/golden/impact_golden.json` (one test per case), reduction to
//! Almgren-Chriss, the square-root law, drift and early finish, VWAP, the
//! pre-trade report and validation.

use std::fs;
use std::path::Path;

use serde_json::Value;

use etrade::impact::{
    impact_expected_cost, impact_variance_cost, optimal_schedule, sqrt_law_cost_bps, vwap_holdings,
    ImpactParams, US_EQUITY_VOLUME_CURVE_13,
};
use etrade::market::{bps, impact_params_from_market, pretrade_report, ImpactMarket};

fn load() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/golden/impact_golden.json");
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).expect("valid JSON")
}

fn num(o: &Value, k: &str) -> f64 {
    o.get(k)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("missing key '{k}'"))
}

fn num_or(o: &Value, k: &str, d: f64) -> f64 {
    o.get(k).and_then(Value::as_f64).unwrap_or(d)
}

fn market_from(v: &Value) -> ImpactMarket {
    ImpactMarket {
        price: num(v, "price"),
        spread: num(v, "spread"),
        adv: num(v, "adv"),
        daily_vol: num(v, "daily_vol"),
        total_shares: num(v, "total_shares"),
        n_intervals: num(v, "n_intervals") as usize,
        tau: num(v, "tau"),
        lambda_ra: num_or(v, "lambda_ra", 0.0),
        alpha: num_or(v, "alpha", 0.5),
        y: num_or(v, "y", 1.0),
        perm_frac: num_or(v, "perm_frac", 0.10),
        drift_bps_per_day: num_or(v, "drift_bps_per_day", 0.0),
        weights: if num(v, "volume_curve") == 1.0 {
            Some(US_EQUITY_VOLUME_CURVE_13.to_vec())
        } else {
            None
        },
    }
}

fn actual(case: &Value) -> Vec<(String, f64)> {
    let kind = case["kind"].as_str().unwrap();
    let inp = &case["inputs"];
    if kind == "sqrt_law" {
        let b = sqrt_law_cost_bps(
            num(inp, "daily_vol"),
            num(inp, "order_shares"),
            num(inp, "adv"),
            num(inp, "y"),
        )
        .unwrap();
        return vec![("bps".to_string(), b)];
    }
    let m = market_from(inp);
    let mut out = Vec::new();
    if kind == "pretrade" {
        let nl = num(inp, "n_lambdas") as usize;
        let lambdas: Vec<f64> = (0..nl).map(|i| num(inp, &format!("lambda_{i}"))).collect();
        let rep = pretrade_report(&m, &lambdas).unwrap();
        out.push(("sqrt_law_bps".to_string(), rep.sqrt_law_bps));
        out.push(("half_spread_bps".to_string(), rep.half_spread_bps));
        for (i, r) in rep.rows().iter().enumerate() {
            out.push((format!("row{i}_impact_bps"), r.impact_bps));
            out.push((format!("row{i}_sd_bps"), r.sd_bps));
            out.push((format!("row{i}_horizon"), r.horizon as f64));
        }
        return out;
    }
    let p = impact_params_from_market(&m).unwrap();
    if kind == "impact_vwap" {
        let x = vwap_holdings(p.total_shares(), p.weights()).unwrap();
        out.push(("e_cost".to_string(), impact_expected_cost(&p, &x).unwrap()));
        out.push((
            "var_cost".to_string(),
            impact_variance_cost(&p, &x).unwrap(),
        ));
        return out;
    }
    let r = optimal_schedule(&p).unwrap();
    for j in 1..p.n_intervals() {
        out.push((format!("xfrac_{j:02}"), r.holdings()[j] / p.total_shares()));
    }
    out.push((
        "e_cost".to_string(),
        impact_expected_cost(&p, r.holdings()).unwrap(),
    ));
    out.push((
        "var_cost".to_string(),
        impact_variance_cost(&p, r.holdings()).unwrap(),
    ));
    out.push(("horizon".to_string(), r.horizon as f64));
    out
}

fn check_case(name: &str) {
    let golden = load();
    let case = golden["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == name)
        .unwrap_or_else(|| panic!("case {name} missing"));
    let expect = case["expect"].as_object().unwrap();
    let tol = num(case, "tol");
    let got = actual(case);
    assert_eq!(
        got.len(),
        expect.len(),
        "{name}: every expect key must be asserted"
    );
    for (key, value) in got {
        let want = num(&case["expect"], &key);
        assert!(
            (value - want).abs() <= tol * want.abs().max(1.0),
            "{name}.{key}: got {value}, want {want}"
        );
    }
}

macro_rules! golden_case {
    ($($test:ident => $name:literal),* $(,)?) => {
        $(#[test] fn $test() { check_case($name); })*
    };
}

golden_case! {
    golden_alpha1_paper => "impact_alpha1_paper",
    golden_sqrt_large_cap => "impact_sqrt_large_cap",
    golden_sqrt_large_cap_curve => "impact_sqrt_large_cap_curve",
    golden_sqrt_drift_early_finish => "impact_sqrt_drift_early_finish",
    golden_alpha06_drift_curve => "impact_alpha06_drift_curve",
    golden_vwap_curve => "impact_vwap_curve",
    golden_pretrade_large_cap_curve => "pretrade_large_cap_curve",
    golden_sqrt_law_small_cap => "sqrt_law_small_cap",
}

#[test]
fn golden_count() {
    assert_eq!(load()["cases"].as_array().unwrap().len(), 8);
}

fn large_cap() -> ImpactMarket {
    ImpactMarket {
        price: 100.0,
        spread: 0.01,
        adv: 5e6,
        daily_vol: 0.015,
        total_shares: 250_000.0,
        n_intervals: 13,
        tau: 1.0 / 13.0,
        ..ImpactMarket::default()
    }
}

#[test]
fn alpha_one_reproduces_almgren_chriss() {
    let ac = etrade::ACParams::new(1e6, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6).unwrap();
    let ip = ImpactParams::new(1e6, 5, 1.0, 0.95, 2.5e-6, 1.0, 2.5e-7, 2e-6, 0.0, None).unwrap();
    let r = optimal_schedule(&ip).unwrap();
    let x = etrade::ac_trajectory(&ac);
    for (a, b) in r.holdings().iter().zip(&x) {
        assert!((a - b).abs() <= 1e-10 * 1e6);
    }
    let e_ac = etrade::expected_cost(&ac, None).unwrap();
    assert!((impact_expected_cost(&ip, r.holdings()).unwrap() / e_ac - 1.0).abs() < 1e-12);
    assert!(r.iterations <= 3);
}

#[test]
fn one_day_twap_is_the_square_root_law() {
    let m = large_cap();
    let p = impact_params_from_market(&m).unwrap();
    let x = etrade::twap_holdings(m.total_shares, m.n_intervals).unwrap();
    let perm = 0.5 * p.gamma_p() * m.total_shares * m.total_shares * (1.0 - 1.0 / 13.0);
    let law = bps(
        impact_expected_cost(&p, &x).unwrap() - perm,
        m.total_shares * m.price,
    )
    .unwrap();
    let want = sqrt_law_cost_bps(0.015, 250_000.0, 5e6, 1.0).unwrap();
    assert!((law / want - 1.0).abs() < 1e-12);
}

#[test]
fn drift_shortens_the_horizon() {
    let mut prev = usize::MAX;
    for drift in [0.0, 50.0, 200.0, 1000.0, 5000.0] {
        let m = ImpactMarket {
            drift_bps_per_day: drift,
            ..large_cap()
        };
        let r = optimal_schedule(&impact_params_from_market(&m).unwrap()).unwrap();
        assert!(r.horizon <= prev);
        prev = r.horizon;
        assert!(r.holdings()[r.horizon..].iter().all(|v| *v == 0.0));
    }
    assert!(prev < 13);
}

#[test]
fn vwap_beats_twap_under_the_curve() {
    let xu = vwap_holdings(1e6, &[1.0; 10]).unwrap();
    let xt = etrade::twap_holdings(1e6, 10).unwrap();
    assert!(xu.iter().zip(&xt).all(|(a, b)| (a - b).abs() < 1e-6));
    let m = ImpactMarket {
        weights: Some(US_EQUITY_VOLUME_CURVE_13.to_vec()),
        ..large_cap()
    };
    let p = impact_params_from_market(&m).unwrap();
    let v = impact_expected_cost(
        &p,
        &vwap_holdings(m.total_shares, &US_EQUITY_VOLUME_CURVE_13).unwrap(),
    )
    .unwrap();
    let t = impact_expected_cost(&p, &etrade::twap_holdings(m.total_shares, 13).unwrap()).unwrap();
    assert!(v < t);
}

#[test]
fn pretrade_report_structure() {
    let rep = pretrade_report(&large_cap(), &[0.0, 1e-6, 1e-5]).unwrap();
    let labels: Vec<&str> = rep.rows().iter().map(|r| r.label).collect();
    assert_eq!(labels, ["twap", "vwap", "optimal", "optimal", "optimal"]);
    for r in rep.rows() {
        assert_eq!(r.total_bps, r.impact_bps + rep.half_spread_bps);
    }
    assert!(
        rep.rows()[2].sd_bps > rep.rows()[3].sd_bps && rep.rows()[3].sd_bps > rep.rows()[4].sd_bps
    );
    assert!(pretrade_report(&large_cap(), &[]).is_err());
}

#[test]
fn validation() {
    let base = |alpha: f64, eta: f64, drift: f64, n: usize, w: Option<Vec<f64>>| {
        ImpactParams::new(1e5, n, 1.0 / 13.0, 1.5, eta, alpha, 2e-8, 0.0, drift, w)
    };
    assert!(base(0.0, 1e-3, 0.0, 13, None).is_err());
    assert!(base(1.5, 1e-3, 0.0, 13, None).is_err());
    assert!(base(0.5, 0.0, 0.0, 13, None).is_err());
    assert!(base(0.5, 1e-3, -1.0, 13, None).is_err());
    assert!(base(0.5, 1e-3, 0.0, 10_001, None).is_err());
    assert!(base(0.5, 1e-3, 0.0, 13, Some(vec![0.1; 12])).is_err());
    let nonconvex = ImpactParams::new(1e8, 13, 1.0 / 13.0, 1.5, 1e-6, 0.5, 1e-3, 0.0, 0.0, None);
    assert!(matches!(
        nonconvex,
        Err(etrade::EtradeError::ModelInvalid(_))
    ));
    let p = impact_params_from_market(&large_cap()).unwrap();
    assert!(impact_expected_cost(&p, &[1.0, 0.0]).is_err());
    let mut up = etrade::twap_holdings(250_000.0, 13).unwrap();
    up[3] = up[1] + 1.0;
    assert!(impact_expected_cost(&p, &up).is_err());
    assert!(sqrt_law_cost_bps(0.0, 1.0, 1.0, 1.0).is_err());
    assert!(vwap_holdings(1.0, &[]).is_err());
    let neg = ImpactMarket {
        drift_bps_per_day: -1.0,
        ..large_cap()
    };
    assert!(impact_params_from_market(&neg).is_err());
}

#[test]
fn zero_shares_is_the_zero_schedule() {
    let m = ImpactMarket {
        total_shares: 0.0,
        lambda_ra: 1e-6,
        ..large_cap()
    };
    let r = optimal_schedule(&impact_params_from_market(&m).unwrap()).unwrap();
    assert!(r.holdings().iter().all(|v| *v == 0.0));
    assert_eq!(r.iterations, 0);
}
