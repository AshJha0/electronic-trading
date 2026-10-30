//! Limit-order placement (API_SPEC.md Section 13): every case of
//! `data/golden/placement_golden.json` (one test per case), the queue fill
//! probability against the Poisson tail and its monotonicity, the
//! passive-or-aggressive rule, the liquidation policy against its Monte
//! Carlo, and validation.

use std::fs;
use std::path::Path;

use serde_json::Value;

use etrade::impact::{impact_expected_cost, ImpactParams};
use etrade::placement::{
    aggressive_cost, aggressive_schedule, liquidation_policy, place_or_cross,
    queue_fill_probability, simulate_liquidation, LiquidationParams,
};
use etrade::EtradeError;

fn load() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/golden/placement_golden.json");
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).expect("valid JSON")
}

fn num(o: &Value, k: &str) -> f64 {
    o.get(k)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("missing key '{k}'"))
}

fn liq_from(inp: &Value) -> LiquidationParams {
    LiquidationParams::new(
        num(inp, "total_units") as usize,
        num(inp, "n_periods") as usize,
        num(inp, "dt"),
        num(inp, "sigma"),
        num(inp, "spread"),
        num(inp, "kappa"),
        num(inp, "p_touch"),
        num(inp, "beta"),
        num(inp, "phi"),
    )
    .unwrap()
}

fn patient(sigma: f64, q: usize, n: usize) -> LiquidationParams {
    LiquidationParams::new(q, n, 0.02, sigma, 0.02, 30.0, 0.3, 0.005, 0.0).unwrap()
}

/// `(actual, own SE)` per key; SE is empty for exact kinds.
type Named = Vec<(String, f64)>;

fn actual(case: &Value) -> (Named, Named) {
    let kind = case["kind"].as_str().unwrap();
    let inp = &case["inputs"];
    let mut out = Vec::new();
    if kind == "fill_probability" || kind == "placement_decision" {
        let fp = queue_fill_probability(
            num(inp, "q_ahead") as usize,
            num(inp, "size") as usize,
            num(inp, "mu"),
            num(inp, "theta"),
            num(inp, "horizon"),
        )
        .unwrap();
        out.push(("fill_prob".to_string(), fp));
        if kind == "placement_decision" {
            let d = place_or_cross(
                num(inp, "child_shares"),
                num(inp, "spread"),
                fp,
                num(inp, "adverse_move"),
                num(inp, "eta"),
                num(inp, "alpha"),
                num(inp, "weight"),
            )
            .unwrap();
            out.push(("passive_cost".to_string(), d.passive_cost));
            out.push(("aggressive_cost".to_string(), d.aggressive_cost));
            out.push(("impact_cost".to_string(), d.impact_cost));
            out.push(("breakeven_fill_prob".to_string(), d.breakeven_fill_prob));
            out.push(("passive".to_string(), if d.passive { 1.0 } else { 0.0 }));
        }
        return (out, Vec::new());
    }
    let p = liq_from(inp);
    if kind == "liquidation_policy" {
        let pol = liquidation_policy(&p);
        let n_market: usize = pol
            .market()
            .iter()
            .map(|row| row.iter().skip(1).filter(|&&m| m).count())
            .sum();
        out.push(("dp_cost".to_string(), pol.expected_cost()));
        out.push(("depth_first".to_string(), pol.depth()[0][p.total_units()]));
        out.push(("depth_last".to_string(), pol.depth()[p.n_periods() - 1][1]));
        out.push(("n_market".to_string(), n_market as f64));
        out.push(("aggressive_cost".to_string(), aggressive_cost(&p)));
        return (out, Vec::new());
    }
    let r = simulate_liquidation(
        &p,
        num(inp, "n_paths_ref") as usize,
        num(inp, "seed") as u64,
    )
    .unwrap();
    out.push(("optimal_mean".to_string(), r.optimal_mean));
    out.push(("aggressive_mean".to_string(), r.aggressive_mean));
    out.push(("passive_mean".to_string(), r.passive_mean));
    out.push((
        "saving_vs_aggressive_mean".to_string(),
        r.saving_vs_aggressive_mean,
    ));
    out.push((
        "saving_vs_passive_mean".to_string(),
        r.saving_vs_passive_mean,
    ));
    let se = vec![
        ("optimal_mean".to_string(), r.optimal_se),
        ("aggressive_mean".to_string(), r.aggressive_se),
        ("passive_mean".to_string(), r.passive_se),
        (
            "saving_vs_aggressive_mean".to_string(),
            r.saving_vs_aggressive_se,
        ),
        ("saving_vs_passive_mean".to_string(), r.saving_vs_passive_se),
    ];
    (out, se)
}

fn lookup(v: &[(String, f64)], k: &str) -> f64 {
    v.iter()
        .find(|(n, _)| n == k)
        .unwrap_or_else(|| panic!("no actual for '{k}'"))
        .1
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
    let (got, se) = actual(case);
    if case["kind"] == "liquidation_sim" {
        let keys: Vec<&String> = expect.keys().filter(|k| !k.ends_with("_se")).collect();
        assert_eq!(keys.len(), got.len(), "{name}: every expect key");
        for key in keys {
            let reference = num(&case["expect"], key);
            let se_ref = num(&case["expect"], &format!("{key}_se"));
            let se_own = lookup(&se, key);
            let value = lookup(&got, key);
            let band = tol * (se_ref * se_ref + se_own * se_own).sqrt();
            assert!(
                (value - reference).abs() <= band,
                "{name}.{key}: got {value}, ref {reference}, band {band}"
            );
        }
        return;
    }
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
    golden_fill_front_of_queue => "fill_front_of_queue",
    golden_fill_no_cancellations => "fill_no_cancellations",
    golden_fill_mid_queue_with_cancellations => "fill_mid_queue_with_cancellations",
    golden_fill_deep_queue_long_horizon => "fill_deep_queue_long_horizon",
    golden_decision_passive_near_front => "decision_passive_near_front",
    golden_decision_aggressive_deep_queue => "decision_aggressive_deep_queue",
    golden_decision_aggressive_zero_spread => "decision_aggressive_zero_spread",
    golden_liquidation_policy_patient => "liquidation_policy_patient",
    golden_liquidation_policy_urgent => "liquidation_policy_urgent",
    golden_liquidation_sim_patient => "liquidation_sim_patient",
    golden_liquidation_sim_urgent => "liquidation_sim_urgent",
}

#[test]
fn golden_count() {
    assert_eq!(load()["cases"].as_array().unwrap().len(), 11);
}

// ------------------------------------------------------------ fill probability

/// `P(Poisson(lam) >= n)` by direct summation of the lower tail (small n).
fn poisson_sf(n: usize, lam: f64) -> f64 {
    let mut term = (-lam).exp();
    let mut cdf = 0.0;
    for k in 0..n {
        cdf += term;
        term *= lam / (k + 1) as f64;
    }
    1.0 - cdf
}

#[test]
fn no_cancellations_is_the_poisson_tail() {
    let v = queue_fill_probability(30, 5, 10.0, 0.0, 2.0).unwrap();
    assert!((v / poisson_sf(35, 20.0) - 1.0).abs() <= 1e-10);
    let one = queue_fill_probability(0, 1, 3.0, 0.0, 0.4).unwrap();
    assert!((one - (1.0 - (-1.2f64).exp())).abs() <= 1e-14);
}

#[test]
fn fill_probability_decreases_with_queue_position_and_size() {
    let probs: Vec<f64> = (0..120)
        .step_by(5)
        .map(|q| queue_fill_probability(q, 10, 20.0, 0.05, 3.0).unwrap())
        .collect();
    assert!(probs.windows(2).all(|w| w[1] < w[0]));
    let sizes: Vec<f64> = (1..40)
        .step_by(3)
        .map(|n| queue_fill_probability(20, n, 20.0, 0.05, 3.0).unwrap())
        .collect();
    assert!(sizes.windows(2).all(|w| w[1] < w[0]));
}

#[test]
fn fill_probability_increases_with_horizon_and_cancellations() {
    let probs: Vec<f64> = (0..30)
        .map(|i| {
            let h = 1.0 + 3.0 * i as f64 / 29.0;
            queue_fill_probability(50, 10, 20.0, 0.05, h).unwrap()
        })
        .collect();
    assert!(probs.windows(2).all(|w| w[1] > w[0]));
    let thetas: Vec<f64> = [0.0, 0.05, 0.2, 1.0]
        .iter()
        .map(|&t| queue_fill_probability(50, 10, 20.0, t, 2.0).unwrap())
        .collect();
    assert!(thetas.windows(2).all(|w| w[1] > w[0]));
}

#[test]
fn fill_probability_edges() {
    assert_eq!(queue_fill_probability(10, 3, 5.0, 0.2, 0.0).unwrap(), 0.0);
    assert_eq!(queue_fill_probability(0, 1, 1e3, 0.0, 100.0).unwrap(), 1.0);
    assert!(queue_fill_probability(4999, 1, 1.0, 0.0, 1.0).unwrap() <= 1e-300);
}

// ------------------------------------------------------------ passive or aggressive

#[test]
fn decision_matches_section_8_and_closed_form_difference() {
    let d = place_or_cross(2000.0, 0.02, 0.7, 0.01, 1e-4, 0.5, 1.0 / 13.0).unwrap();
    let ip = ImpactParams::new(2000.0, 1, 1.0 / 13.0, 0.0, 1e-4, 0.5, 0.0, 0.0, 0.0, None).unwrap();
    let e = impact_expected_cost(&ip, &[2000.0, 0.0]).unwrap();
    assert!((d.impact_cost / (e / 2000.0) - 1.0).abs() <= 1e-12);
    let diff = 0.3 * 0.01 - 2.0 * 0.7 * 0.01;
    assert!((d.passive_cost - d.aggressive_cost - diff).abs() <= 1e-15);
    assert!(d.passive);
}

#[test]
fn aggressive_chosen_when_spread_is_negligible() {
    for fp in [0.0, 0.2, 0.5, 0.9, 1.0] {
        for adv in [0.0, 0.004, 0.05] {
            assert!(
                !place_or_cross(500.0, 0.0, fp, adv, 1e-4, 0.5, 0.1)
                    .unwrap()
                    .passive
            );
        }
    }
}

#[test]
fn breakeven_fill_probability_separates_the_choices() {
    let (adv, spread) = (0.015, 0.01);
    for i in 0..=40 {
        let fp = i as f64 / 40.0;
        let d = place_or_cross(100.0, spread, fp, adv, 1e-4, 0.6, 0.2).unwrap();
        assert!((d.breakeven_fill_prob - adv / (adv + spread)).abs() <= 1e-15);
        if (fp - d.breakeven_fill_prob).abs() > 1e-9 {
            assert_eq!(d.passive, fp > d.breakeven_fill_prob, "{fp}");
        }
    }
    let zero = place_or_cross(100.0, 0.0, 0.5, 0.0, 1e-4, 0.6, 0.2).unwrap();
    assert_eq!(zero.breakeven_fill_prob, 1.0);
}

// ------------------------------------------------------------ liquidation

#[test]
fn policy_closed_form_one_period() {
    let pol = liquidation_policy(&patient(0.5, 1, 1));
    let g = 0.01 + 0.015;
    let d = 1.0 / 30.0 - g;
    assert!((pol.depth()[0][1] - d).abs() <= 1e-15);
    let want = 0.015 - 0.3 * (-30.0 * d).exp() / 30.0;
    assert!((pol.expected_cost() - want).abs() <= 1e-15);
    assert!(!pol.market()[0][1]);
}

#[test]
fn depth_falls_with_inventory() {
    let pol = liquidation_policy(&patient(0.5, 10, 50));
    for (drow, mrow) in pol.depth().iter().zip(pol.market()) {
        let posted: Vec<f64> = (1..drow.len())
            .filter(|&q| !mrow[q])
            .map(|q| drow[q])
            .collect();
        assert!(posted.windows(2).all(|w| w[1] <= w[0] + 1e-15));
    }
}

#[test]
fn monte_carlo_matches_dp_and_beats_baselines_without_penalty() {
    let p = patient(0.5, 10, 50);
    let r = simulate_liquidation(&p, 20_000, 5).unwrap();
    assert!((r.optimal_mean - liquidation_policy(&p).expected_cost()).abs() <= 4.0 * r.optimal_se);
    assert!((r.aggressive_mean - aggressive_cost(&p)).abs() <= 4.0 * r.aggressive_se);
    assert!(r.saving_vs_aggressive_mean > 4.0 * r.saving_vs_aggressive_se);
    assert!(r.saving_vs_passive_mean > 4.0 * r.saving_vs_passive_se);
}

#[test]
fn deterministic_without_volatility() {
    let p = patient(0.0, 4, 4);
    let r = simulate_liquidation(&p, 50, 1).unwrap();
    assert!(r.aggressive_std <= 1e-15);
    assert!((r.aggressive_mean - aggressive_cost(&p)).abs() <= 1e-15);
}

#[test]
fn simulation_is_deterministic() {
    let a = simulate_liquidation(&patient(0.5, 10, 50), 200, 9).unwrap();
    let b = simulate_liquidation(&patient(0.5, 10, 50), 200, 9).unwrap();
    assert_eq!(a, b);
}

#[test]
fn aggressive_schedule_splits_evenly() {
    assert_eq!(aggressive_schedule(10, 4).unwrap(), vec![2, 3, 2, 3]);
    assert_eq!(aggressive_schedule(3, 5).unwrap(), vec![0, 1, 0, 1, 1]);
    assert_eq!(
        aggressive_schedule(997, 50).unwrap().iter().sum::<usize>(),
        997
    );
}

// ------------------------------------------------------------ validation

fn err_msg<T: std::fmt::Debug>(r: Result<T, EtradeError>) -> String {
    match r {
        Err(EtradeError::InvalidInput(m)) => m,
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn fill_probability_validation() {
    type Args = (usize, usize, f64, f64, f64);
    let cases: [(Args, &str); 11] = [
        ((0, 0, 1.0, 0.0, 1.0), "size"),
        ((4999, 2, 1.0, 0.0, 1.0), "q_ahead + size"),
        ((0, 5001, 1.0, 0.0, 1.0), "q_ahead + size"),
        ((0, 1, 0.0, 0.0, 1.0), "mu"),
        ((0, 1, f64::INFINITY, 0.0, 1.0), "mu"),
        ((0, 1, 1.0, -0.1, 1.0), "theta"),
        ((0, 1, 1.0, f64::NAN, 1.0), "theta"),
        ((0, 1, 1.0, 0.0, -1.0), "horizon"),
        ((0, 1, 1.0, 0.0, f64::INFINITY), "horizon"),
        ((0, 1, 1e6, 0.0, 1.0), "horizon too long"),
        ((0, 1, 1e300, 0.0, 1e300), "horizon too long"),
    ];
    for ((q, n, mu, th, h), field) in cases {
        let m = err_msg(queue_fill_probability(q, n, mu, th, h));
        assert!(m.contains(field), "{field}: {m}");
    }
}

#[test]
fn decision_validation() {
    let cases: [(usize, f64, &str); 10] = [
        (0, 0.0, "size"),
        (0, f64::INFINITY, "size"),
        (1, -0.01, "spread"),
        (2, 1.1, "fill_prob"),
        (2, f64::NAN, "fill_prob"),
        (3, -1.0, "adverse_move"),
        (4, -1.0, "eta"),
        (5, 0.0, "alpha"),
        (5, 1.5, "alpha"),
        (6, 0.0, "weight"),
    ];
    for (i, bad, field) in cases {
        let mut a = [100.0, 0.01, 0.5, 0.01, 1e-4, 0.5, 0.1];
        a[i] = bad;
        let m = err_msg(place_or_cross(a[0], a[1], a[2], a[3], a[4], a[5], a[6]));
        assert!(m.contains(field), "{field}: {m}");
    }
}

#[test]
fn liquidation_params_validation() {
    let base = [10.0, 50.0, 0.02, 0.5, 0.02, 30.0, 0.3, 0.005, 0.0];
    let cases: [(usize, f64, &str); 13] = [
        (0, 0.0, "total_units"),
        (0, 1001.0, "total_units"),
        (1, 0.0, "n_periods"),
        (1, 1001.0, "n_periods"),
        (2, 0.0, "dt"),
        (3, -1.0, "sigma"),
        (4, f64::NAN, "spread"),
        (5, 0.0, "kappa"),
        (6, 0.0, "p_touch"),
        (6, 1.01, "p_touch"),
        (7, -1.0, "beta"),
        (8, f64::INFINITY, "phi"),
        (8, -1.0, "phi"),
    ];
    for (i, bad, field) in cases {
        let mut v = base;
        v[i] = bad;
        let r = LiquidationParams::new(
            v[0] as usize,
            v[1] as usize,
            v[2],
            v[3],
            v[4],
            v[5],
            v[6],
            v[7],
            v[8],
        );
        let m = err_msg(r);
        assert!(m.contains(field), "{field}: {m}");
    }
    let p = patient(0.5, 10, 50);
    assert!(err_msg(simulate_liquidation(&p, 1, 0)).contains("n_paths"));
    assert!(err_msg(aggressive_schedule(0, 5)).contains("total_units"));
    assert!(err_msg(aggressive_schedule(5, 0)).contains("n_periods"));
}
