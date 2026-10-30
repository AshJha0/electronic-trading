//! Execution extensions (API_SPEC.md Section 12): every case of
//! `data/golden/execution_golden.json` (one test per case), transient impact
//! and its reduction to the linear case, the Obizhaeva-Wang schedule shape,
//! adaptive re-planning, internalisation and validation.

use std::fs;
use std::path::Path;

use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha12Rng;
use serde_json::Value;

use etrade::execution::{
    internalise, replan, replan_coefficients, simulate_adaptive, transient_expected_cost,
    transient_optimal_schedule, SignalParams, TransientParams, MAX_EXEC_INTERVALS,
};
use etrade::impact::{impact_expected_cost, ImpactParams, US_EQUITY_VOLUME_CURVE_13};
use etrade::market::ImpactMarket;
use etrade::{ac_trajectory, expected_cost, twap_holdings, ACParams, EtradeError};

fn load() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/golden/execution_golden.json");
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

fn market() -> ImpactMarket {
    ImpactMarket {
        price: 100.0,
        spread: 0.01,
        adv: 5e6,
        daily_vol: 0.015,
        n_intervals: 13,
        tau: 1.0 / 13.0,
        ..ImpactMarket::default()
    }
}

/// `(actual, own SE)` per key; SE is empty for exact kinds.
type Named = Vec<(String, f64)>;

fn actual(case: &Value) -> (Named, Named) {
    let kind = case["kind"].as_str().unwrap();
    let inp = &case["inputs"];
    let mut out = Vec::new();
    if kind == "transient_schedule" {
        let p = TransientParams::new(
            num(inp, "total_shares"),
            num(inp, "n_intervals") as usize,
            num(inp, "tau"),
            num(inp, "kappa"),
            num(inp, "rho"),
            num(inp, "gamma_p"),
        )
        .unwrap();
        let x = transient_optimal_schedule(&p);
        for (j, xj) in x.iter().enumerate().take(p.n_intervals()).skip(1) {
            out.push((format!("xfrac_{j:02}"), xj / p.total_shares()));
        }
        out.push((
            "e_cost".to_string(),
            transient_expected_cost(&p, &x).unwrap(),
        ));
        let tw = twap_holdings(p.total_shares(), p.n_intervals()).unwrap();
        out.push((
            "twap_e_cost".to_string(),
            transient_expected_cost(&p, &tw).unwrap(),
        ));
        return (out, Vec::new());
    }
    if kind == "internalise" {
        let buys: Vec<f64> = (0..num(inp, "n_buys") as usize)
            .map(|i| num(inp, &format!("buy_{i}")))
            .collect();
        let sells: Vec<f64> = (0..num(inp, "n_sells") as usize)
            .map(|i| num(inp, &format!("sell_{i}")))
            .collect();
        let m = ImpactMarket {
            price: num(inp, "price"),
            spread: num(inp, "spread"),
            adv: num(inp, "adv"),
            daily_vol: num(inp, "daily_vol"),
            n_intervals: num(inp, "n_intervals") as usize,
            tau: num(inp, "tau"),
            alpha: num_or(inp, "alpha", 0.5),
            y: num_or(inp, "y", 1.0),
            perm_frac: num_or(inp, "perm_frac", 0.10),
            weights: if num(inp, "volume_curve") == 1.0 {
                Some(US_EQUITY_VOLUME_CURVE_13.to_vec())
            } else {
                None
            },
            ..ImpactMarket::default()
        };
        let r = internalise(&buys, &sells, &m).unwrap();
        out.push(("crossed_shares".to_string(), r.crossed_shares));
        out.push(("residual_shares".to_string(), r.residual_shares));
        out.push(("baseline_cost".to_string(), r.baseline_cost));
        out.push(("internal_cost".to_string(), r.internal_cost));
        out.push(("saved_cost".to_string(), r.saved_cost));
        out.push(("saved_bps".to_string(), r.saved_bps));
        for (i, v) in r.buy_saved_bps().iter().enumerate() {
            out.push((format!("buy{i}_saved_bps"), *v));
        }
        for (i, v) in r.sell_saved_bps().iter().enumerate() {
            out.push((format!("sell{i}_saved_bps"), *v));
        }
        return (out, Vec::new());
    }
    assert_eq!(kind, "adaptive_sim");
    let ac = ACParams::new(
        num(inp, "ac_total_shares"),
        num(inp, "ac_n_intervals") as usize,
        num(inp, "ac_tau"),
        num(inp, "ac_sigma"),
        num(inp, "ac_eta"),
        num(inp, "ac_gamma_p"),
        num(inp, "ac_lambda_ra"),
    )
    .unwrap();
    let sig = SignalParams::new(
        num(inp, "phi"),
        num(inp, "signal_sd"),
        num(inp, "fill_risk"),
    )
    .unwrap();
    let r = simulate_adaptive(
        &ac,
        &sig,
        num(inp, "n_paths_ref") as usize,
        num(inp, "seed") as u64,
    )
    .unwrap();
    out.push(("static_mean".to_string(), r.static_mean));
    out.push(("adaptive_mean".to_string(), r.adaptive_mean));
    out.push(("saving_mean".to_string(), r.saving_mean));
    let se = vec![
        ("static_mean".to_string(), r.static_se),
        ("adaptive_mean".to_string(), r.adaptive_se),
        ("saving_mean".to_string(), r.saving_se),
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
    if case["kind"] == "adaptive_sim" {
        let keys: Vec<&String> = expect.keys().filter(|k| !k.ends_with("_se")).collect();
        assert!(!keys.is_empty());
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
    golden_transient_ow_no_permanent => "transient_ow_no_permanent",
    golden_transient_with_permanent => "transient_with_permanent",
    golden_transient_slow_resilience => "transient_slow_resilience",
    golden_internalise_buy_heavy => "internalise_buy_heavy",
    golden_internalise_balanced_curve => "internalise_balanced_curve",
    golden_adaptive_no_signal => "adaptive_no_signal",
    golden_adaptive_signal => "adaptive_signal",
    golden_adaptive_fill_risk => "adaptive_fill_risk",
}

#[test]
fn golden_count() {
    assert_eq!(load()["cases"].as_array().unwrap().len(), 8);
}

// ------------------------------------------------------------ transient impact

fn kms(a: f64, n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| a.powi((i as i32 - j as i32).abs()))
                .collect()
        })
        .collect()
}

/// Dense Gaussian elimination with partial pivoting (test oracle only).
#[allow(clippy::needless_range_loop)]
fn dense_solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))
            .unwrap();
        a.swap(col, piv);
        b.swap(col, piv);
        for r in col + 1..n {
            let f = a[r][col] / a[col][col];
            for c in col..n {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let s: f64 = (r + 1..n).map(|c| a[r][c] * x[c]).sum();
        x[r] = (b[r] - s) / a[r][r];
    }
    x
}

fn trades(x: &[f64]) -> Vec<f64> {
    x.windows(2).map(|w| w[0] - w[1]).collect()
}

#[test]
fn cost_recursion_equals_quadratic_form() {
    let p = TransientParams::new(1e6, 8, 0.25, 3e-6, 1.5, 4e-7).unwrap();
    let mut rng = ChaCha12Rng::seed_from_u64(1);
    let n: Vec<f64> = (0..8)
        .map(|_| (rng.gen::<f64>() - 0.5) * 2e5 + 1.25e5)
        .collect();
    let total: f64 = n.iter().sum();
    let mut x = vec![total];
    for v in &n {
        x.push(x[x.len() - 1] - v);
    }
    x[8] = 0.0;
    let n = trades(&x);
    let k = kms(p.decay(), 8);
    let mut quad = 0.0;
    for i in 0..8 {
        for j in 0..8 {
            let aij = p.kappa() * k[i][j] - if i == j { p.gamma_p() } else { 0.0 };
            quad += n[i] * aij * n[j];
        }
    }
    let want = 0.5 * p.gamma_p() * x[0] * x[0] + 0.5 * quad;
    let got = transient_expected_cost(&p, &x).unwrap();
    assert!((got / want - 1.0).abs() < 1e-12, "{got} vs {want}");
}

#[test]
fn optimum_solves_the_dense_system() {
    for gamma_p in [0.0, 4e-7] {
        let p = TransientParams::new(1e6, 12, 0.25, 3e-6, 1.5, gamma_p).unwrap();
        let k = kms(p.decay(), 12);
        let a: Vec<Vec<f64>> = (0..12)
            .map(|i| {
                (0..12)
                    .map(|j| p.kappa() * k[i][j] - if i == j { gamma_p } else { 0.0 })
                    .collect()
            })
            .collect();
        let y = dense_solve(a, vec![1.0; 12]);
        let s: f64 = y.iter().sum();
        let got = trades(&transient_optimal_schedule(&p));
        for (g, yi) in got.iter().zip(&y) {
            assert!((g - 1e6 * yi / s).abs() <= 1e-9 * 1e6);
        }
    }
}

#[test]
fn obizhaeva_wang_shape_blocks_at_both_ends_constant_between() {
    let p = TransientParams::new(1e6, 10, 0.1, 2e-6, 5.0, 0.0).unwrap();
    let n = trades(&transient_optimal_schedule(&p));
    assert!((n[0] / n[9] - 1.0).abs() < 1e-12);
    let lo = n[1..9].iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = n[1..9].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(hi - lo <= 1e-12 * 1e6);
    assert!((n[1] / n[0] / (1.0 - p.decay()) - 1.0).abs() < 1e-12);
    assert!(n.iter().all(|v| *v > 0.0));
}

#[test]
fn permanent_impact_keeps_symmetry() {
    let p = TransientParams::new(1e6, 10, 0.1, 2e-6, 5.0, 2e-7).unwrap();
    let n = trades(&transient_optimal_schedule(&p));
    for i in 0..10 {
        assert!((n[i] - n[9 - i]).abs() <= 1e-9 * 1e6);
    }
    assert!(n[0] > n[1] && n.iter().all(|v| *v > 0.0));
}

#[test]
fn optimum_beats_twap_and_perturbations() {
    let p = TransientParams::new(2.5e5, 26, 1.0 / 26.0, 4e-6, 2.0, 1e-8).unwrap();
    let x = transient_optimal_schedule(&p);
    let e_opt = transient_expected_cost(&p, &x).unwrap();
    let tw = twap_holdings(2.5e5, 26).unwrap();
    assert!(e_opt < transient_expected_cost(&p, &tw).unwrap());
    let mut rng = ChaCha12Rng::seed_from_u64(3);
    for _ in 0..100 {
        let mut z = x.clone();
        for v in &mut z[1..26] {
            *v += (rng.gen::<f64>() - 0.5) * 2e3;
        }
        assert!(transient_expected_cost(&p, &z).unwrap() >= e_opt);
    }
}

fn liquid(lambda_ra: f64) -> ACParams {
    ACParams::new(1e6, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, lambda_ra).unwrap()
}

#[test]
fn fast_resilience_reduces_to_linear_temporary_impact() {
    let ac = liquid(0.0);
    let kappa = 2.0 * ac.eta() / ac.tau();
    let fast = TransientParams::new(1e6, 5, ac.tau(), kappa, 1e3, ac.gamma_p()).unwrap();
    assert_eq!(fast.decay(), 0.0);
    let x_ow = transient_optimal_schedule(&fast);
    for (a, b) in x_ow.iter().zip(&ac_trajectory(&ac)) {
        assert!((a - b).abs() <= 1e-9 * 1e6);
    }
    let tw = twap_holdings(1e6, 5).unwrap();
    let block = vec![1e6, 0.0, 0.0, 0.0, 0.0, 0.0];
    for x in [&x_ow, &tw, &block] {
        let t = transient_expected_cost(&fast, x).unwrap();
        let e = expected_cost(&ac, Some(x)).unwrap();
        assert!((t / e - 1.0).abs() < 1e-12, "{t} vs {e}");
    }
    // optimal-schedule cost equals the AC closed form
    let e_ac = expected_cost(&ac, None).unwrap();
    let t_opt = transient_expected_cost(&fast, &x_ow).unwrap();
    assert!((t_opt / e_ac - 1.0).abs() < 1e-12);
    // the transient cost converges to it as rho grows
    let mut gaps = Vec::new();
    for rho in [0.5, 2.0, 8.0] {
        let p = TransientParams::new(1e6, 5, ac.tau(), kappa, rho, ac.gamma_p()).unwrap();
        gaps.push(
            transient_expected_cost(&p, &tw).unwrap() - expected_cost(&ac, Some(&tw)).unwrap(),
        );
    }
    assert!(gaps[0] > gaps[1] && gaps[1] > gaps[2] && gaps[2] > 0.0);
    // and the alpha = 1 Section 8 model agrees
    let ip = ImpactParams::new(
        1e6,
        5,
        ac.tau(),
        ac.sigma(),
        ac.eta(),
        1.0,
        ac.gamma_p(),
        0.0,
        0.0,
        None,
    )
    .unwrap();
    let t = transient_expected_cost(&fast, &tw).unwrap();
    assert!((t / impact_expected_cost(&ip, &tw).unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn single_interval_and_zero_shares() {
    let p = TransientParams::new(5.0, 1, 1.0, 1.0, 1.0, 0.0).unwrap();
    assert_eq!(transient_optimal_schedule(&p), vec![5.0, 0.0]);
    assert!((transient_expected_cost(&p, &[5.0, 0.0]).unwrap() - 12.5).abs() < 1e-12);
    let z = TransientParams::new(0.0, 4, 1.0, 1.0, 1.0, 0.0).unwrap();
    assert!(transient_optimal_schedule(&z).iter().all(|v| *v == 0.0));
}

// ------------------------------------------------------------ adaptive

fn ac10() -> ACParams {
    ACParams::new(1e6, 10, 0.5, 0.95, 2.5e-6, 2.5e-7, 2e-6).unwrap()
}

fn none() -> SignalParams {
    SignalParams::new(0.0, 0.0, 0.0).unwrap()
}

#[test]
fn static_plan_is_the_almgren_chriss_trajectory() {
    let ac = ac10();
    let plan = replan(&ac, &none(), 1e6, 0.0, 0).unwrap();
    for (a, b) in plan.iter().zip(&ac_trajectory(&ac)) {
        assert!((a - b).abs() <= 1e-9 * 1e6);
    }
}

#[test]
fn replan_is_time_consistent_without_signal() {
    let ac = ac10();
    let plan = replan(&ac, &none(), 1e6, 0.0, 0).unwrap();
    for k in 1..10 {
        let tail = replan(&ac, &none(), plan[k], 0.0, k).unwrap();
        for (a, b) in tail.iter().zip(&plan[k..]) {
            assert!((a - b).abs() <= 1e-9 * 1e6);
        }
    }
}

#[test]
fn feedback_coefficients_match_replan() {
    let ac = ac10();
    let sig = SignalParams::new(0.6, 0.4, 0.0).unwrap();
    let (c, d) = replan_coefficients(&ac, &sig).unwrap();
    for k in 0..10 {
        let m = 10 - k;
        let x = replan(&ac, &sig, 3e5, 0.7, k).unwrap();
        let want = c[m] * 3e5 - d[m] * 0.7;
        assert!(((x[0] - x[1]) / want - 1.0).abs() < 1e-12);
    }
}

#[test]
fn signal_makes_the_trader_hold_when_price_is_expected_to_rise() {
    let ac = ac10();
    let sig = SignalParams::new(0.7, 0.5, 0.0).unwrap();
    let base = replan(&ac, &sig, 5e5, 0.0, 3).unwrap();
    let up = replan(&ac, &sig, 5e5, 1.0, 3).unwrap();
    assert!(up[1] > base[1]);
}

#[test]
fn adaptive_equals_static_without_signal_or_fill_risk() {
    let r = simulate_adaptive(&ac10(), &none(), 4000, 7).unwrap();
    assert!(r.saving_mean.abs() <= 1e-9 * r.static_mean);
    assert!((r.static_std / r.adaptive_std - 1.0).abs() < 1e-9);
    let e = expected_cost(&ac10(), None).unwrap();
    assert!((r.static_mean - e).abs() <= 4.0 * r.static_se);
}

#[test]
fn adaptive_helps_with_a_signal_or_fill_risk() {
    for sig in [
        SignalParams::new(0.7, 0.5, 0.0).unwrap(),
        SignalParams::new(0.0, 0.0, 0.5).unwrap(),
    ] {
        let r = simulate_adaptive(&ac10(), &sig, 4000, 8).unwrap();
        assert!(
            r.saving_mean > 4.0 * r.saving_se,
            "{} vs {}",
            r.saving_mean,
            r.saving_se
        );
    }
}

#[test]
fn adaptive_is_deterministic() {
    let sig = SignalParams::new(0.5, 0.3, 0.2).unwrap();
    let a = simulate_adaptive(&ac10(), &sig, 100, 5).unwrap();
    let b = simulate_adaptive(&ac10(), &sig, 100, 5).unwrap();
    assert_eq!(a, b);
}

// ------------------------------------------------------------ internalisation

#[test]
fn crossing_never_increases_total_cost() {
    let mut rng = ChaCha12Rng::seed_from_u64(11);
    for _ in 0..200 {
        let nb = rng.gen_range(0..4);
        let ns = rng.gen_range(1..4);
        let buys: Vec<f64> = (0..nb).map(|_| rng.gen_range(1e3..3e5)).collect();
        let sells: Vec<f64> = (0..ns).map(|_| rng.gen_range(1e3..3e5)).collect();
        let r = internalise(&buys, &sells, &market()).unwrap();
        assert!(r.internal_cost <= r.baseline_cost);
        assert!(r.saved_cost >= 0.0);
        assert!(r
            .buy_saved_bps()
            .iter()
            .chain(r.sell_saved_bps())
            .all(|v| *v >= 0.0));
    }
}

#[test]
fn internalisation_accounting() {
    let r = internalise(&[3e5, 1e5], &[2.5e5], &market()).unwrap();
    assert_eq!(r.crossed_shares, 2.5e5);
    assert_eq!(r.residual_shares, 1.5e5);
    assert_eq!(r.residual_side, "buy");
    let q = [3e5, 1e5, 2.5e5];
    let total: f64 = r
        .buy_saved_bps()
        .iter()
        .chain(r.sell_saved_bps())
        .zip(q)
        .map(|(b, qi)| b * qi * 100.0 / 1e4)
        .sum();
    assert!((total / r.saved_cost - 1.0).abs() < 1e-12);
    assert!((r.saved_bps / (1e4 * r.saved_cost / (6.5e5 * 100.0)) - 1.0).abs() < 1e-15);
    let one = internalise(&[], &[1e5, 2e5], &market()).unwrap();
    assert_eq!(one.saved_cost, 0.0);
    assert_eq!(one.residual_side, "sell");
    let even = internalise(&[2e5], &[1e5, 1e5], &market()).unwrap();
    assert_eq!(even.internal_cost, 0.0);
    assert_eq!(even.residual_side, "none");
}

// ------------------------------------------------------------ validation

fn err_msg<T: std::fmt::Debug>(r: Result<T, EtradeError>) -> String {
    match r.unwrap_err() {
        EtradeError::InvalidInput(m)
        | EtradeError::ModelInvalid(m)
        | EtradeError::SolverFailed(m) => m,
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn transient_params_validation() {
    let base = [1e6, 10.0, 0.1, 2e-6, 5.0, 0.0];
    let cases: [(usize, f64, &str); 10] = [
        (0, -1.0, "total_shares"),
        (0, f64::NAN, "total_shares"),
        (1, 0.0, "n_intervals"),
        (1, (MAX_EXEC_INTERVALS + 1) as f64, "n_intervals"),
        (2, 0.0, "tau"),
        (3, 0.0, "kappa"),
        (4, 0.0, "rho"),
        (4, f64::INFINITY, "rho"),
        (5, -1.0, "gamma_p"),
        (5, 1e-6, "gamma_p"), // >= kappa (1-a)/(1+a): not convex
    ];
    for (i, bad, field) in cases {
        let mut v = base;
        v[i] = bad;
        let r = TransientParams::new(v[0], v[1] as usize, v[2], v[3], v[4], v[5]);
        let m = err_msg(r);
        assert!(m.contains(field), "{field}: {m}");
    }
}

#[test]
fn transient_cost_validates_holdings() {
    let p = TransientParams::new(1e6, 4, 0.25, 2e-6, 5.0, 0.0).unwrap();
    assert!(err_msg(transient_expected_cost(&p, &[1.0, 0.0])).contains("length"));
    assert!(err_msg(transient_expected_cost(&p, &[1.0, 0.8, 0.6, 0.4, 0.2])).contains("liquidate"));
    assert!(
        err_msg(transient_expected_cost(&p, &[1.0, f64::NAN, 0.6, 0.4, 0.0])).contains("finite")
    );
}

#[test]
fn signal_params_validation() {
    let cases = [
        ((1.0, 0.0, 0.0), "phi"),
        ((-0.1, 0.0, 0.0), "phi"),
        ((0.0, -1.0, 0.0), "signal_sd"),
        ((0.0, 0.0, 1.0), "fill_risk"),
        ((0.0, 0.0, f64::NAN), "fill_risk"),
    ];
    for ((phi, sd, fr), field) in cases {
        assert!(err_msg(SignalParams::new(phi, sd, fr)).contains(field));
    }
}

#[test]
fn adaptive_validation() {
    let ac = ac10();
    assert!(err_msg(simulate_adaptive(&ac, &none(), 1, 0)).contains("n_paths"));
    assert!(err_msg(replan(&ac, &none(), 1.0, 0.0, 10)).contains("k must"));
    assert!(err_msg(replan(&ac, &none(), f64::INFINITY, 0.0, 0)).contains("holding"));
    assert!(err_msg(replan(&ac, &none(), 1.0, f64::NAN, 0)).contains("signal"));
    let big = ACParams::new(1.0, 1001, 1.0, 1.0, 1.0, 0.0, 0.0).unwrap();
    assert!(err_msg(simulate_adaptive(&big, &none(), 100, 0)).contains("n_intervals"));
    assert!(err_msg(replan(&big, &none(), 1.0, 0.0, 0)).contains("n_intervals"));
    assert!(err_msg(replan_coefficients(&big, &none())).contains("n_intervals"));
}

#[test]
fn internalise_validation() {
    let m = market();
    assert!(err_msg(internalise(&[0.0], &[1.0], &m)).contains("buys"));
    assert!(err_msg(internalise(&[1.0], &[f64::NAN], &m)).contains("sells"));
    assert!(err_msg(internalise(&[], &[], &m)).contains("at least one"));
    let many = vec![1.0; 10_001];
    assert!(err_msg(internalise(&many, &[1.0], &m)).contains("at most"));
    let bad_n = ImpactMarket {
        n_intervals: 1001,
        ..market()
    };
    assert!(err_msg(internalise(&[1.0], &[2.0], &bad_n)).contains("n_intervals"));
    let bad_spread = ImpactMarket {
        spread: 0.0,
        ..market()
    };
    assert!(err_msg(internalise(&[1.0], &[2.0], &bad_spread)).contains("spread"));
}
