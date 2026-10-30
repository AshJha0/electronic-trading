//! Golden-value suite: loads `../data/golden/golden.json` (relative to the
//! rust/ directory) and asserts every case within its tolerance, one test
//! per case.
//!
//! Exact cases: `|actual - expected| <= tol * max(1, |expected|)` (relative
//! for large magnitudes, absolute near zero).  Statistical cases: this port
//! runs its own seeded RNG with `M >= 2000` paths and asserts
//! `|stat - ref| <= 4*sqrt(se_ref^2 + se_own^2)` per API_SPEC.md Section 4.
//!
//! Every input and every expected key of a case must be read: a key the
//! port does not check (or a typo in one) fails the case instead of being
//! skipped silently.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde::Deserialize;

use etrade::{ACSpec, ASSpec, ASStat, Agent, SimConfig};

/// `golden.json`: the pinned case list.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Golden {
    cases: Vec<Case>,
}

/// One golden case; every input and expected value is a number.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    kind: Kind,
    tol: f64,
    inputs: BTreeMap<String, f64>,
    expect: BTreeMap<String, f64>,
}

/// How a case is compared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Kind {
    /// Deterministic: relative tolerance `tol`.
    Exact,
    /// Monte Carlo: 4-SE band.
    Statistical,
}

fn load_golden() -> Golden {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/golden/golden.json");
    let text =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&text).expect("golden.json must match the typed schema")
}

/// Reads a case and records which keys were used, so [`Checker::finish`]
/// can fail on any input or expected value nobody looked at.
struct Checker<'a> {
    case: &'a Case,
    used_inputs: RefCell<BTreeSet<&'a str>>,
    used_expect: RefCell<BTreeSet<&'a str>>,
}

impl<'a> Checker<'a> {
    fn new(case: &'a Case) -> Self {
        Self {
            case,
            used_inputs: RefCell::default(),
            used_expect: RefCell::default(),
        }
    }

    fn name(&self) -> &str {
        &self.case.name
    }

    fn get(
        map: &'a BTreeMap<String, f64>,
        used: &RefCell<BTreeSet<&'a str>>,
        what: &str,
        key: &str,
    ) -> f64 {
        let (k, v) = map
            .get_key_value(key)
            .unwrap_or_else(|| panic!("missing {what} key '{key}'"));
        used.borrow_mut().insert(k.as_str());
        *v
    }

    /// Numeric input.
    fn inp(&self, key: &str) -> f64 {
        Self::get(&self.case.inputs, &self.used_inputs, "input", key)
    }

    /// Integer input (must be a non-negative whole number).
    fn inp_int(&self, key: &str) -> u64 {
        let v = self.inp(key);
        assert!(
            v >= 0.0 && v.fract() == 0.0,
            "input '{key}' = {v} is not a count"
        );
        v as u64
    }

    fn exp(&self, key: &str) -> f64 {
        Self::get(&self.case.expect, &self.used_expect, "expect", key)
    }

    /// Exact-case check with the pinned relative-tolerance semantics.
    fn exact(&self, key: &str, actual: f64) {
        assert_eq!(
            self.case.kind,
            Kind::Exact,
            "{}: not an exact case",
            self.name()
        );
        let expected = self.exp(key);
        let bound = self.case.tol * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= bound,
            "{}.{key}: actual={actual:.15e} expected={expected:.15e} \
             |diff|={:.3e} > bound={bound:.3e}",
            self.name(),
            (actual - expected).abs()
        );
    }

    /// Statistical-case check: 4-SE band combining reference and own SE;
    /// the reference SE is read from `{key}_se`.
    fn stat(&self, key: &str, actual: f64, se_own: f64) {
        assert_eq!(
            self.case.kind,
            Kind::Statistical,
            "{}: not a statistical case",
            self.name()
        );
        let reference = self.exp(key);
        let se_ref = self.exp(&format!("{key}_se"));
        let band = 4.0 * (se_ref * se_ref + se_own * se_own).sqrt();
        assert!(
            (actual - reference).abs() <= band,
            "{}.{key}: actual={actual} ref={reference} |diff|={} > 4SE band={band}",
            self.name(),
            (actual - reference).abs()
        );
    }

    /// Statistical check only when the case pins `key` (the AS simulation
    /// cases pin different subsets of the same statistics).
    fn stat_if_present(&self, key: &str, actual: f64, se_own: f64) {
        if self.case.expect.contains_key(key) {
            self.stat(key, actual, se_own);
        }
    }

    /// Fail unless every input and expected key was read.
    fn finish(self) {
        let unused = |map: &BTreeMap<String, f64>, used: &BTreeSet<&str>| -> Vec<String> {
            map.keys()
                .filter(|k| !used.contains(k.as_str()))
                .cloned()
                .collect()
        };
        let inputs = unused(&self.case.inputs, &self.used_inputs.borrow());
        let expect = unused(&self.case.expect, &self.used_expect.borrow());
        assert!(
            inputs.is_empty() && expect.is_empty(),
            "{}: unchecked inputs {inputs:?}, unchecked expect keys {expect:?}",
            self.name()
        );
    }

    fn ac_params(&self) -> etrade::ACParams {
        ACSpec {
            total_shares: self.inp("ac_total_shares"),
            n_intervals: self.inp_int("ac_n_intervals") as usize,
            tau: self.inp("ac_tau"),
            sigma: self.inp("ac_sigma"),
            eta: self.inp("ac_eta"),
            gamma_p: self.inp("ac_gamma_p"),
            lambda_ra: self.inp("ac_lambda_ra"),
        }
        .build()
        .expect("golden AC params must be valid")
    }

    fn as_params(&self) -> etrade::ASParams {
        ASSpec {
            s0: self.inp("as_s0"),
            sigma: self.inp("as_sigma"),
            horizon: self.inp("as_horizon"),
            dt: self.inp("as_dt"),
            gamma: self.inp("as_gamma"),
            k: self.inp("as_k"),
            a_rate: self.inp("as_a_rate"),
            q_max: self.inp_int("as_q_max") as u32,
            eta_liq: self.inp("as_eta_liq"),
        }
        .build()
        .expect("golden AS params must be valid")
    }
}

/// Number of paths this port uses for AS statistical cases (>= 2000 required).
const AS_PATHS: usize = 4000;

fn check_case(name: &str) {
    let golden = load_golden();
    let case = golden
        .cases
        .iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("golden case '{name}' not found"));
    let c = Checker::new(case);
    run(&c);
    c.finish();
}

fn run(c: &Checker) {
    match c.name() {
        "ac_liquid_solution" | "ac_illiquid_solution" => {
            let p = c.ac_params();
            let x = etrade::ac_trajectory(&p);
            let n = etrade::ac_trade_list(&p);
            let j = c.inp_int("x_index") as usize;
            c.exact("kappa", etrade::ac_kappa(&p));
            c.exact("half_life", etrade::half_life(&p));
            c.exact("x_j", x[j]);
            c.exact("n_1", n[0]);
            c.exact("e_cost", etrade::expected_cost(&p, None).unwrap());
            c.exact("var_cost", etrade::variance_cost(&p, None).unwrap());
        }
        "ac_twap_liquid" => {
            let p = c.ac_params();
            let twap = etrade::twap_holdings(p.total_shares(), p.n_intervals()).unwrap();
            c.exact("e_cost", etrade::expected_cost(&p, Some(&twap)).unwrap());
            c.exact("var_cost", etrade::variance_cost(&p, Some(&twap)).unwrap());
        }
        "ac_pov_liquid" => {
            let p = c.ac_params();
            let pov = etrade::pov_holdings(
                p.total_shares(),
                p.tau(),
                c.inp("pov_market_volume_rate"),
                c.inp("pov_participation"),
            )
            .unwrap();
            c.exact("e_cost", etrade::expected_cost(&p, Some(&pov)).unwrap());
            c.exact("var_cost", etrade::variance_cost(&p, Some(&pov)).unwrap());
            c.exact("n_intervals_used", (pov.len() - 1) as f64);
        }
        "ac_frontier_point_liquid" => {
            let p = c.ac_params();
            let frontier = etrade::efficient_frontier(&p, &[p.lambda_ra()]).unwrap();
            let fp = &frontier[0];
            c.exact("e_cost", fp.e_cost);
            c.exact("std_cost", fp.std_cost);
            c.exact("kappa", fp.kappa);
        }
        "ac_kappa_small_tau" => {
            // kappa_tilde^2*tau^2/2 ~ 2e-9: the ln_1p form matches; the
            // acosh(1+x) form would miss by ~1e-8 relative.
            let p = c.ac_params();
            assert_eq!(p.n_intervals(), 5000);
            c.exact("kappa", etrade::ac_kappa(&p));
            c.exact("half_life", etrade::half_life(&p));
            c.exact("x_1", etrade::ac_trajectory(&p)[1]);
        }
        "ac_lambda_zero_limit_liquid" => {
            let p = c.ac_params();
            let x = etrade::ac_trajectory(&p);
            let twap = etrade::twap_holdings(p.total_shares(), p.n_intervals()).unwrap();
            let max_diff = x
                .iter()
                .zip(&twap)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            c.exact("max_abs_diff_vs_twap", max_diff);
        }
        "ac_instant_liquidation_liquid" => {
            let p = c.ac_params();
            let (e, v) = etrade::instantaneous_cost(&p);
            c.exact("e_cost", e);
            c.exact("var_cost", v);
        }
        "ac_mc_liquid" => {
            let p = c.ac_params();
            let n_paths = c.inp_int("n_paths_ref") as usize;
            let mc = etrade::simulate_is(&p, None, n_paths, c.inp_int("seed")).unwrap();
            c.stat("mean_cost", mc.mean, mc.se_mean);
            c.stat("var_cost", mc.var, mc.se_var);
        }
        "as_reservation_liquid" => {
            let p = c.as_params();
            let r = etrade::reservation_price(&p, c.inp("s"), c.inp("q"), c.inp("t")).unwrap();
            c.exact("reservation_price", r);
        }
        "as_spread_liquid" => {
            let p = c.as_params();
            c.exact("spread", etrade::optimal_spread(&p, c.inp("t")).unwrap());
            c.exact(
                "spread_terminal",
                etrade::optimal_spread(&p, p.horizon()).unwrap(),
            );
        }
        "as_fill_prob_liquid" => {
            let p = c.as_params();
            let delta = c.inp("delta");
            c.exact("intensity", etrade::intensity(&p, delta).unwrap());
            c.exact(
                "fill_probability",
                etrade::fill_probability(&p, delta).unwrap(),
            );
        }
        "as_sim_liquid" | "as_sim_fx" => {
            let p = c.as_params();
            // The reference path count documents the Python run; this port
            // uses its own AS_PATHS (>= 2000) and its own SE.
            assert!(c.inp_int("n_paths_ref") >= 2000);
            let res =
                etrade::simulate(&p, AS_PATHS, c.inp_int("seed"), &SimConfig::default()).unwrap();
            assert!(!res.cap_violated, "{}: inventory cap violated", c.name());
            c.stat_if_present("mean_pnl", res.mean_pnl(), res.se(ASStat::MeanPnl));
            c.stat_if_present("std_pnl", res.std_pnl(), res.se(ASStat::StdPnl));
            c.stat_if_present("mean_abs_q", res.mean_abs_q(), res.se(ASStat::MeanAbsQ));
            c.stat_if_present(
                "mean_max_abs_q",
                res.mean_max_abs_q(),
                res.se(ASStat::MeanMaxAbsQ),
            );
            c.stat_if_present("mean_fills", res.mean_fills(), res.se(ASStat::MeanFills));
        }
        "as_fixed_comparison_liquid" => {
            let p = c.as_params();
            assert!(c.inp_int("n_paths_ref") >= 2000);
            let seed = c.inp_int("seed");
            let res_as = etrade::simulate(&p, AS_PATHS, seed, &SimConfig::default()).unwrap();
            let cfg_fixed = SimConfig {
                agent: Agent::Fixed,
                fixed_spread: Some(c.inp("fixed_spread")),
                ..SimConfig::default()
            };
            let res_fx = etrade::simulate(&p, AS_PATHS, seed, &cfg_fixed).unwrap();
            c.stat("as_rms_q", res_as.rms_q(), res_as.se(ASStat::RmsQ));
            c.stat("fixed_rms_q", res_fx.rms_q(), res_fx.se(ASStat::RmsQ));
        }
        "is_decomposition_sell" | "is_decomposition_buy" => {
            let side = if c.name().ends_with("sell") {
                "sell"
            } else {
                "buy"
            };
            let d = etrade::decompose_shortfall(
                side.parse().unwrap(),
                c.inp("decision_price"),
                c.inp("arrival_price"),
                c.inp("close_price"),
                c.inp("total_qty"),
                &[c.inp("q1"), c.inp("q2"), c.inp("q3")],
                &[c.inp("p1"), c.inp("p2"), c.inp("p3")],
            )
            .unwrap();
            c.exact("total", d.total);
            c.exact("delay", d.delay);
            c.exact("trading", d.trading);
            c.exact("opportunity", d.opportunity);
        }
        "participation_rate" => {
            let part = etrade::participation_rate(c.inp("executed_volume"), c.inp("market_volume"))
                .unwrap();
            c.exact("participation", part);
        }
        "impact_regression" => {
            let (part, cost) = etrade::synthetic_impact_fills(
                c.inp_int("n_obs") as usize,
                c.inp("true_intercept"),
                c.inp("true_slope"),
                c.inp("noise_amp"),
            )
            .unwrap();
            let reg = etrade::estimate_impact(&part, &cost).unwrap();
            c.exact("slope", reg.slope);
            c.exact("intercept", reg.intercept);
            c.exact("r_squared", reg.r_squared);
        }
        other => panic!("unhandled golden case '{other}' — port must cover every case"),
    }
}

macro_rules! golden_case {
    ($($test:ident => $name:literal),* $(,)?) => {
        $(#[test] fn $test() { check_case($name); })*

        /// Every case in the file has a test above, and the count is pinned.
        #[test]
        fn golden_count() {
            let names: Vec<String> = load_golden().cases.into_iter().map(|c| c.name).collect();
            assert_eq!(names.len(), 19, "expected the pinned 19 golden cases");
            let covered: BTreeSet<&str> = [$($name),*].into_iter().collect();
            assert_eq!(covered.len(), names.len(), "duplicate case in golden_case!");
            for n in &names {
                assert!(covered.contains(n.as_str()), "golden case '{n}' has no test");
            }
        }
    };
}

golden_case! {
    golden_ac_liquid_solution => "ac_liquid_solution",
    golden_ac_illiquid_solution => "ac_illiquid_solution",
    golden_ac_twap_liquid => "ac_twap_liquid",
    golden_ac_pov_liquid => "ac_pov_liquid",
    golden_ac_frontier_point_liquid => "ac_frontier_point_liquid",
    golden_ac_kappa_small_tau => "ac_kappa_small_tau",
    golden_ac_lambda_zero_limit_liquid => "ac_lambda_zero_limit_liquid",
    golden_ac_instant_liquidation_liquid => "ac_instant_liquidation_liquid",
    golden_ac_mc_liquid => "ac_mc_liquid",
    golden_as_reservation_liquid => "as_reservation_liquid",
    golden_as_spread_liquid => "as_spread_liquid",
    golden_as_fill_prob_liquid => "as_fill_prob_liquid",
    golden_as_sim_liquid => "as_sim_liquid",
    golden_as_sim_fx => "as_sim_fx",
    golden_as_fixed_comparison_liquid => "as_fixed_comparison_liquid",
    golden_is_decomposition_sell => "is_decomposition_sell",
    golden_is_decomposition_buy => "is_decomposition_buy",
    golden_participation_rate => "participation_rate",
    golden_impact_regression => "impact_regression",
}
