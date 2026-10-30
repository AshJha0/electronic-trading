//! Almgren-Chriss property, edge-case and error-path tests.

use etrade::{ACParams, EtradeError};

/// Liquid large-cap parameter set from data/params_liquid.json.
fn liquid(lambda_ra: f64) -> ACParams {
    ACParams::new(1_000_000.0, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, lambda_ra).unwrap()
}

#[test]
fn trajectory_monotone_decreasing_and_sums_to_x() {
    // Property loop over a risk-aversion grid.
    for &lam in &[0.0, 1e-8, 1e-7, 5e-7, 2e-6, 1e-5, 1e-4] {
        let p = liquid(lam);
        let x = etrade::ac_trajectory(&p);
        assert_eq!(x.len(), p.n_intervals() + 1);
        // x_0 == X exactly: the expm1 ratio is formed before the product.
        assert_eq!(x[0], p.total_shares());
        assert_eq!(*x.last().unwrap(), 0.0);
        for w in x.windows(2) {
            assert!(w[0] >= w[1], "holdings must be non-increasing (sell)");
        }
        let sold: f64 = etrade::ac_trade_list(&p).iter().sum();
        assert!(
            (sold - p.total_shares()).abs() <= 1e-12 * p.total_shares().max(1.0),
            "trades must sum to X (lambda={lam}): sold={sold}"
        );
    }
}

#[test]
fn higher_risk_aversion_front_loads_and_trades_off_e_vs_var() {
    let lambdas = [0.0, 1e-7, 5e-7, 2e-6, 1e-5];
    let mut prev_e = f64::NEG_INFINITY;
    let mut prev_var = f64::INFINITY;
    let mut prev_n1 = 0.0;
    for &lam in &lambdas {
        let p = liquid(lam);
        let e = etrade::expected_cost(&p, None).unwrap();
        let v = etrade::variance_cost(&p, None).unwrap();
        let n1 = etrade::ac_trade_list(&p)[0];
        assert!(e > prev_e, "E[cost] must increase with lambda_ra");
        assert!(v < prev_var, "Var[cost] must decrease with lambda_ra");
        assert!(n1 > prev_n1 - 1e-9, "first clip must grow (front-loading)");
        prev_e = e;
        prev_var = v;
        prev_n1 = n1;
    }
}

#[test]
fn closed_forms_match_monte_carlo_within_4_se() {
    let p = liquid(2e-6);
    let mc = etrade::simulate_is(&p, None, 20_000, 20260829).unwrap();
    let e = etrade::expected_cost(&p, None).unwrap();
    let v = etrade::variance_cost(&p, None).unwrap();
    assert!(
        (mc.mean - e).abs() <= 4.0 * mc.se_mean,
        "MC mean {} vs closed {} beyond 4 SE ({})",
        mc.mean,
        e,
        mc.se_mean
    );
    assert!(
        (mc.var - v).abs() <= 4.0 * mc.se_var,
        "MC var {} vs closed {} beyond 4 SE ({})",
        mc.var,
        v,
        mc.se_var
    );
}

#[test]
fn kappa_limit_branch_is_continuous_near_threshold() {
    let p0 = liquid(0.0);
    // Exact TWAP in the limit branch.
    let twap = etrade::twap_holdings(p0.total_shares(), p0.n_intervals()).unwrap();
    assert_eq!(etrade::ac_trajectory(&p0), twap);
    assert_eq!(etrade::ac_kappa(&p0), 0.0);
    assert!(etrade::half_life(&p0).is_infinite());
    // A lambda putting kappa*T just above the 1e-6 threshold: the sinh branch
    // must agree with the linear limit to well below golden tolerance.
    let eta_tilde = p0.eta_tilde();
    let target_kappa = 2.1e-6 / p0.horizon(); // kappa*T ~= 2.1e-6 > threshold
    let lam = target_kappa.powi(2) * eta_tilde / p0.sigma().powi(2);
    let p_small = liquid(lam);
    let kappa = etrade::ac_kappa(&p_small);
    assert!(kappa * p_small.horizon() >= etrade::KAPPA_T_LIMIT, "must use sinh branch");
    let x = etrade::ac_trajectory(&p_small);
    let max_rel = x
        .iter()
        .zip(&twap)
        .map(|(a, b)| (a - b).abs() / p_small.total_shares())
        .fold(0.0_f64, f64::max);
    assert!(max_rel < 1e-10, "sinh branch discontinuous at threshold: {max_rel}");
}

#[test]
fn single_clip_and_zero_share_edge_cases() {
    // N = 1: single clip [X, 0].
    let p1 = ACParams::new(500.0, 1, 1.0, 0.3, 1e-4, 1e-6, 1e-3).unwrap();
    assert_eq!(etrade::ac_trajectory(&p1), vec![500.0, 0.0]);
    assert_eq!(etrade::ac_trade_list(&p1), vec![500.0]);
    // X = 0: empty trajectory, E = Var = 0.
    let p0 = ACParams::new(0.0, 4, 1.0, 0.3, 1e-4, 1e-6, 1e-3).unwrap();
    assert!(etrade::ac_trajectory(&p0).iter().all(|&v| v == 0.0));
    assert_eq!(etrade::expected_cost(&p0, None).unwrap(), 0.0);
    assert_eq!(etrade::variance_cost(&p0, None).unwrap(), 0.0);
    let (e, v) = etrade::instantaneous_cost(&p0);
    assert_eq!((e, v), (0.0, 0.0));
}

#[test]
fn instantaneous_liquidation_dominates_on_variance_only() {
    let p = liquid(2e-6);
    let (e_inst, v_inst) = etrade::instantaneous_cost(&p);
    assert_eq!(v_inst, 0.0);
    // Instant liquidation costs more in expectation than the optimum.
    assert!(e_inst > etrade::expected_cost(&p, None).unwrap());
}

#[test]
fn pov_schedule_shape_and_moments() {
    let x = etrade::pov_holdings(1_000_000.0, 1.0, 2_500_000.0, 0.15).unwrap();
    // clip = 375k -> ceil(1e6/375k) = 3 intervals, last clip partial.
    assert_eq!(x.len(), 4);
    assert_eq!(x[0], 1_000_000.0);
    assert_eq!(*x.last().unwrap(), 0.0);
    let trades = etrade::trades_from_holdings(&x).unwrap();
    assert!((trades.iter().sum::<f64>() - 1_000_000.0).abs() < 1e-6);
    assert!(trades[2] < trades[0], "last clip must be partial");
    // X = 0 gives a length-2 all-zero path.
    assert_eq!(etrade::pov_holdings(0.0, 1.0, 1e6, 0.1).unwrap(), vec![0.0, 0.0]);
}

#[test]
fn efficient_frontier_is_monotone() {
    let p = liquid(2e-6);
    let grid = [1e-8, 1e-7, 5e-7, 1e-6, 2e-6, 5e-6];
    let frontier = etrade::efficient_frontier(&p, &grid).unwrap();
    assert_eq!(frontier.len(), grid.len());
    for w in frontier.windows(2) {
        assert!(w[1].e_cost > w[0].e_cost);
        assert!(w[1].std_cost < w[0].std_cost);
        assert!(w[1].kappa > w[0].kappa);
    }
}

#[test]
fn mc_simulation_is_deterministic_per_seed() {
    let p = liquid(2e-6);
    let a = etrade::simulate_is(&p, None, 2000, 7).unwrap();
    let b = etrade::simulate_is(&p, None, 2000, 7).unwrap();
    assert_eq!(a.mean, b.mean);
    assert_eq!(a.var, b.var);
    let c = etrade::simulate_is(&p, None, 2000, 8).unwrap();
    assert_ne!(a.mean, c.mean, "different seeds must differ");
}

#[test]
fn error_paths_are_reported_not_panicked() {
    // Model-validity condition (equality included).
    let err = ACParams::new(1e6, 5, 1.0, 0.95, 2.5e-6, 5e-6, 2e-6).unwrap_err();
    assert!(matches!(err, EtradeError::ModelInvalid(_)));
    let err = ACParams::new(1e6, 5, 2.0, 0.95, 1e-6, 1e-6, 0.0).unwrap_err();
    assert!(matches!(err, EtradeError::ModelInvalid(_)));
    // Domain violations.
    assert!(ACParams::new(-1.0, 5, 1.0, 0.95, 2.5e-6, 0.0, 0.0).is_err());
    assert!(ACParams::new(1e6, 0, 1.0, 0.95, 2.5e-6, 0.0, 0.0).is_err());
    assert!(ACParams::new(1e6, 5, 0.0, 0.95, 2.5e-6, 0.0, 0.0).is_err());
    assert!(ACParams::new(1e6, 5, 1.0, -0.1, 2.5e-6, 0.0, 0.0).is_err());
    assert!(ACParams::new(1e6, 5, 1.0, 0.95, 0.0, 0.0, 0.0).is_err());
    assert!(ACParams::new(1e6, 5, 1.0, 0.95, 2.5e-6, -1e-9, 0.0).is_err());
    assert!(ACParams::new(1e6, 5, 1.0, 0.95, 2.5e-6, 0.0, -1.0).is_err());
    assert!(ACParams::new(f64::NAN, 5, 1.0, 0.95, 2.5e-6, 0.0, 0.0).is_err());
    // POV domain.
    assert!(etrade::pov_holdings(1e6, 1.0, 2.5e6, 0.0).is_err());
    assert!(etrade::pov_holdings(1e6, 1.0, 2.5e6, 1.0).is_err());
    assert!(etrade::pov_holdings(1e6, 1.0, 0.0, 0.1).is_err());
    // Frontier and schedule errors.
    let p = liquid(2e-6);
    assert!(etrade::efficient_frontier(&p, &[]).is_err());
    assert!(etrade::efficient_frontier(&p, &[-1.0]).is_err());
    assert!(etrade::expected_cost(&p, Some(&[1e6, 5e5, 100.0])).is_err()); // not liquidated
    assert!(etrade::trades_from_holdings(&[1.0]).is_err());
    assert!(etrade::simulate_is(&p, None, 1, 0).is_err());
    assert!(etrade::twap_holdings(-1.0, 5).is_err());
    assert!(etrade::twap_holdings(1.0, 0).is_err());
}

// ---------------------------------------------------------------------------
// Robustness and literature tests: overflow-free trajectory, log1p kappa, POV size bounds, liquidation checks, paper closed forms
// ---------------------------------------------------------------------------

/// FX-like parameter set from data/params_fx.json.
fn fx_set(n_intervals: usize) -> ACParams {
    ACParams::new(5e7, n_intervals, 0.1, 0.006, 2e-9, 5e-11, 1e-4).unwrap()
}

fn expect_finite_large_kappa_t(p: &ACParams, n_int: usize) {
    let kappa = etrade::ac_kappa(p);
    assert!(kappa * p.horizon() > 710.0, "sinh would overflow here");
    let x = etrade::ac_trajectory(p);
    assert_eq!(x.len(), n_int + 1);
    assert_eq!(x[0], p.total_shares());
    assert_eq!(x[n_int], 0.0);
    assert!(x.iter().all(|v| v.is_finite()), "every x_j must be finite");
    let mut normal = 0;
    for w in x.windows(2) {
        assert!(w[1] <= w[0]);
        if w[1] > f64::MIN_POSITIVE * x[0] {
            assert!(w[1] < w[0], "strictly decreasing while in the normal range");
            normal += 1;
        }
    }
    assert!(normal >= 800);
    let sold: f64 = etrade::ac_trade_list(p).iter().sum();
    assert!((sold - p.total_shares()).abs() <= 1e-12 * p.total_shares());
    let r = (-kappa * p.tau()).exp();
    assert!((x[1] / x[0] - r).abs() <= 1e-9 * r);
    assert!(etrade::expected_cost(p, None).unwrap().is_finite());
    assert!(etrade::variance_cost(p, None).unwrap().is_finite());
    for fp in etrade::efficient_frontier(p, &[1e-8, 1e-6, 1e-4, 1.0]).unwrap() {
        assert!(fp.e_cost.is_finite() && fp.std_cost.is_finite() && fp.kappa.is_finite());
    }
}

#[test]
fn trajectory_finite_for_large_kappa_t() {
    // kappa*T ~ 846 and ~2683: NaN with the textbook sinh ratio.
    expect_finite_large_kappa_t(
        &ACParams::new(1e6, 1000, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6).unwrap(),
        1000,
    );
    expect_finite_large_kappa_t(&fx_set(20000), 20000);
}

#[test]
fn paper_closed_forms_match_discrete_sums() {
    // Almgren-Chriss (2000): closed-form E and V of the optimal strategy
    // (with the discrete kappa) equal the exact sums to 1e-12.
    let illiquid = ACParams::new(5e5, 10, 0.5, 1.5, 1e-5, 5e-7, 5e-6).unwrap();
    for p in [liquid(2e-6), illiquid, fx_set(10)] {
        let (k, t, tau, x) = (etrade::ac_kappa(&p), p.horizon(), p.tau(), p.total_shares());
        let sh = (k * t).sinh();
        let e_paper = 0.5 * p.gamma_p() * x * x
            + p.eta_tilde() * x * x * (0.5 * k * tau).tanh()
                * (tau * (2.0 * k * t).sinh() + 2.0 * t * (k * tau).sinh())
                / (2.0 * tau * tau * sh * sh);
        let v_paper = 0.5 * p.sigma().powi(2) * x * x
            * (tau * sh * (k * (t - tau)).cosh() - t * (k * tau).sinh())
            / (sh * sh * (k * tau).sinh());
        let e = etrade::expected_cost(&p, None).unwrap();
        let v = etrade::variance_cost(&p, None).unwrap();
        assert!((e - e_paper).abs() <= 1e-12 * e_paper, "E: {e} vs paper {e_paper}");
        assert!((v - v_paper).abs() <= 1e-12 * v_paper, "V: {v} vs paper {v_paper}");
    }
}

#[test]
fn kappa_small_tau_accuracy() {
    // kappa = kappa_tilde*(1 - kappa_tilde^2 tau^2/24 + O(tau^4)).
    for lam in [1e-6, 1e-12] {
        let p = ACParams::new(1e6, 5000, 1e-3, 0.95, 2.5e-6, 2.5e-7, lam).unwrap();
        let kt2 = lam * p.sigma().powi(2) / p.eta_tilde();
        let series = kt2.sqrt() * (1.0 - kt2 * p.tau().powi(2) / 24.0);
        let k = etrade::ac_kappa(&p);
        assert!((k - series).abs() <= 1e-9 * series, "kappa {k} vs series {series}");
        assert!((etrade::half_life(&p) - 1.0 / series).abs() <= 1e-9 / series);
        assert_eq!(etrade::characteristic_time(&p), etrade::half_life(&p));
    }
}

#[test]
fn pov_clip_too_small_is_input_error() {
    let err = etrade::pov_holdings(1e6, 1.0, 1.0, 1e-9).unwrap_err();
    assert!(matches!(err, EtradeError::InvalidInput(_)));
    assert!(err.to_string().contains("POV clip too small"));
    assert!(etrade::pov_holdings(etrade::MAX_POV_INTERVALS as f64 + 1.0, 1.0, 1.0, 0.5).is_err());
}

#[test]
fn pov_boundary_interval_counts() {
    let clip = 0.15 * 2.5e6; // 375,000
    let x2 = etrade::pov_holdings(750_000.0, 1.0, 2.5e6, 0.15).unwrap();
    assert_eq!(x2, vec![750_000.0, 375_000.0, 0.0]); // X/clip = 2 exactly
    let x3 = etrade::pov_holdings(750_000.001, 1.0, 2.5e6, 0.15).unwrap();
    assert_eq!(x3.len(), 4); // one share-thousandth more -> 3 intervals
    let n3 = etrade::trades_from_holdings(&x3).unwrap();
    assert_eq!(n3[0], clip);
    assert_eq!(n3[1], clip);
    assert!((n3[2] - 0.001).abs() < 1e-6, "last clip partial: {}", n3[2]);
    assert_eq!(x3[3], 0.0);
}

#[test]
fn simulate_is_rejects_non_liquidating_schedule() {
    let p = liquid(2e-6);
    let err = etrade::simulate_is(&p, Some(&[1e6, 5e5, 1e3]), 10, 0).unwrap_err();
    assert!(err.to_string().contains("liquidate"));
    assert!(etrade::simulate_is(&p, Some(&[1e6, f64::NAN, 0.0]), 10, 0).is_err());
    assert!(etrade::simulate_is(&p, Some(&[1e6]), 10, 0).is_err());
    assert!(etrade::simulate_is(&p, None, etrade::MAX_PATHS + 1, 0).is_err());
    assert!(etrade::simulate_is(&p, None, usize::MAX, 0).is_err()); // no capacity overflow
}

#[test]
fn simulate_is_twap_matches_closed_forms() {
    let p = liquid(2e-6);
    let twap = etrade::twap_holdings(p.total_shares(), p.n_intervals()).unwrap();
    let mc = etrade::simulate_is(&p, Some(&twap), 5000, 17).unwrap();
    let e = etrade::expected_cost(&p, Some(&twap)).unwrap();
    let v = etrade::variance_cost(&p, Some(&twap)).unwrap();
    assert!((mc.mean - e).abs() <= 4.0 * mc.se_mean);
    assert!((mc.var - v).abs() <= 4.0 * mc.se_var);
    assert_eq!(mc.n_paths, 5000);
}

#[test]
fn frontier_rejects_non_finite_lambda_and_keeps_order() {
    let p = liquid(2e-6);
    assert!(etrade::efficient_frontier(&p, &[1e-7, f64::NAN]).is_err());
    assert!(etrade::efficient_frontier(&p, &[f64::INFINITY]).is_err());
    assert!(etrade::efficient_frontier(&p, &[1e-7, -1e-7]).is_err());
    let f = etrade::efficient_frontier(&p, &[2e-6, 1e-8, 2e-6]).unwrap();
    assert_eq!(f.len(), 3);
    assert_eq!(f[0].lambda_ra, 2e-6);
    assert_eq!(f[1].lambda_ra, 1e-8);
    assert_eq!(f[0], f[2]);
    assert!(f[1].e_cost < f[0].e_cost);
}

#[test]
fn twap_and_pov_reject_non_finite() {
    let (nan, inf) = (f64::NAN, f64::INFINITY);
    assert!(etrade::twap_holdings(inf, 5).is_err());
    assert!(etrade::twap_holdings(nan, 5).is_err());
    assert!(etrade::pov_holdings(nan, 1.0, 1e6, 0.1).is_err());
    assert!(etrade::pov_holdings(inf, 1.0, 1e6, 0.1).is_err());
    assert!(etrade::pov_holdings(1e6, inf, 1e6, 0.1).is_err());
    assert!(etrade::pov_holdings(1e6, 1.0, inf, 0.1).is_err());
    assert!(etrade::pov_holdings(1e6, 1.0, 1e6, nan).is_err());
}

#[test]
fn size_bounds_are_input_errors_not_panics() {
    assert!(ACParams::new(1e6, etrade::MAX_AC_INTERVALS + 1, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6)
        .is_err());
    assert!(ACParams::new(1e6, usize::MAX, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6).is_err());
    assert!(etrade::twap_holdings(1e6, etrade::MAX_AC_INTERVALS + 1).is_err());
    assert!(etrade::twap_holdings(1e6, usize::MAX).is_err());
    assert_eq!(
        etrade::twap_holdings(1.0, etrade::MAX_AC_INTERVALS).unwrap().len(),
        etrade::MAX_AC_INTERVALS + 1
    );
}

#[test]
fn cost_functions_validate_holdings() {
    let p = liquid(2e-6);
    assert!(etrade::variance_cost(&p, Some(&[1e6])).is_err());
    assert!(etrade::variance_cost(&p, Some(&[1e6, f64::INFINITY, 0.0])).is_err());
    assert!(etrade::expected_cost(&p, Some(&[1e6, f64::NAN, 0.0])).is_err());
    assert!(etrade::trades_from_holdings(&[1e6, f64::NAN]).is_err());
    // X is taken from the schedule (x_0), not from params (pinned).
    let half = etrade::twap_holdings(5e5, 5).unwrap();
    let e_half = 0.5 * p.gamma_p() * 5e5 * 5e5 + p.eta_tilde() * 5e5 * 5e5 / 5.0;
    let e = etrade::expected_cost(&p, Some(&half)).unwrap();
    assert!((e - e_half).abs() <= 1e-12 * e_half);
}

#[test]
fn trajectory_x0_exact_and_sinh_identity_moderate() {
    for lam in [1e-8, 5e-7, 2e-6, 1e-4] {
        let p = liquid(lam);
        let (k, t) = (etrade::ac_kappa(&p), p.horizon());
        let x = etrade::ac_trajectory(&p);
        assert_eq!(x[0], p.total_shares());
        for j in 1..p.n_intervals() {
            let r = p.total_shares() * (k * (t - j as f64 * p.tau())).sinh() / (k * t).sinh();
            assert!((x[j] - r).abs() <= 1e-14 * r, "j={j}: {} vs {r}", x[j]);
        }
    }
}
