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
        // x_0 = X*sinh(kappa*T)/sinh(kappa*T) can be one ulp off X.
        assert!((x[0] - p.total_shares()).abs() <= 1e-12 * p.total_shares());
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
