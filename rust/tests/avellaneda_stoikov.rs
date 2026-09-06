//! Avellaneda-Stoikov property, edge-case and error-path tests.

use etrade::{ASParams, ASStat, Agent, SimConfig};

/// Liquid large-cap parameter set from data/params_liquid.json.
fn liquid() -> ASParams {
    ASParams::new(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, 20, 0.5).unwrap()
}

#[test]
fn reservation_price_sign_conventions() {
    let p = liquid();
    // Long maker (q > 0) shades below the mid: selling pressure.
    let r_long = etrade::reservation_price(&p, 100.0, 3.0, 0.0).unwrap();
    assert!(r_long < 100.0);
    // Short maker shades above; flat maker quotes the mid.
    let r_short = etrade::reservation_price(&p, 100.0, -3.0, 0.0).unwrap();
    assert!(r_short > 100.0);
    assert_eq!(etrade::reservation_price(&p, 100.0, 0.0, 0.0).unwrap(), 100.0);
    // At t = T the skew vanishes: r = s.
    assert_eq!(
        etrade::reservation_price(&p, 100.0, 5.0, p.horizon()).unwrap(),
        100.0
    );
}

#[test]
fn spread_positive_and_decreasing_to_terminal_floor() {
    let p = liquid();
    let floor = 2.0 / p.gamma() * (p.gamma() / p.k()).ln_1p();
    let mut prev = f64::INFINITY;
    for i in 0..=10 {
        let t = p.horizon() * i as f64 / 10.0;
        let s = etrade::optimal_spread(&p, t).unwrap();
        assert!(s > 0.0);
        assert!(s < prev, "spread must decrease toward terminal");
        assert!(s >= floor - 1e-15);
        prev = s;
    }
    let terminal = etrade::optimal_spread(&p, p.horizon()).unwrap();
    assert!((terminal - floor).abs() < 1e-15, "terminal spread equals the floor");
}

#[test]
fn quotes_bracket_reservation_price() {
    let p = liquid();
    let (bid, ask) = etrade::quotes(&p, 100.0, 3.0, 0.3).unwrap();
    let r = etrade::reservation_price(&p, 100.0, 3.0, 0.3).unwrap();
    let spread = etrade::optimal_spread(&p, 0.3).unwrap();
    assert!(bid < ask);
    assert!((ask - bid - spread).abs() < 1e-12);
    assert!(((ask + bid) / 2.0 - r).abs() < 1e-12);
}

#[test]
fn fill_probability_properties() {
    let p = liquid();
    // Intensity at the mid equals A; decreasing in delta.
    assert!((etrade::intensity(&p, 0.0).unwrap() - p.a_rate()).abs() < 1e-12);
    let mut prev = f64::INFINITY;
    for i in 0..20 {
        let delta = -0.5 + 0.1 * i as f64;
        let prob = etrade::fill_probability(&p, delta).unwrap();
        assert!((0.0..1.0).contains(&prob), "P(fill) must be in [0, 1)");
        assert!(prob < prev, "P(fill) must decrease in delta");
        prev = prob;
    }
    // A quote through the mid (delta < 0) is allowed and fills more often.
    assert!(
        etrade::fill_probability(&p, -0.1).unwrap() > etrade::fill_probability(&p, 0.1).unwrap()
    );
}

#[test]
fn average_spread_matches_time_average() {
    let p = liquid();
    // avg = (spread(0) + terminal floor)/2 because the spread is linear in t.
    let expected =
        0.5 * (etrade::optimal_spread(&p, 0.0).unwrap()
            + etrade::optimal_spread(&p, p.horizon()).unwrap());
    assert!((etrade::average_as_spread(&p) - expected).abs() < 1e-12);
}

#[test]
fn simulation_respects_inventory_cap_and_is_deterministic() {
    let p = liquid();
    let res = etrade::simulate(&p, 500, 11, &SimConfig::default()).unwrap();
    assert!(!res.cap_violated, "hard cap must hold on every step");
    assert!(res.overall_max_abs_q <= p.q_max() as f64);
    assert!(res.mean_fills() > 0.0);
    // Determinism: same seed twice gives identical results.
    let res2 = etrade::simulate(&p, 500, 11, &SimConfig::default()).unwrap();
    assert_eq!(res.pnl, res2.pnl);
    assert_eq!(res.fills_path, res2.fills_path);
    // Different seed differs.
    let res3 = etrade::simulate(&p, 500, 12, &SimConfig::default()).unwrap();
    assert_ne!(res.pnl, res3.pnl);
}

#[test]
fn tight_cap_binds_and_zero_cap_quotes_nothing() {
    // A tight cap must bind under these liquid parameters.
    let tight = ASParams::new(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, 2, 0.5).unwrap();
    let res = etrade::simulate(&tight, 300, 3, &SimConfig::default()).unwrap();
    assert!(!res.cap_violated);
    assert!(res.overall_max_abs_q <= 2.0);
    assert!(res.overall_max_abs_q >= 2.0 - 1e-12, "cap should be reached");
    // q_max = 0: no quoting at all -> no fills, zero PnL.
    let none = ASParams::new(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, 0, 0.5).unwrap();
    let res0 = etrade::simulate(&none, 100, 3, &SimConfig::default()).unwrap();
    assert_eq!(res0.mean_fills(), 0.0);
    assert!(res0.pnl.iter().all(|&v| v == 0.0));
}

#[test]
fn as_agent_controls_inventory_better_than_fixed_spread() {
    let p = liquid();
    let seed = 99;
    let n_paths = 1000;
    let res_as = etrade::simulate(&p, n_paths, seed, &SimConfig::default()).unwrap();
    let cfg_fixed = SimConfig {
        agent: Agent::Fixed,
        ..SimConfig::default()
    };
    let res_fx = etrade::simulate(&p, n_paths, seed, &cfg_fixed).unwrap();
    // Statistical boolean with a wide margin (ref: 1.45 vs 5.02 rms q).
    assert!(
        res_as.rms_q() < 0.5 * res_fx.rms_q(),
        "A-S rms q {} must be far below fixed-spread rms q {}",
        res_as.rms_q(),
        res_fx.rms_q()
    );
    // Similar PnL scale: within a factor ~2 (both agents earn the spread).
    assert!(res_as.mean_pnl() > 0.0 && res_fx.mean_pnl() > 0.0);
    // SE accessors are positive and finite for every statistic.
    for stat in [
        ASStat::MeanPnl,
        ASStat::StdPnl,
        ASStat::MeanAbsQ,
        ASStat::MeanMaxAbsQ,
        ASStat::RmsQ,
        ASStat::MeanFills,
    ] {
        let se = res_as.se(stat);
        assert!(se.is_finite() && se > 0.0);
    }
}

#[test]
fn stress_scenario_widens_pnl_distribution() {
    let p = liquid();
    let base = etrade::simulate(&p, 800, 5, &SimConfig::default()).unwrap();
    let stress_cfg = SimConfig {
        stress_factor: 2.0,
        ..SimConfig::default()
    };
    let stress = etrade::simulate(&p, 800, 5, &stress_cfg).unwrap();
    assert!(
        stress.std_pnl() > base.std_pnl(),
        "doubling realised vol mid-sim must widen the PnL distribution"
    );
}

#[test]
fn error_paths_are_reported_not_panicked() {
    // Parameter validation.
    assert!(ASParams::new(100.0, -1.0, 1.0, 0.005, 0.1, 1.5, 140.0, 20, 0.5).is_err());
    assert!(ASParams::new(100.0, 2.0, 0.0, 0.005, 0.1, 1.5, 140.0, 20, 0.5).is_err());
    assert!(ASParams::new(100.0, 2.0, 1.0, 0.0, 0.1, 1.5, 140.0, 20, 0.5).is_err());
    assert!(ASParams::new(100.0, 2.0, 1.0, 2.0, 0.1, 1.5, 140.0, 20, 0.5).is_err()); // dt > T
    assert!(ASParams::new(100.0, 2.0, 1.0, 0.3, 0.1, 1.5, 140.0, 20, 0.5).is_err()); // T/dt not integral
    assert!(ASParams::new(100.0, 2.0, 1.0, 0.005, 0.0, 1.5, 140.0, 20, 0.5).is_err()); // gamma = 0
    assert!(ASParams::new(100.0, 2.0, 1.0, 0.005, -0.1, 1.5, 140.0, 20, 0.5).is_err());
    assert!(ASParams::new(100.0, 2.0, 1.0, 0.005, 0.1, 0.0, 140.0, 20, 0.5).is_err()); // k = 0
    assert!(ASParams::new(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 0.0, 20, 0.5).is_err()); // A = 0
    assert!(ASParams::new(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, 20, -0.5).is_err());
    assert!(ASParams::new(f64::NAN, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, 20, 0.5).is_err());
    // t outside [0, T].
    let p = liquid();
    assert!(etrade::reservation_price(&p, 100.0, 0.0, -0.1).is_err());
    assert!(etrade::reservation_price(&p, 100.0, 0.0, 1.1).is_err());
    assert!(etrade::optimal_spread(&p, 2.0).is_err());
    assert!(etrade::intensity(&p, f64::NAN).is_err());
    // Simulation config validation.
    assert!(etrade::simulate(&p, 1, 0, &SimConfig::default()).is_err());
    let bad_spread = SimConfig {
        agent: Agent::Fixed,
        fixed_spread: Some(0.0),
        ..SimConfig::default()
    };
    assert!(etrade::simulate(&p, 10, 0, &bad_spread).is_err());
    let bad_stress = SimConfig {
        stress_factor: 0.0,
        ..SimConfig::default()
    };
    assert!(etrade::simulate(&p, 10, 0, &bad_stress).is_err());
    let bad_frac = SimConfig {
        stress_time_frac: 1.5,
        ..SimConfig::default()
    };
    assert!(etrade::simulate(&p, 10, 0, &bad_frac).is_err());
}

// ---------------------------------------------------------------------------
// Robustness tests (review findings CRIT-1, MAJ-8, MIN-16, T-8, T-11, T-14)
// ---------------------------------------------------------------------------

#[test]
fn mean_fills_counts_both_sides() {
    // Saturated book: P(fill) rounds to 1 on both sides every step, so the
    // fill count is exactly 2*n_steps (400), not the step count (200).
    let sat = ASParams::new(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 1e6, 10_000, 0.0).unwrap();
    let r = etrade::simulate(&sat, 50, 1, &SimConfig::default()).unwrap();
    assert_eq!(r.mean_fills(), 2.0 * sat.n_steps() as f64);
    assert!(r.fills_path.iter().all(|&f| f == 400.0));
    // Liquid set: bounded by 2 per step, integral, >= peak inventory.
    let p = liquid();
    let l = etrade::simulate(&p, 400, 8, &SimConfig::default()).unwrap();
    for i in 0..l.n_paths() {
        assert!(l.fills_path[i] <= 2.0 * p.n_steps() as f64);
        assert!(l.fills_path[i] >= l.max_abs_q_path[i]);
        assert_eq!(l.fills_path[i], l.fills_path[i].round());
    }
    assert!(l.se(ASStat::MeanFills) > 0.0);
}

#[test]
fn fixed_spread_must_be_finite_and_positive() {
    let p = liquid();
    for bad in [f64::INFINITY, f64::NAN, 0.0, -1.0] {
        let cfg = SimConfig {
            agent: Agent::Fixed,
            fixed_spread: Some(bad),
            ..SimConfig::default()
        };
        assert!(etrade::simulate(&p, 10, 0, &cfg).is_err(), "fixed_spread {bad}");
    }
    let nan_frac = SimConfig {
        stress_time_frac: f64::NAN,
        ..SimConfig::default()
    };
    assert!(etrade::simulate(&p, 10, 0, &nan_frac).is_err());
    let inf_factor = SimConfig {
        stress_factor: f64::INFINITY,
        ..SimConfig::default()
    };
    assert!(etrade::simulate(&p, 10, 0, &inf_factor).is_err());
    assert!(etrade::simulate(&p, etrade::MAX_AS_PATHS + 1, 0, &SimConfig::default()).is_err());
    assert!(etrade::simulate(&p, usize::MAX, 0, &SimConfig::default()).is_err());
}

#[test]
fn stress_edge_fractions() {
    let p = liquid();
    let base = etrade::simulate(&p, 300, 9, &SimConfig::default()).unwrap();
    // frac = 1.0: no step has t >= T, so bitwise identical to unstressed.
    let late = SimConfig {
        stress_factor: 5.0,
        stress_time_frac: 1.0,
        ..SimConfig::default()
    };
    assert_eq!(etrade::simulate(&p, 300, 9, &late).unwrap().pnl, base.pnl);
    // factor = 1.0 at any fraction: identical.
    let unit = SimConfig {
        stress_factor: 1.0,
        stress_time_frac: 0.0,
        ..SimConfig::default()
    };
    assert_eq!(etrade::simulate(&p, 300, 9, &unit).unwrap().pnl, base.pnl);
    // frac = 0.0 with factor 5: every step stressed, dispersion up.
    let early = SimConfig {
        stress_factor: 5.0,
        stress_time_frac: 0.0,
        ..SimConfig::default()
    };
    assert!(etrade::simulate(&p, 300, 9, &early).unwrap().std_pnl() > base.std_pnl());
}

#[test]
fn quotes_reject_non_finite_state() {
    let p = liquid();
    assert!(etrade::reservation_price(&p, f64::NAN, 0.0, 0.0).is_err());
    assert!(etrade::reservation_price(&p, 100.0, f64::INFINITY, 0.0).is_err());
    assert!(etrade::quotes(&p, 100.0, f64::INFINITY, 0.0).is_err());
    assert!(etrade::quotes(&p, f64::NEG_INFINITY, 0.0, 0.0).is_err());
    assert!(etrade::reservation_price(&p, 100.0, 0.0, f64::NAN).is_err());
    assert!(etrade::fill_probability(&p, f64::NAN).is_err());
}

#[test]
fn n_steps_integrality_float() {
    let p3 = ASParams::new(100.0, 2.0, 1.0, 1.0 / 3.0, 0.1, 1.5, 140.0, 20, 0.5).unwrap();
    assert_eq!(p3.n_steps(), 3);
    assert!(ASParams::new(100.0, 2.0, 1.0, 0.3, 0.1, 1.5, 140.0, 20, 0.5).is_err());
    let p1 = ASParams::new(100.0, 2.0, 1.0, 1.0, 0.1, 1.5, 140.0, 20, 0.5).unwrap(); // dt = T
    assert_eq!(p1.n_steps(), 1);
    let r = etrade::simulate(&p1, 200, 4, &SimConfig::default()).unwrap();
    for i in 0..r.n_paths() {
        // one step: the time average of |q| is the single post-update value
        assert_eq!(r.mean_abs_q_path[i], r.max_abs_q_path[i]);
        assert_eq!(r.rms_q_path[i], r.max_abs_q_path[i]);
        assert!(r.fills_path[i] <= 2.0);
    }
    assert_eq!(r.mean_abs_q(), r.mean_max_abs_q());
    // step-count bound
    let tiny_dt = 1.0 / (10.0 * etrade::MAX_AS_STEPS as f64);
    assert!(ASParams::new(100.0, 2.0, 1.0, tiny_dt, 0.1, 1.5, 140.0, 20, 0.5).is_err());
}
