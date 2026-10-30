//! Property tests (proptest): schedule invariants, IS decomposition
//! identities, monotonicity of impact costs, `bps` linearity, the Section 7
//! defining properties, validation that never panics, and the pinned
//! product bounds of API_SPEC.md Section 0.

use proptest::prelude::*;

use etrade::impact::{impact_expected_cost, optimal_schedule, sqrt_law_cost_bps, ImpactParams};
use etrade::market::{
    ac_params_from_market, as_params_from_market, bps, impact_params_from_market, ACMarket,
    ASMarket, ImpactMarket,
};
use etrade::{
    ac_trajectory, decompose_shortfall, expected_cost, instantaneous_cost, intensity, pov_holdings,
    reservation_price, simulate, simulate_is, twap_holdings, ACParams, ASParams, EtradeError, Side,
    SimConfig, MAX_AS_WORK, MAX_MC_WORK,
};

fn cfg(cases: u32) -> ProptestConfig {
    ProptestConfig {
        cases,
        ..ProptestConfig::default()
    }
}

fn close(a: f64, b: f64, rel: f64, scale: f64) -> bool {
    (a - b).abs() <= rel * scale.max(1.0)
}

/// Assert x_0 == X, x_N == 0, non-increasing, trades sum to X.
fn check_schedule(x: &[f64], total: f64) -> Result<(), TestCaseError> {
    prop_assert!(x.len() >= 2);
    prop_assert!(
        close(x[0], total, 1e-15, total),
        "x_0 {} vs {}",
        x[0],
        total
    );
    prop_assert_eq!(x[x.len() - 1], 0.0);
    for w in x.windows(2) {
        prop_assert!(w[1] <= w[0] + 1e-12 * total.max(1.0), "{} > {}", w[1], w[0]);
    }
    let sum: f64 = x.windows(2).map(|w| w[0] - w[1]).sum();
    prop_assert!(close(sum, total, 1e-9, total), "sum {} vs {}", sum, total);
    Ok(())
}

prop_compose! {
    fn ac_params()(
        x in 0.0f64..1e7,
        n in 1usize..200,
        tau in 0.01f64..2.0,
        sigma in 0.0f64..5.0,
        eta in 1e-8f64..1e-3,
        gfrac in 0.0f64..0.99,
        lambda_ra in prop_oneof![Just(0.0), 1e-9f64..1e-3],
    ) -> ACParams {
        // gamma_p*tau < 2*eta (model validity)
        let gamma_p = gfrac * 2.0 * eta / tau;
        ACParams::new(x, n, tau, sigma, eta, gamma_p, lambda_ra).unwrap()
    }
}

prop_compose! {
    fn impact_params()(
        x in 1e3f64..1e7,
        n in 1usize..25,
        tau in 0.02f64..1.0,
        sigma in 0.0f64..3.0,
        eta in 0.01f64..1.0,
        alpha in 0.3f64..=1.0,
        gfrac in 0.0f64..0.95,
        lambda_ra in prop_oneof![Just(0.0), 1e-9f64..1e-5],
        drift in prop_oneof![Just(0.0), 0.0f64..0.5],
    ) -> ImpactParams {
        // convexity: alpha*(1+alpha)*eta_j*X^(alpha-1) > gamma_p (eta_j = eta*tau^-alpha)
        let bound = alpha * (1.0 + alpha) * eta * tau.powf(-alpha) * x.powf(alpha - 1.0);
        ImpactParams::new(x, n, tau, sigma, eta, alpha, gfrac * bound, lambda_ra, drift, None)
            .unwrap()
    }
}

// ------------------------------------------------------------ schedules

proptest! {
    #![proptest_config(cfg(128))]

    #[test]
    fn ac_trajectory_is_a_liquidation(p in ac_params()) {
        let x = ac_trajectory(&p);
        prop_assert_eq!(x.len(), p.n_intervals() + 1);
        check_schedule(&x, p.total_shares())?;
    }

    #[test]
    fn twap_is_a_liquidation(x in 0.0f64..1e9, n in 1usize..500) {
        check_schedule(&twap_holdings(x, n).unwrap(), x)?;
    }

    #[test]
    fn pov_is_a_liquidation(
        x in 0.0f64..1e6, tau in 0.01f64..1.0, v in 1e3f64..1e6, part in 0.01f64..0.99,
    ) {
        check_schedule(&pov_holdings(x, tau, v, part).unwrap(), x)?;
    }

    #[test]
    fn impact_optimal_schedule_is_a_liquidation(p in impact_params()) {
        let r = optimal_schedule(&p).unwrap();
        prop_assert_eq!(r.holdings.len(), p.n_intervals() + 1);
        check_schedule(&r.holdings, p.total_shares())?;
    }
}

// ------------------------------------------------------------ Section 3.1

prop_compose! {
    fn fills()(n in 0usize..8)(
        qtys in prop::collection::vec(0.0f64..1e4, n),
        prices in prop::collection::vec(50.0f64..150.0, n),
        extra in 0.0f64..1e4,
    ) -> (Vec<f64>, Vec<f64>, f64) {
        let x = qtys.iter().sum::<f64>() + extra;
        (qtys, prices, x)
    }
}

proptest! {
    #![proptest_config(cfg(128))]

    #[test]
    fn is_decomposition_identity_and_shift_invariance(
        (qtys, prices, x) in fills(),
        sell in any::<bool>(),
        pd in 50.0f64..150.0, p0 in 50.0f64..150.0, pt in 50.0f64..150.0,
        c in -1e3f64..1e3,
    ) {
        let side = if sell { Side::Sell } else { Side::Buy };
        let sign = if sell { 1.0 } else { -1.0 };
        let d = decompose_shortfall(side, pd, p0, pt, x, &qtys, &prices).unwrap();
        let scale = x * 200.0;
        prop_assert!(close(d.total, d.delay + d.trading + d.opportunity, 1e-9, scale));
        let notional: f64 = qtys.iter().zip(&prices).map(|(n, p)| n * p).sum();
        let closed = sign * (x * pd - notional - d.remaining_qty * pt);
        prop_assert!(close(d.total, closed, 1e-9, scale), "{} vs {}", d.total, closed);

        let shifted: Vec<f64> = prices.iter().map(|p| p + c).collect();
        let s = decompose_shortfall(side, pd + c, p0 + c, pt + c, x, &qtys, &shifted).unwrap();
        let sscale = x * (200.0 + 2.0 * c.abs());
        prop_assert!(close(s.delay, d.delay, 1e-9, sscale));
        prop_assert!(close(s.trading, d.trading, 1e-9, sscale));
        prop_assert!(close(s.opportunity, d.opportunity, 1e-9, sscale));
        prop_assert!(close(s.total, d.total, 1e-9, sscale));
    }
}

// ------------------------------------------------------------ monotonicity

proptest! {
    #![proptest_config(cfg(128))]

    #[test]
    fn sqrt_law_cost_non_decreasing_in_size(
        vol in 0.001f64..0.1, adv in 1e4f64..1e8, y in 0.1f64..2.0,
        q1 in 0.0f64..1e7, dq in 0.0f64..1e7,
    ) {
        let q2 = q1 + dq;
        let c1 = sqrt_law_cost_bps(vol, q1, adv, y).unwrap() * q1;
        let c2 = sqrt_law_cost_bps(vol, q2, adv, y).unwrap() * q2;
        prop_assert!(c1 <= c2);
    }

    #[test]
    fn impact_twap_cost_non_decreasing_in_size(
        x1 in 1e3f64..1e6, dx in 0.0f64..1e6, n in 1usize..20, tau in 0.05f64..1.0,
        eta in 0.01f64..1.0, alpha in 0.3f64..=1.0, gfrac in 0.0f64..0.95,
        drift in 0.0f64..0.5,
    ) {
        let x2 = x1 + dx;
        // gamma_p fixed for both sizes: convex at the larger X (binding when alpha < 1)
        let bound = alpha * (1.0 + alpha) * eta * tau.powf(-alpha) * x2.powf(alpha - 1.0);
        let g = gfrac * bound;
        let mk = |x: f64| ImpactParams::new(x, n, tau, 1.0, eta, alpha, g, 0.0, drift, None).unwrap();
        let e1 = impact_expected_cost(&mk(x1), &twap_holdings(x1, n).unwrap()).unwrap();
        let e2 = impact_expected_cost(&mk(x2), &twap_holdings(x2, n).unwrap()).unwrap();
        prop_assert!(e1 <= e2 * (1.0 + 1e-12), "{} > {}", e1, e2);
    }

    #[test]
    fn ac_costs_non_decreasing_in_size(p in ac_params(), dx in 0.0f64..1e7) {
        let x1 = p.total_shares();
        let x2 = x1 + dx;
        let p2 = ACParams::new(x2, p.n_intervals(), p.tau(), p.sigma(), p.eta(), p.gamma_p(),
            p.lambda_ra()).unwrap();
        prop_assert!(instantaneous_cost(&p).0 <= instantaneous_cost(&p2).0);
        let n = p.n_intervals();
        let e1 = expected_cost(&p, Some(&twap_holdings(x1, n).unwrap())).unwrap();
        let e2 = expected_cost(&p2, Some(&twap_holdings(x2, n).unwrap())).unwrap();
        prop_assert!(e1 <= e2 * (1.0 + 1e-12), "{} > {}", e1, e2);
    }
}

// ------------------------------------------------------------ bps, Section 7

proptest! {
    #![proptest_config(cfg(128))]

    #[test]
    fn bps_is_linear(a in -1e9f64..1e9, b in -1e9f64..1e9, k in -1e3f64..1e3, n in 1e-3f64..1e12) {
        let (ba, bb) = (bps(a, n).unwrap(), bps(b, n).unwrap());
        let sum = bps(a + b, n).unwrap();
        prop_assert!((sum - (ba + bb)).abs() <= 1e-12 * (ba.abs() + bb.abs()).max(1e-300));
        let scaled = bps(k * a, n).unwrap();
        prop_assert!((scaled - k * ba).abs() <= 1e-12 * (k * ba).abs());
    }

    #[test]
    fn ac_params_from_market_reproduces_calibration(
        price in 1.0f64..1e3, spread_frac in 1e-5f64..1e-2, adv in 1e5f64..1e8,
        vol in 0.002f64..0.08, temp in 0.001f64..=1.0, perm in 0.001f64..=1.0,
        n in 1usize..50, tau in 0.01f64..1.0,
    ) {
        let spread = price * spread_frac;
        let m = ACMarket {
            price, spread, adv, daily_vol: vol, total_shares: 1e5, n_intervals: n, tau,
            lambda_ra: 1e-6, temp_frac: temp, perm_frac: perm,
        };
        match ac_params_from_market(&m) {
            Ok(p) => {
                prop_assert!(close(p.sigma(), price * vol, 1e-12, 0.0) || (p.sigma() / (price * vol) - 1.0).abs() <= 1e-9);
                prop_assert!((p.eta() / (spread / (temp * adv)) - 1.0).abs() <= 1e-9);
                prop_assert!((p.gamma_p() / (spread / (perm * adv)) - 1.0).abs() <= 1e-9);
            }
            // only the model-validity condition may reject valid market inputs
            Err(e) => prop_assert!(matches!(e, EtradeError::ModelInvalid(_)), "{:?}", e),
        }
    }

    #[test]
    fn as_params_from_market_defining_properties(
        price in 1.0f64..1e3, tick_frac in 1e-5f64..1e-2, vol in 0.002f64..0.08,
        touch in 1.0f64..1e4, halving in 0.1f64..10.0, skew in 0.01f64..10.0,
        dt_s in prop::sample::select(vec![1.0, 2.0, 5.0, 10.0, 30.0, 60.0]),
    ) {
        let tick = price * tick_frac;
        let m = ASMarket {
            price, tick, daily_vol: vol, touch_fills_per_day: touch, halving_ticks: halving,
            skew_ticks: skew, q_max: 10, liq_ticks: 1.0, dt_seconds: dt_s,
            ..ASMarket::default()
        };
        let p = as_params_from_market(&m).unwrap();
        let rel = |a: f64, b: f64| (a / b - 1.0).abs() <= 1e-9;
        prop_assert!(rel(intensity(&p, tick / 2.0).unwrap(), touch));
        let d = tick / 2.0;
        prop_assert!(rel(intensity(&p, d + halving * tick).unwrap(), intensity(&p, d).unwrap() / 2.0));
        let shift = price - reservation_price(&p, price, 1.0, 0.0).unwrap();
        prop_assert!((shift - skew * tick).abs() <= 1e-12 * price + 1e-9 * skew * tick);
    }
}

// ------------------------------------------------------------ validation

fn nasty() -> impl Strategy<Value = f64> {
    prop_oneof![
        Just(f64::NAN),
        Just(f64::INFINITY),
        Just(f64::NEG_INFINITY),
        Just(0.0),
        Just(-0.0),
        Just(1e308),
        Just(-1e308),
        Just(f64::MIN_POSITIVE),
        Just(1e-300),
        -10.0f64..10.0,
        -1e6f64..1e6,
    ]
}

fn nasty_usize() -> impl Strategy<Value = usize> {
    prop_oneof![
        Just(0usize),
        Just(1),
        1usize..300,
        Just(10_000_001),
        Just(usize::MAX)
    ]
}

/// Any value is fine; only a panic (caught by proptest) fails the test.
fn ok_or_err<T>(_r: Result<T, EtradeError>) {}

proptest! {
    #![proptest_config(cfg(256))]

    #[test]
    fn validation_never_panics(
        a in nasty(), b in nasty(), c in nasty(), d in nasty(), e in nasty(),
        f in nasty(), g in nasty(), h in nasty(), n in nasty_usize(), q in 0u32..5,
        fq in prop::collection::vec(nasty(), 0..4), fp in prop::collection::vec(nasty(), 0..4),
    ) {
        if let Ok(p) = ACParams::new(a, n, b, c, d, e, f) {
            prop_assert!(ac_trajectory(&p).iter().all(|v| v.is_finite()));
        }
        ok_or_err(ASParams::new(a, b, c, d, e, f, g, q, h));
        // POV: bound the clip so a valid huge path stays cheap
        if (a / (d.abs() * c.abs() * b.abs())).abs() < 1e5 || !a.is_finite() {
            ok_or_err(pov_holdings(a, b, c, d));
        }
        if n <= 300 || n > 10_000_000 {
            ok_or_err(twap_holdings(a, n));
        }
        let side = if q % 2 == 0 { Side::Buy } else { Side::Sell };
        ok_or_err(decompose_shortfall(side, a, b, c, d, &fq, &fp));
        ok_or_err(bps(a, b));
        ok_or_err(ac_params_from_market(&ACMarket {
            price: a, spread: b, adv: c, daily_vol: d, total_shares: e, n_intervals: n,
            tau: f, lambda_ra: g, temp_frac: h, perm_frac: h,
        }));
        ok_or_err(as_params_from_market(&ASMarket {
            price: a, tick: b, daily_vol: c, touch_fills_per_day: d, halving_ticks: e,
            skew_ticks: f, q_max: q, liq_ticks: g, dt_seconds: h, session_seconds: a,
            horizon_days: b,
        }));
        let nn = n.min(400);
        ok_or_err(ImpactParams::new(a, n, b, c, d, e, f, g, h, None));
        ok_or_err(ImpactParams::new(a, nn, b, c, d, e, f, g, h, Some(fq.clone())));
        ok_or_err(impact_params_from_market(&ImpactMarket {
            price: a, spread: b, adv: c, daily_vol: d, total_shares: e, n_intervals: nn,
            tau: f, lambda_ra: g, alpha: h, y: a, perm_frac: b, drift_bps_per_day: c,
            weights: None,
        }));
        ok_or_err(sqrt_law_cost_bps(a, b, c, d));
    }
}

// ------------------------------------------------------------ product bounds

#[test]
fn mc_product_bound_rejects_just_over() {
    // 10_000_000 paths * 101 intervals = 1.01e9 > 1e9: rejected before any draw.
    let p = ACParams::new(1e6, 101, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6).unwrap();
    let err = simulate_is(&p, None, 10_000_000, 1).unwrap_err();
    assert_eq!(
        err,
        EtradeError::InvalidInput("n_paths * n_intervals must be <= 1000000000".to_string())
    );
    // the same bound applies to a supplied schedule (N = its intervals)
    let twap = twap_holdings(1e6, 101).unwrap();
    assert!(simulate_is(&p, Some(&twap), 10_000_000, 1).is_err());
    assert_eq!(MAX_MC_WORK, 1_000_000_000);
    // accepted at a small size
    let small = ACParams::new(1e6, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6).unwrap();
    assert_eq!(simulate_is(&small, None, 10, 1).unwrap().n_paths, 10);
}

#[test]
fn as_product_bound_rejects_just_over() {
    // 10_000_000 paths * 1001 steps = 1.001e10 > 1e10.
    let p = ASParams::new(100.0, 2.0, 1001.0, 1.0, 0.1, 1.5, 140.0, 10, 0.0).unwrap();
    assert_eq!(p.n_steps(), 1001);
    let err = simulate(&p, 10_000_000, 1, &SimConfig::default()).unwrap_err();
    assert_eq!(
        err,
        EtradeError::InvalidInput("n_paths * n_steps must be <= 10000000000".to_string())
    );
    assert_eq!(MAX_AS_WORK, 10_000_000_000);
    let small = ASParams::new(100.0, 2.0, 1.0, 0.1, 0.1, 1.5, 140.0, 10, 0.0).unwrap();
    assert_eq!(
        simulate(&small, 3, 1, &SimConfig::default())
            .unwrap()
            .n_paths(),
        3
    );
}
