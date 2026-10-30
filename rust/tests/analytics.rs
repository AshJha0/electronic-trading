//! Execution-analytics property and error-path tests.

use std::str::FromStr;

use etrade::Side;

#[test]
fn is_decomposition_identity_holds_for_both_sides() {
    // total == delay + trading + opportunity == sign*(X*Pd - sum n*p - R*PT),
    // checked over a small grid of fill patterns (property-style loop).
    let fills: [(&[f64], &[f64]); 3] = [
        (&[300.0, 400.0, 200.0], &[99.4, 99.2, 99.1]),
        (&[1000.0], &[99.0]),
        (&[0.0, 250.0], &[98.0, 99.9]),
    ];
    for side in [Side::Sell, Side::Buy] {
        let sign = if side == Side::Sell { 1.0 } else { -1.0 };
        for (qtys, prices) in fills {
            let (pd, p0, pt, x) = (100.0, 99.5, 98.8, 1000.0);
            let d = etrade::decompose_shortfall(side, pd, p0, pt, x, qtys, prices).unwrap();
            let sum_np: f64 = qtys.iter().zip(prices).map(|(n, p)| n * p).sum();
            let direct = sign * (x * pd - sum_np - d.remaining_qty * pt);
            assert!((d.total - (d.delay + d.trading + d.opportunity)).abs() < 1e-10);
            assert!(
                (d.total - direct).abs() < 1e-10,
                "identity violated: {} vs {}",
                d.total,
                direct
            );
            assert!((d.executed_qty + d.remaining_qty - x).abs() < 1e-12);
        }
    }
}

#[test]
fn buy_and_sell_mirror_each_other() {
    let d_sell =
        etrade::decompose_shortfall(Side::Sell, 100.0, 99.5, 98.8, 1000.0, &[500.0], &[99.3])
            .unwrap();
    // Mirrored prices: a buy with the same magnitudes of adverse moves.
    let d_buy =
        etrade::decompose_shortfall(Side::Buy, 100.0, 100.5, 101.2, 1000.0, &[500.0], &[100.7])
            .unwrap();
    assert!((d_sell.delay - d_buy.delay).abs() < 1e-10);
    assert!((d_sell.trading - d_buy.trading).abs() < 1e-10);
    assert!((d_sell.opportunity - d_buy.opportunity).abs() < 1e-10);
}

#[test]
fn participation_rate_basic_and_errors() {
    assert_eq!(etrade::participation_rate(120_000.0, 1_500_000.0).unwrap(), 0.08);
    assert_eq!(etrade::participation_rate(0.0, 1.0).unwrap(), 0.0);
    assert!(etrade::participation_rate(-1.0, 1.0).is_err());
    assert!(etrade::participation_rate(1.0, 0.0).is_err());
    assert!(etrade::participation_rate(1.0, f64::INFINITY).is_err());
}

#[test]
fn ols_recovers_exact_line() {
    let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|v| 2.5 * v - 1.0).collect();
    let r = etrade::ols(&x, &y).unwrap();
    assert!((r.slope - 2.5).abs() < 1e-12);
    assert!((r.intercept + 1.0).abs() < 1e-12);
    assert!((r.r_squared - 1.0).abs() < 1e-12);
    assert_eq!(r.n_obs, 10);
    // Constant y: SST = 0 -> R^2 defined as 1.
    let y_const = vec![3.0; 10];
    assert_eq!(etrade::ols(&x, &y_const).unwrap().r_squared, 1.0);
}

#[test]
fn synthetic_fills_are_pinned() {
    let (part, cost) = etrade::synthetic_impact_fills(25, 0.5, 8.0, 0.05).unwrap();
    assert_eq!(part.len(), 25);
    assert!((part[0] - 0.02).abs() < 1e-15);
    assert!((part[24] - 0.20).abs() < 1e-15);
    let expected_c0 = 0.5 + 8.0 * 0.02 + 0.05 * (0.3_f64).sin();
    assert!((cost[0] - expected_c0).abs() < 1e-15);
    // Regression slope close to the true slope despite the sine noise.
    let reg = etrade::estimate_impact(&part, &cost).unwrap();
    assert!((reg.slope - 8.0).abs() < 0.1);
    assert!(reg.r_squared > 0.99);
}

#[test]
fn error_paths_are_reported_not_panicked() {
    // Side parsing.
    assert_eq!(Side::from_str("sell").unwrap(), Side::Sell);
    assert_eq!(Side::from_str("buy").unwrap(), Side::Buy);
    assert!(Side::from_str("hold").is_err());
    // Decomposition errors.
    let ok_q = [500.0];
    let ok_p = [99.0];
    assert!(etrade::decompose_shortfall(Side::Sell, f64::NAN, 99.5, 98.8, 1000.0, &ok_q, &ok_p)
        .is_err());
    assert!(etrade::decompose_shortfall(Side::Sell, 100.0, 99.5, 98.8, -1.0, &ok_q, &ok_p)
        .is_err());
    assert!(etrade::decompose_shortfall(
        Side::Sell,
        100.0,
        99.5,
        98.8,
        1000.0,
        &[500.0, 100.0],
        &ok_p
    )
    .is_err()); // length mismatch
    assert!(etrade::decompose_shortfall(Side::Sell, 100.0, 99.5, 98.8, 1000.0, &[-1.0], &ok_p)
        .is_err()); // negative fill
    assert!(etrade::decompose_shortfall(Side::Sell, 100.0, 99.5, 98.8, 100.0, &ok_q, &ok_p)
        .is_err()); // over-execution
    assert!(etrade::decompose_shortfall(
        Side::Sell,
        100.0,
        99.5,
        98.8,
        1000.0,
        &ok_q,
        &[f64::INFINITY]
    )
    .is_err());
    // OLS errors.
    assert!(etrade::ols(&[1.0], &[1.0]).is_err());
    assert!(etrade::ols(&[1.0, 2.0], &[1.0]).is_err());
    assert!(etrade::ols(&[1.0, 1.0], &[1.0, 2.0]).is_err()); // degenerate x
    assert!(etrade::ols(&[1.0, f64::NAN], &[1.0, 2.0]).is_err());
    assert!(etrade::synthetic_impact_fills(1, 0.5, 8.0, 0.05).is_err());
}

// ---------------------------------------------------------------------------
// Robustness tests: price-shift invariance, empty fills, participation above 1
// ---------------------------------------------------------------------------

#[test]
fn is_decomposition_price_shift_invariance() {
    // Adding c to every price leaves every component unchanged (Q + R = X).
    let c = 37.5;
    let qty = [300.0, 400.0, 200.0];
    let px = [99.4, 99.2, 99.1];
    let px_c: Vec<f64> = px.iter().map(|p| p + c).collect();
    for side in [Side::Sell, Side::Buy] {
        let d0 = etrade::decompose_shortfall(side, 100.0, 99.5, 98.8, 1000.0, &qty, &px).unwrap();
        let d1 = etrade::decompose_shortfall(side, 100.0 + c, 99.5 + c, 98.8 + c, 1000.0, &qty, &px_c)
            .unwrap();
        assert!((d0.total - d1.total).abs() < 1e-9);
        assert!((d0.delay - d1.delay).abs() < 1e-9);
        assert!((d0.trading - d1.trading).abs() < 1e-9);
        assert!((d0.opportunity - d1.opportunity).abs() < 1e-9);
    }
}

#[test]
fn is_decomposition_empty_fills() {
    let d = etrade::decompose_shortfall(Side::Sell, 100.0, 99.0, 98.0, 500.0, &[], &[]).unwrap();
    assert_eq!(d.delay, 0.0);
    assert_eq!(d.trading, 0.0);
    assert!((d.opportunity - 1000.0).abs() < 1e-12); // 500 * (100 - 98)
    assert!((d.total - 1000.0).abs() < 1e-12);
    assert_eq!(d.executed_qty, 0.0);
    assert_eq!(d.remaining_qty, 500.0);
    let b = etrade::decompose_shortfall(Side::Buy, 100.0, 99.0, 98.0, 500.0, &[], &[]).unwrap();
    assert!((b.opportunity + 1000.0).abs() < 1e-12); // price fell: a gain for a buyer
}

#[test]
fn participation_rate_above_one_is_error() {
    let err = etrade::participation_rate(2e6, 1.5e6).unwrap_err();
    assert!(err.to_string().contains("exceeds market volume"));
    assert_eq!(etrade::participation_rate(1.5e6, 1.5e6).unwrap(), 1.0); // boundary
    assert!(etrade::participation_rate(f64::NAN, 1.0).is_err());
}
