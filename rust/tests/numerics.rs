//! Standard-error formulas checked numerically (API_SPEC.md Sections 1.7,
//! 2.2 and 9.5): recomputed from the per-path arrays, plus closed-form
//! degenerate cases where every SE is exactly zero.

use etrade::mm::{simulate_mm, MMConfig, MMStat};
use etrade::{
    expected_cost, simulate, simulate_is, variance_cost, ACParams, ASParams, ASSimResult, ASStat,
    SimConfig,
};

fn rel_close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * b.abs().max(1e-300)
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn std1(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() as f64 - 1.0)).sqrt()
}

fn liquid_ac(sigma: f64) -> ACParams {
    ACParams::new(1e6, 5, 1.0, sigma, 2.5e-6, 2.5e-7, 2e-6).unwrap()
}

// ------------------------------------------------------------ Section 1.7

#[test]
fn ac_mc_se_formulas_match_returned_moments() {
    let p = liquid_ac(0.95);
    let m = 4000;
    let r = simulate_is(&p, None, m, 11).unwrap();
    let mf = m as f64;
    assert_eq!(r.n_paths, m);
    assert!(rel_close(r.se_mean, (r.var / mf).sqrt(), 1e-15));
    assert!(rel_close(
        r.se_var,
        r.var * (2.0 / (mf - 1.0)).sqrt(),
        1e-15
    ));
    // the SEs are the right size: closed forms within 4 SE
    let e = expected_cost(&p, None).unwrap();
    let v = variance_cost(&p, None).unwrap();
    assert!(
        (r.mean - e).abs() <= 4.0 * r.se_mean,
        "mean {} vs {e}",
        r.mean
    );
    assert!((r.var - v).abs() <= 4.0 * r.se_var, "var {} vs {v}", r.var);
}

#[test]
fn ac_mc_sigma_zero_is_deterministic() {
    let p = liquid_ac(0.0);
    let r = simulate_is(&p, None, 50, 3).unwrap();
    let e = expected_cost(&p, None).unwrap();
    assert!(rel_close(r.mean, e, 1e-12), "mean {} vs {e}", r.mean);
    assert!(r.var <= 1e-20 * e * e, "var {}", r.var);
    assert!(r.se_mean <= 1e-10 * e.abs());
    assert!(r.se_var <= 1e-20 * e * e);
}

// ------------------------------------------------------------ Section 2.2

fn as_params(sigma: f64, a_rate: f64, q_max: u32) -> ASParams {
    ASParams::new(100.0, sigma, 1.0, 0.005, 0.1, 1.5, a_rate, q_max, 0.5).unwrap()
}

fn check_as_se(r: &ASSimResult) {
    let m = r.n_paths() as f64;
    let pairs: [(ASStat, &[f64]); 5] = [
        (ASStat::MeanPnl, &r.pnl),
        (ASStat::MeanAbsQ, &r.mean_abs_q_path),
        (ASStat::MeanMaxAbsQ, &r.max_abs_q_path),
        (ASStat::RmsQ, &r.rms_q_path),
        (ASStat::MeanFills, &r.fills_path),
    ];
    for (stat, v) in pairs {
        let want = std1(v) / m.sqrt();
        assert!(
            (r.se(stat) - want).abs() <= 1e-13 * want.max(1e-300),
            "{stat:?}: {} vs {want}",
            r.se(stat)
        );
    }
    let want = r.std_pnl() / (2.0 * (m - 1.0)).sqrt();
    assert_eq!(r.se(ASStat::StdPnl), want);
    assert!(rel_close(r.std_pnl(), std1(&r.pnl), 1e-13));
}

#[test]
fn as_se_recomputed_from_per_path_arrays() {
    let r = simulate(&as_params(2.0, 140.0, 10), 300, 5, &SimConfig::default()).unwrap();
    assert!(r.se(ASStat::MeanPnl) > 0.0);
    check_as_se(&r);
}

#[test]
fn as_q_max_zero_quotes_nothing_all_se_zero() {
    let r = simulate(&as_params(2.0, 140.0, 0), 50, 5, &SimConfig::default()).unwrap();
    assert!(r.pnl.iter().all(|&v| v == 0.0));
    assert!(r.fills_path.iter().all(|&v| v == 0.0));
    for stat in [
        ASStat::MeanPnl,
        ASStat::StdPnl,
        ASStat::MeanAbsQ,
        ASStat::MeanMaxAbsQ,
        ASStat::RmsQ,
        ASStat::MeanFills,
    ] {
        assert_eq!(r.se(stat), 0.0, "{stat:?}");
    }
}

#[test]
fn as_saturating_fills_are_deterministic() {
    // sigma = 0 and A huge: P(fill) == 1 on both sides every step, so q
    // stays 0 and each step books the full spread.
    let p = as_params(0.0, 1e300, 5);
    let r = simulate(&p, 20, 9, &SimConfig::default()).unwrap();
    let n = p.n_steps() as f64;
    assert!(r.fills_path.iter().all(|&v| v == 2.0 * n));
    assert_eq!(r.se(ASStat::MeanFills), 0.0);
    assert_eq!(r.se(ASStat::MeanAbsQ), 0.0);
    // identical per-path values: SE is zero up to the rounding of the mean
    assert!(r.pnl.iter().all(|&v| v == r.pnl[0]));
    let scale = r.pnl[0].abs().max(1.0);
    assert!(r.se(ASStat::MeanPnl) <= 1e-14 * scale);
    assert!(r.se(ASStat::StdPnl) <= 1e-14 * scale);
    let spread = 2.0 / p.gamma() * (p.gamma() / p.k()).ln_1p();
    assert!(rel_close(r.mean_pnl(), n * spread, 1e-12));
    check_as_se(&r);
}

// ------------------------------------------------------------ Section 9.5

const MM_STATS: [MMStat; 9] = [
    MMStat::Pnl,
    MMStat::Spread,
    MMStat::Fees,
    MMStat::Inventory,
    MMStat::Liquidation,
    MMStat::Markout,
    MMStat::Fills,
    MMStat::InformedFills,
    MMStat::RmsQ,
];

#[test]
fn mm_se_recomputed_from_per_path_arrays() {
    let cfg = MMConfig {
        tick: 0.01,
        informed_prob: 0.1,
        maker_fee: 0.001,
        ..MMConfig::default()
    };
    let r = simulate_mm(&as_params(2.0, 140.0, 10), 300, 7, &cfg).unwrap();
    let m = r.n_paths() as f64;
    for stat in MM_STATS {
        let v = r.paths(stat);
        assert_eq!(v.len(), 300);
        let want = std1(v) / m.sqrt();
        assert!(
            (r.se(stat) - want).abs() <= 1e-13 * want.max(1e-300),
            "{stat:?}: {} vs {want}",
            r.se(stat)
        );
    }
    assert!(r.se(MMStat::Pnl) > 0.0);
}

#[test]
fn mm_q_max_zero_all_se_zero() {
    let cfg = MMConfig {
        informed_prob: 0.5,
        ..MMConfig::default()
    };
    let r = simulate_mm(&as_params(2.0, 140.0, 0), 40, 7, &cfg).unwrap();
    for stat in MM_STATS {
        assert_eq!(r.se(stat), 0.0, "{stat:?}");
        assert_eq!(r.mean(stat), 0.0, "{stat:?}");
    }
}

#[test]
fn mm_saturating_fills_stat() {
    let p = as_params(0.0, 1e300, 5);
    let r = simulate_mm(&p, 30, 7, &MMConfig::default()).unwrap();
    let n = p.n_steps() as f64;
    assert!(r.paths(MMStat::Fills).iter().all(|&v| v == 2.0 * n));
    assert_eq!(r.mean(MMStat::Fills), 2.0 * n);
    assert_eq!(r.se(MMStat::Fills), 0.0);
    let pnl = r.paths(MMStat::Pnl);
    assert!(pnl.iter().all(|&v| v == pnl[0]));
    assert!(r.se(MMStat::Pnl) <= 1e-14 * pnl[0].abs().max(1.0));
    assert!(r.se(MMStat::RmsQ) <= 1e-14);
}
