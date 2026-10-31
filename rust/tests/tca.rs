//! Post-trade TCA (API_SPEC.md Section 10): every case of
//! `data/golden/tca_golden.json` (one test per case), sign conventions,
//! invariances and validation.

use std::fs;
use std::path::Path;

use serde_json::Value;

use etrade::tca::{
    betainc_reg, ln_gamma, markouts, participation, slippage, student_t_cdf, welch_ttest,
};
use etrade::{decompose_shortfall, EtradeError, Side};

const VOLS: [f64; 4] = [1000.0, 2000.0, 1500.0, 500.0];
const VWAPS: [f64; 4] = [100.00, 100.05, 100.12, 100.08];
const BINS: [usize; 4] = [0, 1, 1, 2];
const QTYS: [f64; 4] = [100.0, 150.0, 50.0, 200.0];
const PXS: [f64; 4] = [100.02, 100.06, 100.07, 100.15];

fn load() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/golden/tca_golden.json");
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).expect("valid JSON")
}

fn num(obj: &Value, key: &str) -> f64 {
    obj.get(key)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("missing key '{key}'"))
}

fn arr(inp: &Value, prefix: &str, n_key: &str) -> Vec<f64> {
    (0..num(inp, n_key) as usize)
        .map(|i| num(inp, &format!("{prefix}_{i}")))
        .collect()
}

fn idx(inp: &Value, prefix: &str, n_key: &str) -> Vec<usize> {
    arr(inp, prefix, n_key)
        .into_iter()
        .map(|v| v as usize)
        .collect()
}

fn side_of(inp: &Value) -> Side {
    if num(inp, "side_buy") == 1.0 {
        Side::Buy
    } else {
        Side::Sell
    }
}

fn indexed(prefix: &str, xs: &[f64]) -> Vec<(String, f64)> {
    xs.iter()
        .enumerate()
        .map(|(i, &v)| (format!("{prefix}_{i}"), v))
        .collect()
}

fn actual(case: &Value) -> Vec<(String, f64)> {
    let inp = &case["inputs"];
    match case["kind"].as_str().unwrap() {
        "slippage" => {
            let r = slippage(
                side_of(inp),
                num(inp, "arrival_price"),
                num(inp, "close_price"),
                &idx(inp, "bin", "n_fills"),
                &arr(inp, "qty", "n_fills"),
                &arr(inp, "px", "n_fills"),
                &arr(inp, "vol", "n_bins"),
                &arr(inp, "vwap", "n_bins"),
            )
            .unwrap();
            [
                ("executed_qty", r.executed_qty),
                ("avg_price", r.avg_price),
                ("market_vwap", r.market_vwap),
                ("interval_vwap", r.interval_vwap),
                ("arrival_bps", r.arrival_bps),
                ("vwap_bps", r.vwap_bps),
                ("interval_vwap_bps", r.interval_vwap_bps),
                ("close_bps", r.close_bps),
            ]
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect()
        }
        "markout" => {
            let m = markouts(
                side_of(inp),
                &idx(inp, "step", "n_fills"),
                &arr(inp, "qty", "n_fills"),
                &arr(inp, "mid", "n_mids"),
                &idx(inp, "h", "n_horizons"),
            )
            .unwrap();
            let mut out = indexed("per_fill", m.per_fill());
            out.extend(indexed("per_share", m.per_share()));
            out
        }
        "participation" => {
            let p = participation(
                &idx(inp, "bin", "n_fills"),
                &arr(inp, "qty", "n_fills"),
                &arr(inp, "vol", "n_bins"),
            )
            .unwrap();
            let mut out = indexed("rate", p.per_bin());
            out.push(("overall".to_string(), p.overall));
            out
        }
        "welch" => {
            let w = welch_ttest(&arr(inp, "a", "n_a"), &arr(inp, "b", "n_b")).unwrap();
            vec![
                ("mean_diff".to_string(), w.mean_diff),
                ("t_stat".to_string(), w.t_stat),
                ("df".to_string(), w.df),
                ("p_value".to_string(), w.p_value),
            ]
        }
        "t_cdf" => vec![(
            "cdf".to_string(),
            student_t_cdf(num(inp, "t"), num(inp, "df")).unwrap(),
        )],
        other => panic!("unknown kind {other}"),
    }
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
    golden_slippage_buy => "slippage_buy",
    golden_slippage_sell => "slippage_sell",
    golden_markout_buy => "markout_buy",
    golden_markout_sell => "markout_sell",
    golden_participation => "participation_with_empty_bin",
    golden_welch_unequal => "welch_unequal_variance",
    golden_welch_constant => "welch_constant_sample",
    golden_t_cdf_cauchy => "t_cdf_cauchy",
    golden_t_cdf_fractional => "t_cdf_negative_fractional_df",
    golden_t_cdf_df30 => "t_cdf_df30",
    golden_t_cdf_small_df => "t_cdf_small_df",
}

#[test]
fn golden_count() {
    assert_eq!(load()["cases"].as_array().unwrap().len(), 11);
}

#[test]
fn buy_sell_symmetry() {
    let b = slippage(Side::Buy, 100.01, 100.09, &BINS, &QTYS, &PXS, &VOLS, &VWAPS).unwrap();
    let s = slippage(
        Side::Sell,
        100.01,
        100.09,
        &BINS,
        &QTYS,
        &PXS,
        &VOLS,
        &VWAPS,
    )
    .unwrap();
    assert!((b.arrival_bps + s.arrival_bps).abs() < 1e-12);
    assert!((b.vwap_bps + s.vwap_bps).abs() < 1e-12);
    assert!((b.interval_vwap_bps + s.interval_vwap_bps).abs() < 1e-12);
    assert!((b.close_bps + s.close_bps).abs() < 1e-12);
}

#[test]
fn price_scale_and_shift_invariance() {
    let (c, k) = (3.7, 25.0);
    let r1 = slippage(Side::Buy, 100.01, 100.09, &BINS, &QTYS, &PXS, &VOLS, &VWAPS).unwrap();
    let px_c: Vec<f64> = PXS.iter().map(|p| p * c).collect();
    let vw_c: Vec<f64> = VWAPS.iter().map(|p| p * c).collect();
    let r2 = slippage(
        Side::Buy,
        100.01 * c,
        100.09 * c,
        &BINS,
        &QTYS,
        &px_c,
        &VOLS,
        &vw_c,
    )
    .unwrap();
    assert!((r2.arrival_bps / r1.arrival_bps - 1.0).abs() < 1e-10);
    assert!((r2.vwap_bps / r1.vwap_bps - 1.0).abs() < 1e-10);
    let px_k: Vec<f64> = PXS.iter().map(|p| p + k).collect();
    let vw_k: Vec<f64> = VWAPS.iter().map(|p| p + k).collect();
    let r3 = slippage(
        Side::Buy,
        100.01 + k,
        100.09 + k,
        &BINS,
        &QTYS,
        &px_k,
        &VOLS,
        &vw_k,
    )
    .unwrap();
    assert!((r3.arrival_bps * (100.01 + k) - r1.arrival_bps * 100.01).abs() < 1e-5);
    assert!((r3.vwap_bps * r3.market_vwap - r1.vwap_bps * r1.market_vwap).abs() < 1e-5);
    assert!((r3.close_bps * (100.09 + k) - r1.close_bps * 100.09).abs() < 1e-5);
}

#[test]
fn arrival_slippage_is_perold_trading_cost() {
    let r = slippage(Side::Buy, 100.01, 100.09, &BINS, &QTYS, &PXS, &VOLS, &VWAPS).unwrap();
    let d = decompose_shortfall(Side::Buy, 100.01, 100.01, 100.09, 500.0, &QTYS, &PXS).unwrap();
    assert!((r.arrival_bps * 100.01 * r.executed_qty / 1e4 - d.trading).abs() < 1e-9);
}

#[test]
fn markout_sign() {
    let mids = [10.0, 10.5, 9.0];
    let buy = markouts(Side::Buy, &[0, 1], &[1.0, 3.0], &mids, &[0, 1]).unwrap();
    assert_eq!(buy.per_fill()[0], 0.0);
    assert!((buy.per_fill()[1] + 0.5).abs() < 1e-15);
    assert!((buy.per_share()[1] + 1.0).abs() < 1e-15);
    let sell = markouts(Side::Sell, &[0, 1], &[1.0, 3.0], &mids, &[0, 1]).unwrap();
    assert!((sell.per_fill()[1] - 0.5).abs() < 1e-15);
}

#[test]
fn known_t_values() {
    assert!((student_t_cdf(1.0, 1.0).unwrap() - 0.75).abs() < 1e-14);
    assert_eq!(student_t_cdf(0.0, 4.0).unwrap(), 0.5);
    let want = 0.5 + 1.5 / (2.0 * 4.25_f64.sqrt());
    assert!((student_t_cdf(1.5, 2.0).unwrap() - want).abs() < 1e-14);
    assert!((ln_gamma(0.5).unwrap() - 0.5 * std::f64::consts::PI.ln()).abs() < 1e-14);
    assert!((ln_gamma(10.0).unwrap() - 362_880.0_f64.ln()).abs() < 1e-13);
}

fn msg<T: std::fmt::Debug>(r: Result<T, EtradeError>) -> String {
    match r {
        Err(EtradeError::InvalidInput(m)) => m,
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn validation_names_the_argument() {
    assert!(msg(slippage(
        Side::Buy,
        0.0,
        1.0,
        &BINS,
        &QTYS,
        &PXS,
        &VOLS,
        &VWAPS
    ))
    .contains("arrival_price"));
    assert!(msg(slippage(
        Side::Buy,
        1.0,
        1.0,
        &[0, 1, 1, 4],
        &QTYS,
        &PXS,
        &VOLS,
        &VWAPS
    ))
    .contains("fill_bins must be integers in [0, 4)"));
    assert!(msg(slippage(
        Side::Buy,
        1.0,
        1.0,
        &BINS,
        &[0.0; 4],
        &PXS,
        &VOLS,
        &VWAPS
    ))
    .contains("executed quantity"));
    assert!(msg(slippage(
        Side::Buy,
        1.0,
        1.0,
        &BINS,
        &QTYS,
        &PXS,
        &[0.0, 0.0, 0.0, 1.0],
        &VWAPS
    ))
    .contains("traded bins"));
    assert!(msg(slippage(Side::Buy, 1.0, 1.0, &[], &[], &[], &VOLS, &VWAPS)).contains("non-empty"));
    assert!(
        msg(markouts(Side::Buy, &[0, 1], &[1.0, 1.0], &[1.0, 1.0], &[1]))
            .contains("exceeds mid series")
    );
    assert!(msg(participation(&[0], &[1001.0], &VOLS)).contains("exceeds market volume"));
    assert!(msg(participation(&[0], &[0.0], &[0.0])).contains("total market volume"));
    assert!(msg(welch_ttest(&[1.0], &[1.0, 2.0])).contains("a must have at least 2"));
    assert!(msg(welch_ttest(&[1.0, 1.0], &[2.0, 2.0])).contains("zero variance"));
    assert!(msg(student_t_cdf(1.0, 0.0)).contains("df"));
    assert!(msg(betainc_reg(1.0, 1.0, 1.5)).contains("x must be in"));
    assert!(msg(ln_gamma(-1.0)).contains("x must be finite"));
}
