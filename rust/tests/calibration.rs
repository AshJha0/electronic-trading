//! Calibration estimators (API_SPEC.md Section 11): every case of
//! `data/golden/calibration_golden.json` (one test per case), the defining
//! properties, the round trip into the market-unit layer, and validation.

use std::fs;
use std::path::Path;

use serde_json::Value;

use etrade::calibration::{
    fill_intensity, spread_stats, volatility_curve, volume_curve, FITTED_VOLUME_CURVE_13,
};
use etrade::market::{as_params_from_market, ASMarket, DEFAULT_SESSION_SECONDS};

fn load(rel: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../data")
        .join(rel);
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).expect("valid JSON")
}

fn num(obj: &Value, key: &str) -> f64 {
    obj.get(key)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("missing key '{key}'"))
}

fn array(inp: &Value, name: &str) -> Vec<f64> {
    (0..num(inp, "n") as usize)
        .map(|i| num(inp, &format!("{name}_{i}")))
        .collect()
}

fn indexed(prefix: &str, values: &[f64]) -> Vec<(String, f64)> {
    values
        .iter()
        .enumerate()
        .map(|(j, &v)| (format!("{prefix}_{j:02}"), v))
        .collect()
}

fn actual(case: &Value) -> Vec<(String, f64)> {
    let inp = &case["inputs"];
    let kind = case["kind"].as_str().unwrap();
    match kind {
        "cal_volume_curve" => {
            let w = volume_curve(
                &array(inp, "times"),
                &array(inp, "sizes"),
                num(inp, "session_start"),
                num(inp, "session_end"),
                num(inp, "n_bins") as usize,
            )
            .unwrap();
            indexed("frac", &w)
        }
        "cal_volatility" => {
            let v = volatility_curve(
                &array(inp, "times"),
                &array(inp, "mids"),
                num(inp, "session_start"),
                num(inp, "session_end"),
                num(inp, "n_bins") as usize,
            )
            .unwrap();
            let mut out = vec![
                ("daily_vol".to_string(), v.daily_vol),
                ("n_returns".to_string(), v.n_returns as f64),
            ];
            out.extend(indexed("bin_vol", &v.bin_vol));
            out
        }
        "cal_spread" => {
            let s = spread_stats(
                &array(inp, "spreads"),
                &array(inp, "mids"),
                num(inp, "tick"),
            )
            .unwrap();
            vec![
                ("mean_ticks".to_string(), s.mean_ticks),
                ("median_ticks".to_string(), s.median_ticks),
                ("mean_bps".to_string(), s.mean_bps),
                ("median_bps".to_string(), s.median_bps),
                ("n".to_string(), s.n as f64),
            ]
        }
        "cal_fill_intensity" => {
            let f = fill_intensity(
                &array(inp, "distances"),
                num(inp, "tick"),
                num(inp, "n_levels") as usize,
                num(inp, "days"),
                num(inp, "sides"),
            )
            .unwrap();
            vec![
                ("a_rate".to_string(), f.a_rate),
                ("k".to_string(), f.k),
                ("touch_fills_per_day".to_string(), f.touch_fills_per_day),
                ("halving_ticks".to_string(), f.halving_ticks),
                ("n_buckets".to_string(), f.n_buckets as f64),
            ]
        }
        "cal_fitted_curve" => indexed("w", &FITTED_VOLUME_CURVE_13),
        other => panic!("unknown kind {other}"),
    }
}

fn check_case(name: &str) {
    let golden = load("golden/calibration_golden.json");
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
    golden_volume_curve_edges => "cal_volume_curve_edges",
    golden_volume_curve_13_bins => "cal_volume_curve_13_bins",
    golden_volatility_edges => "cal_volatility_edges",
    golden_volatility_13_bins => "cal_volatility_13_bins",
    golden_spread_odd => "cal_spread_odd",
    golden_spread_even_effective => "cal_spread_even_effective",
    golden_fill_intensity_exponential => "cal_fill_intensity_exponential",
    golden_fill_intensity_two_days => "cal_fill_intensity_two_days",
    golden_fitted_curve_13 => "cal_fitted_curve_13",
}

#[test]
fn golden_count() {
    let golden = load("golden/calibration_golden.json");
    assert_eq!(golden["cases"].as_array().unwrap().len(), 9);
}

#[test]
fn fitted_curve_sums_to_one_and_matches_fit() {
    let w = FITTED_VOLUME_CURVE_13;
    let mut total = 0.0;
    for &v in &w {
        total += v;
    }
    assert!((total - 1.0).abs() <= 1e-12);
    let fitted = load("calibration/fitted_20191230.json");
    let want: Vec<f64> = fitted["fitted_volume_curve_13"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_eq!(w.to_vec(), want);
    let x = etrade::impact::vwap_holdings(1000.0, &w).unwrap();
    assert_eq!(x[0], 1000.0);
    assert_eq!(*x.last().unwrap(), 0.0);
}

#[test]
fn volume_curve_is_a_distribution() {
    let w = volume_curve(
        &[1.0, 2.0, 3.0, 7.0],
        &[10.0, 20.0, 30.0, 40.0],
        0.0,
        8.0,
        2,
    )
    .unwrap();
    assert!((w[0] / 0.6 - 1.0).abs() <= 1e-15);
    assert!((w[1] / 0.4 - 1.0).abs() <= 1e-15);
}

#[test]
fn volatility_of_a_constant_return() {
    let r = 0.001;
    let times: Vec<f64> = (0..11).map(f64::from).collect();
    let mids: Vec<f64> = (0..11).map(|i| 100.0 * (r * f64::from(i)).exp()).collect();
    let v = volatility_curve(&times, &mids, 0.0, 10.0, 1).unwrap();
    assert!((v.daily_vol / (r * 10.0_f64.sqrt()) - 1.0).abs() <= 1e-9);
    assert_eq!(v.n_returns, 10);
}

fn exponential_distances(tick: f64) -> Vec<f64> {
    let counts = [1024usize, 512, 256, 128, 64];
    let mut dist = Vec::new();
    for (b, &c) in counts.iter().enumerate() {
        let nxt = counts.get(b + 1).copied().unwrap_or(0);
        dist.extend(vec![(b as f64 + 0.5) * tick; c - nxt]);
    }
    dist
}

#[test]
fn fill_intensity_recovers_an_exact_exponential() {
    let (tick, q) = (0.01, 0.5_f64);
    let f = fill_intensity(&exponential_distances(tick), tick, 5, 1.0, 2.0).unwrap();
    assert!((f.k / (-q.ln() / tick) - 1.0).abs() <= 1e-12);
    assert!((f.halving_ticks - 1.0).abs() <= 1e-12);
    assert!((f.touch_fills_per_day / 512.0 - 1.0).abs() <= 1e-12);
    assert_eq!(f.n_buckets, 5);
}

#[test]
fn fill_intensity_round_trips_through_the_market_layer() {
    let golden = load("golden/calibration_golden.json");
    let case = golden["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["kind"] == "cal_fill_intensity")
        .unwrap();
    let inp = &case["inputs"];
    let tick = num(inp, "tick");
    let f = fill_intensity(
        &array(inp, "distances"),
        tick,
        num(inp, "n_levels") as usize,
        1.0,
        2.0,
    )
    .unwrap();
    let a = as_params_from_market(&ASMarket {
        price: 100.0,
        tick,
        daily_vol: 0.01,
        touch_fills_per_day: f.touch_fills_per_day,
        halving_ticks: f.halving_ticks,
        skew_ticks: 1.0,
        q_max: 10,
        liq_ticks: 1.0,
        dt_seconds: 5.0,
        session_seconds: DEFAULT_SESSION_SECONDS,
        horizon_days: 1.0,
    })
    .unwrap();
    assert!((a.k() / f.k - 1.0).abs() <= 1e-12);
    assert!((a.a_rate() / f.a_rate - 1.0).abs() <= 1e-12);
}

#[test]
fn validation_messages_match_python() {
    type Call = Box<dyn Fn() -> Result<(), etrade::EtradeError>>;
    let cases: Vec<(Call, &str)> = vec![
        (
            Box::new(|| volume_curve(&[0.0], &[1.0], 0.0, 0.0, 1).map(drop)),
            "session_end must be finite and > session_start",
        ),
        (
            Box::new(|| volume_curve(&[0.0], &[1.0], 0.0, 1.0, 0).map(drop)),
            "n_bins must be an integer",
        ),
        (
            Box::new(|| volume_curve(&[0.0], &[1.0, 2.0], 0.0, 1.0, 1).map(drop)),
            "times and sizes must have the same length",
        ),
        (
            Box::new(|| volume_curve(&[0.0], &[-1.0], 0.0, 1.0, 1).map(drop)),
            "sizes must be finite and >= 0",
        ),
        (
            Box::new(|| volume_curve(&[5.0], &[1.0], 0.0, 1.0, 1).map(drop)),
            "no volume inside the session",
        ),
        (
            Box::new(|| volatility_curve(&[0.0, 1.0], &[1.0, 0.0], 0.0, 1.0, 1).map(drop)),
            "mids must be finite and > 0",
        ),
        (
            Box::new(|| volatility_curve(&[1.0, 0.0], &[1.0, 1.0], 0.0, 1.0, 1).map(drop)),
            "times must be non-decreasing",
        ),
        (
            Box::new(|| volatility_curve(&[0.0], &[1.0], 0.0, 1.0, 1).map(drop)),
            "no return inside the session",
        ),
        (
            Box::new(|| spread_stats(&[], &[], 0.01).map(drop)),
            "spreads must not be empty",
        ),
        (
            Box::new(|| spread_stats(&[0.01], &[100.0], 0.0).map(drop)),
            "tick must be finite and > 0",
        ),
        (
            Box::new(|| spread_stats(&[-0.01], &[100.0], 0.01).map(drop)),
            "spreads must be finite and >= 0",
        ),
        (
            Box::new(|| fill_intensity(&[0.005], 0.01, 1, 1.0, 2.0).map(drop)),
            "n_levels must be an integer",
        ),
        (
            Box::new(|| fill_intensity(&[0.005], 0.01, 3, 0.0, 2.0).map(drop)),
            "days must be finite and > 0",
        ),
        (
            Box::new(|| fill_intensity(&[0.005], 0.01, 3, 1.0, 0.0).map(drop)),
            "sides must be finite and > 0",
        ),
        (
            Box::new(|| fill_intensity(&[-1.0], 0.01, 3, 1.0, 2.0).map(drop)),
            "distances must be finite and >= 0",
        ),
        (
            Box::new(|| fill_intensity(&[0.005], 0.01, 3, 1.0, 2.0).map(drop)),
            "fewer than two distance buckets have fills",
        ),
        (
            Box::new(|| fill_intensity(&[0.015, 0.015], 0.01, 3, 1.0, 2.0).map(drop)),
            "fitted fill intensity does not decay with distance",
        ),
    ];
    for (call, msg) in cases {
        let err = call().unwrap_err();
        assert!(err.to_string().contains(msg), "{msg}: {err}");
    }
}
