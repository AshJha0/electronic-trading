//! Market-unit layer (API_SPEC.md Section 7): every case of
//! `data/golden/market_golden.json` (one test per case), the paper identity,
//! the defining properties of the A-S conversion, and validation.

use std::fs;
use std::path::Path;

use serde_json::Value;

use etrade::market::{
    ac_params_from_market, as_params_from_market, bps, half_spread_cost, ACMarket, ASMarket,
    DEFAULT_PERM_FRAC, DEFAULT_SESSION_SECONDS, DEFAULT_TEMP_FRAC,
};

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

fn num_or(obj: &Value, key: &str, fallback: f64) -> f64 {
    obj.get(key).and_then(Value::as_f64).unwrap_or(fallback)
}

fn ac_from(v: &Value) -> ACMarket {
    ACMarket {
        price: num(v, "price"),
        spread: num(v, "spread"),
        adv: num(v, "adv"),
        daily_vol: num(v, "daily_vol"),
        total_shares: num(v, "total_shares"),
        n_intervals: num(v, "n_intervals") as usize,
        tau: num(v, "tau"),
        lambda_ra: num(v, "lambda_ra"),
        temp_frac: num_or(v, "temp_frac", DEFAULT_TEMP_FRAC),
        perm_frac: num_or(v, "perm_frac", DEFAULT_PERM_FRAC),
    }
}

fn as_from(v: &Value) -> ASMarket {
    ASMarket {
        price: num(v, "price"),
        tick: num(v, "tick"),
        daily_vol: num(v, "daily_vol"),
        touch_fills_per_day: num(v, "touch_fills_per_day"),
        halving_ticks: num(v, "halving_ticks"),
        skew_ticks: num(v, "skew_ticks"),
        q_max: num(v, "q_max") as u32,
        liq_ticks: num(v, "liq_ticks"),
        dt_seconds: num(v, "dt_seconds"),
        session_seconds: num_or(v, "session_seconds", DEFAULT_SESSION_SECONDS),
        horizon_days: num_or(v, "horizon_days", 1.0),
    }
}

fn actual(case: &Value) -> Vec<(&'static str, f64)> {
    let inp = &case["inputs"];
    if case["kind"] == "market_ac" {
        let m = ac_from(inp);
        let p = ac_params_from_market(&m).unwrap();
        let notional = m.total_shares * m.price;
        let e = etrade::expected_cost(&p, None).unwrap();
        let sd = etrade::variance_cost(&p, None).unwrap().sqrt();
        let hs = half_spread_cost(m.spread, m.total_shares).unwrap();
        vec![
            ("sigma", p.sigma()),
            ("eta", p.eta()),
            ("gamma_p", p.gamma_p()),
            ("kappa", etrade::ac_kappa(&p)),
            ("e_cost_bps", bps(e, notional).unwrap()),
            ("sd_cost_bps", bps(sd, notional).unwrap()),
            ("spread_cost_bps", bps(hs, notional).unwrap()),
        ]
    } else {
        let m = as_from(inp);
        let a = as_params_from_market(&m).unwrap();
        let open = etrade::optimal_spread(&a, 0.0).unwrap();
        let close = etrade::optimal_spread(&a, a.horizon()).unwrap();
        vec![
            ("sigma", a.sigma()),
            ("k", a.k()),
            ("a_rate", a.a_rate()),
            ("gamma", a.gamma()),
            ("eta_liq", a.eta_liq()),
            ("dt", a.dt()),
            ("n_steps", a.n_steps() as f64),
            ("spread_open_bps", bps(open, m.price).unwrap()),
            ("spread_close_bps", bps(close, m.price).unwrap()),
        ]
    }
}

fn check_case(name: &str) {
    let golden = load("golden/market_golden.json");
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
        let want = num(&case["expect"], key);
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
    golden_ac_paper => "market_ac_paper_almgren_chriss_2000",
    golden_ac_large_cap => "market_ac_us_large_cap",
    golden_as_large_cap => "market_as_us_large_cap",
    golden_ac_small_cap => "market_ac_us_small_cap",
    golden_as_small_cap => "market_as_us_small_cap",
    golden_ac_etf => "market_ac_us_etf",
    golden_as_etf => "market_as_us_etf",
    golden_ac_eurusd => "market_ac_eurusd",
    golden_as_eurusd => "market_as_eurusd",
}

#[test]
fn golden_count_and_presets_covered() {
    let golden = load("golden/market_golden.json");
    let cases = golden["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 9);
    let presets = load("presets.json");
    for (name, body) in presets.as_object().unwrap() {
        for side in ["ac", "as"] {
            if body.get(side).is_some() {
                let want = format!("market_{side}_{name}");
                assert!(cases.iter().any(|c| c["name"] == want.as_str()), "{want}");
            }
        }
    }
}

#[test]
fn paper_calibration_reproduces_liquid_set() {
    let presets = load("presets.json");
    let p = ac_params_from_market(&ac_from(&presets["paper_almgren_chriss_2000"]["ac"])).unwrap();
    assert!((p.eta() - 2.5e-6).abs() < 1e-21);
    assert!((p.gamma_p() - 2.5e-7).abs() < 1e-22);
    assert!((p.sigma() - 0.95).abs() < 1e-15);
}

#[test]
fn as_conversion_defining_properties() {
    let presets = load("presets.json");
    let m = as_from(&presets["us_large_cap"]["as"]);
    let a = as_params_from_market(&m).unwrap();
    let d0 = m.tick / 2.0;
    let touch = etrade::intensity(&a, d0).unwrap();
    assert!((touch / m.touch_fills_per_day - 1.0).abs() < 1e-12);
    let far = etrade::intensity(&a, d0 + m.halving_ticks * m.tick).unwrap();
    assert!((far / touch - 0.5).abs() < 1e-12);
    let skew = a.s0() - etrade::reservation_price(&a, a.s0(), 1.0, 0.0).unwrap();
    assert!((skew / (m.skew_ticks * m.tick) - 1.0).abs() < 1e-12);
    assert_eq!(a.n_steps(), 4680);
}

#[test]
fn reporting_helpers() {
    assert_eq!(bps(5.0, 1e4).unwrap(), 5.0);
    assert_eq!(half_spread_cost(0.01, 1000.0).unwrap(), 5.0);
    assert!(bps(1.0, 0.0).is_err());
    assert!(bps(f64::NAN, 1.0).is_err());
    assert!(half_spread_cost(0.0, 1.0).is_err());
}

#[test]
fn ac_validation_names_the_field() {
    let presets = load("presets.json");
    let base = ac_from(&presets["us_large_cap"]["ac"]);
    let cases: Vec<(&str, ACMarket)> = vec![
        ("price", ACMarket { price: 0.0, ..base }),
        (
            "spread",
            ACMarket {
                spread: -0.01,
                ..base
            },
        ),
        (
            "adv",
            ACMarket {
                adv: f64::INFINITY,
                ..base
            },
        ),
        (
            "daily_vol",
            ACMarket {
                daily_vol: f64::NAN,
                ..base
            },
        ),
        (
            "temp_frac",
            ACMarket {
                temp_frac: 0.0,
                ..base
            },
        ),
        (
            "temp_frac",
            ACMarket {
                temp_frac: 1.5,
                ..base
            },
        ),
        (
            "perm_frac",
            ACMarket {
                perm_frac: 0.0,
                ..base
            },
        ),
    ];
    for (field, m) in cases {
        let err = ac_params_from_market(&m).unwrap_err();
        assert!(err.to_string().contains(field), "{field}: {err}");
    }
}

#[test]
fn as_validation_names_the_field() {
    let presets = load("presets.json");
    let base = as_from(&presets["us_large_cap"]["as"]);
    let cases: Vec<(&str, ASMarket)> = vec![
        (
            "price",
            ASMarket {
                price: -1.0,
                ..base
            },
        ),
        ("tick", ASMarket { tick: 0.0, ..base }),
        (
            "daily_vol",
            ASMarket {
                daily_vol: 0.0,
                ..base
            },
        ),
        (
            "touch_fills_per_day",
            ASMarket {
                touch_fills_per_day: 0.0,
                ..base
            },
        ),
        (
            "halving_ticks",
            ASMarket {
                halving_ticks: f64::INFINITY,
                ..base
            },
        ),
        (
            "skew_ticks",
            ASMarket {
                skew_ticks: 0.0,
                ..base
            },
        ),
        (
            "liq_ticks",
            ASMarket {
                liq_ticks: -1.0,
                ..base
            },
        ),
        (
            "dt_seconds",
            ASMarket {
                dt_seconds: 0.0,
                ..base
            },
        ),
        (
            "session_seconds",
            ASMarket {
                session_seconds: f64::NAN,
                ..base
            },
        ),
        (
            "horizon_days",
            ASMarket {
                horizon_days: 0.0,
                ..base
            },
        ),
        (
            "multiple of dt",
            ASMarket {
                dt_seconds: 7.0,
                ..base
            },
        ),
    ];
    for (field, m) in cases {
        let err = as_params_from_market(&m).unwrap_err();
        assert!(err.to_string().contains(field), "{field}: {err}");
    }
}
