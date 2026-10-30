//! Market making with frictions (API_SPEC.md Section 9): every case of
//! `data/golden/mm_golden.json` (one test per case), the attribution identity
//! on every path, determinism and validation.

use std::fs;
use std::path::Path;

use serde_json::Value;

use etrade::market::{as_params_from_market, ASMarket};
use etrade::mm::{
    glft_half_spreads, order_flow_imbalance, round_to_tick, simulate_mm, weighted_mid, MMAgent,
    MMConfig, MMResult, MMStat,
};
use etrade::ASParams;

/// Paths per statistical case (Section 4: at least 2000).
const N_PATHS: usize = 2000;

fn read_json(rel: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).expect("valid JSON")
}

fn load() -> Value {
    read_json("../data/golden/mm_golden.json")
}

fn num(o: &Value, k: &str) -> f64 {
    o.get(k)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("missing key '{k}'"))
}

fn market_from(v: &Value) -> ASMarket {
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
        session_seconds: num(v, "session_seconds"),
        horizon_days: num(v, "horizon_days"),
    }
}

/// The `us_large_cap` A-S preset.  The `mm_sim` inputs reuse the key `tick`
/// for the simulator's tick, so the market tick comes from the preset.
fn large_cap() -> ASParams {
    let preset = read_json("../data/presets.json");
    as_params_from_market(&market_from(&preset["us_large_cap"]["as"])).unwrap()
}

const SIM_STATS: [(&str, MMStat); 7] = [
    ("pnl", MMStat::Pnl),
    ("spread", MMStat::Spread),
    ("fees", MMStat::Fees),
    ("inventory", MMStat::Inventory),
    ("markout", MMStat::Markout),
    ("fills", MMStat::Fills),
    ("informed_fills", MMStat::InformedFills),
];

fn config_from(inp: &Value) -> MMConfig {
    MMConfig {
        agent: match num(inp, "agent_code") as u32 {
            0 => MMAgent::As,
            1 => MMAgent::Glft,
            _ => MMAgent::Fixed,
        },
        tick: num(inp, "tick"),
        maker_fee: num(inp, "maker_fee"),
        latency_steps: num(inp, "latency_steps") as usize,
        informed_prob: num(inp, "informed_prob"),
        signal_accuracy: num(inp, "signal_accuracy"),
        fixed_spread: None,
    }
}

fn assert_identity(r: &MMResult) {
    assert!(r.max_identity_error() <= 1e-9);
    for j in 0..r.n_paths() {
        let pnl = r.paths(MMStat::Pnl)[j];
        let recon =
            r.paths(MMStat::Spread)[j] + r.paths(MMStat::Fees)[j] + r.paths(MMStat::Inventory)[j]
                - r.paths(MMStat::Liquidation)[j];
        assert!(
            (recon - pnl).abs() <= 1e-9 * pnl.abs().max(1.0),
            "path {j}: {recon} vs {pnl}"
        );
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
    let exp = &case["expect"];
    let n_expect = exp.as_object().unwrap().len();
    let inp = &case["inputs"];
    let tol = num(case, "tol");
    let mut exact: Vec<(String, f64)> = Vec::new();
    match case["kind"].as_str().unwrap() {
        "mm_tick" => {
            let (b, a) = round_to_tick(num(inp, "bid"), num(inp, "ask"), num(inp, "tick")).unwrap();
            exact.push(("bid".into(), b));
            exact.push(("ask".into(), a));
        }
        "mm_glft" => {
            let p = as_params_from_market(&market_from(inp)).unwrap();
            let (db, da) = glft_half_spreads(&p, num(inp, "q")).unwrap();
            exact.push(("delta_bid".into(), db));
            exact.push(("delta_ask".into(), da));
        }
        "mm_wmid" => {
            let w = weighted_mid(
                num(inp, "bid"),
                num(inp, "ask"),
                num(inp, "bid_size"),
                num(inp, "ask_size"),
            )
            .unwrap();
            exact.push(("wmid".into(), w));
        }
        "mm_ofi" => {
            let col =
                |k: &str| -> Vec<f64> { (0..5).map(|i| num(inp, &format!("{k}_{i}"))).collect() };
            let e = order_flow_imbalance(
                &col("bid_px"),
                &col("bid_sz"),
                &col("ask_px"),
                &col("ask_sz"),
            )
            .unwrap();
            for (i, v) in e.iter().enumerate() {
                exact.push((format!("e_{}", i + 1), *v));
            }
            exact.push(("ofi".into(), e.iter().sum()));
        }
        "mm_sim" => {
            let r = simulate_mm(
                &large_cap(),
                N_PATHS,
                num(inp, "seed") as u64,
                &config_from(inp),
            )
            .unwrap();
            assert_identity(&r);
            assert_eq!(
                2 * SIM_STATS.len(),
                n_expect,
                "{name}: every expect key must be asserted"
            );
            for (key, stat) in SIM_STATS {
                let (want, se_ref) = (num(exp, key), num(exp, &format!("{key}_se")));
                let (got, se_own) = (r.mean(stat), r.se(stat));
                let bound = tol * (se_ref * se_ref + se_own * se_own).sqrt();
                assert!(
                    (got - want).abs() <= bound + 1e-12,
                    "{name}.{key}: got {got} (se {se_own}), want {want} (se {se_ref})"
                );
            }
            return;
        }
        k => panic!("unknown kind {k}"),
    }
    assert_eq!(
        exact.len(),
        n_expect,
        "{name}: every expect key must be asserted"
    );
    for (key, value) in exact {
        let want = num(exp, &key);
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
    golden_tick_round_away => "mm_tick_round_away",
    golden_tick_on_grid => "mm_tick_on_grid",
    golden_tick_continuous => "mm_tick_continuous",
    golden_glft_qp0 => "mm_glft_qp0",
    golden_glft_qp3 => "mm_glft_qp3",
    golden_glft_qm2 => "mm_glft_qm2",
    golden_weighted_mid => "mm_weighted_mid",
    golden_ofi_five_states => "mm_ofi_five_states",
    golden_sim_frictionless => "mm_sim_frictionless",
    golden_sim_tick_rebate => "mm_sim_tick_rebate",
    golden_sim_informed => "mm_sim_informed",
    golden_sim_informed_signal => "mm_sim_informed_signal",
    golden_sim_latency => "mm_sim_latency",
    golden_sim_glft => "mm_sim_glft",
}

#[test]
fn golden_count() {
    assert_eq!(load()["cases"].as_array().unwrap().len(), 14);
}

fn small() -> ASParams {
    ASParams::new(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, 10, 0.05).unwrap()
}

#[test]
fn identity_holds_for_every_agent_and_friction() {
    for agent in [MMAgent::As, MMAgent::Glft, MMAgent::Fixed] {
        let cfg = MMConfig {
            agent,
            tick: 0.05,
            maker_fee: -0.01,
            latency_steps: 3,
            informed_prob: 0.3,
            signal_accuracy: 0.7,
            fixed_spread: None,
        };
        let r = simulate_mm(&small(), 300, 11, &cfg).unwrap();
        assert_identity(&r);
        assert!(r.mean(MMStat::InformedFills) > 0.0);
        assert!(r.mean(MMStat::RmsQ) > 0.0);
    }
}

#[test]
fn deterministic_per_seed() {
    let cfg = MMConfig {
        informed_prob: 0.2,
        ..MMConfig::default()
    };
    let a = simulate_mm(&small(), 50, 5, &cfg).unwrap();
    let b = simulate_mm(&small(), 50, 5, &cfg).unwrap();
    let c = simulate_mm(&small(), 50, 6, &cfg).unwrap();
    assert_eq!(a.paths(MMStat::Pnl), b.paths(MMStat::Pnl));
    assert_ne!(a.paths(MMStat::Pnl), c.paths(MMStat::Pnl));
}

#[test]
fn informed_flow_is_adverse() {
    let base = simulate_mm(&small(), 400, 9, &MMConfig::default()).unwrap();
    let cfg = MMConfig {
        informed_prob: 0.5,
        ..MMConfig::default()
    };
    let inf = simulate_mm(&small(), 400, 9, &cfg).unwrap();
    assert!(inf.mean(MMStat::Markout) < base.mean(MMStat::Markout));
    assert_eq!(base.mean(MMStat::InformedFills), 0.0);
}

#[test]
fn helpers() {
    assert_eq!(round_to_tick(1.234, 1.236, 0.0).unwrap(), (1.234, 1.236));
    let (db0, da0) = glft_half_spreads(&small(), 0.0).unwrap();
    assert!((db0 - da0).abs() < 1e-15);
    let (db, da) = glft_half_spreads(&small(), 2.0).unwrap();
    assert!(db > db0 && da < da0);
    assert_eq!(weighted_mid(99.0, 101.0, 1.0, 1.0).unwrap(), 100.0);
    assert_eq!(
        order_flow_imbalance(&[1.0, 1.0], &[5.0, 7.0], &[2.0, 2.0], &[4.0, 4.0]).unwrap(),
        vec![2.0]
    );
}

#[test]
fn validation_errors() {
    let p = small();
    let bad = |cfg: MMConfig| simulate_mm(&p, 10, 1, &cfg).is_err();
    let d = MMConfig::default();
    assert!(bad(MMConfig { tick: -0.01, ..d }));
    assert!(bad(MMConfig {
        tick: f64::NAN,
        ..d
    }));
    assert!(bad(MMConfig {
        maker_fee: f64::INFINITY,
        ..d
    }));
    assert!(bad(MMConfig {
        latency_steps: 201,
        ..d
    }));
    assert!(bad(MMConfig {
        informed_prob: 1.5,
        ..d
    }));
    assert!(bad(MMConfig {
        signal_accuracy: 0.4,
        ..d
    }));
    assert!(bad(MMConfig {
        fixed_spread: Some(0.0),
        ..d
    }));
    assert!(simulate_mm(
        &p,
        10,
        1,
        &MMConfig {
            latency_steps: 200,
            ..d
        }
    )
    .is_ok());
    assert!(simulate_mm(&p, 1, 1, &d).is_err());
    assert!(round_to_tick(1.0, 2.0, -1.0).is_err());
    assert!(round_to_tick(f64::NAN, 2.0, 0.01).is_err());
    assert!(glft_half_spreads(&p, f64::NAN).is_err());
    assert!(weighted_mid(101.0, 99.0, 1.0, 1.0).is_err());
    assert!(weighted_mid(99.0, 101.0, 0.0, 0.0).is_err());
    assert!(weighted_mid(99.0, 101.0, -1.0, 2.0).is_err());
    assert!(order_flow_imbalance(&[1.0], &[1.0], &[1.0], &[1.0]).is_err());
    assert!(order_flow_imbalance(&[1.0, 1.0], &[1.0], &[1.0, 1.0], &[1.0, 1.0]).is_err());
    assert!(order_flow_imbalance(&[1.0, f64::NAN], &[1.0; 2], &[1.0; 2], &[1.0; 2]).is_err());
    assert!(order_flow_imbalance(&[1.0; 2], &[1.0, -1.0], &[1.0; 2], &[1.0; 2]).is_err());
}
