//! Hedged market making and risk controls (API_SPEC.md Section 14): every
//! case of `data/golden/hedging_golden.json` (one test per case), the
//! attribution identity on every path, no hedge = Section 9 in distribution,
//! the loss-limit overshoot bound, the effect of each control, determinism
//! and validation.

use std::fs;
use std::path::Path;

use serde_json::Value;

use etrade::hedging::{
    hedge_trade, quantile, simulate_hedged, HedgeConfig, HedgeStat, HedgedResult, RiskConfig,
};
use etrade::market::{as_params_from_market, ASMarket};
use etrade::mm::{simulate_mm, MMAgent, MMConfig, MMStat};
use etrade::ASParams;

/// Paths per statistical case (Section 4: at least 2000).
const N_PATHS: usize = 2000;

fn read_json(rel: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).expect("valid JSON")
}

fn load() -> Value {
    read_json("../data/golden/hedging_golden.json")
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

/// The `us_large_cap` A-S preset.
fn large_cap() -> ASParams {
    let preset = read_json("../data/presets.json");
    as_params_from_market(&market_from(&preset["us_large_cap"]["as"])).unwrap()
}

const SIM_STATS: [(&str, HedgeStat); 15] = [
    ("pnl", HedgeStat::Pnl),
    ("spread", HedgeStat::Spread),
    ("fees", HedgeStat::Fees),
    ("inventory", HedgeStat::Inventory),
    ("hedge_pnl", HedgeStat::HedgePnl),
    ("hedge_costs", HedgeStat::HedgeCosts),
    ("risk_costs", HedgeStat::RiskCosts),
    ("fills", HedgeStat::Fills),
    ("rms_q", HedgeStat::RmsQ),
    ("rms_x", HedgeStat::RmsX),
    ("rms_skew", HedgeStat::RmsSkew),
    ("quote_updates", HedgeStat::QuoteUpdates),
    ("loss_stop", HedgeStat::LossStop),
    ("kill_stop", HedgeStat::KillStop),
    ("pos_flattens", HedgeStat::PosFlattens),
];

fn configs_from(inp: &Value) -> (MMConfig, HedgeConfig, RiskConfig) {
    let cfg = MMConfig {
        agent: match num(inp, "agent_code") as u32 {
            0 => MMAgent::As,
            1 => MMAgent::Glft,
            _ => MMAgent::Fixed,
        },
        tick: num(inp, "quote_tick"),
        maker_fee: num(inp, "maker_fee"),
        latency_steps: num(inp, "latency_steps") as usize,
        informed_prob: num(inp, "informed_prob"),
        signal_accuracy: num(inp, "signal_accuracy"),
        fixed_spread: None,
    };
    let hedge = HedgeConfig {
        rho: num(inp, "rho"),
        hedge_ratio: num(inp, "hedge_ratio"),
        hedge_cost: num(inp, "hedge_cost"),
        hedge_threshold: num(inp, "hedge_threshold"),
        hedge_target: num(inp, "hedge_target"),
    };
    let risk = RiskConfig {
        loss_limit: num(inp, "loss_limit"),
        position_limit: num(inp, "position_limit") as u32,
        flatten_penalty: num(inp, "flatten_penalty"),
        throttle_updates: num(inp, "throttle_updates") as usize,
        throttle_window: num(inp, "throttle_window") as usize,
        kill_step: num(inp, "kill_step") as i64,
    };
    (cfg, hedge, risk)
}

fn assert_identity(r: &HedgedResult) {
    assert!(r.max_identity_error() <= 1e-9);
    let p = |s: HedgeStat, j: usize| r.paths(s)[j];
    for j in 0..r.n_paths() {
        let pnl = p(HedgeStat::Pnl, j);
        let recon = p(HedgeStat::Spread, j) + p(HedgeStat::Fees, j) + p(HedgeStat::Inventory, j)
            - p(HedgeStat::Liquidation, j)
            + p(HedgeStat::HedgePnl, j)
            - p(HedgeStat::HedgeCosts, j)
            - p(HedgeStat::RiskCosts, j);
        assert!(
            (recon - pnl).abs() <= 1e-9 * pnl.abs().max(1.0),
            "path {j}: {recon} vs {pnl}"
        );
        assert!(p(HedgeStat::HedgeCosts, j) >= 0.0 && p(HedgeStat::RiskCosts, j) >= 0.0);
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
    let exact: Vec<(String, f64)> = match case["kind"].as_str().unwrap() {
        "hedge_trade" => vec![(
            "dy".into(),
            hedge_trade(
                num(inp, "q"),
                num(inp, "y"),
                num(inp, "hedge_ratio"),
                num(inp, "hedge_threshold"),
                num(inp, "hedge_target"),
            )
            .unwrap(),
        )],
        "hedge_quantile" => {
            let v: Vec<f64> = (0..10).map(|i| num(inp, &format!("v_{i}"))).collect();
            vec![("q".into(), quantile(&v, num(inp, "p")).unwrap())]
        }
        "hedge_sim" => {
            let (cfg, hedge, risk) = configs_from(inp);
            let p = as_params_from_market(&market_from(inp)).unwrap();
            let seed = num(inp, "seed") as u64 + 1;
            let r = simulate_hedged(&p, N_PATHS, seed, &cfg, &hedge, &risk).unwrap();
            assert_identity(&r);
            assert_eq!(
                2 * (SIM_STATS.len() + 1),
                n_expect,
                "{name}: every expect key must be asserted"
            );
            let mut rows: Vec<(String, f64, f64)> = SIM_STATS
                .iter()
                .map(|(key, stat)| (key.to_string(), r.mean(*stat), r.se(*stat)))
                .collect();
            rows.push((
                "pnl_var".into(),
                r.var(HedgeStat::Pnl),
                r.var_se(HedgeStat::Pnl),
            ));
            for (key, got, se_own) in rows {
                let (want, se_ref) = (num(exp, &key), num(exp, &format!("{key}_se")));
                let bound = tol * se_ref.hypot(se_own);
                assert!(
                    (got - want).abs() <= bound + 1e-12,
                    "{name}.{key}: got {got} (se {se_own}), want {want} (se {se_ref})"
                );
            }
            return;
        }
        k => panic!("unknown kind {k}"),
    };
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
    golden_trade_no_hedge => "hedge_trade_no_hedge",
    golden_trade_full => "hedge_trade_full",
    golden_trade_inside_band => "hedge_trade_inside_band",
    golden_trade_partial_target => "hedge_trade_partial_target",
    golden_quantile_p005 => "hedge_quantile_p005",
    golden_quantile_p050 => "hedge_quantile_p050",
    golden_quantile_p095 => "hedge_quantile_p095",
    golden_quantile_p100 => "hedge_quantile_p100",
    golden_sim_as_unhedged => "hedge_sim_as_unhedged",
    golden_sim_as_hedged => "hedge_sim_as_hedged",
    golden_sim_as_band => "hedge_sim_as_band",
    golden_sim_fixed_unhedged => "hedge_sim_fixed_unhedged",
    golden_sim_fixed_hedged => "hedge_sim_fixed_hedged",
    golden_risk_sim_loss_limit => "risk_sim_loss_limit",
    golden_risk_sim_position_limit => "risk_sim_position_limit",
    golden_risk_sim_throttle => "risk_sim_throttle",
    golden_risk_sim_kill_switch => "risk_sim_kill_switch",
    golden_risk_sim_all_controls_hedged => "risk_sim_all_controls_hedged",
}

#[test]
fn golden_count() {
    let g = load();
    let cases = g["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 18);
    let sims = cases.iter().filter(|c| c["kind"] == "hedge_sim").count();
    assert_eq!((cases.len() - sims, sims), (8, 10));
}

// ------------------------------------------------------------------ helpers

fn book() -> MMConfig {
    MMConfig {
        tick: 0.01,
        maker_fee: 0.002,
        informed_prob: 0.15,
        ..MMConfig::default()
    }
}

fn fixed() -> MMConfig {
    MMConfig {
        agent: MMAgent::Fixed,
        ..book()
    }
}

fn hedge() -> HedgeConfig {
    HedgeConfig {
        rho: 0.9,
        hedge_ratio: 0.9,
        hedge_cost: 0.005,
        ..HedgeConfig::default()
    }
}

fn sim(cfg: MMConfig, h: HedgeConfig, r: RiskConfig, n: usize, seed: u64) -> HedgedResult {
    simulate_hedged(&large_cap(), n, seed, &cfg, &h, &r).unwrap()
}

fn sim600(cfg: MMConfig, h: HedgeConfig, r: RiskConfig) -> HedgedResult {
    sim(cfg, h, r, 600, 11)
}

fn no_risk() -> RiskConfig {
    RiskConfig::default()
}

fn no_hedge() -> HedgeConfig {
    HedgeConfig::default()
}

// ------------------------------------------------------------------ attribution

#[test]
fn attribution_identity_holds_on_every_path() {
    let cases = [
        (MMConfig::default(), no_hedge(), no_risk()),
        (book(), hedge(), no_risk()),
        (
            MMConfig {
                agent: MMAgent::Glft,
                tick: 0.01,
                latency_steps: 2,
                ..MMConfig::default()
            },
            hedge(),
            RiskConfig {
                position_limit: 3,
                flatten_penalty: 0.02,
                ..no_risk()
            },
        ),
        (
            fixed(),
            HedgeConfig {
                rho: -0.5,
                hedge_ratio: 0.4,
                hedge_cost: 0.01,
                hedge_threshold: 2.0,
                hedge_target: 0.5,
            },
            RiskConfig {
                loss_limit: 2.0,
                flatten_penalty: 0.01,
                throttle_updates: 2,
                throttle_window: 5,
                kill_step: 3000,
                ..no_risk()
            },
        ),
    ];
    for (cfg, h, r) in cases {
        assert_identity(&sim(cfg, h, r, 300, 9));
    }
}

// ------------------------------------------------------------------ properties

#[test]
fn no_hedge_no_controls_is_section_9_in_distribution() {
    let p = large_cap();
    let mm_stats = [
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
    for cfg in [book(), fixed()] {
        let a = simulate_hedged(&p, N_PATHS, 31, &cfg, &no_hedge(), &no_risk()).unwrap();
        let b = simulate_mm(&p, N_PATHS, 32, &cfg).unwrap();
        for (i, stat) in mm_stats.iter().enumerate() {
            let hs = HedgeStat::ALL[i];
            let band = 4.0 * a.se(hs).hypot(b.se(*stat));
            assert!(
                (a.mean(hs) - b.mean(*stat)).abs() <= band + 1e-12,
                "{}",
                hs.name()
            );
        }
        for stat in [
            HedgeStat::HedgePnl,
            HedgeStat::HedgeCosts,
            HedgeStat::RiskCosts,
            HedgeStat::LossStop,
            HedgeStat::KillStop,
        ] {
            assert!(a.paths(stat).iter().all(|v| *v == 0.0));
        }
        assert_eq!(a.paths(HedgeStat::RmsX), a.paths(HedgeStat::RmsQ));
        assert!(a
            .paths(HedgeStat::QuoteUpdates)
            .iter()
            .all(|v| *v == p.n_steps() as f64));
    }
}

#[test]
fn loss_limit_overshoot_is_at_most_one_step_move() {
    let r = sim(
        fixed(),
        no_hedge(),
        RiskConfig {
            loss_limit: 3.0,
            flatten_penalty: 0.01,
            ..no_risk()
        },
        800,
        5,
    );
    let stop = r.paths(HedgeStat::LossStop);
    assert!(stop.iter().filter(|v| **v == 1.0).count() > 100);
    for (j, &st) in stop.iter().enumerate() {
        let (ov, mv) = (
            r.paths(HedgeStat::Overshoot)[j],
            r.paths(HedgeStat::StopMove)[j],
        );
        assert!(ov <= mv + 1e-9, "path {j}: {ov} > {mv}");
        if st == 1.0 {
            // once stopped, the book is flat and no longer quotes
            assert_eq!(r.paths(HedgeStat::Liquidation)[j], 0.0);
            assert!(r.paths(HedgeStat::QuoteUpdates)[j] < 4680.0);
        } else {
            assert_eq!(ov, 0.0);
        }
    }
}

#[test]
fn hedging_cuts_variance_and_skew_at_a_cost() {
    let fixed_un = sim600(fixed(), no_hedge(), no_risk());
    let fixed_h = sim600(fixed(), hedge(), no_risk());
    assert!(fixed_h.var(HedgeStat::Pnl) < 0.5 * fixed_un.var(HedgeStat::Pnl));
    assert!(fixed_h.mean(HedgeStat::RmsX) < 0.5 * fixed_h.mean(HedgeStat::RmsQ));
    assert!(fixed_h.mean(HedgeStat::HedgeCosts) > 0.0);
    assert!(fixed_h.mean(HedgeStat::HedgeTrades) > 0.0);
    let as_un = sim600(book(), no_hedge(), no_risk());
    let as_h = sim600(book(), hedge(), no_risk());
    assert!(as_h.mean(HedgeStat::RmsSkew) < 0.6 * as_un.mean(HedgeStat::RmsSkew));
    assert!(as_h.var(HedgeStat::Pnl) < as_un.var(HedgeStat::Pnl));
}

#[test]
fn each_control_triggers_and_cuts_the_tail() {
    let q05 = |r: &HedgedResult| r.quantile(HedgeStat::Pnl, 0.05).unwrap();
    let base = sim600(fixed(), no_hedge(), no_risk());
    let pos = sim600(
        fixed(),
        no_hedge(),
        RiskConfig {
            position_limit: 10,
            flatten_penalty: 0.01,
            ..no_risk()
        },
    );
    assert!(pos.mean(HedgeStat::PosFlattens) > 1.0 && pos.mean(HedgeStat::RiskCosts) > 0.0);
    assert!(q05(&pos) > q05(&base));
    let loss = sim600(
        fixed(),
        no_hedge(),
        RiskConfig {
            loss_limit: 5.0,
            flatten_penalty: 0.01,
            ..no_risk()
        },
    );
    assert!(loss.mean(HedgeStat::LossStop) > 0.2);
    assert!(q05(&loss) > q05(&base));
    let kill = sim600(
        fixed(),
        no_hedge(),
        RiskConfig {
            kill_step: 1000,
            ..no_risk()
        },
    );
    assert!(kill.paths(HedgeStat::KillStop).iter().all(|v| *v == 1.0));
    assert!(kill
        .paths(HedgeStat::QuoteUpdates)
        .iter()
        .all(|v| *v == 1000.0));
    let thr = sim600(
        book(),
        no_hedge(),
        RiskConfig {
            throttle_updates: 1,
            throttle_window: 3,
            ..no_risk()
        },
    );
    // ceil(4680 / 3)
    assert!(thr
        .paths(HedgeStat::QuoteUpdates)
        .iter()
        .all(|v| *v == 1560.0));
    // stale quotes are picked off
    assert!(thr.mean(HedgeStat::Pnl) < sim600(book(), no_hedge(), no_risk()).mean(HedgeStat::Pnl));
}

#[test]
fn deterministic_per_seed() {
    let risk = RiskConfig {
        loss_limit: 4.0,
        throttle_updates: 2,
        throttle_window: 3,
        ..no_risk()
    };
    let a = sim(fixed(), hedge(), risk, 100, 11);
    let b = sim(fixed(), hedge(), risk, 100, 11);
    let c = sim(fixed(), hedge(), risk, 100, 12);
    for stat in HedgeStat::ALL {
        assert_eq!(a.paths(stat), b.paths(stat), "{}", stat.name());
    }
    assert_ne!(a.paths(HedgeStat::Pnl), c.paths(HedgeStat::Pnl));
}

// ------------------------------------------------------------------ deterministic pieces

#[test]
fn hedge_trade_leaves_target_fraction_uncovered() {
    let (q, y, h) = (9.0, -2.0, 0.8);
    let u = q + y / h;
    let dy = hedge_trade(q, y, h, 1.0, 0.25).unwrap();
    let left = q + (y + dy) / h;
    assert!((left - 0.25 * u).abs() <= 1e-14 * u.abs());
    assert_eq!(hedge_trade(1.0, 0.0, 0.9, 1.0, 0.0).unwrap(), 0.0); // |u| <= threshold
    assert_eq!(hedge_trade(5.0, 0.0, 0.0, 0.0, 0.0).unwrap(), 0.0);
}

#[test]
fn quantile_is_type_7() {
    let v = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    // sorted: 1 1 2 3 4 5 6 9
    assert_eq!(quantile(&v, 0.0).unwrap(), 1.0);
    assert_eq!(quantile(&v, 1.0).unwrap(), 9.0);
    assert!((quantile(&v, 0.5).unwrap() - 3.5).abs() < 1e-15);
    // h = 7 * 0.9 = 6.3 -> 6 + 0.3 * 3
    assert!((quantile(&v, 0.9).unwrap() - 6.9).abs() < 1e-14);
    assert_eq!(quantile(&[4.0], 0.3).unwrap(), 4.0);
    let mut prev = f64::NEG_INFINITY;
    for i in 0..=20 {
        let x = quantile(&v, i as f64 / 20.0).unwrap();
        assert!(x >= prev);
        prev = x;
    }
}

#[test]
fn stats_by_name() {
    let r = sim(fixed(), hedge(), no_risk(), 50, 3);
    for stat in HedgeStat::ALL {
        assert_eq!(HedgeStat::from_name(stat.name()).unwrap(), stat);
        assert_eq!(r.mean_by_name(stat.name()).unwrap(), r.mean(stat));
    }
    assert_eq!(r.se_by_name("pnl").unwrap(), r.se(HedgeStat::Pnl));
    assert_eq!(r.var_by_name("pnl").unwrap(), r.var(HedgeStat::Pnl));
    assert_eq!(r.var_se_by_name("pnl").unwrap(), r.var_se(HedgeStat::Pnl));
    assert_eq!(
        r.quantile_by_name("pnl", 0.5).unwrap(),
        r.quantile(HedgeStat::Pnl, 0.5).unwrap()
    );
    assert_eq!(r.paths_by_name("fills").unwrap(), r.paths(HedgeStat::Fills));
    assert!(r.mean_by_name("nope").is_err());
    assert!(r.paths_by_name("nope").is_err());
}

// ------------------------------------------------------------------ validation

#[test]
fn hedge_config_validation() {
    let d = HedgeConfig::default();
    for bad in [
        HedgeConfig { rho: 1.5, ..d },
        HedgeConfig {
            hedge_ratio: -0.1,
            ..d
        },
        HedgeConfig {
            hedge_cost: f64::NAN,
            ..d
        },
        HedgeConfig {
            hedge_threshold: -1.0,
            ..d
        },
        HedgeConfig {
            hedge_target: 1.0,
            ..d
        },
    ] {
        assert!(bad.validate().is_err(), "{bad:?}");
        assert!(
            simulate_hedged(&large_cap(), 10, 0, &MMConfig::default(), &bad, &no_risk()).is_err()
        );
    }
    assert!(d.validate().is_ok());
}

#[test]
fn risk_config_validation() {
    let d = RiskConfig::default();
    for bad in [
        RiskConfig {
            loss_limit: -1.0,
            ..d
        },
        RiskConfig {
            loss_limit: f64::INFINITY,
            ..d
        },
        RiskConfig {
            flatten_penalty: -0.01,
            ..d
        },
        RiskConfig {
            throttle_window: 0,
            ..d
        },
        RiskConfig { kill_step: -2, ..d },
    ] {
        assert!(bad.validate().is_err(), "{bad:?}");
        assert!(
            simulate_hedged(&large_cap(), 10, 0, &MMConfig::default(), &no_hedge(), &bad).is_err()
        );
    }
    assert!(d.validate().is_ok());
}

#[test]
fn simulation_and_helper_validation() {
    let p = large_cap();
    let d = MMConfig::default();
    let h = no_hedge();
    assert!(simulate_hedged(&p, 1, 0, &d, &h, &no_risk()).is_err());
    let kill = RiskConfig {
        kill_step: 10_000,
        ..no_risk()
    };
    assert!(simulate_hedged(&p, 10, 0, &d, &h, &kill).is_err());
    let thr = RiskConfig {
        throttle_updates: 1,
        throttle_window: 10_000,
        ..no_risk()
    };
    assert!(simulate_hedged(&p, 10, 0, &d, &h, &thr).is_err());
    let lat = MMConfig {
        latency_steps: 10_000,
        ..d
    };
    assert!(simulate_hedged(&p, 10, 0, &lat, &h, &no_risk()).is_err());
    let r = simulate_hedged(&p, 10, 0, &d, &h, &no_risk()).unwrap();
    assert!(r.quantile(HedgeStat::Pnl, 1.5).is_err());
    assert!(quantile(&[], 0.5).is_err());
    assert!(quantile(&[1.0, 2.0], 1.5).is_err());
    assert!(quantile(&[1.0, f64::NAN], 0.5).is_err());
    assert!(hedge_trade(1.0, 0.0, 0.9, 0.0, 1.0).is_err());
    assert!(hedge_trade(f64::NAN, 0.0, 0.9, 0.0, 0.0).is_err());
    assert!(hedge_trade(1.0, 0.0, -0.9, 0.0, 0.0).is_err());
    assert!(hedge_trade(1.0, 0.0, 0.9, -1.0, 0.0).is_err());
}
