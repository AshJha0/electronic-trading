//! Draw-order guards (API_SPEC.md Sections 2.2 and 9.4).
//!
//! A reference re-implementation of each simulation loop, using the pinned
//! RNG (`ChaCha12Rng::seed_from_u64`, one stream for all paths, path-major),
//! must reproduce the library bit for bit; deliberately wrong draw orders on
//! the same seed must diverge.

use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rand_distr::StandardNormal;

use etrade::mm::{simulate_mm, MMAgent, MMConfig, MMStat};
use etrade::{simulate, ASParams, SimConfig};

/// Tight enough that informed traders pick off quotes inside the move.
const FIXED_SPREAD: f64 = 0.2;

const N_PATHS: usize = 4;
const SEED: u64 = 2024;

/// 50 steps, q_max = 1 so the cap binds often.
fn params() -> ASParams {
    ASParams::new(100.0, 2.0, 1.0, 0.02, 0.1, 1.5, 140.0, 1, 0.5).unwrap()
}

#[derive(Clone, Copy, PartialEq)]
enum AsOrder {
    Right,
    BidBeforeAsk,
    SkipSuppressed,
}

fn ref_as(p: &ASParams, order: AsOrder) -> Vec<f64> {
    let mut rng = ChaCha12Rng::seed_from_u64(SEED);
    let n_steps = p.n_steps();
    let (g, big_t, dt) = (p.gamma(), p.horizon(), p.dt());
    let sig2 = p.sigma().powi(2);
    let markup = 2.0 / g * (g / p.k()).ln_1p();
    let sqdt = dt.sqrt();
    let q_cap = p.q_max() as i64;
    let mut out = Vec::new();
    for _ in 0..N_PATHS {
        let (mut s, mut q, mut cash) = (p.s0(), 0_i64, 0.0);
        for i in 0..n_steps {
            let t = i as f64 * dt;
            let r = s - q as f64 * g * sig2 * (big_t - t);
            let half = 0.5 * (g * sig2 * (big_t - t) + markup);
            let (ask, bid) = (r + half, r - half);
            let quote_ask = q > -q_cap;
            let quote_bid = q < q_cap;
            let (u_ask, u_bid): (f64, f64) = match order {
                AsOrder::Right => {
                    let a = rng.gen();
                    (a, rng.gen())
                }
                AsOrder::BidBeforeAsk => {
                    let b = rng.gen();
                    (rng.gen(), b)
                }
                AsOrder::SkipSuppressed => {
                    let a = if quote_ask { rng.gen() } else { 1.0 };
                    let b = if quote_bid { rng.gen() } else { 1.0 };
                    (a, b)
                }
            };
            let p_ask = -(-p.a_rate() * (-p.k() * (ask - s)).exp() * dt).exp_m1();
            let p_bid = -(-p.a_rate() * (-p.k() * (s - bid)).exp() * dt).exp_m1();
            if quote_ask && u_ask < p_ask {
                q -= 1;
                cash += ask;
            }
            if quote_bid && u_bid < p_bid {
                q += 1;
                cash -= bid;
            }
            let z: f64 = rng.sample(StandardNormal);
            s += p.sigma() * 1.0 * sqdt * z;
        }
        out.push(cash + q as f64 * s - p.eta_liq() * q.abs() as f64);
    }
    out
}

#[test]
fn as_reference_loop_reproduces_library_bitwise() {
    let p = params();
    let lib = simulate(&p, N_PATHS, SEED, &SimConfig::default()).unwrap();
    assert_eq!(ref_as(&p, AsOrder::Right), lib.pnl());
    // the cap binds in this case (otherwise the skip test proves nothing)
    assert_eq!(lib.overall_max_abs_q, 1.0);
}

#[test]
fn as_wrong_orders_diverge() {
    let p = params();
    let lib = simulate(&p, N_PATHS, SEED, &SimConfig::default()).unwrap();
    assert_ne!(ref_as(&p, AsOrder::BidBeforeAsk), lib.pnl());
    assert_ne!(ref_as(&p, AsOrder::SkipSuppressed), lib.pnl());
}

#[derive(Clone, Copy, PartialEq)]
enum MmOrder {
    Right,
    SwapAskBid,
    ZLast,
}

/// Section 9.4 loop for the fixed agent, tick 0, no latency, no signal
/// (`signal_accuracy = 0.5`, so `fair` is the current mid), informed flow on.
fn ref_mm(p: &ASParams, informed_prob: f64, order: MmOrder) -> (Vec<f64>, Vec<f64>) {
    let mut rng = ChaCha12Rng::seed_from_u64(SEED);
    let (dt, k, a, sig) = (p.dt(), p.k(), p.a_rate(), p.sigma());
    let qmax = p.q_max() as f64;
    let sqdt = dt.sqrt();
    let fixed_half = 0.5 * FIXED_SPREAD;
    let (mut pnls, mut fills_out) = (Vec::new(), Vec::new());
    for _ in 0..N_PATHS {
        let (mut s, mut q, mut cash, mut fills) = (p.s0(), 0.0_f64, 0.0, 0.0);
        for _ in 0..p.n_steps() {
            let (z, u_ask, u_bid, u_inf): (f64, f64, f64, f64) = match order {
                MmOrder::Right => {
                    let z = rng.sample(StandardNormal);
                    let _u_sig: f64 = rng.gen();
                    let ua = rng.gen();
                    let ub = rng.gen();
                    (z, ua, ub, rng.gen())
                }
                MmOrder::SwapAskBid => {
                    let z = rng.sample(StandardNormal);
                    let _u_sig: f64 = rng.gen();
                    let ub = rng.gen();
                    let ua = rng.gen();
                    (z, ua, ub, rng.gen())
                }
                MmOrder::ZLast => {
                    let _u_sig: f64 = rng.gen();
                    let ua = rng.gen();
                    let ub = rng.gen();
                    let ui = rng.gen();
                    (rng.sample(StandardNormal), ua, ub, ui)
                }
            };
            let fair = s;
            let (bid, ask) = (fair - fixed_half, fair + fixed_half);
            let can_ask = q > -qmax;
            let can_bid = q < qmax;
            let p_ask = -(-a * (-k * (ask - s)).exp() * dt).exp_m1();
            let p_bid = -(-a * (-k * (s - bid)).exp() * dt).exp_m1();
            let fill_ask = can_ask && u_ask < p_ask;
            let fill_bid = can_bid && u_bid < p_bid;
            let mv = sig * sqdt * z;
            let informed = u_inf < informed_prob;
            let inf_ask = informed && z > 0.0 && can_ask && !fill_ask && ask < s + mv;
            let inf_bid = informed && z < 0.0 && can_bid && !fill_bid && bid > s + mv;
            let fa = fill_ask || inf_ask;
            let fb = fill_bid || inf_bid;
            let fa_f = if fa { 1.0 } else { 0.0 };
            let fb_f = if fb { 1.0 } else { 0.0 };
            q += fb_f - fa_f;
            cash +=
                (if fa { ask } else { 0.0 }) - (if fb { bid } else { 0.0 }) + 0.0 * (fa_f + fb_f);
            fills += fa_f + fb_f;
            s += mv;
        }
        let liq = p.eta_liq() * q.abs();
        pnls.push(cash + q * s - liq);
        fills_out.push(fills);
    }
    (pnls, fills_out)
}

fn mm_cfg(informed_prob: f64) -> MMConfig {
    MMConfig {
        agent: MMAgent::Fixed,
        informed_prob,
        fixed_spread: Some(FIXED_SPREAD),
        ..MMConfig::default()
    }
}

#[test]
fn mm_reference_loop_reproduces_library_bitwise() {
    let p = params();
    let lib = simulate_mm(&p, N_PATHS, SEED, &mm_cfg(0.3)).unwrap();
    let (pnl, fills) = ref_mm(&p, 0.3, MmOrder::Right);
    assert_eq!(pnl, lib.paths(MMStat::Pnl));
    assert_eq!(fills, lib.paths(MMStat::Fills));
    assert!(lib.mean(MMStat::InformedFills) > 0.0);
}

#[test]
fn mm_wrong_orders_diverge() {
    let p = params();
    let lib = simulate_mm(&p, N_PATHS, SEED, &mm_cfg(0.3)).unwrap();
    for order in [MmOrder::SwapAskBid, MmOrder::ZLast] {
        let (pnl, _) = ref_mm(&p, 0.3, order);
        assert_ne!(pnl, lib.paths(MMStat::Pnl));
    }
}
