//! Market making that can lose money: informed flow, ticks, fees, latency,
//! fair-value signals, PnL attribution and markouts (API_SPEC.md Section 9).
//!
//! The Section 2 simulator ([`crate::avellaneda_stoikov::simulate`]) is the
//! paper's world: fills are independent of where the price goes next, quotes
//! may sit at any real price, trading is free and quotes never go stale.  In
//! that world a market maker cannot lose money on average.  This module keeps
//! the A-S quoting logic and adds, one switch at a time, the frictions that
//! decide profit in a real book, plus the accounting a desk uses to see where
//! the money went.
//!
//! Per path, per step `i` (`t = i*dt`), five random numbers are drawn in the
//! pinned order (API_SPEC.md Section 9.4), always, whatever the configuration:
//! `z ~ N(0,1)`, then `u_sig`, `u_ask`, `u_bid`, `u_inf ~ U(0,1)`.  This port
//! uses its own RNG per Section 4 (`ChaCha12Rng::seed_from_u64`), so its
//! sample values differ from Python while all statistics agree within SE
//! bands.
//!
//! Attribution (exact per path, Section 9.5):
//!
//! ```text
//! spread      = sum over fills of (ask - mid) or (mid - bid), mid before the move
//! fees        = maker_fee * fills
//! inventory   = sum over steps of q_after_fills * (mid move)
//! liquidation = eta_liq * |q_T|
//! pnl         = cash + q_T*S_T - liquidation = spread + fees + inventory - liquidation
//! ```
//!
//! `markout` is the one-step markout of the fills (part of `inventory`): the
//! mid move right after each fill, signed so that a negative value is adverse
//! selection.

use std::collections::VecDeque;

use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rand_distr::StandardNormal;

use crate::almgren_chriss::MAX_PATHS;
use crate::avellaneda_stoikov::{average_as_spread, ASParams};
use crate::error::{require, EtradeError};

const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;

// ------------------------------------------------------------ deterministic pieces

/// Round a quote pair away from fair value to the tick grid (API_SPEC 9.2).
///
/// `bid -> floor(bid/tick + 1e-9)*tick`, `ask -> ceil(ask/tick - 1e-9)*tick`;
/// `tick = 0` leaves the quotes unchanged.  Bid and ask finite, `tick >= 0`.
pub fn round_to_tick(bid: f64, ask: f64, tick: f64) -> Result<(f64, f64), EtradeError> {
    require(
        tick.is_finite() && tick >= 0.0,
        "tick must be finite and >= 0",
    )?;
    require(
        bid.is_finite() && ask.is_finite(),
        "bid and ask must be finite",
    )?;
    if tick == 0.0 {
        return Ok((bid, ask));
    }
    Ok(round_raw(bid, ask, tick))
}

fn round_raw(bid: f64, ask: f64, tick: f64) -> (f64, f64) {
    (
        (bid / tick + 1e-9).floor() * tick,
        (ask / tick - 1e-9).ceil() * tick,
    )
}

fn glft_coeffs(params: &ASParams) -> (f64, f64) {
    let (g, k, a) = (params.gamma(), params.k(), params.a_rate());
    let s2 = params.sigma().powi(2);
    let c1 = (g / k).ln_1p() / g;
    let c2 = (s2 * g / (2.0 * k * a) * (1.0 + g / k).powf(1.0 + k / g)).sqrt();
    (c1, c2)
}

/// Closed-form asymptotic GLFT distances `(delta_bid, delta_ask)` (API_SPEC 9.3).
///
/// Gueant, Lehalle and Fernandez-Tapia (2013) asymptotic approximation for
/// the inventory-constrained market maker:
///
/// ```text
/// c1 = (1/gamma) ln(1 + gamma/k)
/// c2 = sqrt( sigma^2 gamma / (2 k A) * (1 + gamma/k)^(1 + k/gamma) )
/// delta_bid = c1 + (2q + 1)/2 * c2,   delta_ask = c1 - (2q - 1)/2 * c2
/// ```
///
/// Time-independent; quotes are `bid = s - delta_bid`, `ask = s + delta_ask`.
/// `q` must be finite.
pub fn glft_half_spreads(params: &ASParams, q: f64) -> Result<(f64, f64), EtradeError> {
    require(q.is_finite(), "q must be finite")?;
    let (c1, c2) = glft_coeffs(params);
    Ok((
        c1 + (2.0 * q + 1.0) / 2.0 * c2,
        c1 - (2.0 * q - 1.0) / 2.0 * c2,
    ))
}

/// Size-weighted mid `(bid*ask_size + ask*bid_size)/(bid_size + ask_size)`.
///
/// A simple proxy for the microprice (Stoikov 2018): a large bid queue pulls
/// the fair value towards the ask.  Prices finite with `ask >= bid`; sizes
/// finite `>= 0` with a positive sum.
pub fn weighted_mid(bid: f64, ask: f64, bid_size: f64, ask_size: f64) -> Result<f64, EtradeError> {
    require(bid.is_finite(), "bid must be finite")?;
    require(ask.is_finite(), "ask must be finite")?;
    require(ask >= bid, "ask must be >= bid")?;
    require(
        bid_size.is_finite() && bid_size >= 0.0 && ask_size.is_finite() && ask_size >= 0.0,
        "sizes must be finite and >= 0",
    )?;
    require(bid_size + ask_size > 0.0, "bid_size + ask_size must be > 0")?;
    Ok((bid * ask_size + ask * bid_size) / (bid_size + ask_size))
}

/// Per-update order-flow imbalance of Cont, Kukanov and Stoikov (2014).
///
/// For consecutive top-of-book states `n-1 -> n`:
///
/// ```text
/// e_n = 1[Pb_n >= Pb_{n-1}] Qb_n - 1[Pb_n <= Pb_{n-1}] Qb_{n-1}
///     - 1[Pa_n <= Pa_{n-1}] Qa_n + 1[Pa_n >= Pa_{n-1}] Qa_{n-1}
/// ```
///
/// Returns the `len - 1` values `e_1..e_n`; their sum over a window is the
/// OFI.  At least 2 states, equal lengths, finite, sizes `>= 0`.
pub fn order_flow_imbalance(
    bid_px: &[f64],
    bid_sz: &[f64],
    ask_px: &[f64],
    ask_sz: &[f64],
) -> Result<Vec<f64>, EtradeError> {
    let n = bid_px.len();
    require(n >= 2, "need at least 2 book states")?;
    require(
        bid_sz.len() == n && ask_px.len() == n && ask_sz.len() == n,
        "book series must be 1-D and equal length",
    )?;
    require(
        [bid_px, bid_sz, ask_px, ask_sz]
            .iter()
            .all(|a| a.iter().all(|v| v.is_finite())),
        "book series must be finite",
    )?;
    require(
        bid_sz.iter().chain(ask_sz).all(|v| *v >= 0.0),
        "sizes must be >= 0",
    )?;
    let ind = |c: bool| if c { 1.0 } else { 0.0 };
    Ok((1..n)
        .map(|j| {
            let (pb0, pb1, pa0, pa1) = (bid_px[j - 1], bid_px[j], ask_px[j - 1], ask_px[j]);
            ind(pb1 >= pb0) * bid_sz[j]
                - ind(pb1 <= pb0) * bid_sz[j - 1]
                - ind(pa1 <= pa0) * ask_sz[j]
                + ind(pa1 >= pa0) * ask_sz[j - 1]
        })
        .collect())
}

// ------------------------------------------------------------ simulation

/// Quoting agent for [`simulate_mm`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MMAgent {
    /// Avellaneda-Stoikov reservation price and time-dependent spread.
    As,
    /// Gueant-Lehalle-Fernandez-Tapia asymptotic quotes ([`glft_half_spreads`]).
    Glft,
    /// Symmetric fixed spread around fair value, no inventory skew.
    Fixed,
}

impl MMAgent {
    /// Lowercase name, the inverse of [`FromStr`](std::str::FromStr).
    pub fn as_str(self) -> &'static str {
        match self {
            MMAgent::As => "as",
            MMAgent::Glft => "glft",
            MMAgent::Fixed => "fixed",
        }
    }
}

impl std::str::FromStr for MMAgent {
    type Err = EtradeError;

    /// Parse the exact lowercase name (mirrors the reference).
    ///
    /// ```
    /// use etrade::mm::MMAgent;
    /// let a: MMAgent = "glft".parse().unwrap();
    /// assert_eq!(a, MMAgent::Glft);
    /// assert_eq!(a.to_string(), "glft");
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "as" => Ok(MMAgent::As),
            "glft" => Ok(MMAgent::Glft),
            "fixed" => Ok(MMAgent::Fixed),
            _ => Err(EtradeError::InvalidInput(
                "agent must be 'as', 'glft' or 'fixed'".to_string(),
            )),
        }
    }
}

impl std::fmt::Display for MMAgent {
    /// Writes the same lowercase name [`FromStr`](std::str::FromStr) accepts.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Frictions and agent for [`simulate_mm`] (API_SPEC 9.1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MMConfig {
    /// Quoting agent ([`MMAgent::As`] by default).
    pub agent: MMAgent,
    /// Tick size, `>= 0` (`0` = continuous prices).
    pub tick: f64,
    /// Per-share fee per fill, finite (positive = rebate received).
    pub maker_fee: f64,
    /// Quote staleness in steps, `<= n_steps`.
    pub latency_steps: usize,
    /// Probability per step that an informed trader arrives, in `[0, 1]`.
    pub informed_prob: f64,
    /// Accuracy of the fair-value signal, in `[0.5, 1]` (`0.5` = no signal).
    pub signal_accuracy: f64,
    /// Fixed agent's total spread, `> 0`; `None` uses [`average_as_spread`].
    pub fixed_spread: Option<f64>,
}

impl Default for MMConfig {
    fn default() -> Self {
        Self {
            agent: MMAgent::As,
            tick: 0.0,
            maker_fee: 0.0,
            latency_steps: 0,
            informed_prob: 0.0,
            signal_accuracy: 0.5,
            fixed_spread: None,
        }
    }
}

impl MMConfig {
    fn validate(&self) -> Result<(), EtradeError> {
        require(
            self.tick.is_finite() && self.tick >= 0.0,
            "tick must be finite and >= 0",
        )?;
        require(self.maker_fee.is_finite(), "maker_fee must be finite")?;
        require(
            self.informed_prob.is_finite() && (0.0..=1.0).contains(&self.informed_prob),
            "informed_prob must be in [0, 1]",
        )?;
        require(
            self.signal_accuracy.is_finite() && (0.5..=1.0).contains(&self.signal_accuracy),
            "signal_accuracy must be in [0.5, 1]",
        )?;
        if let Some(f) = self.fixed_spread {
            require(
                f.is_finite() && f > 0.0,
                "fixed_spread must be finite and > 0",
            )?;
        }
        Ok(())
    }
}

/// Statistic selector for [`MMResult`] (API_SPEC 9.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MMStat {
    /// Terminal PnL after liquidation.
    Pnl,
    /// Spread captured against the mid before the move.
    Spread,
    /// Maker fees (positive = rebates).
    Fees,
    /// Inventory PnL `sum q * move`.
    Inventory,
    /// Terminal liquidation penalty `eta_liq |q_T|`.
    Liquidation,
    /// One-step markout of the fills (negative = adverse selection).
    Markout,
    /// Number of fills, both sides counted.
    Fills,
    /// Number of fills against informed traders.
    InformedFills,
    /// `sqrt(time average of q^2)`.
    RmsQ,
}

/// Per-path statistics of a [`simulate_mm`] run, plus the largest
/// attribution residual seen.
#[derive(Debug, Clone)]
pub struct MMResult {
    pnl: Vec<f64>,
    spread: Vec<f64>,
    fees: Vec<f64>,
    inventory: Vec<f64>,
    liquidation: Vec<f64>,
    markout: Vec<f64>,
    fills: Vec<f64>,
    informed_fills: Vec<f64>,
    rms_q: Vec<f64>,
    max_identity_error: f64,
}

impl MMResult {
    /// Number of simulated paths.
    pub fn n_paths(&self) -> usize {
        self.pnl.len()
    }
    /// Per-path values of a statistic (length `n_paths`).
    pub fn paths(&self, stat: MMStat) -> &[f64] {
        match stat {
            MMStat::Pnl => &self.pnl,
            MMStat::Spread => &self.spread,
            MMStat::Fees => &self.fees,
            MMStat::Inventory => &self.inventory,
            MMStat::Liquidation => &self.liquidation,
            MMStat::Markout => &self.markout,
            MMStat::Fills => &self.fills,
            MMStat::InformedFills => &self.informed_fills,
            MMStat::RmsQ => &self.rms_q,
        }
    }
    /// Mean over paths of a statistic.
    pub fn mean(&self, stat: MMStat) -> f64 {
        let v = self.paths(stat);
        v.iter().sum::<f64>() / v.len() as f64
    }
    /// Standard error of the mean: sample std (ddof 1) / sqrt(n_paths).
    pub fn se(&self, stat: MMStat) -> f64 {
        let v = self.paths(stat);
        let m = self.mean(stat);
        let n = v.len() as f64;
        let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0);
        var.sqrt() / n.sqrt()
    }
    /// Largest `|spread + fees + inventory - liquidation - pnl| / max(1, |pnl|)`
    /// over paths (the attribution identity residual).
    pub fn max_identity_error(&self) -> f64 {
        self.max_identity_error
    }
}

/// Simulate the market maker with frictions (API_SPEC 9.4); deterministic
/// per `(seed, config, n_paths)` with `ChaCha12Rng`.
///
/// Per step: draw `z, u_sig, u_ask, u_bid, u_inf` (always all five); shift
/// fair value by the signal from the mid `latency_steps` steps ago; quote
/// with the chosen agent and round to the tick grid; uninformed fills
/// against the true current mid; an informed trader takes a quote inside the
/// coming move; account; move the mid.  `2 <= n_paths <= MAX_PATHS`.
///
/// ```
/// use etrade::mm::{simulate_mm, MMConfig, MMStat};
/// use etrade::ASParams;
///
/// let p = ASParams::new(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, 10, 0.0).unwrap();
/// let cfg = MMConfig { tick: 0.01, informed_prob: 0.1, ..MMConfig::default() };
/// let r = simulate_mm(&p, 200, 7, &cfg).unwrap();
/// assert_eq!(r.n_paths(), 200);
/// assert!(r.max_identity_error() < 1e-9);
/// assert!(r.mean(MMStat::Fills) > 0.0);
/// ```
pub fn simulate_mm(
    params: &ASParams,
    n_paths: usize,
    seed: u64,
    config: &MMConfig,
) -> Result<MMResult, EtradeError> {
    config.validate()?;
    require(
        (2..=MAX_PATHS).contains(&n_paths),
        "n_paths must be an integer in [2, 10000000]",
    )?;
    let n_steps = params.n_steps();
    require(
        config.latency_steps <= n_steps,
        "latency_steps must be <= n_steps",
    )?;

    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    let (g, sig, big_t, dt, k, a_rate) = (
        params.gamma(),
        params.sigma(),
        params.horizon(),
        params.dt(),
        params.k(),
        params.a_rate(),
    );
    let qmax = params.q_max() as f64;
    let sig2 = sig * sig;
    let sqdt = dt.sqrt();
    let markup = 2.0 / g * (g / k).ln_1p();
    let fixed_half = 0.5
        * config
            .fixed_spread
            .unwrap_or_else(|| average_as_spread(params));
    let (c1, c2) = glft_coeffs(params);
    let shift_scale = (2.0 * config.signal_accuracy - 1.0) * sig * sqdt * SQRT_2_OVER_PI;
    let tick = config.tick;
    let fee = config.maker_fee;

    let mut r = MMResult {
        pnl: Vec::with_capacity(n_paths),
        spread: Vec::with_capacity(n_paths),
        fees: Vec::with_capacity(n_paths),
        inventory: Vec::with_capacity(n_paths),
        liquidation: Vec::with_capacity(n_paths),
        markout: Vec::with_capacity(n_paths),
        fills: Vec::with_capacity(n_paths),
        informed_fills: Vec::with_capacity(n_paths),
        rms_q: Vec::with_capacity(n_paths),
        max_identity_error: 0.0,
    };
    let mut hist: VecDeque<f64> = VecDeque::with_capacity(config.latency_steps + 1);

    for _ in 0..n_paths {
        let mut s = params.s0();
        hist.clear();
        hist.extend(std::iter::repeat(s).take(config.latency_steps + 1));
        let mut q = 0.0_f64;
        let mut cash = 0.0;
        let (mut spread, mut fees, mut inventory, mut markout) = (0.0, 0.0, 0.0, 0.0);
        let (mut fills, mut informed_fills, mut sum_q2) = (0.0, 0.0, 0.0);

        for i in 0..n_steps {
            let t = i as f64 * dt;
            // 1. draws, always all five, in the pinned order
            let z: f64 = rng.sample(StandardNormal);
            let u_sig: f64 = rng.gen();
            let u_ask: f64 = rng.gen();
            let u_bid: f64 = rng.gen();
            let u_inf: f64 = rng.gen();
            // 2. fair value from the stale mid plus the signal
            let sz = if z > 0.0 {
                1.0
            } else if z < 0.0 {
                -1.0
            } else {
                0.0
            };
            let pred = if u_sig < config.signal_accuracy {
                sz
            } else {
                -sz
            };
            let fair = hist[0] + shift_scale * pred;
            // 3. quotes
            let (mut bid, mut ask) = match config.agent {
                MMAgent::As => {
                    let res = fair - q * g * sig2 * (big_t - t);
                    let half = 0.5 * (g * sig2 * (big_t - t) + markup);
                    (res - half, res + half)
                }
                MMAgent::Glft => (
                    fair - (c1 + (2.0 * q + 1.0) / 2.0 * c2),
                    fair + (c1 - (2.0 * q - 1.0) / 2.0 * c2),
                ),
                MMAgent::Fixed => (fair - fixed_half, fair + fixed_half),
            };
            if tick > 0.0 {
                (bid, ask) = round_raw(bid, ask, tick);
            }
            // 4. uninformed fills against the true current mid
            let can_ask = q > -qmax;
            let can_bid = q < qmax;
            let p_ask = -(-a_rate * (-k * (ask - s)).exp() * dt).exp_m1();
            let p_bid = -(-a_rate * (-k * (s - bid)).exp() * dt).exp_m1();
            let fill_ask = can_ask && u_ask < p_ask;
            let fill_bid = can_bid && u_bid < p_bid;
            // 5. informed fills
            let mv = sig * sqdt * z;
            let informed = u_inf < config.informed_prob;
            let inf_ask = informed && z > 0.0 && can_ask && !fill_ask && ask < s + mv;
            let inf_bid = informed && z < 0.0 && can_bid && !fill_bid && bid > s + mv;
            // 6. updates
            let fa = fill_ask || inf_ask;
            let fb = fill_bid || inf_bid;
            let fa_f = if fa { 1.0 } else { 0.0 };
            let fb_f = if fb { 1.0 } else { 0.0 };
            let n_fill = fa_f + fb_f;
            q += fb_f - fa_f;
            cash += (if fa { ask } else { 0.0 }) - (if fb { bid } else { 0.0 }) + fee * n_fill;
            spread += (if fa { ask - s } else { 0.0 }) + (if fb { s - bid } else { 0.0 });
            fees += fee * n_fill;
            fills += n_fill;
            informed_fills += (inf_ask as u8 + inf_bid as u8) as f64;
            inventory += q * mv;
            markout += (fb_f - fa_f) * mv;
            sum_q2 += q * q;
            // 7. mid move
            s += mv;
            hist.pop_front();
            hist.push_back(s);
        }

        let liq = params.eta_liq() * q.abs();
        let pnl = cash + q * s - liq;
        let recon = spread + fees + inventory - liq;
        let err = (recon - pnl).abs() / pnl.abs().max(1.0);
        r.max_identity_error = r.max_identity_error.max(err);
        r.pnl.push(pnl);
        r.spread.push(spread);
        r.fees.push(fees);
        r.inventory.push(inventory);
        r.liquidation.push(liq);
        r.markout.push(markout);
        r.fills.push(fills);
        r.informed_fills.push(informed_fills);
        r.rms_q.push((sum_q2 / n_steps as f64).sqrt());
    }
    Ok(r)
}
