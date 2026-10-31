//! Hedged market making and risk controls (API_SPEC.md Section 14).
//!
//! The Section 9 maker ([`crate::mm`]) can only shed inventory by skewing its
//! own quotes.  A desk that makes markets in an ETF also trades the future or
//! the basket: it lays inventory risk off in a correlated instrument B, pays
//! B's half spread and fee to do so, and keeps the basis risk.  The same desk
//! runs hard risk controls outside the quoting model: a loss limit, a
//! position limit, a quote throttle and a kill switch.  This module adds both
//! to the Section 9 loop and keeps the Section 9 accounting exact.
//!
//! B moves `sigma*sqrt(dt)*(rho*z + sqrt(1 - rho^2)*z_b)` per step (quoted in
//! A-equivalent units), so the minimum-variance hedge ratio is `rho`
//! (Ederington 1979).  The quoting agent skews on the risk-equivalent
//! exposure `x = q + rho*y` (the multi-asset reservation price, Gueant 2017).
//!
//! Per path, per step, six numbers are drawn in the pinned order, always:
//! `z, u_sig, u_ask, u_bid, u_inf` (Section 9.4), then `z_b ~ N(0,1)`.  This
//! port uses its own RNG per Section 4 (`ChaCha12Rng::seed_from_u64`), so its
//! sample values differ from Python while all statistics agree within SE
//! bands.
//!
//! Attribution (exact per path, Section 14.5):
//!
//! ```text
//! pnl = spread + fees + inventory - liquidation + hedge_pnl - hedge_costs - risk_costs
//! ```
//!
//! `hedge_pnl` is the sum of `y * (B move)`; `hedge_costs` is
//! `hedge_cost * |dy|` over every B trade including the terminal unwind;
//! `risk_costs` is `flatten_penalty * |q|` over every forced flatten of A.

use std::collections::VecDeque;

use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rand_distr::StandardNormal;

use crate::almgren_chriss::MAX_PATHS;
use crate::avellaneda_stoikov::{average_as_spread, ASParams};
use crate::error::{require, EtradeError};
use crate::mm::{MMAgent, MMConfig};

const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;

fn add(stat: HedgeStat, v: f64, acc: &mut [f64; 21]) {
    acc[stat as usize] += v;
}

// ------------------------------------------------------------ deterministic pieces

/// B shares to trade under the pinned hedging rule (API_SPEC 14.2).
///
/// With `u = q + y/hedge_ratio` the A shares not covered at the chosen ratio,
/// trade `dy = -hedge_ratio * (1 - target_frac) * u` when `|u| > threshold`
/// (leaving `target_frac * u` uncovered), else `0`.  `hedge_ratio = 0`
/// disables hedging and returns `0`.  `q`, `y` finite; `hedge_ratio >= 0`,
/// `threshold >= 0` finite; `target_frac` in `[0, 1)`.
pub fn hedge_trade(
    q: f64,
    y: f64,
    hedge_ratio: f64,
    threshold: f64,
    target_frac: f64,
) -> Result<f64, EtradeError> {
    require(q.is_finite() && y.is_finite(), "q and y must be finite")?;
    require(
        hedge_ratio.is_finite() && hedge_ratio >= 0.0,
        "hedge_ratio must be finite and >= 0",
    )?;
    require(
        threshold.is_finite() && threshold >= 0.0,
        "hedge_threshold must be finite and >= 0",
    )?;
    require(
        target_frac.is_finite() && (0.0..1.0).contains(&target_frac),
        "hedge_target must be in [0, 1)",
    )?;
    Ok(hedge_raw(q, y, hedge_ratio, threshold, target_frac))
}

fn hedge_raw(q: f64, y: f64, h: f64, threshold: f64, target: f64) -> f64 {
    if h == 0.0 {
        return 0.0;
    }
    let u = q + y / h;
    if u.abs() <= threshold {
        0.0
    } else {
        -h * (1.0 - target) * u
    }
}

/// Pinned sample quantile (API_SPEC 14.6): Hyndman and Fan (1996) type 7,
/// the default of NumPy and R.
///
/// Sort ascending to `v`, `h = (n - 1) p`, `j = floor(h)`;
/// `Q = v[j] + (h - j)(v[j+1] - v[j])` (`Q = v[n-1]` when `j >= n - 1`).
/// `values` non-empty and finite, `p` in `[0, 1]`.
pub fn quantile(values: &[f64], p: f64) -> Result<f64, EtradeError> {
    require(!values.is_empty(), "values must be a non-empty 1-D array")?;
    require(
        values.iter().all(|v| v.is_finite()),
        "values must be finite",
    )?;
    require(
        p.is_finite() && (0.0..=1.0).contains(&p),
        "p must be in [0, 1]",
    )?;
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let h = (v.len() - 1) as f64 * p;
    let j = h.floor() as usize;
    if j >= v.len() - 1 {
        return Ok(v[v.len() - 1]);
    }
    Ok(v[j] + (h - j as f64) * (v[j + 1] - v[j]))
}

// ------------------------------------------------------------ configuration

/// Hedge instrument and rule (API_SPEC 14.1); `hedge_ratio = 0` is no hedge.
/// Defaults are all `0`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct HedgeConfig {
    /// Correlation of B's move with A's, in `[-1, 1]`.
    pub rho: f64,
    /// B shares per A share, `>= 0` (`0` = no hedging).
    pub hedge_ratio: f64,
    /// Cost per B share traded (half spread plus fee), `>= 0`.
    pub hedge_cost: f64,
    /// Uncovered A shares tolerated before hedging, `>= 0`.
    pub hedge_threshold: f64,
    /// Fraction of the uncovered shares left unhedged, in `[0, 1)`.
    pub hedge_target: f64,
}

impl HedgeConfig {
    /// Check the bounds of API_SPEC 14.1.
    pub fn validate(&self) -> Result<(), EtradeError> {
        require(
            self.rho.is_finite() && (-1.0..=1.0).contains(&self.rho),
            "rho must be in [-1, 1]",
        )?;
        require(
            self.hedge_ratio.is_finite() && self.hedge_ratio >= 0.0,
            "hedge_ratio must be finite and >= 0",
        )?;
        require(
            self.hedge_cost.is_finite() && self.hedge_cost >= 0.0,
            "hedge_cost must be finite and >= 0",
        )?;
        require(
            self.hedge_threshold.is_finite() && self.hedge_threshold >= 0.0,
            "hedge_threshold must be finite and >= 0",
        )?;
        require(
            self.hedge_target.is_finite() && (0.0..1.0).contains(&self.hedge_target),
            "hedge_target must be in [0, 1)",
        )
    }
}

/// Risk controls (API_SPEC 14.1); every default is off.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RiskConfig {
    /// Stop when the marked PnL falls below `-loss_limit`, `>= 0` (`0` = off).
    pub loss_limit: f64,
    /// Flatten A when `|q| >= position_limit` (`0` = off).
    pub position_limit: u32,
    /// Penalty per A share flattened by a control, `>= 0`.
    pub flatten_penalty: f64,
    /// Quote updates allowed per `throttle_window` steps (`0` = off).
    pub throttle_updates: usize,
    /// Throttle window in steps, in `[1, n_steps]`.
    pub throttle_window: usize,
    /// Step at which the kill switch flattens and stops, in `[-1, n_steps]`
    /// (`-1` = off).
    pub kill_step: i64,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            loss_limit: 0.0,
            position_limit: 0,
            flatten_penalty: 0.0,
            throttle_updates: 0,
            throttle_window: 1,
            kill_step: -1,
        }
    }
}

impl RiskConfig {
    /// Check the bounds of API_SPEC 14.1 that do not depend on `n_steps`.
    pub fn validate(&self) -> Result<(), EtradeError> {
        require(
            self.loss_limit.is_finite() && self.loss_limit >= 0.0,
            "loss_limit must be finite and >= 0",
        )?;
        require(
            self.flatten_penalty.is_finite() && self.flatten_penalty >= 0.0,
            "flatten_penalty must be finite and >= 0",
        )?;
        require(
            self.throttle_window >= 1,
            "throttle_window must be an integer >= 1",
        )?;
        require(self.kill_step >= -1, "kill_step must be an integer >= -1")
    }
}

/// Statistic selector for [`HedgedResult`] (API_SPEC 14.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HedgeStat {
    /// Terminal PnL after liquidation and the hedge unwind.
    Pnl,
    /// Spread captured against the mid before the move.
    Spread,
    /// Maker fees (positive = rebates).
    Fees,
    /// Inventory PnL of A, `sum q * move`.
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
    /// PnL of the B position, `sum y * move_b`.
    HedgePnl,
    /// `hedge_cost * |dy|` over every B trade, the unwind included.
    HedgeCosts,
    /// Number of B trades, the unwind included.
    HedgeTrades,
    /// `flatten_penalty * |q|` over every forced flatten of A.
    RiskCosts,
    /// `sqrt(time average of x^2)`, `x = q + rho y`.
    RmsX,
    /// `sqrt(sum of skew^2 while active / n_steps)`.
    RmsSkew,
    /// Number of steps whose desired quotes became the standing quotes.
    QuoteUpdates,
    /// 1 if the loss limit stopped the path.
    LossStop,
    /// 1 if the kill switch stopped the path.
    KillStop,
    /// Number of position-limit flattens.
    PosFlattens,
    /// `max(0, -loss_limit - mark)` at a loss stop (else 0).
    Overshoot,
    /// `|mark - mark_prev|` at a loss stop (else 0).
    StopMove,
}

impl HedgeStat {
    /// Every statistic, in the order of API_SPEC 14.5.
    pub const ALL: [HedgeStat; 21] = [
        HedgeStat::Pnl,
        HedgeStat::Spread,
        HedgeStat::Fees,
        HedgeStat::Inventory,
        HedgeStat::Liquidation,
        HedgeStat::Markout,
        HedgeStat::Fills,
        HedgeStat::InformedFills,
        HedgeStat::RmsQ,
        HedgeStat::HedgePnl,
        HedgeStat::HedgeCosts,
        HedgeStat::HedgeTrades,
        HedgeStat::RiskCosts,
        HedgeStat::RmsX,
        HedgeStat::RmsSkew,
        HedgeStat::QuoteUpdates,
        HedgeStat::LossStop,
        HedgeStat::KillStop,
        HedgeStat::PosFlattens,
        HedgeStat::Overshoot,
        HedgeStat::StopMove,
    ];

    /// The spec name of the statistic (`"pnl"`, `"hedge_pnl"`, ...).
    pub fn name(self) -> &'static str {
        match self {
            HedgeStat::Pnl => "pnl",
            HedgeStat::Spread => "spread",
            HedgeStat::Fees => "fees",
            HedgeStat::Inventory => "inventory",
            HedgeStat::Liquidation => "liquidation",
            HedgeStat::Markout => "markout",
            HedgeStat::Fills => "fills",
            HedgeStat::InformedFills => "informed_fills",
            HedgeStat::RmsQ => "rms_q",
            HedgeStat::HedgePnl => "hedge_pnl",
            HedgeStat::HedgeCosts => "hedge_costs",
            HedgeStat::HedgeTrades => "hedge_trades",
            HedgeStat::RiskCosts => "risk_costs",
            HedgeStat::RmsX => "rms_x",
            HedgeStat::RmsSkew => "rms_skew",
            HedgeStat::QuoteUpdates => "quote_updates",
            HedgeStat::LossStop => "loss_stop",
            HedgeStat::KillStop => "kill_stop",
            HedgeStat::PosFlattens => "pos_flattens",
            HedgeStat::Overshoot => "overshoot",
            HedgeStat::StopMove => "stop_move",
        }
    }

    /// Look a statistic up by its spec name; unknown names are input errors.
    pub fn from_name(name: &str) -> Result<HedgeStat, EtradeError> {
        HedgeStat::ALL
            .iter()
            .copied()
            .find(|s| s.name() == name)
            .ok_or_else(|| EtradeError::InvalidInput(format!("unknown statistic '{name}'")))
    }
}

/// Per-path statistics of a [`simulate_hedged`] run (one vector per
/// [`HedgeStat`]), plus the largest attribution residual seen.
#[derive(Debug, Clone)]
pub struct HedgedResult {
    paths: Vec<Vec<f64>>,
    max_identity_error: f64,
}

impl HedgedResult {
    /// Number of simulated paths.
    pub fn n_paths(&self) -> usize {
        self.paths[0].len()
    }
    /// Per-path values of a statistic (length `n_paths`).
    pub fn paths(&self, stat: HedgeStat) -> &[f64] {
        &self.paths[stat as usize]
    }
    /// Per-path values of a statistic named as in API_SPEC 14.5.
    pub fn paths_by_name(&self, name: &str) -> Result<&[f64], EtradeError> {
        Ok(self.paths(HedgeStat::from_name(name)?))
    }
    /// Mean over paths of a statistic.
    pub fn mean(&self, stat: HedgeStat) -> f64 {
        let v = self.paths(stat);
        v.iter().sum::<f64>() / v.len() as f64
    }
    /// Standard error of the mean: sample std (ddof 1) / sqrt(n_paths).
    pub fn se(&self, stat: HedgeStat) -> f64 {
        let v = self.paths(stat);
        let m = self.mean(stat);
        let n = v.len() as f64;
        let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0);
        var.sqrt() / n.sqrt()
    }
    /// Population variance `m2 = mean((v - mean)^2)` (API_SPEC 14.5).
    pub fn var(&self, stat: HedgeStat) -> f64 {
        self.moments(stat).0
    }
    /// Delta-method standard error of [`HedgedResult::var`]:
    /// `sqrt(max(0, m4 - m2^2)/n_paths)`.
    pub fn var_se(&self, stat: HedgeStat) -> f64 {
        let (m2, m4) = self.moments(stat);
        ((m4 - m2 * m2).max(0.0) / self.n_paths() as f64).sqrt()
    }
    /// Pinned type-7 [`quantile`] of a statistic over paths; `p` in `[0, 1]`.
    pub fn quantile(&self, stat: HedgeStat, p: f64) -> Result<f64, EtradeError> {
        quantile(self.paths(stat), p)
    }
    /// [`HedgedResult::mean`] by name; unknown names are input errors.
    pub fn mean_by_name(&self, name: &str) -> Result<f64, EtradeError> {
        Ok(self.mean(HedgeStat::from_name(name)?))
    }
    /// [`HedgedResult::se`] by name; unknown names are input errors.
    pub fn se_by_name(&self, name: &str) -> Result<f64, EtradeError> {
        Ok(self.se(HedgeStat::from_name(name)?))
    }
    /// [`HedgedResult::var`] by name; unknown names are input errors.
    pub fn var_by_name(&self, name: &str) -> Result<f64, EtradeError> {
        Ok(self.var(HedgeStat::from_name(name)?))
    }
    /// [`HedgedResult::var_se`] by name; unknown names are input errors.
    pub fn var_se_by_name(&self, name: &str) -> Result<f64, EtradeError> {
        Ok(self.var_se(HedgeStat::from_name(name)?))
    }
    /// [`HedgedResult::quantile`] by name; unknown names are input errors.
    pub fn quantile_by_name(&self, name: &str, p: f64) -> Result<f64, EtradeError> {
        self.quantile(HedgeStat::from_name(name)?, p)
    }
    /// Largest `|spread + fees + inventory - liquidation + hedge_pnl
    /// - hedge_costs - risk_costs - pnl| / max(1, |pnl|)` over paths.
    pub fn max_identity_error(&self) -> f64 {
        self.max_identity_error
    }

    fn moments(&self, stat: HedgeStat) -> (f64, f64) {
        let v = self.paths(stat);
        let m = self.mean(stat);
        let n = v.len() as f64;
        let m2 = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
        let m4 = v.iter().map(|x| (x - m).powi(4)).sum::<f64>() / n;
        (m2, m4)
    }
}

// ------------------------------------------------------------ simulation

/// Simulate the Section 9 maker with a hedge and risk controls (API_SPEC
/// 14.4); deterministic per `(seed, configs, n_paths)` with `ChaCha12Rng`.
///
/// Per step: draw `z, u_sig, u_ask, u_bid, u_inf, z_b` (always all six);
/// stops (kill switch, then loss limit on the marked PnL); desired quotes
/// skewed on `x = q + rho y`; throttle; Section 9 fills against the standing
/// quotes; position limit; hedge in B; accounting and the moves of A and B.
/// With the default `hedge` and `risk` the result equals [`crate::mm::simulate_mm`]
/// in distribution (not sample by sample: the sixth draw shifts the stream).
/// `2 <= n_paths <= MAX_PATHS`, `latency_steps`, `kill_step` and
/// `throttle_window` `<= n_steps`.
///
/// ```
/// use etrade::hedging::{simulate_hedged, HedgeConfig, HedgeStat, RiskConfig};
/// use etrade::mm::MMConfig;
/// use etrade::ASParams;
///
/// let p = ASParams::new(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, 10, 0.0).unwrap();
/// let hedge = HedgeConfig { rho: 0.9, hedge_ratio: 0.9, hedge_cost: 0.005, ..HedgeConfig::default() };
/// let risk = RiskConfig { loss_limit: 2.0, ..RiskConfig::default() };
/// let r = simulate_hedged(&p, 200, 7, &MMConfig::default(), &hedge, &risk).unwrap();
/// assert_eq!(r.n_paths(), 200);
/// assert!(r.max_identity_error() < 1e-9);
/// assert!(r.mean(HedgeStat::HedgeTrades) > 0.0);
/// ```
pub fn simulate_hedged(
    params: &ASParams,
    n_paths: usize,
    seed: u64,
    config: &MMConfig,
    hedge: &HedgeConfig,
    risk: &RiskConfig,
) -> Result<HedgedResult, EtradeError> {
    config.validate()?;
    hedge.validate()?;
    risk.validate()?;
    require(
        (2..=MAX_PATHS).contains(&n_paths),
        "n_paths must be an integer in [2, 10000000]",
    )?;
    let n_steps = params.n_steps();
    require(
        config.latency_steps <= n_steps,
        "latency_steps must be <= n_steps",
    )?;
    require(
        risk.kill_step <= n_steps as i64,
        "kill_step must be <= n_steps",
    )?;
    require(
        risk.throttle_window <= n_steps,
        "throttle_window must be <= n_steps",
    )?;
    // A ring longer than n_steps is never read past its fresh slots, so
    // capping it keeps the semantics and bounds the allocation.
    let m = risk.throttle_updates.min(n_steps);

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
    let c1 = (g / k).ln_1p() / g;
    let c2 = (sig2 * g / (2.0 * k * a_rate) * (1.0 + g / k).powf(1.0 + k / g)).sqrt();
    let shift_scale = (2.0 * config.signal_accuracy - 1.0) * sig * sqdt * SQRT_2_OVER_PI;
    let tick = config.tick;
    let fee = config.maker_fee;
    let (h, rho) = (hedge.hedge_ratio, hedge.rho);
    let rho_perp = (1.0 - rho * rho).max(0.0).sqrt();
    let w = risk.throttle_window as i64;
    let pen = risk.flatten_penalty;
    let hcost = hedge.hedge_cost;
    let loss_limit = risk.loss_limit;
    let pos_limit = risk.position_limit as f64;

    let n_stats = HedgeStat::ALL.len();
    let mut out: Vec<Vec<f64>> = (0..n_stats).map(|_| Vec::with_capacity(n_paths)).collect();
    let mut max_err = 0.0_f64;
    let mut hist: VecDeque<f64> = VecDeque::with_capacity(config.latency_steps + 1);
    // Steps of the last `m` quote updates (ring buffer), "-inf" initially.
    let mut upd: Vec<i64> = vec![i64::MIN / 2; m.max(1)];

    for _ in 0..n_paths {
        let mut s = params.s0();
        let mut b = params.s0();
        hist.clear();
        hist.extend(std::iter::repeat(s).take(config.latency_steps + 1));
        upd.iter_mut().for_each(|v| *v = i64::MIN / 2);
        let mut ptr = 0_usize;
        let (mut q, mut y, mut cash) = (0.0_f64, 0.0_f64, 0.0_f64);
        let mut active = true;
        let (mut bid_q, mut ask_q) = (params.s0(), params.s0());
        let mut mark_prev = 0.0_f64;
        let mut acc = [0.0_f64; 21];
        let (mut sum_q2, mut sum_x2, mut sum_skew2) = (0.0, 0.0, 0.0);

        for i in 0..n_steps {
            let t = i as f64 * dt;
            // 1. draws, always all six, in the pinned order
            let z: f64 = rng.sample(StandardNormal);
            let u_sig: f64 = rng.gen();
            let u_ask: f64 = rng.gen();
            let u_bid: f64 = rng.gen();
            let u_inf: f64 = rng.gen();
            let z_b: f64 = rng.sample(StandardNormal);
            // 2. stops: kill switch first, then the loss limit on the mark
            let mark = cash + q * s + y * b;
            let kill = active && i as i64 == risk.kill_step;
            let loss = active && !kill && loss_limit > 0.0 && mark < -loss_limit;
            if kill {
                acc[HedgeStat::KillStop as usize] = 1.0;
            }
            if loss {
                acc[HedgeStat::LossStop as usize] = 1.0;
                acc[HedgeStat::Overshoot as usize] = (-loss_limit - mark).max(0.0);
                acc[HedgeStat::StopMove as usize] = (mark - mark_prev).abs();
            }
            if kill || loss {
                let cost_a = pen * q.abs();
                cash += q * s - cost_a;
                add(HedgeStat::RiskCosts, cost_a, &mut acc);
                q = 0.0;
                let cost_b = hcost * y.abs();
                cash += y * b - cost_b;
                add(HedgeStat::HedgeCosts, cost_b, &mut acc);
                if y != 0.0 {
                    add(HedgeStat::HedgeTrades, 1.0, &mut acc);
                }
                y = 0.0;
                active = false;
            }
            mark_prev = mark;
            // 3. desired quotes, skewed on the risk-equivalent exposure x
            let x = q + rho * y;
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
            let (skew, mut bid, mut ask) = match config.agent {
                MMAgent::As => {
                    let skew = x * g * sig2 * (big_t - t);
                    let half = 0.5 * (g * sig2 * (big_t - t) + markup);
                    (skew, fair - skew - half, fair - skew + half)
                }
                MMAgent::Glft => (
                    x * c2,
                    fair - (c1 + (2.0 * x + 1.0) / 2.0 * c2),
                    fair + (c1 - (2.0 * x - 1.0) / 2.0 * c2),
                ),
                MMAgent::Fixed => (0.0, fair - fixed_half, fair + fixed_half),
            };
            if tick > 0.0 {
                bid = (bid / tick + 1e-9).floor() * tick;
                ask = (ask / tick - 1e-9).ceil() * tick;
            }
            if active {
                sum_skew2 += skew * skew;
            }
            // 4. throttle: at most m updates in any w consecutive steps
            let allow = if m > 0 {
                let ok = active && upd[ptr] <= i as i64 - w;
                if ok {
                    upd[ptr] = i as i64;
                    ptr = (ptr + 1) % m;
                }
                ok
            } else {
                active
            };
            if allow {
                bid_q = bid;
                ask_q = ask;
                add(HedgeStat::QuoteUpdates, 1.0, &mut acc);
            }
            // 5. Section 9 fills against the standing quotes
            let can_ask = active && q > -qmax;
            let can_bid = active && q < qmax;
            let p_ask = -(-a_rate * (-k * (ask_q - s)).exp() * dt).exp_m1();
            let p_bid = -(-a_rate * (-k * (s - bid_q)).exp() * dt).exp_m1();
            let fill_ask = can_ask && u_ask < p_ask;
            let fill_bid = can_bid && u_bid < p_bid;
            let mv = sig * sqdt * z;
            let informed = u_inf < config.informed_prob;
            let inf_ask = informed && z > 0.0 && can_ask && !fill_ask && ask_q < s + mv;
            let inf_bid = informed && z < 0.0 && can_bid && !fill_bid && bid_q > s + mv;
            let fa = fill_ask || inf_ask;
            let fb = fill_bid || inf_bid;
            let fa_f = if fa { 1.0 } else { 0.0 };
            let fb_f = if fb { 1.0 } else { 0.0 };
            let n_fill = fa_f + fb_f;
            q += fb_f - fa_f;
            cash += (if fa { ask_q } else { 0.0 }) - (if fb { bid_q } else { 0.0 }) + fee * n_fill;
            add(
                HedgeStat::Spread,
                (if fa { ask_q - s } else { 0.0 }) + (if fb { s - bid_q } else { 0.0 }),
                &mut acc,
            );
            add(HedgeStat::Fees, fee * n_fill, &mut acc);
            add(HedgeStat::Fills, n_fill, &mut acc);
            add(
                HedgeStat::InformedFills,
                (inf_ask as u8 + inf_bid as u8) as f64,
                &mut acc,
            );
            // 6. position limit: flatten A at the mid less the penalty
            if active && pos_limit > 0.0 && q.abs() >= pos_limit {
                let cost = pen * q.abs();
                cash += q * s - cost;
                add(HedgeStat::RiskCosts, cost, &mut acc);
                add(HedgeStat::PosFlattens, 1.0, &mut acc);
                q = 0.0;
            }
            // 7. hedge in B at its mid plus the hedge cost
            if active && h > 0.0 {
                let u = q + y / h;
                if u.abs() > hedge.hedge_threshold {
                    let dy = hedge_raw(q, y, h, hedge.hedge_threshold, hedge.hedge_target);
                    let cost = hcost * dy.abs();
                    cash -= dy * b + cost;
                    add(HedgeStat::HedgeCosts, cost, &mut acc);
                    add(HedgeStat::HedgeTrades, 1.0, &mut acc);
                    y += dy;
                }
            }
            // 8. accounting, then the moves of A and B
            let mv_b = sig * sqdt * (rho * z + rho_perp * z_b);
            add(HedgeStat::Inventory, q * mv, &mut acc);
            add(HedgeStat::Markout, (fb_f - fa_f) * mv, &mut acc);
            add(HedgeStat::HedgePnl, y * mv_b, &mut acc);
            let x = q + rho * y;
            sum_q2 += q * q;
            sum_x2 += x * x;
            s += mv;
            b += mv_b;
            hist.pop_front();
            hist.push_back(s);
        }

        let liq = params.eta_liq() * q.abs();
        let unwind = hcost * y.abs();
        add(HedgeStat::HedgeCosts, unwind, &mut acc);
        if y != 0.0 {
            add(HedgeStat::HedgeTrades, 1.0, &mut acc);
        }
        let pnl = cash + q * s + y * b - liq - unwind;
        let ns = n_steps as f64;
        acc[HedgeStat::Liquidation as usize] = liq;
        acc[HedgeStat::Pnl as usize] = pnl;
        acc[HedgeStat::RmsQ as usize] = (sum_q2 / ns).sqrt();
        acc[HedgeStat::RmsX as usize] = (sum_x2 / ns).sqrt();
        acc[HedgeStat::RmsSkew as usize] = (sum_skew2 / ns).sqrt();
        let a = |st: HedgeStat| acc[st as usize];
        let recon = a(HedgeStat::Spread) + a(HedgeStat::Fees) + a(HedgeStat::Inventory) - liq
            + a(HedgeStat::HedgePnl)
            - a(HedgeStat::HedgeCosts)
            - a(HedgeStat::RiskCosts);
        max_err = max_err.max((recon - pnl).abs() / pnl.abs().max(1.0));
        for (col, v) in out.iter_mut().zip(acc.iter()) {
            col.push(*v);
        }
    }
    Ok(HedgedResult {
        paths: out,
        max_identity_error: max_err,
    })
}
