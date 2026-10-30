//! Execution extensions: transient impact, adaptive re-planning and
//! internalisation (API_SPEC.md Section 12).
//!
//! Sections 1 and 8 treat temporary impact as instantaneous.  Real order
//! books refill at a finite speed, so a trade also raises the price paid by
//! the trades that follow it.  Obizhaeva and Wang (2013) model this with a
//! limit-order book of constant depth whose displacement decays
//! exponentially (resilience `rho`); the optimal schedule then trades a block
//! at the start, a constant rate in between and a block at the end.  This
//! module provides that model in discrete time, a Monte Carlo comparison of
//! a static schedule with one re-planned every interval, and the crossing
//! (internalisation) of opposite parent orders before any reaches the market.
//!
//! # Transient impact (API_SPEC 12.1)
//!
//! Sell `X` over `N` intervals of length `tau`; trade `n_j` at the start of
//! interval `j`.  The book displacement obeys
//! `D_1 = 0`, `D_{j+1} = a*(D_j + kappa*n_j)`, `a = exp(-rho*tau)`.  A trade
//! pays `n*D_j + kappa*n^2/2`; with permanent impact `gamma_p`,
//! `E[IS] = gamma_p*X^2/2 + n'An/2`, `A = kappa*K - gamma_p*I`,
//! `K_ij = a^|i-j|` (Kac-Murdock-Szego, tridiagonal inverse), so the static
//! optimum is one exact tridiagonal (Thomas) solve.  With `gamma_p = 0` the
//! solution is proportional to `[1, 1-a, .., 1-a, 1]` (Obizhaeva and Wang
//! 2013, Proposition 1 in discrete time; Alfonsi, Fruth and Schied 2010).  As
//! `rho -> infinity` the cost is Section 1's linear temporary impact with
//! `kappa = 2*eta/tau` (Almgren and Chriss 2000).
//!
//! # Adaptive re-planning (API_SPEC 12.2)
//!
//! Section 1 dynamics plus an observable AR(1) drift signal and random
//! partial fills.  The adaptive trader re-solves the remaining mean-variance
//! problem every interval; the static trader sends the time-0 plan.  Without
//! signal and with full fills the two coincide path by path (the
//! Almgren-Chriss schedule is time consistent).  Almgren and Lorenz (2007)
//! describe a separate gain from non-time-consistent rules under a global
//! mean-variance criterion, which this module does not model.
//!
//! # Internalisation (API_SPEC 12.3)
//!
//! Opposite parent orders are crossed at the mid; only the net residual is
//! sent to the market and costed with the Section 8 model; the saving is
//! attributed pro rata.  Crossing never increases total impact cost.
//!
//! # References
//!
//! * Obizhaeva, A. A. and Wang, J. (2013). Optimal trading strategy and
//!   supply/demand dynamics. Journal of Financial Markets 16(1), 1-32.
//! * Alfonsi, A., Fruth, A. and Schied, A. (2010). Optimal execution
//!   strategies in limit order books with general shape functions.
//!   Quantitative Finance 10(2), 143-157.
//! * Almgren, R. and Chriss, N. (2000). Optimal execution of portfolio
//!   transactions. Journal of Risk 3(2), 5-39.
//! * Almgren, R. and Lorenz, J. (2007). Adaptive arrival price. Trading
//!   2007(1), 59-66.

use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rand_distr::StandardNormal;

use crate::almgren_chriss::{ACParams, MAX_PATHS};
use crate::error::{require, EtradeError};
use crate::impact::{impact_expected_cost, vwap_holdings};
use crate::market::{bps, impact_params_from_market, ImpactMarket};

/// Upper bound on intervals for every Section 12 entry point (pinned).
pub const MAX_EXEC_INTERVALS: usize = 1_000;

/// Upper bound on the number of parent orders per side in [`internalise`].
pub const MAX_ORDERS: usize = 10_000;

const N_INTERVALS_MSG: &str = "n_intervals must be an integer in [1, 1000]";

/// Solve a symmetric tridiagonal system (`off[k]` couples k and k+1).
fn thomas(diag: &[f64], off: &[f64], rhs: &[f64]) -> Vec<f64> {
    let m = diag.len();
    let mut c = vec![0.0; m];
    let mut d = vec![0.0; m];
    c[0] = if m > 1 { off[0] / diag[0] } else { 0.0 };
    d[0] = rhs[0] / diag[0];
    for k in 1..m {
        let denom = diag[k] - off[k - 1] * c[k - 1];
        c[k] = if k < m - 1 { off[k] / denom } else { 0.0 };
        d[k] = (rhs[k] - off[k - 1] * d[k - 1]) / denom;
    }
    let mut out = vec![0.0; m];
    out[m - 1] = d[m - 1];
    for k in (0..m - 1).rev() {
        out[k] = d[k] - c[k] * out[k + 1];
    }
    out
}

// ------------------------------------------------------------------ transient

/// Transient-impact problem (API_SPEC 12.1); validated on construction.
///
/// `kappa` is the price displacement per share of a trade (inverse book
/// depth, currency/share^2), `rho` the resilience rate per time unit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransientParams {
    total_shares: f64,
    n_intervals: usize,
    tau: f64,
    kappa: f64,
    rho: f64,
    gamma_p: f64,
}

impl TransientParams {
    /// Validated constructor.
    ///
    /// * `total_shares` — `X >= 0`, finite;
    /// * `n_intervals`  — `1 <= N <= MAX_EXEC_INTERVALS`;
    /// * `tau`, `kappa` — `> 0`, finite;
    /// * `rho`          — resilience, finite and `> 0`;
    /// * `gamma_p`      — permanent impact, `>= 0`.
    ///
    /// Returns [`EtradeError::InvalidInput`] for every violation, including
    /// the convexity condition `gamma_p < kappa*(1-a)/(1+a)`.
    pub fn new(
        total_shares: f64,
        n_intervals: usize,
        tau: f64,
        kappa: f64,
        rho: f64,
        gamma_p: f64,
    ) -> Result<Self, EtradeError> {
        require(
            total_shares.is_finite() && total_shares >= 0.0,
            "total_shares must be finite and >= 0",
        )?;
        require(
            (1..=MAX_EXEC_INTERVALS).contains(&n_intervals),
            N_INTERVALS_MSG,
        )?;
        require(tau.is_finite() && tau > 0.0, "tau must be > 0")?;
        require(kappa.is_finite() && kappa > 0.0, "kappa must be > 0")?;
        require(rho.is_finite() && rho > 0.0, "rho must be finite and > 0")?;
        require(
            gamma_p.is_finite() && gamma_p >= 0.0,
            "gamma_p must be >= 0",
        )?;
        let p = Self {
            total_shares,
            n_intervals,
            tau,
            kappa,
            rho,
            gamma_p,
        };
        let a = p.decay();
        require(
            gamma_p < kappa * (1.0 - a) / (1.0 + a),
            "model invalid: gamma_p >= kappa*(1-a)/(1+a) (cost not convex in the trades)",
        )?;
        Ok(p)
    }

    /// Shares to sell `X`.
    pub fn total_shares(&self) -> f64 {
        self.total_shares
    }
    /// Number of intervals `N`.
    pub fn n_intervals(&self) -> usize {
        self.n_intervals
    }
    /// Interval length.
    pub fn tau(&self) -> f64 {
        self.tau
    }
    /// Displacement per share traded (inverse book depth).
    pub fn kappa(&self) -> f64 {
        self.kappa
    }
    /// Resilience rate per time unit.
    pub fn rho(&self) -> f64 {
        self.rho
    }
    /// Permanent impact coefficient.
    pub fn gamma_p(&self) -> f64 {
        self.gamma_p
    }
    /// Per-interval displacement decay `a = exp(-rho*tau)`.
    pub fn decay(&self) -> f64 {
        (-self.rho * self.tau).exp()
    }
}

fn check_path(p: &TransientParams, x: &[f64]) -> Result<(), EtradeError> {
    require(
        x.len() == p.n_intervals + 1,
        "holdings must have length n_intervals + 1",
    )?;
    require(x.iter().all(|v| v.is_finite()), "holdings must be finite")?;
    require(
        x[x.len() - 1].abs() <= 1e-9 * x[0].abs().max(1.0),
        "holdings path must fully liquidate (x_N = 0)",
    )
}

/// `E[IS]` of any fully liquidating schedule by the displacement recursion
/// (API_SPEC 12.1).  Trades may have either sign; `X` in the permanent term
/// is the schedule's `x_0`.
pub fn transient_expected_cost(p: &TransientParams, holdings: &[f64]) -> Result<f64, EtradeError> {
    check_path(p, holdings)?;
    let a = p.decay();
    let half = 0.5 * (p.kappa - p.gamma_p);
    let mut d = 0.0;
    let mut total = 0.0;
    for j in 1..holdings.len() {
        let n = holdings[j - 1] - holdings[j];
        total += half * n * n + n * d;
        d = a * (d + p.kappa * n);
    }
    let x0 = holdings[0];
    Ok(0.5 * p.gamma_p * x0 * x0 + total)
}

/// Static schedule minimising `E[IS]` (holdings, length `N + 1`).
///
/// Solves `A n = c 1` with `1'n = X` exactly through the tridiagonal system
/// `((1-a^2)*kappa*I - gamma_p*T) y = T 1` with `T = (1-a^2) K^{-1}`
/// (API_SPEC 12.1).  `x_0 = X` and `x_N = 0` exactly.
pub fn transient_optimal_schedule(p: &TransientParams) -> Vec<f64> {
    let n_int = p.n_intervals;
    let big_x = p.total_shares;
    let mut x = vec![0.0; n_int + 1];
    x[0] = big_x;
    if big_x == 0.0 || n_int == 1 {
        return x;
    }
    let a = p.decay();
    let b = -(-p.rho * p.tau).exp_m1(); // 1 - a without cancellation
    let ka = b * (1.0 + a) * p.kappa; // (1 - a^2) * kappa
    let g = p.gamma_p;
    let mut diag = vec![ka - g * (1.0 + a * a); n_int];
    diag[0] = ka - g;
    diag[n_int - 1] = ka - g;
    let off = vec![g * a; n_int - 1];
    let mut rhs = vec![b * b; n_int];
    rhs[0] = b;
    rhs[n_int - 1] = b;
    let y = thomas(&diag, &off, &rhs);
    let total: f64 = y.iter().sum();
    for j in 1..n_int {
        x[j] = x[j - 1] - big_x * y[j - 1] / total;
    }
    x[n_int] = 0.0;
    x
}

// ------------------------------------------------------------------ adaptive

/// Observable AR(1) drift signal and fill risk (API_SPEC 12.2).
///
/// `s_j = phi*s_{j-1} + sqrt(1-phi^2)*signal_sd*eps_j`, `s_0 = 0`; the price
/// drifts by `s_{j-1}*tau` over interval `j`.  A child order in interval
/// `j < N` fills the fraction `1 - fill_risk*u_j`; the last interval always
/// fills completely.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SignalParams {
    phi: f64,
    signal_sd: f64,
    fill_risk: f64,
}

impl SignalParams {
    /// Validated constructor: `phi` in `[0, 1)`, `signal_sd >= 0`,
    /// `fill_risk` in `[0, 1)`, all finite.
    pub fn new(phi: f64, signal_sd: f64, fill_risk: f64) -> Result<Self, EtradeError> {
        require(
            phi.is_finite() && (0.0..1.0).contains(&phi),
            "phi must be in [0, 1)",
        )?;
        require(
            signal_sd.is_finite() && signal_sd >= 0.0,
            "signal_sd must be >= 0",
        )?;
        require(
            fill_risk.is_finite() && (0.0..1.0).contains(&fill_risk),
            "fill_risk must be in [0, 1)",
        )?;
        Ok(Self {
            phi,
            signal_sd,
            fill_risk,
        })
    }
    /// AR(1) persistence.
    pub fn phi(&self) -> f64 {
        self.phi
    }
    /// Stationary standard deviation of the signal.
    pub fn signal_sd(&self) -> f64 {
        self.signal_sd
    }
    /// Maximum unfilled fraction of a child order.
    pub fn fill_risk(&self) -> f64 {
        self.fill_risk
    }
}

fn check_ac(p: &ACParams) -> Result<(), EtradeError> {
    require(p.n_intervals() <= MAX_EXEC_INTERVALS, N_INTERVALS_MSG)
}

fn replan_system(p: &ACParams, m: usize) -> (Vec<f64>, Vec<f64>) {
    let e = p.eta_tilde() / p.tau();
    let diag = vec![2.0 * e + p.lambda_ra() * p.sigma() * p.sigma() * p.tau(); m - 1];
    let off = vec![-e; m - 1];
    (diag, off)
}

/// Optimal remaining holdings `[x_k, x_{k+1}, .., x_N = 0]` from interval
/// `k` (0-based) given the current holding and signal (API_SPEC 12.2).
pub fn replan(
    p: &ACParams,
    sig: &SignalParams,
    holding: f64,
    signal: f64,
    k: usize,
) -> Result<Vec<f64>, EtradeError> {
    check_ac(p)?;
    require(
        k < p.n_intervals(),
        "k must be an integer in [0, n_intervals)",
    )?;
    require(holding.is_finite(), "holding must be finite")?;
    require(signal.is_finite(), "signal must be finite")?;
    let m = p.n_intervals() - k;
    if m == 1 {
        return Ok(vec![holding, 0.0]);
    }
    let (diag, off) = replan_system(p, m);
    let e = p.eta_tilde() / p.tau();
    let mut rhs: Vec<f64> = (0..m - 1)
        .map(|i| 0.5 * p.tau() * sig.phi.powi(i as i32) * signal)
        .collect();
    rhs[0] += e * holding;
    let mut out = Vec::with_capacity(m + 1);
    out.push(holding);
    out.extend(thomas(&diag, &off, &rhs));
    out.push(0.0);
    Ok(out)
}

/// Feedback form of [`replan`]: with `m` intervals left the next child
/// order is `c[m]*holding - d[m]*signal` (index 0 unused).
///
/// The re-plan is linear in `(holding, signal)`, so `c` and `d` are the
/// first-step responses to the two right-hand sides (API_SPEC 12.2).
pub fn replan_coefficients(
    p: &ACParams,
    sig: &SignalParams,
) -> Result<(Vec<f64>, Vec<f64>), EtradeError> {
    check_ac(p)?;
    let n_int = p.n_intervals();
    let mut c = vec![0.0; n_int + 1];
    let mut d = vec![0.0; n_int + 1];
    if n_int >= 1 {
        c[1] = 1.0;
    }
    let e = p.eta_tilde() / p.tau();
    for m in 2..=n_int {
        let (diag, off) = replan_system(p, m);
        let mut r_hold = vec![0.0; m - 1];
        r_hold[0] = e;
        let r_sig: Vec<f64> = (0..m - 1)
            .map(|i| 0.5 * p.tau() * sig.phi.powi(i as i32))
            .collect();
        c[m] = 1.0 - thomas(&diag, &off, &r_hold)[0];
        d[m] = thomas(&diag, &off, &r_sig)[0];
    }
    Ok((c, d))
}

/// Static vs adaptive implementation shortfall over `n_paths` paths.
///
/// `saving` is the paired per-path difference `static - adaptive`
/// (positive = adaptive cheaper); SEs are sample std (ddof 1) / sqrt(M).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveSimResult {
    /// Mean static shortfall.
    pub static_mean: f64,
    /// Standard error of `static_mean`.
    pub static_se: f64,
    /// Sample standard deviation of the static shortfall.
    pub static_std: f64,
    /// Mean adaptive shortfall.
    pub adaptive_mean: f64,
    /// Standard error of `adaptive_mean`.
    pub adaptive_se: f64,
    /// Sample standard deviation of the adaptive shortfall.
    pub adaptive_std: f64,
    /// Mean paired saving `static - adaptive`.
    pub saving_mean: f64,
    /// Standard error of `saving_mean`.
    pub saving_se: f64,
    /// Number of paths simulated.
    pub n_paths: usize,
}

fn mean_std(v: &[f64]) -> (f64, f64) {
    let m = v.len() as f64;
    let mean = v.iter().sum::<f64>() / m;
    let ss: f64 = v.iter().map(|x| (x - mean) * (x - mean)).sum();
    (mean, (ss / (m - 1.0)).sqrt())
}

/// Monte Carlo of static vs adaptive execution (API_SPEC 12.2).  Both
/// strategies see the same draws on each path.
///
/// Per interval `j = 1..N`, for each strategy: child order `q` (static: the
/// time-0 plan's `n_j`; adaptive: `c[m]*x - d[m]*s` with `m = N - j + 1`; at
/// `j = N` both send the whole remainder); then draw `u`, `xi`, `eps` (in
/// that order, shared by the two strategies); fill `e = f*q` with
/// `f = 1 - fill_risk*u` (`f = 1` at `j = N`); proceeds `+= e*(S - eta*e/tau)`;
/// then `S += s*tau + sigma*sqrt(tau)*xi - gamma_p*e` and
/// `s = phi*s + sqrt(1-phi^2)*signal_sd*eps`.  `IS = -proceeds` (`S_0 = 0`).
///
/// Paths are simulated one at a time (paths outer, intervals inner) with
/// `ChaCha12Rng::seed_from_u64(seed)`: results match other ports only
/// statistically.
pub fn simulate_adaptive(
    p: &ACParams,
    sig: &SignalParams,
    n_paths: usize,
    seed: u64,
) -> Result<AdaptiveSimResult, EtradeError> {
    check_ac(p)?;
    require(
        (2..=MAX_PATHS).contains(&n_paths),
        "n_paths must be an integer in [2, 10000000]",
    )?;
    let n_int = p.n_intervals();
    let plan = replan(p, sig, p.total_shares(), 0.0, 0)?;
    let (c, d) = replan_coefficients(p, sig)?;
    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    let srt = p.sigma() * p.tau().sqrt();
    let innov = (1.0 - sig.phi * sig.phi).sqrt() * sig.signal_sd;
    let mut is_st = Vec::with_capacity(n_paths);
    let mut is_ad = Vec::with_capacity(n_paths);
    let mut sav = Vec::with_capacity(n_paths);
    for _ in 0..n_paths {
        let (mut x_st, mut x_ad) = (p.total_shares(), p.total_shares());
        let (mut px_st, mut px_ad) = (0.0, 0.0);
        let (mut pr_st, mut pr_ad) = (0.0, 0.0);
        let mut sgn = 0.0;
        for j in 1..=n_int {
            let m = n_int - j + 1;
            let (q_st, q_ad) = if j == n_int {
                (x_st, x_ad)
            } else {
                (plan[j - 1] - plan[j], c[m] * x_ad - d[m] * sgn)
            };
            let u: f64 = rng.gen();
            let xi: f64 = rng.sample(StandardNormal);
            let eps: f64 = rng.sample(StandardNormal);
            let f = if j == n_int {
                1.0
            } else {
                1.0 - sig.fill_risk * u
            };
            let e_st = f * q_st;
            let e_ad = f * q_ad;
            pr_st += e_st * (px_st - p.eta() * e_st / p.tau());
            pr_ad += e_ad * (px_ad - p.eta() * e_ad / p.tau());
            let common = sgn * p.tau() + srt * xi;
            px_st = px_st + common - p.gamma_p() * e_st;
            px_ad = px_ad + common - p.gamma_p() * e_ad;
            sgn = sig.phi * sgn + innov * eps;
            x_st -= e_st;
            x_ad -= e_ad;
        }
        is_st.push(-pr_st);
        is_ad.push(-pr_ad);
        sav.push(-pr_st + pr_ad);
    }
    let (m_st, sd_st) = mean_std(&is_st);
    let (m_ad, sd_ad) = mean_std(&is_ad);
    let (m_sav, sd_sav) = mean_std(&sav);
    let root = (n_paths as f64).sqrt();
    Ok(AdaptiveSimResult {
        static_mean: m_st,
        static_se: sd_st / root,
        static_std: sd_st,
        adaptive_mean: m_ad,
        adaptive_se: sd_ad / root,
        adaptive_std: sd_ad,
        saving_mean: m_sav,
        saving_se: sd_sav / root,
        n_paths,
    })
}

// ------------------------------------------------------------------ internalisation

/// Crossing of opposite parent orders (API_SPEC 12.3).  Costs are currency;
/// `*_saved_bps` are per order in bps of its own notional and `saved_bps` is
/// in bps of the gross notional `(B + S)*price`.
#[derive(Debug, Clone, PartialEq)]
pub struct InternalisationResult {
    /// Shares crossed at the mid, `min(B, S)`.
    pub crossed_shares: f64,
    /// Net residual sent to the market, `|B - S|`.
    pub residual_shares: f64,
    /// Side of the residual: `"buy"`, `"sell"` or `"none"`.
    pub residual_side: &'static str,
    /// Impact cost of sending both sides to the market.
    pub baseline_cost: f64,
    /// Impact cost of the residual only.
    pub internal_cost: f64,
    /// `baseline_cost - internal_cost`.
    pub saved_cost: f64,
    /// Saving in bps of the gross notional.
    pub saved_bps: f64,
    /// Saving per buy order in bps of its own notional.
    pub buy_saved_bps: Vec<f64>,
    /// Saving per sell order in bps of its own notional.
    pub sell_saved_bps: Vec<f64>,
}

fn orders(name: &str, q: &[f64]) -> Result<(), EtradeError> {
    require(
        q.len() <= MAX_ORDERS,
        &format!("{name} must have at most {MAX_ORDERS} orders"),
    )?;
    require(
        q.iter().all(|v| v.is_finite() && *v > 0.0),
        &format!("{name} must be finite and > 0"),
    )
}

/// Cross buy and sell parent orders at the mid, send the residual to the
/// market and report the Section 8 impact cost saved (API_SPEC 12.3).
///
/// Market inputs come from `m` (`price`, `spread`, `adv`, `daily_vol`,
/// `n_intervals`, `tau`, `alpha`, `y`, `perm_frac`, `weights`); its
/// `total_shares`, `lambda_ra` and `drift_bps_per_day` are ignored (each
/// side is costed at its own size with zero risk aversion and zero drift).
/// Each program is the volume-weighted schedule over `weights` (TWAP when
/// `None`).
pub fn internalise(
    buys: &[f64],
    sells: &[f64],
    m: &ImpactMarket,
) -> Result<InternalisationResult, EtradeError> {
    orders("buys", buys)?;
    orders("sells", sells)?;
    require(
        buys.len() + sells.len() >= 1,
        "at least one order is required",
    )?;
    require(
        (1..=MAX_EXEC_INTERVALS).contains(&m.n_intervals),
        N_INTERVALS_MSG,
    )?;
    let cost = |q: f64| -> Result<f64, EtradeError> {
        if q == 0.0 {
            return Ok(0.0);
        }
        let mk = ImpactMarket {
            total_shares: q,
            lambda_ra: 0.0,
            drift_bps_per_day: 0.0,
            ..m.clone()
        };
        let ip = impact_params_from_market(&mk)?;
        impact_expected_cost(&ip, &vwap_holdings(q, ip.weights())?)
    };
    let big_b: f64 = buys.iter().sum();
    let big_s: f64 = sells.iter().sum();
    let crossed = big_b.min(big_s);
    let residual = (big_b - big_s).abs();
    let side = if big_b > big_s {
        "buy"
    } else if big_s > big_b {
        "sell"
    } else {
        "none"
    };
    let c_b = cost(big_b)?;
    let c_s = cost(big_s)?;
    let c_r = cost(residual)?;
    let baseline = c_b + c_s;
    let saved = baseline - c_r;
    let per_order = |qs: &[f64], total: f64, c_side: f64, majority: bool| {
        qs.iter()
            .map(|&q| {
                let base_i = c_side * q / total;
                let int_i = if majority { c_r * q / total } else { 0.0 };
                bps(base_i - int_i, q * m.price)
            })
            .collect::<Result<Vec<f64>, EtradeError>>()
    };
    Ok(InternalisationResult {
        crossed_shares: crossed,
        residual_shares: residual,
        residual_side: side,
        baseline_cost: baseline,
        internal_cost: c_r,
        saved_cost: saved,
        saved_bps: bps(saved, (big_b + big_s) * m.price)?,
        buy_saved_bps: per_order(buys, big_b, c_b, side == "buy")?,
        sell_saved_bps: per_order(sells, big_s, c_s, side == "sell")?,
    })
}
