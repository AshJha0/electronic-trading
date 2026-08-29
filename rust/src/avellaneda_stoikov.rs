//! Avellaneda-Stoikov market making (2008 paper, finite horizon).
//!
//! The mid price is an arithmetic Brownian motion `dS = sigma dW` (prices may
//! go negative — documented model property).  A maker with exponential
//! utility quotes bid/ask around the mid; market orders lift a quote at
//! distance `delta` with Poisson intensity `lambda(delta) = A*exp(-k*delta)`.
//!
//! Closed forms (pinned in API_SPEC.md Section 2.1):
//!
//! ```text
//! r(s,q,t)  = s - q*gamma*sigma^2*(T - t)                    (reservation price)
//! spread(t) = gamma*sigma^2*(T - t) + (2/gamma)*ln(1 + gamma/k)
//! ask = r + spread/2 ,  bid = r - spread/2
//! P(fill)   = 1 - exp(-lambda(delta)*dt)                     (per-step, thinned Poisson)
//! ```
//!
//! A long maker (`q > 0`) shades quotes *down* to attract buyers of its
//! inventory; a short maker shades up.  The first spread term is
//! inventory-risk compensation (zero at `t = T`); the second is the
//! monopolist-dealer markup (the terminal floor).  `gamma > 0` strictly — the
//! `gamma -> 0` limit degenerates the utility derivation and is rejected.
//!
//! The simulation loop order (quote → fill draws → inventory/cash update →
//! mid move) and the draw order (`u_ask`, then `u_bid`, then `z`, per step,
//! both uniforms always drawn even for a suppressed side) are pinned in
//! API_SPEC.md Section 2.2; this port uses its own RNG per Section 4, so its
//! sample values differ from Python while all statistics agree within SE
//! bands.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use crate::error::{require, EtradeError};

/// Avellaneda-Stoikov model parameters (validated at construction).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ASParams {
    s0: f64,
    sigma: f64,
    horizon: f64,
    dt: f64,
    gamma: f64,
    k: f64,
    a_rate: f64,
    q_max: u32,
    eta_liq: f64,
}

impl ASParams {
    /// Validate and build the parameter set.
    ///
    /// * `s0`      — initial mid price (any finite real; arithmetic model);
    /// * `sigma`   — arithmetic mid volatility per sqrt(time unit), `>= 0`;
    /// * `horizon` — terminal time `T > 0`;
    /// * `dt`      — simulation step, `0 < dt <= T`, `T/dt` integral (1e-9 tol);
    /// * `gamma`   — risk aversion, strictly `> 0`;
    /// * `k`       — intensity decay, `> 0`;
    /// * `a_rate`  — intensity scale `A`, `> 0`;
    /// * `q_max`   — hard inventory cap in units (`0` quotes nothing);
    /// * `eta_liq` — terminal liquidation penalty per unit inventory, `>= 0`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        s0: f64,
        sigma: f64,
        horizon: f64,
        dt: f64,
        gamma: f64,
        k: f64,
        a_rate: f64,
        q_max: u32,
        eta_liq: f64,
    ) -> Result<Self, EtradeError> {
        require(s0.is_finite(), "s0 must be finite")?;
        require(sigma.is_finite() && sigma >= 0.0, "sigma must be >= 0")?;
        require(horizon.is_finite() && horizon > 0.0, "horizon must be > 0")?;
        require(
            dt.is_finite() && dt > 0.0 && dt <= horizon,
            "dt must satisfy 0 < dt <= horizon",
        )?;
        require(
            gamma.is_finite() && gamma > 0.0,
            "gamma must be > 0 (gamma = 0 is outside the A-S model)",
        )?;
        require(k.is_finite() && k > 0.0, "k must be > 0")?;
        require(a_rate.is_finite() && a_rate > 0.0, "a_rate (A) must be > 0")?;
        require(eta_liq.is_finite() && eta_liq >= 0.0, "eta_liq must be >= 0")?;
        let n = horizon / dt;
        require(
            (n - n.round()).abs() < 1e-9,
            "horizon must be an integer multiple of dt",
        )?;
        Ok(Self {
            s0,
            sigma,
            horizon,
            dt,
            gamma,
            k,
            a_rate,
            q_max,
            eta_liq,
        })
    }

    /// Initial mid price `s0`.
    pub fn s0(&self) -> f64 {
        self.s0
    }
    /// Mid volatility `sigma`.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }
    /// Terminal time `T`.
    pub fn horizon(&self) -> f64 {
        self.horizon
    }
    /// Simulation step `dt`.
    pub fn dt(&self) -> f64 {
        self.dt
    }
    /// Risk aversion `gamma`.
    pub fn gamma(&self) -> f64 {
        self.gamma
    }
    /// Intensity decay `k`.
    pub fn k(&self) -> f64 {
        self.k
    }
    /// Intensity scale `A`.
    pub fn a_rate(&self) -> f64 {
        self.a_rate
    }
    /// Hard inventory cap `q_max`.
    pub fn q_max(&self) -> u32 {
        self.q_max
    }
    /// Terminal liquidation penalty `eta_liq`.
    pub fn eta_liq(&self) -> f64 {
        self.eta_liq
    }

    /// Number of simulation steps `round(T/dt)`.
    pub fn n_steps(&self) -> usize {
        (self.horizon / self.dt).round() as usize
    }
}

/// Reservation price `r(s,q,t) = s - q*gamma*sigma^2*(T - t)`.
/// Requires `0 <= t <= T`.
pub fn reservation_price(params: &ASParams, s: f64, q: f64, t: f64) -> Result<f64, EtradeError> {
    require(t >= 0.0 && t <= params.horizon(), "t must be in [0, horizon]")?;
    Ok(s - q * params.gamma() * params.sigma().powi(2) * (params.horizon() - t))
}

/// Total optimal spread `gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/k)`.
///
/// Strictly positive; decreases linearly in `t` to the terminal floor
/// `(2/gamma)*ln(1 + gamma/k)` at `t = T`.  Requires `0 <= t <= T`.
pub fn optimal_spread(params: &ASParams, t: f64) -> Result<f64, EtradeError> {
    require(t >= 0.0 && t <= params.horizon(), "t must be in [0, horizon]")?;
    Ok(params.gamma() * params.sigma().powi(2) * (params.horizon() - t)
        + 2.0 / params.gamma() * (params.gamma() / params.k()).ln_1p())
}

/// `(bid, ask)` = reservation price `-/+` half the optimal spread.
pub fn quotes(params: &ASParams, s: f64, q: f64, t: f64) -> Result<(f64, f64), EtradeError> {
    let r = reservation_price(params, s, q, t)?;
    let half = 0.5 * optimal_spread(params, t)?;
    Ok((r - half, r + half))
}

/// Arrival intensity `lambda(delta) = A*exp(-k*delta)`.
///
/// `delta` is signed (a quote through the mid has `delta < 0` and a higher
/// intensity); it must be finite.
pub fn intensity(params: &ASParams, delta: f64) -> Result<f64, EtradeError> {
    require(delta.is_finite(), "delta must be finite")?;
    Ok(params.a_rate() * (-params.k() * delta).exp())
}

/// Per-step fill probability `1 - exp(-lambda(delta)*dt)`, in `[0, 1)`.
pub fn fill_probability(params: &ASParams, delta: f64) -> Result<f64, EtradeError> {
    Ok(-(-intensity(params, delta)? * params.dt()).exp_m1())
}

/// Time average over `[0, T]` of the A-S spread (the fixed agent's default):
/// `gamma*sigma^2*T/2 + (2/gamma)*ln(1 + gamma/k)`.
pub fn average_as_spread(params: &ASParams) -> f64 {
    0.5 * params.gamma() * params.sigma().powi(2) * params.horizon()
        + 2.0 / params.gamma() * (params.gamma() / params.k()).ln_1p()
}

/// Quoting agent for the market-making simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Agent {
    /// Avellaneda-Stoikov: quotes around the reservation price with the
    /// time-dependent optimal spread (inventory skew).
    As,
    /// Comparison agent: symmetric fixed spread around the mid, no skew.
    Fixed,
}

/// Configuration of a market-making simulation run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimConfig {
    /// Which agent quotes ([`Agent::As`] by default).
    pub agent: Agent,
    /// Fixed agent's total spread; `None` uses [`average_as_spread`].
    pub fixed_spread: Option<f64>,
    /// Multiplier on the *realised* mid volatility for `t >= stress_time_frac*T`
    /// (the agent keeps quoting with the base `sigma` — a model
    /// misspecification scenario).  `1.0` = stress off.
    pub stress_factor: f64,
    /// Fraction of the horizon after which the stress applies, in `[0, 1]`.
    pub stress_time_frac: f64,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            agent: Agent::As,
            fixed_spread: None,
            stress_factor: 1.0,
            stress_time_frac: 0.5,
        }
    }
}

/// Per-path results and aggregate statistics of a market-making simulation.
///
/// Per-path vectors have length `n_paths`; aggregates are means/stds over
/// paths (see the accessor methods and [`ASSimResult::se`]).
#[derive(Debug, Clone)]
pub struct ASSimResult {
    /// Terminal PnL per path (after liquidation penalty).
    pub pnl: Vec<f64>,
    /// Per-path time average of `|q|` (over the post-update values).
    pub mean_abs_q_path: Vec<f64>,
    /// Per-path maximum of `|q|`.
    pub max_abs_q_path: Vec<f64>,
    /// Per-path `sqrt(time average of q^2)` — the "inventory std" measure.
    pub rms_q_path: Vec<f64>,
    /// Per-path count of fill events: steps where at least one side filled
    /// (a step where both sides fill counts once — the Python reference's
    /// convention, pinned by the golden `mean_fills` values).
    pub fills_path: Vec<f64>,
    /// Maximum `|q|` seen across all paths and steps.
    pub overall_max_abs_q: f64,
    /// True if the hard cap was ever breached (must always be false — the
    /// simulation asserts the cap by construction).
    pub cap_violated: bool,
}

/// Statistic selector for [`ASSimResult::se`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ASStat {
    /// Mean terminal PnL.
    MeanPnl,
    /// Std of terminal PnL.
    StdPnl,
    /// Mean over paths of time-averaged `|q|`.
    MeanAbsQ,
    /// Mean over paths of max `|q|`.
    MeanMaxAbsQ,
    /// Mean over paths of per-path RMS inventory.
    RmsQ,
    /// Mean over paths of fill events.
    MeanFills,
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn sample_std(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() as f64 - 1.0)).sqrt()
}

impl ASSimResult {
    /// Number of simulated paths.
    pub fn n_paths(&self) -> usize {
        self.pnl.len()
    }
    /// Mean terminal PnL over paths.
    pub fn mean_pnl(&self) -> f64 {
        mean(&self.pnl)
    }
    /// Sample std (ddof = 1) of terminal PnL.
    pub fn std_pnl(&self) -> f64 {
        sample_std(&self.pnl)
    }
    /// Sharpe-like ratio `mean_pnl/std_pnl` (`inf` when the std is zero).
    pub fn sharpe(&self) -> f64 {
        let s = self.std_pnl();
        if s == 0.0 {
            f64::INFINITY
        } else {
            self.mean_pnl() / s
        }
    }
    /// Mean over paths of the per-path time average of `|q|`.
    pub fn mean_abs_q(&self) -> f64 {
        mean(&self.mean_abs_q_path)
    }
    /// Mean over paths of the per-path maximum `|q|`.
    pub fn mean_max_abs_q(&self) -> f64 {
        mean(&self.max_abs_q_path)
    }
    /// Mean over paths of the per-path RMS inventory.
    pub fn rms_q(&self) -> f64 {
        mean(&self.rms_q_path)
    }
    /// Mean over paths of per-path fill events (steps with at least one fill).
    pub fn mean_fills(&self) -> f64 {
        mean(&self.fills_path)
    }

    /// Standard error of an aggregate statistic: sample std of the per-path
    /// values `/ sqrt(M)` for per-path-mean statistics;
    /// `std_pnl/sqrt(2*(M-1))` (normal approximation) for [`ASStat::StdPnl`].
    pub fn se(&self, stat: ASStat) -> f64 {
        let m = self.n_paths() as f64;
        match stat {
            ASStat::MeanPnl => sample_std(&self.pnl) / m.sqrt(),
            ASStat::StdPnl => self.std_pnl() / (2.0 * (m - 1.0)).sqrt(),
            ASStat::MeanAbsQ => sample_std(&self.mean_abs_q_path) / m.sqrt(),
            ASStat::MeanMaxAbsQ => sample_std(&self.max_abs_q_path) / m.sqrt(),
            ASStat::RmsQ => sample_std(&self.rms_q_path) / m.sqrt(),
            ASStat::MeanFills => sample_std(&self.fills_path) / m.sqrt(),
        }
    }
}

/// Simulate the market maker over `n_paths` seeded paths.
///
/// Per step, in the pinned order (API_SPEC.md Section 2.2):
/// 1. quote from the current `s`, `q`, remaining time (inventory cap: bid
///    only if `q < q_max`, ask only if `q > -q_max`);
/// 2. draw `u_ask` then `u_bid` (always both, even for a suppressed side —
///    the stream must not depend on the cap state); a side fills iff quoted
///    and `u < P(fill)` with `delta` measured from the current mid;
/// 3. ask fill: `q -= 1`, `cash += ask`; bid fill: `q += 1`, `cash -= bid`;
/// 4. mid move `s += sigma_t*sqrt(dt)*z` with the stressed volatility for
///    `t >= stress_time_frac*T` (the agent keeps the base `sigma`).
///
/// Terminal: `PnL = cash + q*S_T - eta_liq*|q|`.  Deterministic per
/// `(seed, config, n_paths)` with this crate's own RNG.
pub fn simulate(
    params: &ASParams,
    n_paths: usize,
    seed: u64,
    config: &SimConfig,
) -> Result<ASSimResult, EtradeError> {
    require(n_paths >= 2, "n_paths must be >= 2")?;
    require(
        config.stress_factor.is_finite() && config.stress_factor > 0.0,
        "stress_factor must be > 0",
    )?;
    require(
        config.stress_time_frac >= 0.0 && config.stress_time_frac <= 1.0,
        "stress_time_frac must be in [0, 1]",
    )?;
    let fixed_spread = config.fixed_spread.unwrap_or_else(|| average_as_spread(params));
    require(
        fixed_spread.is_finite() && fixed_spread > 0.0,
        "fixed_spread must be > 0",
    )?;

    let mut rng = StdRng::seed_from_u64(seed);
    let n_steps = params.n_steps();
    let (g, big_t, dt) = (params.gamma(), params.horizon(), params.dt());
    let sig2 = params.sigma().powi(2);
    let markup = 2.0 / g * (g / params.k()).ln_1p();
    let sqdt = dt.sqrt();
    let q_cap = params.q_max() as i64;
    let stress_from = config.stress_time_frac * big_t;

    let mut pnl = Vec::with_capacity(n_paths);
    let mut mean_abs_q_path = Vec::with_capacity(n_paths);
    let mut max_abs_q_path = Vec::with_capacity(n_paths);
    let mut rms_q_path = Vec::with_capacity(n_paths);
    let mut fills_path = Vec::with_capacity(n_paths);
    let mut overall_max_abs_q: i64 = 0;
    let mut cap_violated = false;

    for _ in 0..n_paths {
        let mut s = params.s0();
        let mut q: i64 = 0;
        let mut cash = 0.0;
        let mut sum_abs_q = 0.0;
        let mut sum_q2 = 0.0;
        let mut max_abs_q: i64 = 0;
        let mut fills: u64 = 0;

        for i in 0..n_steps {
            let t = i as f64 * dt;
            // 1. quote from the current state
            let (r, half) = match config.agent {
                Agent::As => (
                    s - q as f64 * g * sig2 * (big_t - t),
                    0.5 * (g * sig2 * (big_t - t) + markup),
                ),
                Agent::Fixed => (s, 0.5 * fixed_spread),
            };
            let ask = r + half;
            let bid = r - half;
            let quote_ask = q > -q_cap;
            let quote_bid = q < q_cap;
            // 2. fill draws (u_ask first, then u_bid; both always drawn)
            let u_ask: f64 = rng.gen();
            let u_bid: f64 = rng.gen();
            let p_ask = -(-params.a_rate() * (-params.k() * (ask - s)).exp() * dt).exp_m1();
            let p_bid = -(-params.a_rate() * (-params.k() * (s - bid)).exp() * dt).exp_m1();
            let fill_ask = quote_ask && u_ask < p_ask;
            let fill_bid = quote_bid && u_bid < p_bid;
            // 3. inventory / cash update (unit fill size, both sides may fill)
            if fill_ask {
                q -= 1;
                cash += ask;
            }
            if fill_bid {
                q += 1;
                cash -= bid;
            }
            if fill_ask || fill_bid {
                // Fill count matches the Python reference (which sums numpy
                // bool arrays, i.e. logical OR): a step where BOTH sides
                // fill counts as ONE fill event.  The golden `mean_fills`
                // values pin this convention.
                fills += 1;
            }
            let abs_q = q.abs();
            if abs_q > q_cap {
                cap_violated = true; // must never happen by construction
            }
            sum_abs_q += abs_q as f64;
            sum_q2 += (q * q) as f64;
            max_abs_q = max_abs_q.max(abs_q);
            // 4. mid move (stressed vol only affects the realised path)
            let z: f64 = rng.sample(StandardNormal);
            let sig_t = params.sigma() * if t >= stress_from { config.stress_factor } else { 1.0 };
            s += sig_t * sqdt * z;
        }

        pnl.push(cash + q as f64 * s - params.eta_liq() * q.abs() as f64);
        mean_abs_q_path.push(sum_abs_q / n_steps as f64);
        max_abs_q_path.push(max_abs_q as f64);
        rms_q_path.push((sum_q2 / n_steps as f64).sqrt());
        fills_path.push(fills as f64);
        overall_max_abs_q = overall_max_abs_q.max(max_abs_q);
    }

    Ok(ASSimResult {
        pnl,
        mean_abs_q_path,
        max_abs_q_path,
        rms_q_path,
        fills_path,
        overall_max_abs_q: overall_max_abs_q as f64,
        cap_violated,
    })
}
