//! Almgren-Chriss discrete-time optimal execution (Almgren & Chriss, 2000).
//!
//! Sell `X` shares over `N` intervals of length `tau` (horizon `T = N*tau`,
//! times `t_j = j*tau`).  Holdings `x_0 = X, ..., x_N = 0`; trade list
//! `n_j = x_{j-1} - x_j`.  The mid price follows an arithmetic random walk
//! with linear permanent impact `g(v) = gamma_p*v`; trade `j` executes at
//! `S_{j-1} - eta*n_j/tau` (linear temporary impact `h(v) = eta*v`).
//! Implementation shortfall is `IS = X*S_0 - sum_j n_j*S~_j` (independent of
//! `S_0`; prices may go negative — a documented model property, not an error).
//!
//! Exact discrete moments for any fully liquidating schedule (pinned sums):
//!
//! ```text
//! E[IS]   = (1/2)*gamma_p*X^2 + (eta_tilde/tau) * sum_j n_j^2
//! Var[IS] = sigma^2 * tau * sum_{j=1}^{N} x_j^2
//! eta_tilde = eta - gamma_p*tau/2            (must be > 0, paper footnote)
//! ```
//!
//! The sums coincide with the paper's hyperbolic closed forms (with the
//! discrete `kappa`) to machine precision; they are used because they also
//! cover TWAP/POV/arbitrary schedules.  The paper's fixed-cost term
//! `epsilon*sum|n_j|` is omitted (schedule-independent for a one-sided
//! program).
//!
//! Minimising `U = E + lambda_ra*Var` gives the sinh trajectory
//! `x_j = X*sinh(kappa*(T - t_j))/sinh(kappa*T)` where the discrete `kappa`
//! solves `(2/tau^2)*(cosh(kappa*tau) - 1) = kappa_tilde^2` with
//! `kappa_tilde^2 = lambda_ra*sigma^2/eta_tilde`.  Pinned evaluation
//! (API_SPEC.md Section 1.4): `kappa = ln_1p(x + sqrt(x*(2+x)))/tau` with
//! `x = kappa_tilde^2*tau^2/2` (no cancellation for small `x`), and
//! `x_j = X*exp(-kappa*t_j)*exp_m1(-2*kappa*(T-t_j))/exp_m1(-2*kappa*T)`
//! (no `sinh` overflow for large `kappa*T`).  When `kappa*T <`
//! [`KAPPA_T_LIMIT`] the exact linear TWAP limit `x_j = X*(N-j)/N` is used
//! instead (covers `lambda_ra = 0` and `sigma = 0` with no 0/0).
//!
//! RNG: `ChaCha12Rng::seed_from_u64` (the `rand_chacha` crate) — pinned
//! explicitly so "same seed, same results" holds across `rand` releases.

use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rand_distr::StandardNormal;

use crate::error::{require, EtradeError};

/// Threshold on `kappa*T` below which the exact linear (TWAP) limit branch
/// of the optimal trajectory is used (pinned in API_SPEC.md Section 1.4).
pub const KAPPA_T_LIMIT: f64 = 1e-6;

/// Upper bound on the number of AC trading intervals `N` (pinned, API_SPEC.md Section 0).
pub const MAX_AC_INTERVALS: usize = 10_000_000;

/// Upper bound on the number of POV intervals `M = ceil(X/clip)` (pinned).
pub const MAX_POV_INTERVALS: usize = 10_000_000;

/// Upper bound on Monte Carlo path counts (pinned, API_SPEC.md Section 0).
pub const MAX_PATHS: usize = 10_000_000;

/// Upper bound on the total Monte Carlo work `n_paths * N` of [`simulate_is`]
/// (pinned, API_SPEC.md Section 0).
pub const MAX_MC_WORK: u64 = 1_000_000_000;

/// Parameters of the discrete Almgren-Chriss problem (sell `X` shares).
///
/// Constructed through [`ACParams::new`], which validates every field and
/// the model-validity condition `gamma_p*tau < 2*eta`; the fields are
/// immutable afterwards, so a constructed value is always well posed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ACParams {
    total_shares: f64,
    n_intervals: usize,
    tau: f64,
    sigma: f64,
    eta: f64,
    gamma_p: f64,
    lambda_ra: f64,
}

impl ACParams {
    /// Validate and build the parameter set.
    ///
    /// * `total_shares` — `X >= 0`, shares to sell (`0` gives the empty trajectory);
    /// * `n_intervals`  — `N` trading intervals, `1 <= N <= MAX_AC_INTERVALS`;
    /// * `tau`          — interval length, `> 0`;
    /// * `sigma`        — arithmetic volatility per sqrt(time unit), `>= 0`;
    /// * `eta`          — linear temporary impact coefficient, `> 0`;
    /// * `gamma_p`      — linear permanent impact coefficient, `>= 0`;
    /// * `lambda_ra`    — risk aversion, `>= 0` (`0` gives the TWAP limit).
    ///
    /// Returns [`EtradeError::ModelInvalid`] when `gamma_p*tau >= 2*eta`
    /// (`eta_tilde <= 0`), [`EtradeError::InvalidInput`] for other violations.
    pub fn new(
        total_shares: f64,
        n_intervals: usize,
        tau: f64,
        sigma: f64,
        eta: f64,
        gamma_p: f64,
        lambda_ra: f64,
    ) -> Result<Self, EtradeError> {
        require(
            total_shares.is_finite() && total_shares >= 0.0,
            "total_shares must be finite and >= 0",
        )?;
        require(
            (1..=MAX_AC_INTERVALS).contains(&n_intervals),
            "n_intervals must be in [1, 10000000]",
        )?;
        require(tau.is_finite() && tau > 0.0, "tau must be > 0")?;
        require(sigma.is_finite() && sigma >= 0.0, "sigma must be >= 0")?;
        require(eta.is_finite() && eta > 0.0, "eta must be > 0")?;
        require(
            gamma_p.is_finite() && gamma_p >= 0.0,
            "gamma_p must be >= 0",
        )?;
        require(
            lambda_ra.is_finite() && lambda_ra >= 0.0,
            "lambda_ra must be >= 0",
        )?;
        // Paper validity condition: eta_tilde = eta - gamma_p*tau/2 must be > 0.
        if gamma_p * tau >= 2.0 * eta {
            return Err(EtradeError::ModelInvalid(
                "gamma_p*tau >= 2*eta (eta_tilde <= 0); see Almgren-Chriss (2000) footnote"
                    .to_string(),
            ));
        }
        Ok(Self {
            total_shares,
            n_intervals,
            tau,
            sigma,
            eta,
            gamma_p,
            lambda_ra,
        })
    }

    /// Shares to sell `X`.
    pub fn total_shares(&self) -> f64 {
        self.total_shares
    }
    /// Number of trading intervals `N`.
    pub fn n_intervals(&self) -> usize {
        self.n_intervals
    }
    /// Interval length `tau`.
    pub fn tau(&self) -> f64 {
        self.tau
    }
    /// Arithmetic volatility `sigma`.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }
    /// Temporary impact coefficient `eta`.
    pub fn eta(&self) -> f64 {
        self.eta
    }
    /// Permanent impact coefficient `gamma_p`.
    pub fn gamma_p(&self) -> f64 {
        self.gamma_p
    }
    /// Risk aversion `lambda_ra`.
    pub fn lambda_ra(&self) -> f64 {
        self.lambda_ra
    }

    /// Total horizon `T = N * tau`.
    pub fn horizon(&self) -> f64 {
        self.n_intervals as f64 * self.tau
    }

    /// Adjusted temporary impact `eta_tilde = eta*(1 - gamma_p*tau/(2*eta))`.
    pub fn eta_tilde(&self) -> f64 {
        self.eta * (1.0 - self.gamma_p * self.tau / (2.0 * self.eta))
    }

    /// Copy of these parameters with a different risk aversion (revalidated).
    pub fn with_lambda(&self, lambda_ra: f64) -> Result<Self, EtradeError> {
        Self::new(
            self.total_shares,
            self.n_intervals,
            self.tau,
            self.sigma,
            self.eta,
            self.gamma_p,
            lambda_ra,
        )
    }
}

/// Named-field form of [`ACParams::new`]: the same arguments, labelled, so a
/// call site cannot silently swap two numbers of the same type.
///
/// [`ACSpec::build`] (or `ACParams::try_from(spec)`) runs exactly the validation of
/// [`ACParams::new`] and returns the same errors.
///
/// ```
/// use etrade::{ACParams, ACSpec};
///
/// let p = ACSpec {
///     total_shares: 1e6,
///     n_intervals: 10,
///     tau: 1.0,
///     sigma: 0.95,
///     eta: 2.5e-6,
///     gamma_p: 2.5e-7,
///     lambda_ra: 1e-6,
/// }
/// .build()?;
/// let x = etrade::ac_trajectory(&p);
/// assert_eq!(x.len(), 11);
/// assert_eq!((x[0], x[10]), (1e6, 0.0));
/// // `ACParams::try_from(spec)` is the same validation.
/// assert!(ACParams::try_from(ACSpec { eta: -1.0, ..ACSpec::from(p) }).is_err());
/// # Ok::<(), etrade::EtradeError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ACSpec {
    /// Shares to sell, `X >= 0`.
    pub total_shares: f64,
    /// Trading intervals `N`, `1 <= N <= MAX_AC_INTERVALS`.
    pub n_intervals: usize,
    /// Interval length, `> 0`.
    pub tau: f64,
    /// Arithmetic volatility per sqrt(time unit), `>= 0`.
    pub sigma: f64,
    /// Linear temporary impact coefficient, `> 0`.
    pub eta: f64,
    /// Linear permanent impact coefficient, `>= 0`.
    pub gamma_p: f64,
    /// Risk aversion, `>= 0`.
    pub lambda_ra: f64,
}

impl ACSpec {
    /// Validate and build the [`ACParams`] (same checks as [`ACParams::new`]).
    pub fn build(self) -> Result<ACParams, EtradeError> {
        ACParams::try_from(self)
    }
}

impl TryFrom<ACSpec> for ACParams {
    type Error = EtradeError;

    fn try_from(s: ACSpec) -> Result<Self, Self::Error> {
        ACParams::new(
            s.total_shares,
            s.n_intervals,
            s.tau,
            s.sigma,
            s.eta,
            s.gamma_p,
            s.lambda_ra,
        )
    }
}

impl From<ACParams> for ACSpec {
    /// The spec a validated parameter set was built from.
    fn from(p: ACParams) -> Self {
        Self {
            total_shares: p.total_shares,
            n_intervals: p.n_intervals,
            tau: p.tau,
            sigma: p.sigma,
            eta: p.eta,
            gamma_p: p.gamma_p,
            lambda_ra: p.lambda_ra,
        }
    }
}

/// Discrete urgency parameter `kappa`.
///
/// Solves `(2/tau^2)*(cosh(kappa*tau) - 1) = kappa_tilde^2` with
/// `kappa_tilde^2 = lambda_ra*sigma^2/eta_tilde`, i.e.
/// `kappa = arccosh(1 + kappa_tilde^2*tau^2/2)/tau`, evaluated in the
/// cancellation-free form `ln_1p(x + sqrt(x*(2 + x)))/tau` with
/// `x = kappa_tilde^2*tau^2/2` (pinned); returns `0.0` in the
/// `lambda_ra*sigma^2 = 0` limit.
pub fn ac_kappa(params: &ACParams) -> f64 {
    let kt2 = params.lambda_ra() * params.sigma().powi(2) / params.eta_tilde();
    if kt2 == 0.0 {
        return 0.0;
    }
    if !kt2.is_finite() {
        return f64::INFINITY;
    }
    let x = 0.5 * kt2 * params.tau().powi(2);
    let kappa = (x + (x * (2.0 + x)).sqrt()).ln_1p() / params.tau();
    if kappa.is_finite() {
        kappa
    } else {
        f64::INFINITY
    }
}

/// Characteristic decay time `theta = 1/kappa` of the optimal trajectory
/// (the paper's e-folding time, not `ln(2)/kappa`).
///
/// Returns `f64::INFINITY` in the TWAP limit (`kappa == 0`) — reported as
/// infinity, never an error.  [`characteristic_time`] is an alias.
pub fn half_life(params: &ACParams) -> f64 {
    let kappa = ac_kappa(params);
    if kappa == 0.0 {
        f64::INFINITY
    } else {
        1.0 / kappa
    }
}

/// Alias of [`half_life`] under the paper's name (`theta = 1/kappa`).
pub fn characteristic_time(params: &ACParams) -> f64 {
    half_life(params)
}

/// Optimal holdings path `x_0..x_N` (length `N+1`, `x_0 = X` exactly,
/// `x_N = 0` exactly, every entry finite for any valid parameters).
///
/// Uses the sinh solution in the overflow-free `exp/exp_m1` ratio form;
/// switches to the exact linear (TWAP) branch when `kappa*T < KAPPA_T_LIMIT`
/// (covers `lambda_ra = 0` and `sigma = 0`).
///
/// ```
/// use etrade::{ac_trade_list, ac_trajectory, ACParams};
///
/// let p = ACParams::new(1e6, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6)?;
/// let x = ac_trajectory(&p);
/// assert_eq!((x[0], x[5]), (1e6, 0.0));
/// // Risk aversion front-loads: holdings fall faster than TWAP.
/// assert!(x[1] < 0.8e6);
/// let n = ac_trade_list(&p);
/// assert!((n.iter().sum::<f64>() - 1e6).abs() < 1e-6);
/// # Ok::<(), etrade::EtradeError>(())
/// ```
pub fn ac_trajectory(params: &ACParams) -> Vec<f64> {
    let n = params.n_intervals();
    let x0 = params.total_shares();
    if x0 == 0.0 {
        return vec![0.0; n + 1];
    }
    let kappa = ac_kappa(params);
    if !kappa.is_finite() {
        // kappa overflowed: infinite-urgency limit, everything in interval 1.
        let mut x = vec![0.0; n + 1];
        x[0] = x0;
        return x;
    }
    let big_t = params.horizon();
    let linear = kappa == 0.0 || kappa * big_t < KAPPA_T_LIMIT; // kappa = 0 even if T overflows
                                                                // sinh(k(T-t))/sinh(kT) = exp(-k t) * expm1(-2k(T-t)) / expm1(-2kT)
    let denom = (-2.0 * kappa * big_t).exp_m1();
    let mut x: Vec<f64> = (0..=n)
        .map(|j| {
            if linear {
                // Exact TWAP limit — no 0/0 from the sinh ratio.
                x0 * ((n - j) as f64 / n as f64)
            } else {
                let t_j = j as f64 * params.tau();
                // ratio formed first so that x_0 == X exactly
                let ratio = (-2.0 * kappa * (big_t - t_j)).exp_m1() / denom;
                x0 * (-kappa * t_j).exp() * ratio
            }
        })
        .collect();
    x[n] = 0.0; // pinned: terminal holding set to exactly zero
    x
}

/// Optimal trade list `n_1..n_N` (shares sold per interval; sums to `X`).
pub fn ac_trade_list(params: &ACParams) -> Vec<f64> {
    // The trajectory always has length N + 1 >= 2, so the differences are
    // taken directly (no validation path, no panic path).
    ac_trajectory(params)
        .windows(2)
        .map(|w| w[0] - w[1])
        .collect()
}

/// Validate a user-supplied holdings path: length `>= 2`, all finite.
fn require_holdings(x: &[f64]) -> Result<(), EtradeError> {
    require(x.len() >= 2, "holdings must have length >= 2")?;
    require(x.iter().all(|v| v.is_finite()), "holdings must be finite")
}

/// Pinned liquidation check `|x_N| <= 1e-9 * max(1, |x_0|)`.
fn require_liquidating(x: &[f64]) -> Result<(), EtradeError> {
    require(
        x[x.len() - 1].abs() <= 1e-9 * x[0].abs().max(1.0),
        "holdings path must fully liquidate (x_N = 0)",
    )
}

/// Trade list `n_j = x_{j-1} - x_j` from a finite holdings path (length `>= 2`).
pub fn trades_from_holdings(holdings: &[f64]) -> Result<Vec<f64>, EtradeError> {
    require_holdings(holdings)?;
    Ok(holdings.windows(2).map(|w| w[0] - w[1]).collect())
}

/// Linear TWAP holdings path `x_j = X*(N-j)/N` (length `N+1`).  Requires
/// finite `total_shares >= 0` and `1 <= n_intervals <= MAX_AC_INTERVALS`.
pub fn twap_holdings(total_shares: f64, n_intervals: usize) -> Result<Vec<f64>, EtradeError> {
    require(
        total_shares.is_finite() && total_shares >= 0.0,
        "total_shares must be finite and >= 0",
    )?;
    require(
        (1..=MAX_AC_INTERVALS).contains(&n_intervals),
        "n_intervals must be in [1, 10000000]",
    )?;
    Ok((0..=n_intervals)
        .map(|j| total_shares * ((n_intervals - j) as f64 / n_intervals as f64))
        .collect())
}

/// POV (percentage-of-volume) holdings path.
///
/// Sells `clip = participation * market_volume_rate * tau` shares per
/// interval until the parent order is done; the last clip is partial.  The
/// number of intervals is `M = ceil(X/clip - 1e-12)` (pinned) and the path
/// length is `M+1` with `x_M = 0` exactly.  All inputs must be finite,
/// `participation` must lie in the open interval `(0, 1)`,
/// `market_volume_rate` must be `> 0`, and `X/clip <= MAX_POV_INTERVALS`
/// (a clip too small for the order is an input error, never a huge
/// allocation).
pub fn pov_holdings(
    total_shares: f64,
    tau: f64,
    market_volume_rate: f64,
    participation: f64,
) -> Result<Vec<f64>, EtradeError> {
    require(
        total_shares.is_finite() && total_shares >= 0.0,
        "total_shares must be finite and >= 0",
    )?;
    require(tau.is_finite() && tau > 0.0, "tau must be > 0")?;
    require(
        market_volume_rate.is_finite() && market_volume_rate > 0.0,
        "market_volume_rate must be > 0",
    )?;
    require(
        participation.is_finite() && participation > 0.0 && participation < 1.0,
        "participation must be in (0, 1)",
    )?;
    let clip = participation * market_volume_rate * tau;
    require(
        clip.is_finite(),
        "POV clip participation*market_volume_rate*tau must be finite",
    )?;
    if total_shares == 0.0 {
        return Ok(vec![0.0, 0.0]);
    }
    let ratio = total_shares / clip; // bounded BEFORE any cast/allocation
    require(
        ratio <= MAX_POV_INTERVALS as f64,
        "POV clip too small: would need > 10000000 intervals",
    )?;
    let n_int = ((ratio - 1e-12).ceil() as usize).max(1); // >= 1 for X > 0
    let mut x: Vec<f64> = (0..=n_int)
        .map(|i| (total_shares - clip * i as f64).max(0.0))
        .collect();
    x[n_int] = 0.0;
    Ok(x)
}

/// `E[IS] = 0.5*gamma_p*X^2 + (eta_tilde/tau)*sum n_j^2` (exact discrete sum).
///
/// `holdings = None` uses the optimal trajectory; any finite, fully
/// liquidating path (`|x_N| <= 1e-9*max(1, X)`) may be supplied (TWAP, POV,
/// ...).  The order size `X` in the permanent-impact term is the schedule's
/// `x_0`, not `params.total_shares()` (pinned).
pub fn expected_cost(params: &ACParams, holdings: Option<&[f64]>) -> Result<f64, EtradeError> {
    let traj;
    let x: &[f64] = match holdings {
        Some(h) => h,
        None => {
            traj = ac_trajectory(params);
            &traj
        }
    };
    require_holdings(x)?;
    require_liquidating(x)?;
    let trades = trades_from_holdings(x)?;
    let x0 = x[0];
    let sum_n2: f64 = trades.iter().map(|n| n * n).sum();
    Ok(0.5 * params.gamma_p() * x0 * x0 + params.eta_tilde() / params.tau() * sum_n2)
}

/// `Var[IS] = sigma^2 * tau * sum_{j=1}^{N} x_j^2` (exact discrete sum;
/// `x_N = 0` contributes nothing).  `holdings = None` uses the optimal
/// trajectory; a supplied path must be finite with length `>= 2`.
pub fn variance_cost(params: &ACParams, holdings: Option<&[f64]>) -> Result<f64, EtradeError> {
    let traj;
    let x: &[f64] = match holdings {
        Some(h) => h,
        None => {
            traj = ac_trajectory(params);
            &traj
        }
    };
    require_holdings(x)?;
    let sum_x2: f64 = x.iter().skip(1).map(|v| v * v).sum();
    if sum_x2 == 0.0 {
        return Ok(0.0); // no residual exposure (avoids inf * 0)
    }
    Ok(params.sigma().powi(2) * params.tau() * sum_x2)
}

/// `(E, Var)` of instantaneous liquidation (all `X` in the first interval):
/// `E = 0.5*gamma_p*X^2 + eta_tilde*X^2/tau`, `Var = 0`.
pub fn instantaneous_cost(params: &ACParams) -> (f64, f64) {
    let x0 = params.total_shares();
    let e = 0.5 * params.gamma_p() * x0 * x0 + params.eta_tilde() * x0 * x0 / params.tau();
    (e, 0.0)
}

/// One point on the efficient frontier of execution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrontierPoint {
    /// Risk aversion at this point.
    pub lambda_ra: f64,
    /// Urgency parameter `kappa`.
    pub kappa: f64,
    /// Expected cost `E[IS]`.
    pub e_cost: f64,
    /// Cost variance `Var[IS]`.
    pub var_cost: f64,
    /// Cost standard deviation `sqrt(Var[IS])`.
    pub std_cost: f64,
}

/// Sweep a risk-aversion grid; each point carries
/// `(lambda_ra, kappa, E, Var, sqrt(Var))` of the optimal strategy.
/// The grid must be non-empty and every `lambda_ra` finite and `>= 0`;
/// input order is preserved (duplicates and unsorted grids are reproduced).
pub fn efficient_frontier(
    params: &ACParams,
    lambdas: &[f64],
) -> Result<Vec<FrontierPoint>, EtradeError> {
    require(!lambdas.is_empty(), "lambdas grid must be non-empty")?;
    let mut out = Vec::with_capacity(lambdas.len());
    for &lam in lambdas {
        let p = params.with_lambda(lam)?;
        let e = expected_cost(&p, None)?;
        let v = variance_cost(&p, None)?;
        out.push(FrontierPoint {
            lambda_ra: lam,
            kappa: ac_kappa(&p),
            e_cost: e,
            var_cost: v,
            std_cost: v.sqrt(),
        });
    }
    Ok(out)
}

/// Monte Carlo implementation-shortfall summary (arithmetic-walk simulation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ISSimResult {
    /// Sample mean of IS across paths.
    pub mean: f64,
    /// Sample variance of IS (ddof = 1).
    pub var: f64,
    /// Standard error of the mean, `sqrt(var/M)`.
    pub se_mean: f64,
    /// Standard error of the variance, `var*sqrt(2/(M-1))` (normal approx —
    /// exact here since IS is Gaussian in this model).
    pub se_var: f64,
    /// Number of simulated paths `M`.
    pub n_paths: usize,
}

/// Simulate the implementation shortfall of a schedule under the arithmetic
/// random walk (API_SPEC.md Section 1.7; per path, per interval `j`):
/// execute `n_j` at `S_{j-1} - eta*n_j/tau`, then
/// `S_j = S_{j-1} + sigma*sqrt(tau)*xi_j - gamma_p*n_j`.  `IS` is independent
/// of `S_0`, which is taken as `0`.  The schedule must fully liquidate (same
/// check as [`expected_cost`]) and `2 <= n_paths <= MAX_PATHS`.
/// Deterministic per seed with `ChaCha12Rng`; sample values differ from the
/// Python reference, but the closed forms must match within a few standard
/// errors.
pub fn simulate_is(
    params: &ACParams,
    holdings: Option<&[f64]>,
    n_paths: usize,
    seed: u64,
) -> Result<ISSimResult, EtradeError> {
    require(
        (2..=MAX_PATHS).contains(&n_paths),
        "n_paths must be in [2, 10000000]",
    )?;
    let traj;
    let x: &[f64] = match holdings {
        Some(h) => h,
        None => {
            traj = ac_trajectory(params);
            &traj
        }
    };
    require_holdings(x)?;
    require_liquidating(x)?;
    require(
        (n_paths as u64).saturating_mul((x.len() - 1) as u64) <= MAX_MC_WORK,
        "n_paths * n_intervals must be <= 1000000000",
    )?;
    let trades = trades_from_holdings(x)?;
    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    let srt = params.sigma() * params.tau().sqrt();
    let mut shortfalls = Vec::with_capacity(n_paths);
    for _ in 0..n_paths {
        let mut s_prev = 0.0;
        let mut proceeds = 0.0;
        for &nj in &trades {
            let exec_price = s_prev - params.eta() * nj / params.tau();
            proceeds += nj * exec_price;
            let xi: f64 = rng.sample(StandardNormal);
            s_prev += srt * xi - params.gamma_p() * nj;
        }
        shortfalls.push(-proceeds); // X*S_0 - proceeds with S_0 = 0
    }
    let m = n_paths as f64;
    let mean = shortfalls.iter().sum::<f64>() / m;
    let var = shortfalls.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (m - 1.0);
    Ok(ISSimResult {
        mean,
        var,
        se_mean: (var / m).sqrt(),
        se_var: var * (2.0 / (m - 1.0)).sqrt(),
        n_paths,
    })
}
