//! Power-law impact execution: square-root costs, volume curves, alpha-aware
//! (drift) schedules and a pre-trade report (API_SPEC.md Section 8).
//!
//! Temporary cost per interval is `eta_j * n^(1+alpha)` with
//! `eta_j = eta * w_j^(-alpha)` (`w_j` = fraction of daily volume in the
//! interval).  `alpha = 1` with uniform weights is exactly Almgren-Chriss;
//! `alpha = 1/2` is the square-root law.  The optimal schedule is found by a
//! pinned Newton method (tridiagonal Hessian, Armijo line search) with a
//! bisection over horizons when an adverse drift makes it optimal to finish
//! early, verified by a KKT check.

use crate::error::{require, EtradeError};

/// Upper bound on intervals for the power-law solver (pinned).
pub const MAX_IMPACT_INTERVALS: usize = 10_000;
/// Newton iteration cap per solve (pinned).
pub const NEWTON_MAX_ITER: usize = 100;
/// Convergence: Newton step below this times X in every coordinate (pinned).
pub const NEWTON_STEP_TOL: f64 = 1e-12;
/// Convergence: Newton decrement below this times max(1, |U|) (pinned).
pub const NEWTON_DECREMENT_TOL: f64 = 1e-14;
/// Armijo sufficient-decrease constant (pinned).
pub const ARMIJO_C: f64 = 1e-4;
/// Smallest line-search step, 2^-40 (pinned).
pub const MIN_STEP: f64 = 9.094_947_017_729_282e-13;

/// Illustrative US-equity intraday volume curve: thirteen 30-minute
/// fractions of daily volume (U-shaped, sums to 1; not fitted to data).
pub const US_EQUITY_VOLUME_CURVE_13: [f64; 13] = [
    0.12, 0.09, 0.075, 0.065, 0.06, 0.055, 0.05, 0.055, 0.06, 0.065, 0.075, 0.09, 0.14,
];

/// Power-law impact execution problem (API_SPEC 8.1); validated on construction.
#[derive(Debug, Clone, PartialEq)]
pub struct ImpactParams {
    total_shares: f64,
    n_intervals: usize,
    tau: f64,
    sigma: f64,
    eta: f64,
    alpha: f64,
    gamma_p: f64,
    lambda_ra: f64,
    drift: f64,
    weights: Vec<f64>,
    etas: Vec<f64>,
}

impl ImpactParams {
    /// Validated constructor.  `weights: None` selects `tau` per interval.
    /// Returns [`EtradeError::ModelInvalid`] when the convexity condition
    /// `alpha*(1+alpha)*eta_j*X^(alpha-1) > gamma_p` fails.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        total_shares: f64,
        n_intervals: usize,
        tau: f64,
        sigma: f64,
        eta: f64,
        alpha: f64,
        gamma_p: f64,
        lambda_ra: f64,
        drift: f64,
        weights: Option<Vec<f64>>,
    ) -> Result<Self, EtradeError> {
        require(
            total_shares.is_finite() && total_shares >= 0.0,
            "total_shares must be finite and >= 0",
        )?;
        require(
            (1..=MAX_IMPACT_INTERVALS).contains(&n_intervals),
            "n_intervals must be an integer in [1, 10000]",
        )?;
        require(tau.is_finite() && tau > 0.0, "tau must be > 0")?;
        require(sigma.is_finite() && sigma >= 0.0, "sigma must be >= 0")?;
        require(eta.is_finite() && eta > 0.0, "eta must be > 0")?;
        require(
            alpha.is_finite() && alpha > 0.0 && alpha <= 1.0,
            "alpha must be in (0, 1]",
        )?;
        require(
            gamma_p.is_finite() && gamma_p >= 0.0,
            "gamma_p must be >= 0",
        )?;
        require(
            lambda_ra.is_finite() && lambda_ra >= 0.0,
            "lambda_ra must be >= 0",
        )?;
        require(drift.is_finite() && drift >= 0.0, "drift must be >= 0")?;
        let weights = match weights {
            None => vec![tau; n_intervals],
            Some(w) => {
                require(
                    w.len() == n_intervals,
                    "weights must have length n_intervals",
                )?;
                require(
                    w.iter().all(|v| v.is_finite() && *v > 0.0),
                    "weights must be finite and > 0",
                )?;
                w
            }
        };
        let etas: Vec<f64> = weights.iter().map(|w| eta * w.powf(-alpha)).collect();
        if total_shares > 0.0 {
            let x_pow = total_shares.powf(alpha - 1.0);
            for eta_j in &etas {
                if alpha * (1.0 + alpha) * eta_j * x_pow <= gamma_p {
                    return Err(EtradeError::ModelInvalid(
                        "temporary impact not convex enough against permanent impact \
                         (alpha*(1+alpha)*eta_j*X**(alpha-1) <= gamma_p)"
                            .to_string(),
                    ));
                }
            }
        }
        Ok(Self {
            total_shares,
            n_intervals,
            tau,
            sigma,
            eta,
            alpha,
            gamma_p,
            lambda_ra,
            drift,
            weights,
            etas,
        })
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
    /// Arithmetic volatility.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }
    /// Temporary-impact coefficient.
    pub fn eta(&self) -> f64 {
        self.eta
    }
    /// Impact exponent.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
    /// Permanent-impact coefficient.
    pub fn gamma_p(&self) -> f64 {
        self.gamma_p
    }
    /// Risk aversion.
    pub fn lambda_ra(&self) -> f64 {
        self.lambda_ra
    }
    /// Adverse drift per share per time unit.
    pub fn drift(&self) -> f64 {
        self.drift
    }
    /// Liquidity weights (fraction of daily volume per interval).
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }
    /// Per-interval coefficients `eta * w_j^(-alpha)`.
    pub fn etas(&self) -> &[f64] {
        &self.etas
    }
}

/// Named-field form of [`ImpactParams::new`]: the same arguments, labelled, so a
/// call site cannot silently swap two numbers of the same type.
///
/// [`ImpactSpec::build`] (or `ImpactParams::try_from(spec)`) runs exactly the validation of
/// [`ImpactParams::new`] and returns the same errors.
///
/// ```
/// use etrade::impact::{optimal_schedule, ImpactSpec};
///
/// let p = ImpactSpec {
///     total_shares: 1e6,
///     n_intervals: 5,
///     tau: 1.0,
///     sigma: 0.95,
///     eta: 2.5e-6,
///     alpha: 1.0,
///     gamma_p: 2.5e-7,
///     lambda_ra: 2e-6,
///     drift: 0.0,
///     weights: None,
/// }
/// .build()?;
/// let s = optimal_schedule(&p)?;
/// assert_eq!(s.holdings().len(), 6);
/// assert_eq!(s.holdings()[0], 1e6);
/// # Ok::<(), etrade::EtradeError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ImpactSpec {
    /// Shares to sell, `X >= 0`.
    pub total_shares: f64,
    /// Trading intervals `N`.
    pub n_intervals: usize,
    /// Interval length, `> 0`.
    pub tau: f64,
    /// Arithmetic volatility per sqrt(time unit), `>= 0`.
    pub sigma: f64,
    /// Temporary impact coefficient, `> 0`.
    pub eta: f64,
    /// Impact exponent, `0 < alpha <= 1`.
    pub alpha: f64,
    /// Linear permanent impact coefficient, `>= 0`.
    pub gamma_p: f64,
    /// Risk aversion, `>= 0`.
    pub lambda_ra: f64,
    /// Expected price drift per time unit.
    pub drift: f64,
    /// Volume weights per interval; `None` selects `tau` per interval.
    pub weights: Option<Vec<f64>>,
}

impl ImpactSpec {
    /// Validate and build the [`ImpactParams`] (same checks as [`ImpactParams::new`]).
    pub fn build(self) -> Result<ImpactParams, EtradeError> {
        ImpactParams::try_from(self)
    }
}

impl TryFrom<ImpactSpec> for ImpactParams {
    type Error = EtradeError;

    fn try_from(s: ImpactSpec) -> Result<Self, Self::Error> {
        ImpactParams::new(
            s.total_shares,
            s.n_intervals,
            s.tau,
            s.sigma,
            s.eta,
            s.alpha,
            s.gamma_p,
            s.lambda_ra,
            s.drift,
            s.weights,
        )
    }
}

fn check_holdings(p: &ImpactParams, x: &[f64]) -> Result<(), EtradeError> {
    require(
        x.len() == p.n_intervals + 1,
        "holdings must have length n_intervals + 1",
    )?;
    require(x.iter().all(|v| v.is_finite()), "holdings must be finite")?;
    let scale = x[0].abs().max(1.0);
    require(
        x[x.len() - 1].abs() <= 1e-9 * scale,
        "holdings path must fully liquidate (x_N = 0)",
    )?;
    require(
        x.windows(2).all(|w| w[1] - w[0] <= 1e-12 * scale),
        "holdings must be non-increasing (a sell program)",
    )
}

/// `E[IS]` of a non-increasing, fully liquidating schedule (length `N+1`).
pub fn impact_expected_cost(p: &ImpactParams, holdings: &[f64]) -> Result<f64, EtradeError> {
    check_holdings(p, holdings)?;
    let a1 = 1.0 + p.alpha;
    let mut temp = 0.0;
    for j in 1..holdings.len() {
        let n = (holdings[j - 1] - holdings[j]).max(0.0);
        temp += p.etas[j - 1] * n.powf(a1) - 0.5 * p.gamma_p * n * n;
    }
    let hold: f64 = holdings[1..holdings.len() - 1].iter().sum();
    Ok(0.5 * p.gamma_p * holdings[0] * holdings[0] + temp + p.drift * p.tau * hold)
}

/// `Var[IS] = sigma^2 * tau * sum_{j>=1} x_j^2` (same path checks).
pub fn impact_variance_cost(p: &ImpactParams, holdings: &[f64]) -> Result<f64, EtradeError> {
    check_holdings(p, holdings)?;
    let sq: f64 = holdings[1..].iter().map(|v| v * v).sum();
    Ok(p.sigma * p.sigma * p.tau * sq)
}

/// Volume-weighted schedule: trade `X * w_j / sum(w)`; `x_N = 0` exactly.
pub fn vwap_holdings(total_shares: f64, weights: &[f64]) -> Result<Vec<f64>, EtradeError> {
    require(
        total_shares.is_finite() && total_shares >= 0.0,
        "total_shares must be finite and >= 0",
    )?;
    require(
        !weights.is_empty(),
        "weights must be a non-empty 1-D sequence",
    )?;
    require(
        weights.iter().all(|v| v.is_finite() && *v > 0.0),
        "weights must be finite and > 0",
    )?;
    let sum: f64 = weights.iter().sum();
    let mut x = Vec::with_capacity(weights.len() + 1);
    x.push(total_shares);
    let mut cum = 0.0;
    for w in weights {
        cum += w;
        x.push(total_shares * (1.0 - cum / sum));
    }
    let last = x.len() - 1;
    x[last] = 0.0;
    Ok(x)
}

/// Result of [`optimal_schedule`].
#[derive(Debug, Clone, PartialEq)]
pub struct ScheduleResult {
    /// Holdings path, length `N + 1`.
    holdings: Vec<f64>,
    /// Horizon `m` actually used (`x_k = 0` for `k >= m`).
    pub horizon: usize,
    /// Total Newton iterations.
    pub iterations: usize,
}

impl ScheduleResult {
    /// Holdings path, length `N + 1`.
    pub fn holdings(&self) -> &[f64] {
        &self.holdings
    }
}

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

fn objective(p: &ImpactParams, etas: &[f64], x: &[f64]) -> f64 {
    let big_x = x[0];
    let mut total = 0.5 * p.gamma_p * big_x * big_x;
    let a1 = 1.0 + p.alpha;
    for j in 1..x.len() {
        let n = x[j - 1] - x[j];
        total += etas[j - 1] * n.powf(a1) - 0.5 * p.gamma_p * n * n;
    }
    let mut hold = 0.0;
    for v in &x[1..x.len() - 1] {
        hold += v;
    }
    let mut sq = 0.0;
    for v in &x[1..] {
        sq += v * v;
    }
    total + p.drift * p.tau * hold + p.lambda_ra * (p.sigma * p.sigma) * p.tau * sq
}

fn solve_horizon(p: &ImpactParams, m: usize) -> Option<(Vec<f64>, usize)> {
    let big_x = p.total_shares;
    if m == 1 {
        return Some((vec![big_x, 0.0], 0));
    }
    let (a, a1, g2) = (p.alpha, 1.0 + p.alpha, p.gamma_p);
    let v2 = 2.0 * p.lambda_ra * (p.sigma * p.sigma) * p.tau;
    let dt = p.drift * p.tau;
    let e = &p.etas[..m];
    let mut x: Vec<f64> = (0..=m).map(|k| big_x * (m - k) as f64 / m as f64).collect();
    for it in 1..=NEWTON_MAX_ITER {
        let n: Vec<f64> = (0..m).map(|j| x[j] - x[j + 1]).collect();
        let c1: Vec<f64> = (0..m)
            .map(|j| a1 * e[j] * n[j].powf(a) - g2 * n[j])
            .collect();
        let c2: Vec<f64> = (0..m)
            .map(|j| a * a1 * e[j] * n[j].powf(a - 1.0) - g2)
            .collect();
        let grad: Vec<f64> = (1..m).map(|k| c1[k] - c1[k - 1] + v2 * x[k] + dt).collect();
        let diag: Vec<f64> = (1..m).map(|k| c2[k - 1] + c2[k] + v2).collect();
        let off: Vec<f64> = (1..m).map(|k| -c2[k]).collect();
        let rhs: Vec<f64> = grad.iter().map(|g| -g).collect();
        let step = thomas(&diag, &off, &rhs);
        let u0 = objective(p, e, &x);
        let mut slope = 0.0;
        for (g, s) in grad.iter().zip(&step) {
            slope += g * s;
        }
        let max_step = step.iter().fold(0.0_f64, |acc, s| acc.max(s.abs()));
        if max_step <= NEWTON_STEP_TOL * big_x || -slope <= NEWTON_DECREMENT_TOL * u0.abs().max(1.0)
        {
            return Some((x, it));
        }
        let mut t = 1.0;
        let mut accepted = None;
        while t >= MIN_STEP {
            let mut xn = Vec::with_capacity(m + 1);
            xn.push(x[0]);
            for k in 1..m {
                xn.push(x[k] + t * step[k - 1]);
            }
            xn.push(0.0);
            let positive = (1..=m).all(|j| xn[j - 1] - xn[j] > 0.0);
            if positive && objective(p, e, &xn) <= u0 + ARMIJO_C * t * slope {
                accepted = Some(xn);
                break;
            }
            t *= 0.5;
        }
        x = accepted?;
    }
    None
}

/// Minimise `E + lambda_ra*Var` over non-increasing liquidating schedules
/// (API_SPEC 8.2).  Returns [`EtradeError::SolverFailed`] if no optimum is found.
pub fn optimal_schedule(p: &ImpactParams) -> Result<ScheduleResult, EtradeError> {
    let n_int = p.n_intervals;
    let mut holdings = vec![0.0; n_int + 1];
    if p.total_shares == 0.0 {
        return Ok(ScheduleResult {
            holdings,
            horizon: n_int,
            iterations: 0,
        });
    }
    if let Some((x, it)) = solve_horizon(p, n_int) {
        holdings[..x.len()].copy_from_slice(&x);
        return Ok(ScheduleResult {
            holdings,
            horizon: n_int,
            iterations: it,
        });
    }
    if p.drift <= 0.0 {
        return Err(EtradeError::SolverFailed(
            "optimal schedule: Newton solve did not converge".to_string(),
        ));
    }
    let (mut lo, mut hi, mut iters) = (1_usize, n_int, 0_usize);
    let mut best = solve_horizon(p, 1)
        .ok_or_else(|| EtradeError::SolverFailed("single clip infeasible".to_string()))?;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        match solve_horizon(p, mid) {
            Some(r) => {
                lo = mid;
                iters += r.1;
                best = r;
            }
            None => hi = mid,
        }
    }
    iters += best.1;
    let n_last = best.0[lo - 1] - best.0[lo];
    let marginal = (1.0 + p.alpha) * p.etas[lo - 1] * n_last.powf(p.alpha) - p.gamma_p * n_last;
    if marginal > p.drift * p.tau * (1.0 + 1e-9) + 1e-12 {
        return Err(EtradeError::SolverFailed(
            "optimal schedule: KKT check failed for the early-finish horizon".to_string(),
        ));
    }
    holdings[..best.0.len()].copy_from_slice(&best.0);
    Ok(ScheduleResult {
        holdings,
        horizon: lo,
        iterations: iters,
    })
}

/// Square-root-law estimate `1e4 * y * daily_vol * sqrt(Q / adv)` in bps.
pub fn sqrt_law_cost_bps(
    daily_vol: f64,
    order_shares: f64,
    adv: f64,
    y: f64,
) -> Result<f64, EtradeError> {
    require(
        daily_vol.is_finite() && daily_vol > 0.0,
        "daily_vol must be finite and > 0",
    )?;
    require(
        order_shares.is_finite() && order_shares >= 0.0,
        "order_shares must be >= 0",
    )?;
    require(adv.is_finite() && adv > 0.0, "adv must be finite and > 0")?;
    require(y.is_finite() && y > 0.0, "y must be finite and > 0")?;
    Ok(1e4 * y * daily_vol * (order_shares / adv).sqrt())
}
