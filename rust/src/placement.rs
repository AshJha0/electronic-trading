//! Limit-order placement for execution: queue-position fill probability, the
//! passive-or-aggressive choice for one child order, and liquidation with
//! limit and market orders (API_SPEC.md Section 13).
//!
//! Sections 1, 8 and 12 decide how much to trade per interval and assume
//! every child order crosses the spread.  A resting limit order at the touch
//! saves the half spread but may not fill, and an unfilled remainder must
//! still be completed, usually after the price has moved away.
//!
//! # Fill probability by queue position (API_SPEC 13.1)
//!
//! A FIFO queue holds `q_ahead` units ahead of our order of `size` units.
//! Unit market orders arrive as a Poisson process of rate `mu`; each unit
//! ahead cancels at rate `theta`; our units never cancel.  The fill time is
//! `sum_{k=1}^{q_ahead} Exp(mu + theta k) + sum_{i=1}^{size} Exp(mu)` (the
//! pure-death queue of Cont, Stoikov and Talreja 2010; Cont and de Larrard
//! 2013 use the same order flow), evaluated by pinned uniformisation.
//!
//! # Passive or aggressive (API_SPEC 13.2)
//!
//! Crossing pays the half spread plus the Section 8 impact; posting earns the
//! half spread with probability `P` and otherwise crosses after an adverse
//! move.  The impact is paid either way (the square-root law holds for both
//! order types, Toth et al. 2011), so posting wins exactly when
//! `P > adverse_move / (adverse_move + spread)` (Moallemi and Yuan 2016 value
//! queue position on the same terms).
//!
//! # Liquidation with limit and market orders (API_SPEC 13.3)
//!
//! A discrete Cartea and Jaimungal (2015) policy by backward induction: post
//! one unit at depth `max(0, 1/kappa - g)` or send a market order, with a
//! seeded Monte Carlo against all-aggressive and all-passive-then-sweep.
//!
//! # References
//!
//! * Cont, R., Stoikov, S. and Talreja, R. (2010). A stochastic model for
//!   order book dynamics. Operations Research 58(3), 549-563.
//! * Cont, R. and de Larrard, A. (2013). Price dynamics in a Markovian limit
//!   order market. SIAM Journal on Financial Mathematics 4(1), 1-25.
//! * Moallemi, C. C. and Yuan, K. (2016). A model for queue position
//!   valuation in a limit order book. Working paper, Columbia Business School.
//! * Cartea, A. and Jaimungal, S. (2015). Optimal execution with limit and
//!   market orders. Quantitative Finance 15(8), 1279-1291.
//! * Toth, B. et al. (2011). Anomalous price impact and the critical nature
//!   of liquidity in financial markets. Physical Review X 1(2), 021006.

use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rand_distr::StandardNormal;

use crate::almgren_chriss::MAX_PATHS;
use crate::error::{require, EtradeError};

/// Upper bound on `q_ahead + size` in [`queue_fill_probability`] (pinned).
pub const MAX_QUEUE_UNITS: usize = 5_000;

/// Upper bound on uniformisation steps in [`queue_fill_probability`] (pinned).
pub const MAX_UNIFORMISATION_STEPS: usize = 200_000;

/// Upper bound on `total_units` of a liquidation (pinned).
pub const MAX_LIQ_UNITS: usize = 1_000;

/// Upper bound on `n_periods` of a liquidation (pinned).
pub const MAX_LIQ_PERIODS: usize = 1_000;

const TOTAL_UNITS_MSG: &str = "total_units must be an integer in [1, 1000]";
const N_PERIODS_MSG: &str = "n_periods must be an integer in [1, 1000]";

// ------------------------------------------------------------------ fill probability

/// Probability that a resting order of `size` units with `q_ahead` units
/// ahead fills completely within `horizon` (API_SPEC 13.1).
///
/// Market orders of one unit arrive at rate `mu > 0`; each unit ahead
/// cancels at rate `theta >= 0`.  Returns [`EtradeError::InvalidInput`] for
/// `size == 0`, `q_ahead + size > MAX_QUEUE_UNITS`, non-finite or
/// out-of-domain rates or horizon, and a horizon needing more than
/// `MAX_UNIFORMISATION_STEPS` steps.
pub fn queue_fill_probability(
    q_ahead: usize,
    size: usize,
    mu: f64,
    theta: f64,
    horizon: f64,
) -> Result<f64, EtradeError> {
    require(size >= 1, "size must be an integer >= 1")?;
    require(
        size <= MAX_QUEUE_UNITS && q_ahead <= MAX_QUEUE_UNITS - size,
        "q_ahead + size must be <= 5000",
    )?;
    require(mu.is_finite() && mu > 0.0, "mu must be finite and > 0")?;
    require(
        theta.is_finite() && theta >= 0.0,
        "theta must be finite and >= 0",
    )?;
    require(
        horizon.is_finite() && horizon >= 0.0,
        "horizon must be finite and >= 0",
    )?;
    let lam = mu + theta * q_ahead as f64;
    let big_l = lam * horizon;
    require(
        big_l.is_finite(),
        "horizon too long: (mu + theta*q_ahead)*horizon overflows",
    )?;
    if big_l == 0.0 {
        return Ok(0.0);
    }
    let m_f = big_l.floor();
    let half = (10.0 * big_l.sqrt() + 10.0).ceil();
    require(
        m_f + half <= MAX_UNIFORMISATION_STEPS as f64,
        "horizon too long: (mu + theta*q_ahead)*horizon needs more than 200000 uniformisation steps",
    )?;
    let m = m_f as usize;
    let hf = half as usize;
    let k_lo = m.saturating_sub(hf);
    let k_hi = m + hf;
    let n_st = q_ahead + size;
    let go: Vec<f64> = (0..n_st)
        .map(|s| {
            let r = if s < q_ahead {
                mu + theta * (q_ahead - s) as f64
            } else {
                mu
            };
            r / lam
        })
        .collect();
    let stay: Vec<f64> = go.iter().map(|g| 1.0 - g).collect();
    let mut w = vec![0.0; k_hi + 1];
    w[m] = 1.0;
    for k in m + 1..=k_hi {
        w[k] = w[k - 1] * big_l / k as f64;
    }
    for k in (k_lo..m).rev() {
        w[k] = w[k + 1] * (k + 1) as f64 / big_l;
    }
    let mut p = vec![0.0; n_st];
    p[0] = 1.0;
    let mut absorbed = 0.0;
    let mut acc = 0.0;
    let mut total_w = 0.0;
    for (k, &wk) in w.iter().enumerate() {
        if k > 0 {
            let mut prev_move = 0.0;
            for ((ps, g), st) in p.iter_mut().zip(&go).zip(&stay) {
                let mv = *ps * g;
                *ps = *ps * st + prev_move;
                prev_move = mv;
            }
            absorbed += prev_move;
        }
        if k >= k_lo {
            acc += wk * absorbed;
            total_w += wk;
        }
    }
    Ok((acc / total_w).clamp(0.0, 1.0))
}

// ------------------------------------------------------------------ passive or aggressive

/// Expected cost per share (currency, relative to the mid) of posting at the
/// touch and of crossing now (API_SPEC 13.2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlacementDecision {
    /// Expected cost of posting at the touch.
    pub passive_cost: f64,
    /// Cost of crossing now: half spread plus impact.
    pub aggressive_cost: f64,
    /// Section 8 temporary impact per share.
    pub impact_cost: f64,
    /// Fill probability above which posting is cheaper.
    pub breakeven_fill_prob: f64,
    /// True when posting is strictly cheaper (ties cross).
    pub passive: bool,
}

/// Compare posting one sell child order at the touch with crossing now
/// (API_SPEC 13.2).  `eta`, `alpha` and `weight` are the Section 8 impact
/// parameters of the interval the child order trades in.
pub fn place_or_cross(
    size: f64,
    spread: f64,
    fill_prob: f64,
    adverse_move: f64,
    eta: f64,
    alpha: f64,
    weight: f64,
) -> Result<PlacementDecision, EtradeError> {
    require(
        size.is_finite() && size > 0.0,
        "size must be finite and > 0",
    )?;
    require(
        spread.is_finite() && spread >= 0.0,
        "spread must be finite and >= 0",
    )?;
    require(
        fill_prob.is_finite() && (0.0..=1.0).contains(&fill_prob),
        "fill_prob must be in [0, 1]",
    )?;
    require(
        adverse_move.is_finite() && adverse_move >= 0.0,
        "adverse_move must be finite and >= 0",
    )?;
    require(eta.is_finite() && eta >= 0.0, "eta must be finite and >= 0")?;
    require(
        alpha.is_finite() && alpha > 0.0 && alpha <= 1.0,
        "alpha must be in (0, 1]",
    )?;
    require(
        weight.is_finite() && weight > 0.0,
        "weight must be finite and > 0",
    )?;
    let hs = 0.5 * spread;
    let impact = eta * weight.powf(-alpha) * size.powf(alpha);
    let aggressive = hs + impact;
    let passive = fill_prob * (impact - hs) + (1.0 - fill_prob) * (adverse_move + hs + impact);
    let denom = adverse_move + spread;
    let breakeven = if denom > 0.0 {
        adverse_move / denom
    } else {
        1.0
    };
    Ok(PlacementDecision {
        passive_cost: passive,
        aggressive_cost: aggressive,
        impact_cost: impact,
        breakeven_fill_prob: breakeven,
        passive: passive < aggressive,
    })
}

// ------------------------------------------------------------------ liquidation

/// Sell `total_units` over `n_periods` with limit and market orders
/// (API_SPEC 13.3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LiquidationParams {
    total_units: usize,
    n_periods: usize,
    dt: f64,
    sigma: f64,
    spread: f64,
    kappa: f64,
    p_touch: f64,
    beta: f64,
    phi: f64,
}

impl LiquidationParams {
    /// Validated constructor.
    ///
    /// * `total_units` — `Q` in `[1, MAX_LIQ_UNITS]`;
    /// * `n_periods`   — `N` in `[1, MAX_LIQ_PERIODS]`;
    /// * `dt`, `kappa` — `> 0`; `sigma`, `spread`, `beta`, `phi` — `>= 0`;
    /// * `p_touch`     — per-period fill probability at the touch, `(0, 1]`.
    ///
    /// All finite; returns [`EtradeError::InvalidInput`] for every violation.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        total_units: usize,
        n_periods: usize,
        dt: f64,
        sigma: f64,
        spread: f64,
        kappa: f64,
        p_touch: f64,
        beta: f64,
        phi: f64,
    ) -> Result<Self, EtradeError> {
        require((1..=MAX_LIQ_UNITS).contains(&total_units), TOTAL_UNITS_MSG)?;
        require((1..=MAX_LIQ_PERIODS).contains(&n_periods), N_PERIODS_MSG)?;
        require(dt.is_finite() && dt > 0.0, "dt must be finite and > 0")?;
        require(
            sigma.is_finite() && sigma >= 0.0,
            "sigma must be finite and >= 0",
        )?;
        require(
            spread.is_finite() && spread >= 0.0,
            "spread must be finite and >= 0",
        )?;
        require(
            kappa.is_finite() && kappa > 0.0,
            "kappa must be finite and > 0",
        )?;
        require(
            p_touch.is_finite() && p_touch > 0.0 && p_touch <= 1.0,
            "p_touch must be in (0, 1]",
        )?;
        require(
            beta.is_finite() && beta >= 0.0,
            "beta must be finite and >= 0",
        )?;
        require(phi.is_finite() && phi >= 0.0, "phi must be finite and >= 0")?;
        Ok(Self {
            total_units,
            n_periods,
            dt,
            sigma,
            spread,
            kappa,
            p_touch,
            beta,
            phi,
        })
    }

    /// Units to sell `Q`.
    pub fn total_units(&self) -> usize {
        self.total_units
    }

    /// Number of periods `N`.
    pub fn n_periods(&self) -> usize {
        self.n_periods
    }

    /// Period length.
    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Mid volatility per sqrt time unit.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Quoted spread.
    pub fn spread(&self) -> f64 {
        self.spread
    }

    /// Fill decay per unit of depth.
    pub fn kappa(&self) -> f64 {
        self.kappa
    }

    /// Per-period fill probability at the touch.
    pub fn p_touch(&self) -> f64 {
        self.p_touch
    }

    /// Book walk per unit of a market order.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Running inventory penalty.
    pub fn phi(&self) -> f64 {
        self.phi
    }
}

/// Backward-induction policy (API_SPEC 13.3).
///
/// `cost[j][q]` (`(N + 1) x (Q + 1)`) is the optimal expected cost to go
/// including the running penalty; `depth[j][q]` and `market[j][q]`
/// (`N x (Q + 1)`) the action (`depth` is 0 where `market` is true).
#[derive(Debug, Clone, PartialEq)]
pub struct LiquidationPolicy {
    /// Cost to go.
    pub cost: Vec<Vec<f64>>,
    /// Posting depth beyond the touch.
    pub depth: Vec<Vec<f64>>,
    /// True where a market order is sent.
    pub market: Vec<Vec<bool>>,
}

impl LiquidationPolicy {
    /// `C_0(Q)`: the policy's expected shortfall plus penalties.
    pub fn expected_cost(&self) -> f64 {
        *self.cost[0].last().expect("Q >= 1")
    }
}

/// Backward induction of API_SPEC 13.3 (`j` from `N - 1` down to 0, `q` from
/// 1 up to `Q`).
pub fn liquidation_policy(p: &LiquidationParams) -> LiquidationPolicy {
    let (n_per, big_q) = (p.n_periods, p.total_units);
    let hs = 0.5 * p.spread;
    let mut cost = vec![vec![0.0; big_q + 1]; n_per + 1];
    let mut depth = vec![vec![0.0; big_q + 1]; n_per];
    let mut market = vec![vec![false; big_q + 1]; n_per];
    for (q, c) in cost[n_per].iter_mut().enumerate() {
        let qd = q as f64;
        *c = qd * (hs + p.beta * qd);
    }
    let inv_k = 1.0 / p.kappa;
    for j in (0..n_per).rev() {
        for q in 1..=big_q {
            let c_hold = cost[j + 1][q];
            let c_less = cost[j + 1][q - 1];
            let g = hs + (c_hold - c_less);
            let d = (inv_k - g).max(0.0);
            let gain = p.p_touch * (-p.kappa * d).exp() * (d + g);
            let post = c_hold - gain;
            let mkt = hs + p.beta + c_less;
            let base = if mkt < post {
                market[j][q] = true;
                mkt
            } else {
                depth[j][q] = d;
                post
            };
            let qd = q as f64;
            cost[j][q] = p.phi * qd * qd * p.dt + base;
        }
    }
    LiquidationPolicy {
        cost,
        depth,
        market,
    }
}

/// Even market-order schedule `n_j = floor(Q(j+1)/N) - floor(Qj/N)`.
pub fn aggressive_schedule(
    total_units: usize,
    n_periods: usize,
) -> Result<Vec<usize>, EtradeError> {
    require((1..=MAX_LIQ_UNITS).contains(&total_units), TOTAL_UNITS_MSG)?;
    require((1..=MAX_LIQ_PERIODS).contains(&n_periods), N_PERIODS_MSG)?;
    Ok((0..n_periods)
        .map(|j| total_units * (j + 1) / n_periods - total_units * j / n_periods)
        .collect())
}

/// Expected shortfall of the all-aggressive baseline,
/// `sum_j n_j (half_spread + beta n_j)`.
pub fn aggressive_cost(p: &LiquidationParams) -> f64 {
    let hs = 0.5 * p.spread;
    aggressive_schedule(p.total_units, p.n_periods)
        .expect("validated params")
        .into_iter()
        .map(|n| {
            let nd = n as f64;
            nd * (hs + p.beta * nd)
        })
        .sum()
}

/// Shortfall (currency, `S_0 = 0`) of the policy and the two baselines;
/// savings are paired per path (baseline - optimal, positive = policy
/// cheaper); SEs are sample std (ddof 1) / sqrt(M).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LiquidationSimResult {
    /// Mean shortfall of the policy.
    pub optimal_mean: f64,
    /// Standard error of `optimal_mean`.
    pub optimal_se: f64,
    /// Sample standard deviation of the policy's shortfall.
    pub optimal_std: f64,
    /// Mean shortfall of all-aggressive.
    pub aggressive_mean: f64,
    /// Standard error of `aggressive_mean`.
    pub aggressive_se: f64,
    /// Sample standard deviation of all-aggressive.
    pub aggressive_std: f64,
    /// Mean shortfall of all-passive-then-sweep.
    pub passive_mean: f64,
    /// Standard error of `passive_mean`.
    pub passive_se: f64,
    /// Sample standard deviation of all-passive-then-sweep.
    pub passive_std: f64,
    /// Mean paired saving `aggressive - optimal`.
    pub saving_vs_aggressive_mean: f64,
    /// Standard error of `saving_vs_aggressive_mean`.
    pub saving_vs_aggressive_se: f64,
    /// Mean paired saving `passive - optimal`.
    pub saving_vs_passive_mean: f64,
    /// Standard error of `saving_vs_passive_mean`.
    pub saving_vs_passive_se: f64,
    /// Number of paths simulated.
    pub n_paths: usize,
}

fn mean_std(v: &[f64]) -> (f64, f64) {
    let m = v.len() as f64;
    let mean = v.iter().sum::<f64>() / m;
    let ss: f64 = v.iter().map(|x| (x - mean) * (x - mean)).sum();
    (mean, (ss / (m - 1.0)).sqrt())
}

/// Seeded Monte Carlo of the policy against all-aggressive and
/// all-passive-then-sweep (API_SPEC 13.3).
///
/// Per period: draw `u ~ U(0,1)` then `z ~ N(0,1)` (shared by the three
/// strategies); trade at the current mid; then `S += sigma sqrt(dt) z`;
/// sweep the remainder at the end.  Paths are simulated one at a time
/// (paths outer, periods inner) with `ChaCha12Rng::seed_from_u64(seed)`:
/// results match other ports only statistically.
pub fn simulate_liquidation(
    p: &LiquidationParams,
    n_paths: usize,
    seed: u64,
) -> Result<LiquidationSimResult, EtradeError> {
    require(
        (2..=MAX_PATHS).contains(&n_paths),
        "n_paths must be an integer in [2, 10000000]",
    )?;
    let pol = liquidation_policy(p);
    let sched = aggressive_schedule(p.total_units, p.n_periods)?;
    let hs = 0.5 * p.spread;
    let srt = p.sigma * p.dt.sqrt();
    let fill_at: Vec<Vec<f64>> = pol
        .depth
        .iter()
        .map(|row| {
            row.iter()
                .map(|d| p.p_touch * (-p.kappa * d).exp())
                .collect()
        })
        .collect();
    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    let mut is_opt = Vec::with_capacity(n_paths);
    let mut is_agg = Vec::with_capacity(n_paths);
    let mut is_pas = Vec::with_capacity(n_paths);
    let mut sav_a = Vec::with_capacity(n_paths);
    let mut sav_p = Vec::with_capacity(n_paths);
    for _ in 0..n_paths {
        let mut s = 0.0;
        let (mut q_opt, mut q_pas) = (p.total_units, p.total_units);
        let (mut pr_opt, mut pr_agg, mut pr_pas) = (0.0, 0.0, 0.0);
        for (j, &n_j) in sched.iter().enumerate() {
            let u: f64 = rng.gen();
            let z: f64 = rng.sample(StandardNormal);
            if q_opt > 0 {
                if pol.market[j][q_opt] {
                    pr_opt += s - hs - p.beta;
                    q_opt -= 1;
                } else if u < fill_at[j][q_opt] {
                    pr_opt += s + hs + pol.depth[j][q_opt];
                    q_opt -= 1;
                }
            }
            if q_pas > 0 && u < p.p_touch {
                pr_pas += s + hs;
                q_pas -= 1;
            }
            if n_j > 0 {
                let nd = n_j as f64;
                pr_agg += nd * (s - hs - p.beta * nd);
            }
            s += srt * z;
        }
        let (qo, qp) = (q_opt as f64, q_pas as f64);
        pr_opt += qo * (s - hs - p.beta * qo);
        pr_pas += qp * (s - hs - p.beta * qp);
        is_opt.push(-pr_opt);
        is_agg.push(-pr_agg);
        is_pas.push(-pr_pas);
        sav_a.push(-pr_agg + pr_opt);
        sav_p.push(-pr_pas + pr_opt);
    }
    let (m_o, sd_o) = mean_std(&is_opt);
    let (m_a, sd_a) = mean_std(&is_agg);
    let (m_p, sd_p) = mean_std(&is_pas);
    let (m_va, sd_va) = mean_std(&sav_a);
    let (m_vp, sd_vp) = mean_std(&sav_p);
    let root = (n_paths as f64).sqrt();
    Ok(LiquidationSimResult {
        optimal_mean: m_o,
        optimal_se: sd_o / root,
        optimal_std: sd_o,
        aggressive_mean: m_a,
        aggressive_se: sd_a / root,
        aggressive_std: sd_a,
        passive_mean: m_p,
        passive_se: sd_p / root,
        passive_std: sd_p,
        saving_vs_aggressive_mean: m_va,
        saving_vs_aggressive_se: sd_va / root,
        saving_vs_passive_mean: m_vp,
        saving_vs_passive_se: sd_vp / root,
        n_paths,
    })
}
