//! Market-unit layer: build model parameters from quoted market quantities
//! and report results in basis points (API_SPEC.md Section 7).
//!
//! Almgren-Chriss impact follows the paper's own calibration: trading 1% of
//! ADV per day costs one spread of temporary impact, 10% of ADV moves the
//! price permanently by one spread.  Avellaneda-Stoikov flow is described by
//! the fill rate of a quote at the touch and the distance over which that
//! rate halves; risk aversion by the opening skew per unit of inventory.
//!
//! ```
//! use etrade::market::{ac_params_from_market, ACMarket};
//! // Almgren and Chriss (2000) Section 4 example.
//! let m = ACMarket {
//!     price: 50.0, spread: 0.125, adv: 5.0e6, daily_vol: 0.019,
//!     total_shares: 1.0e6, n_intervals: 5, tau: 1.0, lambda_ra: 2.0e-6,
//!     ..ACMarket::default()
//! };
//! let p = ac_params_from_market(&m).unwrap();
//! assert!((p.eta() - 2.5e-6).abs() < 1e-20);
//! ```

use crate::error::{require, EtradeError};
use crate::impact::{
    impact_expected_cost, impact_variance_cost, optimal_schedule, sqrt_law_cost_bps, vwap_holdings,
    ImpactParams,
};
use crate::{twap_holdings, ACParams, ASParams};

/// Default fraction of ADV per day that costs one spread of temporary impact.
pub const DEFAULT_TEMP_FRAC: f64 = 0.01;
/// Default fraction of ADV that moves the price permanently by one spread.
pub const DEFAULT_PERM_FRAC: f64 = 0.10;
/// Default trading-session length in seconds (US equities, 6.5 hours).
pub const DEFAULT_SESSION_SECONDS: f64 = 23_400.0;

/// Almgren-Chriss inputs in market units (API_SPEC 7.1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ACMarket {
    /// Price, > 0.
    pub price: f64,
    /// Full quoted spread in currency, > 0.
    pub spread: f64,
    /// Average daily volume in shares per day, > 0.
    pub adv: f64,
    /// Daily volatility as a fraction (0.015 = 1.5%), > 0.
    pub daily_vol: f64,
    /// Shares to sell, >= 0.
    pub total_shares: f64,
    /// Number of trading intervals.
    pub n_intervals: usize,
    /// Interval length in trading days.
    pub tau: f64,
    /// Risk aversion (1/currency), >= 0.
    pub lambda_ra: f64,
    /// Fraction of ADV per day costing one spread of temporary impact, in (0, 1].
    pub temp_frac: f64,
    /// Fraction of ADV moving the price permanently by one spread, in (0, 1].
    pub perm_frac: f64,
}

impl Default for ACMarket {
    fn default() -> Self {
        Self {
            price: 0.0,
            spread: 0.0,
            adv: 0.0,
            daily_vol: 0.0,
            total_shares: 0.0,
            n_intervals: 1,
            tau: 1.0,
            lambda_ra: 0.0,
            temp_frac: DEFAULT_TEMP_FRAC,
            perm_frac: DEFAULT_PERM_FRAC,
        }
    }
}

/// Avellaneda-Stoikov inputs in market units (API_SPEC 7.2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ASMarket {
    /// Price (initial mid), > 0.
    pub price: f64,
    /// Tick size, > 0.
    pub tick: f64,
    /// Daily volatility as a fraction, > 0.
    pub daily_vol: f64,
    /// Fills per side per day of a quote half a tick from the mid, > 0.
    pub touch_fills_per_day: f64,
    /// Ticks over which the intensity halves, > 0.
    pub halving_ticks: f64,
    /// Opening skew per unit of inventory in ticks, > 0.
    pub skew_ticks: f64,
    /// Hard inventory cap.
    pub q_max: u32,
    /// Terminal liquidation penalty in ticks, >= 0.
    pub liq_ticks: f64,
    /// Step length in seconds, > 0.
    pub dt_seconds: f64,
    /// Session length in seconds, > 0.
    pub session_seconds: f64,
    /// Horizon in trading days, > 0.
    pub horizon_days: f64,
}

impl Default for ASMarket {
    fn default() -> Self {
        Self {
            price: 0.0,
            tick: 0.0,
            daily_vol: 0.0,
            touch_fills_per_day: 0.0,
            halving_ticks: 0.0,
            skew_ticks: 0.0,
            q_max: 0,
            liq_ticks: 0.0,
            dt_seconds: 0.0,
            session_seconds: DEFAULT_SESSION_SECONDS,
            horizon_days: 1.0,
        }
    }
}

fn positive(name: &str, v: f64) -> Result<(), EtradeError> {
    require(
        v.is_finite() && v > 0.0,
        &format!("{name} must be finite and > 0"),
    )
}

/// `sigma = price*daily_vol`, `eta = spread/(temp_frac*adv)`,
/// `gamma_p = spread/(perm_frac*adv)`; the result is validated as any
/// [`ACParams`].
pub fn ac_params_from_market(m: &ACMarket) -> Result<ACParams, EtradeError> {
    positive("price", m.price)?;
    positive("spread", m.spread)?;
    positive("adv", m.adv)?;
    positive("daily_vol", m.daily_vol)?;
    require(
        m.temp_frac.is_finite() && m.temp_frac > 0.0 && m.temp_frac <= 1.0,
        "temp_frac must be in (0, 1]",
    )?;
    require(
        m.perm_frac.is_finite() && m.perm_frac > 0.0 && m.perm_frac <= 1.0,
        "perm_frac must be in (0, 1]",
    )?;
    ACParams::new(
        m.total_shares,
        m.n_intervals,
        m.tau,
        m.price * m.daily_vol,
        m.spread / (m.temp_frac * m.adv),
        m.spread / (m.perm_frac * m.adv),
        m.lambda_ra,
    )
}

/// `k = ln2/(halving_ticks*tick)`, `A = touch_fills_per_day*exp(k*tick/2)`,
/// `gamma = skew_ticks*tick/(sigma^2*horizon_days)`,
/// `eta_liq = liq_ticks*tick`, `dt = dt_seconds/session_seconds`; the result
/// is validated as any [`ASParams`].
pub fn as_params_from_market(m: &ASMarket) -> Result<ASParams, EtradeError> {
    positive("price", m.price)?;
    positive("tick", m.tick)?;
    positive("daily_vol", m.daily_vol)?;
    positive("touch_fills_per_day", m.touch_fills_per_day)?;
    positive("halving_ticks", m.halving_ticks)?;
    positive("skew_ticks", m.skew_ticks)?;
    require(
        m.liq_ticks.is_finite() && m.liq_ticks >= 0.0,
        "liq_ticks must be finite and >= 0",
    )?;
    positive("dt_seconds", m.dt_seconds)?;
    positive("session_seconds", m.session_seconds)?;
    positive("horizon_days", m.horizon_days)?;
    let sigma = m.price * m.daily_vol;
    let k = 2.0_f64.ln() / (m.halving_ticks * m.tick);
    ASParams::new(
        m.price,
        sigma,
        m.horizon_days,
        m.dt_seconds / m.session_seconds,
        m.skew_ticks * m.tick / (sigma * sigma * m.horizon_days),
        k,
        m.touch_fills_per_day * (k * m.tick / 2.0).exp(),
        m.q_max,
        m.liq_ticks * m.tick,
    )
}

/// Basis points of a notional: `1e4 * amount / notional` (`notional > 0`).
pub fn bps(amount: f64, notional: f64) -> Result<f64, EtradeError> {
    require(amount.is_finite(), "amount must be finite")?;
    positive("notional", notional)?;
    Ok(1e4 * amount / notional)
}

/// Cost of crossing half the quoted spread on every share: `0.5*spread*X`.
pub fn half_spread_cost(spread: f64, total_shares: f64) -> Result<f64, EtradeError> {
    positive("spread", spread)?;
    require(
        total_shares.is_finite() && total_shares >= 0.0,
        "total_shares must be finite and >= 0",
    )?;
    Ok(0.5 * spread * total_shares)
}

/// Power-law impact inputs in market units (API_SPEC 8.3).
#[derive(Debug, Clone, PartialEq)]
pub struct ImpactMarket {
    /// Price, > 0.
    pub price: f64,
    /// Full quoted spread, > 0.
    pub spread: f64,
    /// Average daily volume, > 0.
    pub adv: f64,
    /// Daily volatility as a fraction, > 0.
    pub daily_vol: f64,
    /// Shares to sell, >= 0.
    pub total_shares: f64,
    /// Number of intervals.
    pub n_intervals: usize,
    /// Interval length in trading days.
    pub tau: f64,
    /// Risk aversion, >= 0.
    pub lambda_ra: f64,
    /// Impact exponent in (0, 1].
    pub alpha: f64,
    /// Square-root-law prefactor Y, > 0.
    pub y: f64,
    /// Fraction of ADV moving the price permanently by one spread, in (0, 1].
    pub perm_frac: f64,
    /// Expected adverse drift in bps per day, >= 0.
    pub drift_bps_per_day: f64,
    /// Volume fractions per interval (`None` = `tau` each).
    pub weights: Option<Vec<f64>>,
}

impl Default for ImpactMarket {
    fn default() -> Self {
        Self {
            price: 0.0,
            spread: 0.0,
            adv: 0.0,
            daily_vol: 0.0,
            total_shares: 0.0,
            n_intervals: 1,
            tau: 1.0,
            lambda_ra: 0.0,
            alpha: 0.5,
            y: 1.0,
            perm_frac: DEFAULT_PERM_FRAC,
            drift_bps_per_day: 0.0,
            weights: None,
        }
    }
}

/// `eta = y*sigma/adv^alpha`, `gamma_p = spread/(perm_frac*adv)`,
/// `drift = price*drift_bps_per_day*1e-4` (API_SPEC 8.3).
pub fn impact_params_from_market(m: &ImpactMarket) -> Result<ImpactParams, EtradeError> {
    positive("price", m.price)?;
    positive("spread", m.spread)?;
    positive("adv", m.adv)?;
    positive("daily_vol", m.daily_vol)?;
    positive("y", m.y)?;
    require(
        m.perm_frac.is_finite() && m.perm_frac > 0.0 && m.perm_frac <= 1.0,
        "perm_frac must be in (0, 1]",
    )?;
    require(
        m.drift_bps_per_day.is_finite() && m.drift_bps_per_day >= 0.0,
        "drift_bps_per_day must be finite and >= 0",
    )?;
    require(
        m.alpha.is_finite() && m.alpha > 0.0 && m.alpha <= 1.0,
        "alpha must be in (0, 1]",
    )?;
    let sigma = m.price * m.daily_vol;
    ImpactParams::new(
        m.total_shares,
        m.n_intervals,
        m.tau,
        sigma,
        m.y * sigma / m.adv.powf(m.alpha),
        m.alpha,
        m.spread / (m.perm_frac * m.adv),
        m.lambda_ra,
        m.price * m.drift_bps_per_day * 1e-4,
        m.weights.clone(),
    )
}

/// One strategy in a pre-trade report (bps of `X * price`).
#[derive(Debug, Clone, PartialEq)]
pub struct PretradeRow {
    /// `"twap"`, `"vwap"` or `"optimal"`.
    pub label: &'static str,
    /// Risk aversion of the row (0 for TWAP and VWAP).
    pub lambda_ra: f64,
    /// `bps(E[IS])`.
    pub impact_bps: f64,
    /// `bps(sqrt(Var))`.
    pub sd_bps: f64,
    /// `impact_bps + half_spread_bps`.
    pub total_bps: f64,
    /// Horizon used.
    pub horizon: usize,
}

/// Pre-trade report (API_SPEC 8.4).
#[derive(Debug, Clone, PartialEq)]
pub struct PretradeReport {
    /// Closed-form square-root-law estimate.
    pub sqrt_law_bps: f64,
    /// Half-spread cost in bps.
    pub half_spread_bps: f64,
    /// TWAP, VWAP, then optimal per lambda.
    pub rows: Vec<PretradeRow>,
}

/// TWAP, VWAP and the optimal schedule per lambda, costed on the power-law
/// model; `m.lambda_ra` is ignored (API_SPEC 8.4).
pub fn pretrade_report(m: &ImpactMarket, lambdas: &[f64]) -> Result<PretradeReport, EtradeError> {
    require(!lambdas.is_empty(), "lambdas must be non-empty")?;
    let p0 = impact_params_from_market(&ImpactMarket {
        lambda_ra: 0.0,
        ..m.clone()
    })?;
    let notional = m.total_shares * m.price;
    positive("total_shares * price", notional)?;
    let hs = bps(half_spread_cost(m.spread, m.total_shares)?, notional)?;
    let row = |label: &'static str,
               lam: f64,
               x: &[f64],
               horizon: usize,
               p: &ImpactParams|
     -> Result<PretradeRow, EtradeError> {
        let e = bps(impact_expected_cost(p, x)?, notional)?;
        let sd = bps(impact_variance_cost(p, x)?.sqrt(), notional)?;
        Ok(PretradeRow {
            label,
            lambda_ra: lam,
            impact_bps: e,
            sd_bps: sd,
            total_bps: e + hs,
            horizon,
        })
    };
    let mut rows = vec![
        row(
            "twap",
            0.0,
            &twap_holdings(m.total_shares, m.n_intervals)?,
            m.n_intervals,
            &p0,
        )?,
        row(
            "vwap",
            0.0,
            &vwap_holdings(m.total_shares, p0.weights())?,
            m.n_intervals,
            &p0,
        )?,
    ];
    for &lam in lambdas {
        require(
            lam.is_finite() && lam >= 0.0,
            "lambdas must be finite and >= 0",
        )?;
        let p = impact_params_from_market(&ImpactMarket {
            lambda_ra: lam,
            ..m.clone()
        })?;
        let s = optimal_schedule(&p)?;
        rows.push(row("optimal", lam, &s.holdings, s.horizon, &p)?);
    }
    Ok(PretradeReport {
        sqrt_law_bps: sqrt_law_cost_bps(m.daily_vol, m.total_shares, m.adv, m.y)?,
        half_spread_bps: hs,
        rows,
    })
}
