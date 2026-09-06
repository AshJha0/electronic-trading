//! Execution analytics: implementation-shortfall decomposition (Perold
//! convention), participation rate, and a tiny native OLS for market-impact
//! estimation.
//!
//! Implementation-shortfall decomposition: parent order of `X` shares,
//! decision price `P_d`, arrival price `P_0`, close price `P_T`, fills
//! `(n_j, p_j)` with executed `Q = sum n_j`, remainder `R = X - Q`.  All
//! components are currency costs (positive = money lost versus the paper
//! portfolio).  For a sell (`sign = +1`; a buy mirrors with `sign = -1`):
//!
//! ```text
//! delay       = Q * sign*(P_d - P_0)         (drift before trading started)
//! trading     = sum_j n_j * sign*(P_0 - p_j) (execution slippage vs arrival)
//! opportunity = R * sign*(P_d - P_T)         (unfilled shares vs decision)
//! total       = delay + trading + opportunity
//!             = sign*(X*P_d - sum_j n_j*p_j - R*P_T)   (algebraic identity)
//! ```
//!
//! Market-impact estimation: closed-form OLS of per-share cost on
//! participation rate — `slope = Sxy/Sxx`, `intercept = mean(y) -
//! slope*mean(x)`, `R^2 = 1 - SSR/SST` (no external fitting library).

use std::str::FromStr;

use crate::error::{require, EtradeError};

/// Order side of a parent order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// Buying parent order (costs mirror the sell convention).
    Buy,
    /// Selling parent order (`sign = +1` in the decomposition).
    Sell,
}

impl FromStr for Side {
    type Err = EtradeError;

    /// Parse `"buy"` / `"sell"` (exact, lowercase — mirrors the reference).
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "buy" => Ok(Side::Buy),
            "sell" => Ok(Side::Sell),
            _ => Err(EtradeError::InvalidInput(
                "side must be 'buy' or 'sell'".to_string(),
            )),
        }
    }
}

/// Implementation-shortfall components (currency costs; positive = loss).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ISDecomposition {
    /// Total shortfall `delay + trading + opportunity`.
    pub total: f64,
    /// Cost of price drift between decision and arrival.
    pub delay: f64,
    /// Execution slippage of the fills versus the arrival price.
    pub trading: f64,
    /// Opportunity cost of the unexecuted remainder versus the decision price.
    pub opportunity: f64,
    /// Executed quantity `Q = sum n_j`.
    pub executed_qty: f64,
    /// Remaining quantity `R = X - Q`.
    pub remaining_qty: f64,
}

/// Decompose implementation shortfall into delay/trading/opportunity.
///
/// Errors on non-finite prices, negative quantities, mismatched fill arrays,
/// or over-execution (`Q > X` beyond rounding tolerance).  The identity
/// `total == delay + trading + opportunity` holds exactly (algebraic).
pub fn decompose_shortfall(
    side: Side,
    decision_price: f64,
    arrival_price: f64,
    close_price: f64,
    total_qty: f64,
    fill_qtys: &[f64],
    fill_prices: &[f64],
) -> Result<ISDecomposition, EtradeError> {
    require(decision_price.is_finite(), "decision_price must be finite")?;
    require(arrival_price.is_finite(), "arrival_price must be finite")?;
    require(close_price.is_finite(), "close_price must be finite")?;
    require(
        total_qty.is_finite() && total_qty >= 0.0,
        "total_qty must be >= 0",
    )?;
    require(
        fill_qtys.len() == fill_prices.len(),
        "fill_qtys and fill_prices must have equal length",
    )?;
    require(
        fill_qtys.iter().all(|&n| n >= 0.0),
        "fill quantities must be >= 0",
    )?;
    require(
        fill_qtys.iter().all(|n| n.is_finite()) && fill_prices.iter().all(|p| p.is_finite()),
        "fills must be finite",
    )?;
    let executed: f64 = fill_qtys.iter().sum();
    require(
        executed <= total_qty * (1.0 + 1e-12) + 1e-12,
        "executed quantity exceeds total_qty",
    )?;
    let remaining = total_qty - executed;
    let sign = match side {
        Side::Sell => 1.0,
        Side::Buy => -1.0,
    };
    let delay = executed * sign * (decision_price - arrival_price);
    let trading: f64 = fill_qtys
        .iter()
        .zip(fill_prices)
        .map(|(n, p)| n * sign * (arrival_price - p))
        .sum();
    let opportunity = remaining * sign * (decision_price - close_price);
    Ok(ISDecomposition {
        total: delay + trading + opportunity,
        delay,
        trading,
        opportunity,
        executed_qty: executed,
        remaining_qty: remaining,
    })
}

/// Executed volume as a fraction of market volume over the same window.
/// Requires `executed_volume >= 0`, `market_volume > 0` and
/// `executed_volume <= market_volume` (market volume includes own fills; a
/// rate above 1 is an input error).
pub fn participation_rate(executed_volume: f64, market_volume: f64) -> Result<f64, EtradeError> {
    require(
        executed_volume.is_finite() && executed_volume >= 0.0,
        "executed_volume must be >= 0",
    )?;
    require(
        market_volume.is_finite() && market_volume > 0.0,
        "market_volume must be > 0",
    )?;
    require(
        executed_volume <= market_volume,
        "executed volume exceeds market volume - market volume must include own fills",
    )?;
    Ok(executed_volume / market_volume)
}

/// Simple linear regression `y = intercept + slope * x`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OLSResult {
    /// Fitted slope `Sxy/Sxx`.
    pub slope: f64,
    /// Fitted intercept `mean(y) - slope*mean(x)`.
    pub intercept: f64,
    /// Coefficient of determination `1 - SSR/SST` (`1.0` when `SST = 0`).
    pub r_squared: f64,
    /// Number of observations.
    pub n_obs: usize,
}

/// Tiny native OLS: `slope = cov(x,y)/var(x)` in closed form.
///
/// Requires equal-length finite inputs, `>= 2` observations, and
/// non-degenerate `x` (positive `Sxx`).
pub fn ols(x: &[f64], y: &[f64]) -> Result<OLSResult, EtradeError> {
    require(x.len() == y.len(), "x and y must have equal length")?;
    require(x.len() >= 2, "need at least 2 observations")?;
    require(
        x.iter().all(|v| v.is_finite()) && y.iter().all(|v| v.is_finite()),
        "x and y must be finite",
    )?;
    let n = x.len() as f64;
    let xm = x.iter().sum::<f64>() / n;
    let ym = y.iter().sum::<f64>() / n;
    let sxx: f64 = x.iter().map(|v| (v - xm).powi(2)).sum();
    require(sxx > 0.0, "x is degenerate (zero variance)")?;
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - xm) * (b - ym)).sum();
    let slope = sxy / sxx;
    let intercept = ym - slope * xm;
    let ssr: f64 = x
        .iter()
        .zip(y)
        .map(|(a, b)| (b - (intercept + slope * a)).powi(2))
        .sum();
    let sst: f64 = y.iter().map(|v| (v - ym).powi(2)).sum();
    let r_squared = if sst == 0.0 { 1.0 } else { 1.0 - ssr / sst };
    Ok(OLSResult {
        slope,
        intercept,
        r_squared,
        n_obs: x.len(),
    })
}

/// Deterministic synthetic `(participation, per-share cost)` fills.
///
/// Pinned RNG-free construction, identical in every language
/// (API_SPEC.md Section 3.3):
///
/// ```text
/// part_i = 0.02 + 0.18*i/(n_obs - 1),                     i = 0..n_obs-1
/// cost_i = intercept + slope*part_i + noise_amp*sin(2.5*i + 0.3)
/// ```
pub fn synthetic_impact_fills(
    n_obs: usize,
    intercept: f64,
    slope: f64,
    noise_amp: f64,
) -> Result<(Vec<f64>, Vec<f64>), EtradeError> {
    require(n_obs >= 2, "n_obs must be >= 2")?;
    let mut part = Vec::with_capacity(n_obs);
    let mut cost = Vec::with_capacity(n_obs);
    for i in 0..n_obs {
        let fi = i as f64;
        let p = 0.02 + 0.18 * fi / (n_obs as f64 - 1.0);
        part.push(p);
        cost.push(intercept + slope * p + noise_amp * (2.5 * fi + 0.3).sin());
    }
    Ok((part, cost))
}

/// Market-impact regression: per-share cost on participation rate.
pub fn estimate_impact(participations: &[f64], costs: &[f64]) -> Result<OLSResult, EtradeError> {
    ols(participations, costs)
}
