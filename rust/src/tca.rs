//! Post-trade transaction-cost analysis (API_SPEC.md Section 10): benchmark
//! slippage, markouts, participation and a Welch A/B test.
//!
//! Slippage is `1e4 * sign * (B - avg_price) / B` bps of each benchmark `B`
//! (arrival, market VWAP, interval VWAP over the bins the order traded in,
//! close) with `sign = +1` for a sell and `-1` for a buy, the Perold
//! convention of [`crate::analytics`]: positive = cost.  Markouts are
//! `d * (mid[t + h] - mid[t])` with `d = +1` for a buy, negative = adverse.
//! The Student-t tail uses a pinned Lanczos log-gamma and the modified Lentz
//! continued fraction for the incomplete beta (Numerical Recipes, 3rd ed.),
//! so every port runs the same algorithm.
//!
//! ```
//! use etrade::tca::slippage;
//! use etrade::Side;
//! // A buy filled at 100.10 against arrival 100.00 costs 10 bps.
//! let r = slippage(Side::Buy, 100.0, 100.0, &[0], &[10.0], &[100.10], &[100.0], &[100.0]).unwrap();
//! assert!((r.arrival_bps - 10.0).abs() < 1e-9);
//! ```

use crate::error::{require, EtradeError};
use crate::Side;

/// Iteration cap of the incomplete-beta continued fraction.
pub const MAX_BETACF_ITER: usize = 10_000;

const BETACF_EPS: f64 = 1e-15;
const FPMIN: f64 = 1e-300;
#[allow(clippy::excessive_precision)]
const LANCZOS: [f64; 9] = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
];

fn sell_sign(side: Side) -> f64 {
    match side {
        Side::Sell => 1.0,
        Side::Buy => -1.0,
    }
}

fn check_fills(bins: &[usize], qtys: &[f64], n_bins: usize, name: &str) -> Result<(), EtradeError> {
    require(!qtys.is_empty(), "fills must be non-empty")?;
    require(
        bins.len() == qtys.len(),
        "fill arrays must have equal length",
    )?;
    require(
        bins.iter().all(|&b| b < n_bins),
        &format!("{name} must be integers in [0, {n_bins})"),
    )?;
    require(
        qtys.iter().all(|n| n.is_finite() && *n >= 0.0),
        "fill_qtys must be finite and >= 0",
    )
}

fn check_volumes(v: &[f64]) -> Result<f64, EtradeError> {
    require(!v.is_empty(), "bins must be non-empty")?;
    let mut total = 0.0;
    for &x in v {
        require(
            x.is_finite() && x >= 0.0,
            "bin_volumes must be finite and >= 0",
        )?;
        total += x;
    }
    require(total > 0.0, "total market volume must be > 0")?;
    Ok(total)
}

/// Average price, benchmarks and slippage in bps (positive = cost).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlippageReport {
    /// Executed quantity `Q = sum n_j`.
    pub executed_qty: f64,
    /// Quantity-weighted average fill price.
    pub avg_price: f64,
    /// Full-period market VWAP.
    pub market_vwap: f64,
    /// Market VWAP over the bins in which the order traded.
    pub interval_vwap: f64,
    /// Slippage against the arrival price, bps.
    pub arrival_bps: f64,
    /// Slippage against the market VWAP, bps.
    pub vwap_bps: f64,
    /// Slippage against the interval VWAP, bps.
    pub interval_vwap_bps: f64,
    /// Slippage against the close, bps.
    pub close_bps: f64,
}

/// Slippage against arrival, VWAP, interval VWAP and close (API_SPEC 10.1).
/// `bin_volumes`/`bin_vwaps` describe the whole market per bin, including
/// the order's own fills.
#[allow(clippy::too_many_arguments)]
pub fn slippage(
    side: Side,
    arrival_price: f64,
    close_price: f64,
    fill_bins: &[usize],
    fill_qtys: &[f64],
    fill_prices: &[f64],
    bin_volumes: &[f64],
    bin_vwaps: &[f64],
) -> Result<SlippageReport, EtradeError> {
    let sign = sell_sign(side);
    require(
        arrival_price.is_finite() && arrival_price > 0.0,
        "arrival_price must be finite and > 0",
    )?;
    require(
        close_price.is_finite() && close_price > 0.0,
        "close_price must be finite and > 0",
    )?;
    require(
        bin_vwaps.len() == bin_volumes.len(),
        "bin arrays must have equal length",
    )?;
    let total_v = check_volumes(bin_volumes)?;
    require(
        bin_vwaps.iter().all(|p| p.is_finite() && *p > 0.0),
        "bin_vwaps must be finite and > 0",
    )?;
    check_fills(fill_bins, fill_qtys, bin_volumes.len(), "fill_bins")?;
    require(
        fill_prices.len() == fill_qtys.len(),
        "fill arrays must have equal length",
    )?;
    require(
        fill_prices.iter().all(|p| p.is_finite() && *p > 0.0),
        "fill_prices must be finite and > 0",
    )?;

    let mut q = 0.0;
    let mut paid = 0.0;
    let mut traded = vec![false; bin_volumes.len()];
    for j in 0..fill_qtys.len() {
        q += fill_qtys[j];
        paid += fill_qtys[j] * fill_prices[j];
        if fill_qtys[j] > 0.0 {
            traded[fill_bins[j]] = true;
        }
    }
    require(q > 0.0, "executed quantity must be > 0")?;
    let avg = paid / q;

    let mut vp = 0.0;
    let mut iv = 0.0;
    let mut ivp = 0.0;
    for b in 0..bin_volumes.len() {
        vp += bin_volumes[b] * bin_vwaps[b];
        if traded[b] {
            iv += bin_volumes[b];
            ivp += bin_volumes[b] * bin_vwaps[b];
        }
    }
    require(iv > 0.0, "market volume in traded bins must be > 0")?;
    let vwap = vp / total_v;
    let ivwap = ivp / iv;
    let cost = |bench: f64| 1e4 * sign * (bench - avg) / bench;
    Ok(SlippageReport {
        executed_qty: q,
        avg_price: avg,
        market_vwap: vwap,
        interval_vwap: ivwap,
        arrival_bps: cost(arrival_price),
        vwap_bps: cost(vwap),
        interval_vwap_bps: cost(ivwap),
        close_bps: cost(close_price),
    })
}

/// Mean markout per fill and per share at each horizon (negative = adverse).
#[derive(Debug, Clone, PartialEq)]
pub struct MarkoutResult {
    /// Horizons in steps, in input order.
    horizons: Vec<usize>,
    /// Plain mean over fills, currency per share.
    per_fill: Vec<f64>,
    /// Quantity-weighted mean, currency per share.
    per_share: Vec<f64>,
}

impl MarkoutResult {
    /// Horizons in steps, in input order.
    pub fn horizons(&self) -> &[usize] {
        &self.horizons
    }
    /// Plain mean over fills, currency per share.
    pub fn per_fill(&self) -> &[f64] {
        &self.per_fill
    }
    /// Quantity-weighted mean, currency per share.
    pub fn per_share(&self) -> &[f64] {
        &self.per_share
    }
}

/// Markouts `d * (mids[t_j + h] - mids[t_j])`, `d = +1` buy, `-1` sell
/// (API_SPEC 10.2).
pub fn markouts(
    side: Side,
    fill_steps: &[usize],
    fill_qtys: &[f64],
    mids: &[f64],
    horizons: &[usize],
) -> Result<MarkoutResult, EtradeError> {
    let d = -sell_sign(side);
    require(!mids.is_empty(), "mids must be non-empty")?;
    require(mids.iter().all(|m| m.is_finite()), "mids must be finite")?;
    check_fills(fill_steps, fill_qtys, mids.len(), "fill_steps")?;
    require(!horizons.is_empty(), "horizons must be non-empty")?;
    for &h in horizons {
        for &t in fill_steps {
            require(h < mids.len() - t, "fill step + horizon exceeds mid series")?;
        }
    }
    let mut q = 0.0;
    for &n in fill_qtys {
        q += n;
    }
    require(q > 0.0, "executed quantity must be > 0")?;
    let mut per_fill = Vec::with_capacity(horizons.len());
    let mut per_share = Vec::with_capacity(horizons.len());
    for &h in horizons {
        let mut s = 0.0;
        let mut w = 0.0;
        for (&t, &n) in fill_steps.iter().zip(fill_qtys) {
            let mk = d * (mids[t + h] - mids[t]);
            s += mk;
            w += n * mk;
        }
        per_fill.push(s / fill_steps.len() as f64);
        per_share.push(w / q);
    }
    Ok(MarkoutResult {
        horizons: horizons.to_vec(),
        per_fill,
        per_share,
    })
}

/// Order volume as a fraction of market volume, per bin and overall.
#[derive(Debug, Clone, PartialEq)]
pub struct ParticipationResult {
    /// `q_b / V_b` per bin (`0` for a bin with no market volume).
    per_bin: Vec<f64>,
    /// `Q / sum V`.
    pub overall: f64,
}

impl ParticipationResult {
    /// `q_b / V_b` per bin (`0` for a bin with no market volume).
    pub fn per_bin(&self) -> &[f64] {
        &self.per_bin
    }
}

/// Per-bin and overall participation (API_SPEC 10.3); `q_b <= V_b` required.
pub fn participation(
    fill_bins: &[usize],
    fill_qtys: &[f64],
    bin_volumes: &[f64],
) -> Result<ParticipationResult, EtradeError> {
    let total_v = check_volumes(bin_volumes)?;
    check_fills(fill_bins, fill_qtys, bin_volumes.len(), "fill_bins")?;
    let mut qb = vec![0.0; bin_volumes.len()];
    let mut q = 0.0;
    for (&b, &n) in fill_bins.iter().zip(fill_qtys) {
        qb[b] += n;
        q += n;
    }
    let mut per_bin = Vec::with_capacity(bin_volumes.len());
    for (&qv, &v) in qb.iter().zip(bin_volumes) {
        require(qv <= v, "fill volume exceeds market volume in a bin")?;
        per_bin.push(if v > 0.0 { qv / v } else { 0.0 });
    }
    Ok(ParticipationResult {
        per_bin,
        overall: q / total_v,
    })
}

/// Natural log of the gamma function, pinned Lanczos (`g = 7`, nine terms).
pub fn ln_gamma(x: f64) -> Result<f64, EtradeError> {
    require(x.is_finite() && x > 0.0, "x must be finite and > 0")?;
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return Ok((pi / (pi * x).sin().abs()).ln() - ln_gamma(1.0 - x)?);
    }
    let z = x - 1.0;
    let mut s = LANCZOS[0];
    for (i, c) in LANCZOS.iter().enumerate().skip(1) {
        s += c / (z + i as f64);
    }
    let t = z + 7.5;
    Ok(0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + s.ln())
}

fn fix(v: f64) -> f64 {
    if v.abs() < FPMIN {
        FPMIN
    } else {
        v
    }
}

fn betacf(a: f64, b: f64, x: f64) -> Result<f64, EtradeError> {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 / fix(1.0 - qab * x / qap);
    let mut h = d;
    for m in 1..=MAX_BETACF_ITER {
        let md = m as f64;
        let m2 = 2.0 * md;
        let mut aa = md * (b - md) * x / ((qam + m2) * (a + m2));
        d = 1.0 / fix(1.0 + aa * d);
        c = fix(1.0 + aa / c);
        h *= d * c;
        aa = -(a + md) * (qab + md) * x / ((a + m2) * (qap + m2));
        d = 1.0 / fix(1.0 + aa * d);
        c = fix(1.0 + aa / c);
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() <= BETACF_EPS {
            return Ok(h);
        }
    }
    Err(EtradeError::SolverFailed(
        "incomplete beta continued fraction did not converge".to_string(),
    ))
}

/// Regularized incomplete beta `I_x(a, b)` (API_SPEC 10.4).
pub fn betainc_reg(a: f64, b: f64, x: f64) -> Result<f64, EtradeError> {
    require(a.is_finite() && a > 0.0, "a must be finite and > 0")?;
    require(b.is_finite() && b > 0.0, "b must be finite and > 0")?;
    require(
        x.is_finite() && (0.0..=1.0).contains(&x),
        "x must be in [0, 1]",
    )?;
    if x == 0.0 {
        return Ok(0.0);
    }
    if x == 1.0 {
        return Ok(1.0);
    }
    let ln_bt = ln_gamma(a + b)? - ln_gamma(a)? - ln_gamma(b)? + a * x.ln() + b * (-x).ln_1p();
    let bt = ln_bt.exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        Ok(bt * betacf(a, b, x)? / a)
    } else {
        Ok(1.0 - bt * betacf(b, a, 1.0 - x)? / b)
    }
}

/// Student-t CDF with `df > 0` degrees of freedom (non-integer allowed).
pub fn student_t_cdf(t: f64, df: f64) -> Result<f64, EtradeError> {
    require(t.is_finite(), "t must be finite")?;
    require(df.is_finite() && df > 0.0, "df must be finite and > 0")?;
    let tail = 0.5 * betainc_reg(0.5 * df, 0.5, df / (df + t * t))?;
    Ok(if t >= 0.0 { 1.0 - tail } else { tail })
}

/// Welch two-sample t-test of `mean(a) - mean(b)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WelchResult {
    /// `mean(a) - mean(b)`.
    pub mean_diff: f64,
    /// Welch t statistic.
    pub t_stat: f64,
    /// Welch-Satterthwaite degrees of freedom.
    pub df: f64,
    /// Two-sided p-value.
    pub p_value: f64,
}

fn mean_var(xs: &[f64], name: &str) -> Result<(f64, f64), EtradeError> {
    require(
        xs.len() >= 2,
        &format!("{name} must have at least 2 values"),
    )?;
    let mut s = 0.0;
    for &v in xs {
        require(v.is_finite(), &format!("{name} must be finite"))?;
        s += v;
    }
    let n = xs.len() as f64;
    let mean = s / n;
    let mut ss = 0.0;
    for &v in xs {
        ss += (v - mean) * (v - mean);
    }
    Ok((mean, ss / (n - 1.0)))
}

/// Welch's t-test on per-order costs of strategies A and B (API_SPEC 10.4).
pub fn welch_ttest(a: &[f64], b: &[f64]) -> Result<WelchResult, EtradeError> {
    let (ma, va) = mean_var(a, "a")?;
    let (mb, vb) = mean_var(b, "b")?;
    let na = a.len() as f64;
    let nb = b.len() as f64;
    let ea = va / na;
    let eb = vb / nb;
    let se2 = ea + eb;
    require(se2 > 0.0, "samples have zero variance")?;
    let mean_diff = ma - mb;
    let t_stat = mean_diff / se2.sqrt();
    let df = se2 * se2 / (ea * ea / (na - 1.0) + eb * eb / (nb - 1.0));
    let p_value = betainc_reg(0.5 * df, 0.5, df / (df + t_stat * t_stat))?;
    Ok(WelchResult {
        mean_diff,
        t_stat,
        df,
        p_value,
    })
}
