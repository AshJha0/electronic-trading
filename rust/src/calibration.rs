//! Calibration estimators: market quantities measured from recorded data
//! (API_SPEC.md Section 11).
//!
//! The market-unit layer (Section 7) and the power-law impact model
//! (Section 8) take inputs a desk quotes: daily volatility, quoted spread,
//! ADV, an intraday volume curve, the rate at which a resting quote is
//! filled.  This module measures those quantities from plain slices of
//! timestamps, prices and sizes, so it needs no data access and every port
//! computes the same numbers.
//!
//! Binning rule (shared by the curves): a session `[t0, t1]` is split into
//! `n` equal bins; an observation at `t0 <= t <= t1` falls in bin
//! `j = min(floor((t - t0) * n / (t1 - t0)), n - 1)`; observations outside
//! the session are ignored.  The closing instant belongs to the last bin.
//!
//! * [`volume_curve`] — fraction of session volume per bin, `V_j / sum(V)`;
//! * [`volatility_curve`] — realised variance (Andersen, Bollerslev, Diebold
//!   and Labys 2001) of the mid per bin and for the session;
//! * [`spread_stats`] — mean and median of a spread sample in ticks and bps;
//! * [`fill_intensity`] — weighted least-squares fit of the Avellaneda and
//!   Stoikov (2008) intensity `lambda(delta) = A exp(-k delta)`, reported in
//!   model units and in the Section 7.2 market units.
//!
//! ```
//! use etrade::calibration::volume_curve;
//! let w = volume_curve(&[1.0, 2.0, 3.0, 7.0], &[10.0, 20.0, 30.0, 40.0], 0.0, 8.0, 2).unwrap();
//! assert!((w[0] - 0.6).abs() < 1e-15 && (w[1] - 0.4).abs() < 1e-15);
//! ```

use crate::error::{require, EtradeError};

/// Largest number of bins accepted by the curve estimators.
pub const MAX_CALIBRATION_BINS: usize = 10_000;

/// Largest number of distance buckets accepted by [`fill_intensity`].
pub const MAX_FILL_LEVELS: usize = 1_000;

/// Slack, in ticks, when comparing a distance with a bucket edge, so that a
/// touch print at exactly half a tick is not lost to rounding.
pub const FILL_EDGE_SLACK: f64 = 1e-9;

/// Volume curve fitted to the NASDAQ TotalView-ITCH sample day 2019-12-30:
/// thirteen 30-minute fractions of continuous-session NASDAQ volume, the
/// equal-weighted mean of AAPL, MSFT and QQQ, rounded to 1e-4 (API_SPEC 11.6).
pub const FITTED_VOLUME_CURVE_13: [f64; 13] = [
    0.1987, 0.1247, 0.0742, 0.0680, 0.0589, 0.0604, 0.0414, 0.0430, 0.0484, 0.0466, 0.0376, 0.0535,
    0.1446,
];

fn session(session_start: f64, session_end: f64, n_bins: usize) -> Result<(), EtradeError> {
    require(session_start.is_finite(), "session_start must be finite")?;
    require(
        session_end.is_finite() && session_end > session_start,
        "session_end must be finite and > session_start",
    )?;
    require(
        (1..=MAX_CALIBRATION_BINS).contains(&n_bins),
        &format!("n_bins must be an integer in [1, {MAX_CALIBRATION_BINS}]"),
    )
}

fn bin_of(t: f64, session_start: f64, session_end: f64, n_bins: usize) -> usize {
    let j = ((t - session_start) * n_bins as f64 / (session_end - session_start)).floor();
    // t >= session_start, so j >= 0; the cast saturates and is then clamped.
    (j as usize).min(n_bins - 1)
}

/// Fraction of session volume per bin (API_SPEC 11.2); sums to 1.
pub fn volume_curve(
    times: &[f64],
    sizes: &[f64],
    session_start: f64,
    session_end: f64,
    n_bins: usize,
) -> Result<Vec<f64>, EtradeError> {
    session(session_start, session_end, n_bins)?;
    require(
        times.len() == sizes.len(),
        "times and sizes must have the same length",
    )?;
    let mut vol = vec![0.0; n_bins];
    for (&t, &s) in times.iter().zip(sizes) {
        require(t.is_finite(), "times must be finite")?;
        require(s.is_finite() && s >= 0.0, "sizes must be finite and >= 0")?;
        if session_start <= t && t <= session_end {
            vol[bin_of(t, session_start, session_end, n_bins)] += s;
        }
    }
    let mut total = 0.0;
    for &v in &vol {
        total += v;
    }
    require(total > 0.0, "no volume inside the session")?;
    Ok(vol.iter().map(|v| v / total).collect())
}

/// Realised volatility of the session and of each bin (fractions).
#[derive(Debug, Clone, PartialEq)]
pub struct VolatilityCurve {
    /// `sqrt(sum_j RV_j)`: open-to-close volatility of the session.
    pub daily_vol: f64,
    /// `sqrt(RV_j)` per bin.
    bin_vol: Vec<f64>,
    /// Number of returns inside the session.
    pub n_returns: usize,
}

impl VolatilityCurve {
    /// `sqrt(RV_j)` per bin.
    pub fn bin_vol(&self) -> &[f64] {
        &self.bin_vol
    }
}

/// Realised volatility per bin and for the session (API_SPEC 11.3).
///
/// A return from `t_{i-1}` to `t_i` counts when both instants are inside
/// the session and is assigned to the bin of `t_i`.
pub fn volatility_curve(
    times: &[f64],
    mids: &[f64],
    session_start: f64,
    session_end: f64,
    n_bins: usize,
) -> Result<VolatilityCurve, EtradeError> {
    session(session_start, session_end, n_bins)?;
    require(
        times.len() == mids.len(),
        "times and mids must have the same length",
    )?;
    let mut rv = vec![0.0; n_bins];
    let mut n_returns = 0usize;
    for (i, (&t, &m)) in times.iter().zip(mids).enumerate() {
        require(t.is_finite(), "times must be finite")?;
        require(m.is_finite() && m > 0.0, "mids must be finite and > 0")?;
        if i == 0 {
            continue;
        }
        require(t >= times[i - 1], "times must be non-decreasing")?;
        if times[i - 1] >= session_start && t <= session_end {
            let r = (m / mids[i - 1]).ln();
            rv[bin_of(t, session_start, session_end, n_bins)] += r * r;
            n_returns += 1;
        }
    }
    require(n_returns > 0, "no return inside the session")?;
    let mut total = 0.0;
    for &v in &rv {
        total += v;
    }
    Ok(VolatilityCurve {
        daily_vol: total.sqrt(),
        bin_vol: rv.iter().map(|v| v.sqrt()).collect(),
        n_returns,
    })
}

/// Mean and median of a spread sample, in ticks and in bps of the mid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpreadStats {
    /// Mean spread in ticks.
    pub mean_ticks: f64,
    /// Median spread in ticks.
    pub median_ticks: f64,
    /// Mean spread in basis points of the mid.
    pub mean_bps: f64,
    /// Median spread in basis points of the mid.
    pub median_bps: f64,
    /// Sample size.
    pub n: usize,
}

fn median(values: &[f64]) -> f64 {
    let mut s = values.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

/// Spread statistics of a sample (API_SPEC 11.4).
///
/// `spreads` in currency (`ask - bid` for the quoted spread, `2 |p - m|`
/// for the effective spread), `mids` the mid of each observation.
pub fn spread_stats(spreads: &[f64], mids: &[f64], tick: f64) -> Result<SpreadStats, EtradeError> {
    require(
        tick.is_finite() && tick > 0.0,
        "tick must be finite and > 0",
    )?;
    require(
        spreads.len() == mids.len(),
        "spreads and mids must have the same length",
    )?;
    require(!spreads.is_empty(), "spreads must not be empty")?;
    let mut ticks = Vec::with_capacity(spreads.len());
    let mut bps = Vec::with_capacity(spreads.len());
    for (&s, &m) in spreads.iter().zip(mids) {
        require(s.is_finite() && s >= 0.0, "spreads must be finite and >= 0")?;
        require(m.is_finite() && m > 0.0, "mids must be finite and > 0")?;
        ticks.push(s / tick);
        bps.push(1e4 * s / m);
    }
    let mut sum_t = 0.0;
    let mut sum_b = 0.0;
    for (&a, &b) in ticks.iter().zip(&bps) {
        sum_t += a;
        sum_b += b;
    }
    let n = ticks.len();
    Ok(SpreadStats {
        mean_ticks: sum_t / n as f64,
        median_ticks: median(&ticks),
        mean_bps: sum_b / n as f64,
        median_bps: median(&bps),
        n,
    })
}

/// Fitted `lambda(delta) = A exp(-k delta)`, in model and market units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FillIntensity {
    /// `A`, fills per day at zero distance.
    pub a_rate: f64,
    /// `k`, decay per unit of price.
    pub k: f64,
    /// `lambda(tick/2)`: fill rate of a quote at the touch, per day.
    pub touch_fills_per_day: f64,
    /// `ln 2 / (k tick)`: ticks over which the rate halves.
    pub halving_ticks: f64,
    /// Number of buckets with a non-zero count used in the fit.
    pub n_buckets: usize,
}

/// Fit the exponential fill intensity (API_SPEC 11.5).
///
/// `distances` holds, for each market order, the largest distance from the
/// prevailing mid at which it printed (currency, >= 0); `days` is the length
/// of the sample in trading days and `sides` the number of quote sides the
/// sample pools (2 for buys and sells together; Python defaults 1 and 2).
/// The rate of orders reaching `delta_b = (b + 1/2) tick` is fitted by
/// weighted least squares of `ln(rate_b)` on `delta_b`, weight `c_b`, over
/// the buckets with a non-zero count.  The market units invert through
/// [`crate::market::as_params_from_market`].
pub fn fill_intensity(
    distances: &[f64],
    tick: f64,
    n_levels: usize,
    days: f64,
    sides: f64,
) -> Result<FillIntensity, EtradeError> {
    require(
        tick.is_finite() && tick > 0.0,
        "tick must be finite and > 0",
    )?;
    require(
        (2..=MAX_FILL_LEVELS).contains(&n_levels),
        &format!("n_levels must be an integer in [2, {MAX_FILL_LEVELS}]"),
    )?;
    require(
        days.is_finite() && days > 0.0,
        "days must be finite and > 0",
    )?;
    require(
        sides.is_finite() && sides > 0.0,
        "sides must be finite and > 0",
    )?;
    let mut counts = vec![0u64; n_levels];
    for &d in distances {
        require(
            d.is_finite() && d >= 0.0,
            "distances must be finite and >= 0",
        )?;
        let x = d / tick;
        for (b, c) in counts.iter_mut().enumerate() {
            if x >= b as f64 + 0.5 - FILL_EDGE_SLACK {
                *c += 1;
            } else {
                break;
            }
        }
    }
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut ws = Vec::new();
    for (b, &c) in counts.iter().enumerate() {
        if c > 0 {
            xs.push((b as f64 + 0.5) * tick);
            ys.push((c as f64 / (sides * days)).ln());
            ws.push(c as f64);
        }
    }
    let n = xs.len();
    require(n >= 2, "fewer than two distance buckets have fills")?;
    let mut w_sum = 0.0;
    let mut x_bar = 0.0;
    let mut y_bar = 0.0;
    for i in 0..n {
        w_sum += ws[i];
        x_bar += ws[i] * xs[i];
        y_bar += ws[i] * ys[i];
    }
    x_bar /= w_sum;
    y_bar /= w_sum;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for i in 0..n {
        sxx += ws[i] * (xs[i] - x_bar) * (xs[i] - x_bar);
        sxy += ws[i] * (xs[i] - x_bar) * (ys[i] - y_bar);
    }
    let k = -sxy / sxx;
    require(
        k > 0.0,
        "fitted fill intensity does not decay with distance",
    )?;
    let a_rate = (y_bar + k * x_bar).exp();
    Ok(FillIntensity {
        a_rate,
        k,
        touch_fills_per_day: a_rate * (-k * tick / 2.0).exp(),
        halving_ticks: 2.0_f64.ln() / (k * tick),
        n_buckets: n,
    })
}
