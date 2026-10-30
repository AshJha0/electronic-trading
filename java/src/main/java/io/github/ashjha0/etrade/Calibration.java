package io.github.ashjha0.etrade;

import static io.github.ashjha0.etrade.Validation.require;

import java.util.Arrays;
import java.util.List;

/**
 * Calibration estimators: market quantities measured from recorded data (API_SPEC Section 11).
 *
 * <p>A session {@code [t0, t1]} is split into {@code n} equal bins; an observation at time
 * {@code t} with {@code t0 <= t <= t1} falls in bin
 * {@code min(floor((t - t0) * n / (t1 - t0)), n - 1)} and observations outside the session are
 * ignored. The volume curve is the fraction of session volume per bin; the volatility curve is
 * realised variance (sum of squared log mid returns, each assigned to the bin of its end
 * instant); spread statistics are the mean and median in ticks and in bps of the mid; the
 * fill intensity fits {@code lambda(delta) = A exp(-k delta)} (Avellaneda-Stoikov 2008) by
 * count-weighted least squares on {@code ln(rate_b)} at {@code delta_b = (b + 1/2) tick}.
 */
public final class Calibration {

    /** Largest number of bins accepted by the curve estimators. */
    public static final int MAX_CALIBRATION_BINS = 10_000;
    /** Largest number of distance buckets accepted by {@link #fillIntensity}. */
    public static final int MAX_FILL_LEVELS = 1_000;
    /** Slack, in ticks, when comparing a distance with a bucket edge. */
    public static final double FILL_EDGE_SLACK = 1e-9;
    /**
     * Volume curve fitted to the NASDAQ TotalView-ITCH sample day 2019-12-30: thirteen
     * 30-minute fractions of continuous-session volume, the equal-weighted mean of AAPL, MSFT
     * and QQQ, rounded to 1e-4 (API_SPEC 11.6). Unmodifiable.
     */
    public static final List<Double> FITTED_VOLUME_CURVE_13 = List.of(
            0.1987, 0.1247, 0.0742, 0.0680, 0.0589, 0.0604, 0.0414,
            0.0430, 0.0484, 0.0466, 0.0376, 0.0535, 0.1446);

    private Calibration() {
    }

    /**
     * The fitted 13-bin volume curve as a fresh array.
     *
     * @return a copy of {@link #FITTED_VOLUME_CURVE_13}
     */
    public static double[] fittedVolumeCurve13() {
        return FITTED_VOLUME_CURVE_13.stream().mapToDouble(Double::doubleValue).toArray();
    }

    /**
     * Realised volatility of the session and of each bin (fractions).
     *
     * @param dailyVol session open-to-close volatility
     * @param binVol per-bin volatility
     * @param nReturns number of returns inside the session
     */
    public record VolatilityCurve(double dailyVol, double[] binVol, int nReturns) {
    }

    /**
     * Mean and median of a spread sample, in ticks and in bps of the mid.
     *
     * @param meanTicks mean in ticks
     * @param medianTicks median in ticks
     * @param meanBps mean in basis points of the mid
     * @param medianBps median in basis points of the mid
     * @param n sample size
     */
    public record SpreadStats(double meanTicks, double medianTicks, double meanBps,
                              double medianBps, int n) {
    }

    /**
     * Fitted {@code lambda(delta) = A exp(-k delta)}, in model and market units.
     *
     * @param aRate A, fills per side per day
     * @param k decay per unit of price
     * @param touchFillsPerDay {@code lambda(tick/2)}
     * @param halvingTicks {@code ln 2 / (k tick)}
     * @param nBuckets buckets with a non-zero count used in the fit
     */
    public record FillIntensity(double aRate, double k, double touchFillsPerDay,
                                double halvingTicks, int nBuckets) {
    }

    private static void session(double start, double end, int nBins) {
        require(Double.isFinite(start), "session_start must be finite");
        require(Double.isFinite(end) && end > start,
                "session_end must be finite and > session_start");
        require(1 <= nBins && nBins <= MAX_CALIBRATION_BINS,
                "n_bins must be an integer in [1, " + MAX_CALIBRATION_BINS + "]");
    }

    private static int bin(double t, double start, double end, int nBins) {
        int j = (int) Math.floor((t - start) * nBins / (end - start));
        return Math.min(j, nBins - 1);
    }

    /**
     * Fraction of session volume per bin (API_SPEC 11.2); sums to 1.
     *
     * @param times observation times
     * @param sizes traded sizes, finite and &ge; 0
     * @param sessionStart session open
     * @param sessionEnd session close, &gt; sessionStart
     * @param nBins number of bins in [1, MAX_CALIBRATION_BINS]
     * @return the volume fractions
     */
    public static double[] volumeCurve(double[] times, double[] sizes, double sessionStart,
                                       double sessionEnd, int nBins) {
        session(sessionStart, sessionEnd, nBins);
        require(times.length == sizes.length, "times and sizes must have the same length");
        double[] vol = new double[nBins];
        for (int i = 0; i < times.length; i++) {
            double t = times[i];
            double s = sizes[i];
            require(Double.isFinite(t), "times must be finite");
            require(Double.isFinite(s) && s >= 0.0, "sizes must be finite and >= 0");
            if (sessionStart <= t && t <= sessionEnd) {
                vol[bin(t, sessionStart, sessionEnd, nBins)] += s;
            }
        }
        double total = 0.0;
        for (double v : vol) {
            total += v;
        }
        require(total > 0.0, "no volume inside the session");
        double[] out = new double[nBins];
        for (int j = 0; j < nBins; j++) {
            out[j] = vol[j] / total;
        }
        return out;
    }

    /**
     * Realised volatility per bin and for the session (API_SPEC 11.3).
     *
     * @param times observation times, non-decreasing
     * @param mids mid prices, finite and &gt; 0
     * @param sessionStart session open
     * @param sessionEnd session close, &gt; sessionStart
     * @param nBins number of bins in [1, MAX_CALIBRATION_BINS]
     * @return the volatility curve
     */
    public static VolatilityCurve volatilityCurve(double[] times, double[] mids,
                                                  double sessionStart, double sessionEnd,
                                                  int nBins) {
        session(sessionStart, sessionEnd, nBins);
        require(times.length == mids.length, "times and mids must have the same length");
        double[] rv = new double[nBins];
        int nReturns = 0;
        for (int i = 0; i < times.length; i++) {
            double t = times[i];
            double m = mids[i];
            require(Double.isFinite(t), "times must be finite");
            require(Double.isFinite(m) && m > 0.0, "mids must be finite and > 0");
            if (i == 0) {
                continue;
            }
            require(t >= times[i - 1], "times must be non-decreasing");
            if (times[i - 1] >= sessionStart && t <= sessionEnd) {
                double r = Math.log(m / mids[i - 1]);
                rv[bin(t, sessionStart, sessionEnd, nBins)] += r * r;
                nReturns++;
            }
        }
        require(nReturns > 0, "no return inside the session");
        double total = 0.0;
        for (double v : rv) {
            total += v;
        }
        double[] binVol = new double[nBins];
        for (int j = 0; j < nBins; j++) {
            binVol[j] = Math.sqrt(rv[j]);
        }
        return new VolatilityCurve(Math.sqrt(total), binVol, nReturns);
    }

    private static double median(double[] values) {
        double[] s = values.clone();
        Arrays.sort(s);
        int n = s.length;
        if (n % 2 == 1) {
            return s[n / 2];
        }
        return 0.5 * (s[n / 2 - 1] + s[n / 2]);
    }

    /**
     * Spread statistics of a sample (API_SPEC 11.4).
     *
     * @param spreads spreads in currency ({@code ask - bid} or {@code 2 |p - m|}), &ge; 0
     * @param mids mid of each observation, &gt; 0
     * @param tick tick size, &gt; 0
     * @return the statistics
     */
    public static SpreadStats spreadStats(double[] spreads, double[] mids, double tick) {
        require(Double.isFinite(tick) && tick > 0.0, "tick must be finite and > 0");
        require(spreads.length == mids.length, "spreads and mids must have the same length");
        require(spreads.length > 0, "spreads must not be empty");
        int n = spreads.length;
        double[] ticks = new double[n];
        double[] bps = new double[n];
        for (int i = 0; i < n; i++) {
            double s = spreads[i];
            double m = mids[i];
            require(Double.isFinite(s) && s >= 0.0, "spreads must be finite and >= 0");
            require(Double.isFinite(m) && m > 0.0, "mids must be finite and > 0");
            ticks[i] = s / tick;
            bps[i] = 1e4 * s / m;
        }
        double sumT = 0.0;
        double sumB = 0.0;
        for (int i = 0; i < n; i++) {
            sumT += ticks[i];
            sumB += bps[i];
        }
        return new SpreadStats(sumT / n, median(ticks), sumB / n, median(bps), n);
    }

    /**
     * Fit the exponential fill intensity over one day pooling two sides (API_SPEC 11.5).
     *
     * @param distances largest distance from the mid each market order printed at, &ge; 0
     * @param tick tick size, &gt; 0
     * @param nLevels number of distance buckets in [2, MAX_FILL_LEVELS]
     * @return the fit
     */
    public static FillIntensity fillIntensity(double[] distances, double tick, int nLevels) {
        return fillIntensity(distances, tick, nLevels, 1.0, 2.0);
    }

    /**
     * Fit the exponential fill intensity (API_SPEC 11.5).
     *
     * @param distances largest distance from the mid each market order printed at, &ge; 0
     * @param tick tick size, &gt; 0
     * @param nLevels number of distance buckets in [2, MAX_FILL_LEVELS]
     * @param days sample length in trading days, &gt; 0
     * @param sides number of quote sides pooled in the sample, &gt; 0
     * @return the fit
     */
    public static FillIntensity fillIntensity(double[] distances, double tick, int nLevels,
                                              double days, double sides) {
        require(Double.isFinite(tick) && tick > 0.0, "tick must be finite and > 0");
        require(2 <= nLevels && nLevels <= MAX_FILL_LEVELS,
                "n_levels must be an integer in [2, " + MAX_FILL_LEVELS + "]");
        require(Double.isFinite(days) && days > 0.0, "days must be finite and > 0");
        require(Double.isFinite(sides) && sides > 0.0, "sides must be finite and > 0");
        int[] counts = new int[nLevels];
        for (double d : distances) {
            require(Double.isFinite(d) && d >= 0.0, "distances must be finite and >= 0");
            double x = d / tick;
            for (int b = 0; b < nLevels; b++) {
                if (x >= b + 0.5 - FILL_EDGE_SLACK) {
                    counts[b]++;
                } else {
                    break;
                }
            }
        }
        double[] xs = new double[nLevels];
        double[] ys = new double[nLevels];
        double[] ws = new double[nLevels];
        int n = 0;
        for (int b = 0; b < nLevels; b++) {
            if (counts[b] > 0) {
                xs[n] = (b + 0.5) * tick;
                ys[n] = Math.log(counts[b] / (sides * days));
                ws[n] = counts[b];
                n++;
            }
        }
        require(n >= 2, "fewer than two distance buckets have fills");
        double wSum = 0.0;
        double xBar = 0.0;
        double yBar = 0.0;
        for (int i = 0; i < n; i++) {
            wSum += ws[i];
            xBar += ws[i] * xs[i];
            yBar += ws[i] * ys[i];
        }
        xBar /= wSum;
        yBar /= wSum;
        double sxx = 0.0;
        double sxy = 0.0;
        for (int i = 0; i < n; i++) {
            sxx += ws[i] * (xs[i] - xBar) * (xs[i] - xBar);
            sxy += ws[i] * (xs[i] - xBar) * (ys[i] - yBar);
        }
        double k = -sxy / sxx;
        require(k > 0.0, "fitted fill intensity does not decay with distance");
        double aRate = Math.exp(yBar + k * xBar);
        return new FillIntensity(aRate, k, aRate * Math.exp(-k * tick / 2.0),
                Math.log(2.0) / (k * tick), n);
    }
}
