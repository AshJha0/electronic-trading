package io.github.ashjha0.etrade;

import static io.github.ashjha0.etrade.Validation.require;

/**
 * Post-trade transaction-cost analysis (API_SPEC Section 10): benchmark slippage, markouts,
 * participation and a Welch A/B test.
 *
 * <p>Slippage is {@code 1e4 * sign * (B - avgPrice) / B} bps of each benchmark {@code B}
 * (arrival, market VWAP, interval VWAP over the bins the order traded in, close) with
 * {@code sign = +1} for a sell and {@code -1} for a buy, the Perold convention of
 * {@link Analytics}: positive = cost. Markouts are {@code d * (mid[t + h] - mid[t])} with
 * {@code d = +1} for a buy, negative = adverse. The Student-t tail uses a pinned Lanczos
 * log-gamma and the modified Lentz continued fraction for the incomplete beta (Numerical
 * Recipes, 3rd ed.), so every port runs the same algorithm.
 */
public final class Tca {

    /** Iteration cap of the incomplete-beta continued fraction. */
    public static final int MAX_BETACF_ITER = 10_000;

    private static final double BETACF_EPS = 1e-15;
    private static final double FPMIN = 1e-300;
    private static final double[] LANCZOS = {
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    };

    private Tca() {
    }

    /**
     * Average price, benchmarks and slippage in bps (positive = cost).
     *
     * @param executedQty executed quantity {@code Q}
     * @param avgPrice quantity-weighted average fill price
     * @param marketVwap full-period market VWAP
     * @param intervalVwap market VWAP over the bins in which the order traded
     * @param arrivalBps slippage against the arrival price
     * @param vwapBps slippage against the market VWAP
     * @param intervalVwapBps slippage against the interval VWAP
     * @param closeBps slippage against the close
     */
    public record SlippageReport(double executedQty, double avgPrice, double marketVwap,
                                 double intervalVwap, double arrivalBps, double vwapBps,
                                 double intervalVwapBps, double closeBps) {
    }

    /**
     * Mean markout per fill and per share at each horizon (negative = adverse).
     *
     * @param horizons horizons in steps, in input order
     * @param perFill plain mean over fills, currency per share
     * @param perShare quantity-weighted mean, currency per share
     */
    public record MarkoutResult(int[] horizons, double[] perFill, double[] perShare) {
    }

    /**
     * Order volume as a fraction of market volume.
     *
     * @param perBin {@code q_b / V_b} per bin (0 for a bin with no market volume)
     * @param overall {@code Q / sum V}
     */
    public record ParticipationResult(double[] perBin, double overall) {
    }

    /**
     * Welch two-sample t-test of {@code mean(a) - mean(b)}.
     *
     * @param meanDiff {@code mean(a) - mean(b)}
     * @param tStat Welch t statistic
     * @param df Welch-Satterthwaite degrees of freedom
     * @param pValue two-sided p-value
     */
    public record WelchResult(double meanDiff, double tStat, double df, double pValue) {
    }

    private static double sellSign(String side) {
        require("buy".equals(side) || "sell".equals(side), "side must be 'buy' or 'sell'");
        return "sell".equals(side) ? 1.0 : -1.0;
    }

    private static void checkFills(int[] bins, double[] qtys, int nBins, String name) {
        require(qtys.length > 0, "fills must be non-empty");
        require(bins.length == qtys.length, "fill arrays must have equal length");
        for (int b : bins) {
            require(b >= 0 && b < nBins, name + " must be integers in [0, " + nBins + ")");
        }
        for (double n : qtys) {
            require(Double.isFinite(n) && n >= 0.0, "fill_qtys must be finite and >= 0");
        }
    }

    private static double checkVolumes(double[] v) {
        require(v.length > 0, "bins must be non-empty");
        double total = 0.0;
        for (double x : v) {
            require(Double.isFinite(x) && x >= 0.0, "bin_volumes must be finite and >= 0");
            total += x;
        }
        require(total > 0.0, "total market volume must be > 0");
        return total;
    }

    /**
     * Slippage against arrival, VWAP, interval VWAP and close (API_SPEC 10.1). The bin
     * arrays describe the whole market per bin, including the order's own fills.
     *
     * @param side {@code "buy"} or {@code "sell"}
     * @param arrivalPrice arrival price, &gt; 0
     * @param closePrice close price, &gt; 0
     * @param fillBins bin of each fill, in [0, number of bins)
     * @param fillQtys quantity of each fill, &ge; 0
     * @param fillPrices price of each fill, &gt; 0
     * @param binVolumes market volume per bin, &ge; 0
     * @param binVwaps market VWAP per bin, &gt; 0
     * @return the slippage report
     */
    public static SlippageReport slippage(String side, double arrivalPrice, double closePrice,
                                          int[] fillBins, double[] fillQtys,
                                          double[] fillPrices, double[] binVolumes,
                                          double[] binVwaps) {
        double sign = sellSign(side);
        require(Double.isFinite(arrivalPrice) && arrivalPrice > 0.0,
                "arrival_price must be finite and > 0");
        require(Double.isFinite(closePrice) && closePrice > 0.0,
                "close_price must be finite and > 0");
        require(binVwaps.length == binVolumes.length, "bin arrays must have equal length");
        double totalV = checkVolumes(binVolumes);
        for (double p : binVwaps) {
            require(Double.isFinite(p) && p > 0.0, "bin_vwaps must be finite and > 0");
        }
        checkFills(fillBins, fillQtys, binVolumes.length, "fill_bins");
        require(fillPrices.length == fillQtys.length, "fill arrays must have equal length");
        for (double p : fillPrices) {
            require(Double.isFinite(p) && p > 0.0, "fill_prices must be finite and > 0");
        }

        double q = 0.0;
        double paid = 0.0;
        boolean[] traded = new boolean[binVolumes.length];
        for (int j = 0; j < fillQtys.length; j++) {
            q += fillQtys[j];
            paid += fillQtys[j] * fillPrices[j];
            if (fillQtys[j] > 0.0) {
                traded[fillBins[j]] = true;
            }
        }
        require(q > 0.0, "executed quantity must be > 0");
        double avg = paid / q;

        double vp = 0.0;
        double iv = 0.0;
        double ivp = 0.0;
        for (int b = 0; b < binVolumes.length; b++) {
            vp += binVolumes[b] * binVwaps[b];
            if (traded[b]) {
                iv += binVolumes[b];
                ivp += binVolumes[b] * binVwaps[b];
            }
        }
        require(iv > 0.0, "market volume in traded bins must be > 0");
        double vwap = vp / totalV;
        double ivwap = ivp / iv;
        return new SlippageReport(q, avg, vwap, ivwap, cost(sign, arrivalPrice, avg),
                cost(sign, vwap, avg), cost(sign, ivwap, avg), cost(sign, closePrice, avg));
    }

    private static double cost(double sign, double bench, double avg) {
        return 1e4 * sign * (bench - avg) / bench;
    }

    /**
     * Markouts {@code d * (mids[t_j + h] - mids[t_j])}, {@code d = +1} buy, {@code -1} sell
     * (API_SPEC 10.2).
     *
     * @param side {@code "buy"} or {@code "sell"}
     * @param fillSteps step of each fill in the mid series
     * @param fillQtys quantity of each fill, &ge; 0
     * @param mids mid series indexed by step
     * @param horizons horizons in steps, &ge; 0
     * @return mean markouts per fill and per share
     */
    public static MarkoutResult markouts(String side, int[] fillSteps, double[] fillQtys,
                                         double[] mids, int[] horizons) {
        double d = -sellSign(side);
        require(mids.length > 0, "mids must be non-empty");
        for (double m : mids) {
            require(Double.isFinite(m), "mids must be finite");
        }
        checkFills(fillSteps, fillQtys, mids.length, "fill_steps");
        require(horizons.length > 0, "horizons must be non-empty");
        for (int h : horizons) {
            require(h >= 0, "horizons must be integers >= 0");
            for (int t : fillSteps) {
                require((long) t + h < mids.length, "fill step + horizon exceeds mid series");
            }
        }
        double q = 0.0;
        for (double n : fillQtys) {
            q += n;
        }
        require(q > 0.0, "executed quantity must be > 0");
        double[] perFill = new double[horizons.length];
        double[] perShare = new double[horizons.length];
        for (int k = 0; k < horizons.length; k++) {
            int h = horizons[k];
            double s = 0.0;
            double w = 0.0;
            for (int j = 0; j < fillSteps.length; j++) {
                int t = fillSteps[j];
                double mk = d * (mids[t + h] - mids[t]);
                s += mk;
                w += fillQtys[j] * mk;
            }
            perFill[k] = s / fillSteps.length;
            perShare[k] = w / q;
        }
        return new MarkoutResult(horizons.clone(), perFill, perShare);
    }

    /**
     * Per-bin {@code q_b / V_b} (0 for an empty bin) and overall {@code Q / sum V}
     * (API_SPEC 10.3); {@code q_b <= V_b} is required.
     *
     * @param fillBins bin of each fill
     * @param fillQtys quantity of each fill, &ge; 0
     * @param binVolumes market volume per bin, &ge; 0
     * @return participation per bin and overall
     */
    public static ParticipationResult participation(int[] fillBins, double[] fillQtys,
                                                    double[] binVolumes) {
        double totalV = checkVolumes(binVolumes);
        checkFills(fillBins, fillQtys, binVolumes.length, "fill_bins");
        double[] qb = new double[binVolumes.length];
        double q = 0.0;
        for (int j = 0; j < fillQtys.length; j++) {
            qb[fillBins[j]] += fillQtys[j];
            q += fillQtys[j];
        }
        double[] rates = new double[binVolumes.length];
        for (int b = 0; b < binVolumes.length; b++) {
            require(qb[b] <= binVolumes[b], "fill volume exceeds market volume in a bin");
            rates[b] = binVolumes[b] > 0.0 ? qb[b] / binVolumes[b] : 0.0;
        }
        return new ParticipationResult(rates, q / totalV);
    }

    /**
     * Natural log of the gamma function, pinned Lanczos ({@code g = 7}, nine terms).
     *
     * @param x argument, &gt; 0
     * @return {@code ln Gamma(x)}
     */
    public static double lnGamma(double x) {
        require(Double.isFinite(x) && x > 0.0, "x must be finite and > 0");
        if (x < 0.5) {
            return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lnGamma(1.0 - x);
        }
        double z = x - 1.0;
        double s = LANCZOS[0];
        for (int i = 1; i < 9; i++) {
            s += LANCZOS[i] / (z + i);
        }
        double t = z + 7.5;
        return 0.5 * Math.log(2.0 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(s);
    }

    private static double fix(double v) {
        return Math.abs(v) < FPMIN ? FPMIN : v;
    }

    private static double betacf(double a, double b, double x) {
        double qab = a + b;
        double qap = a + 1.0;
        double qam = a - 1.0;
        double c = 1.0;
        double d = 1.0 / fix(1.0 - qab * x / qap);
        double h = d;
        for (int m = 1; m <= MAX_BETACF_ITER; m++) {
            double m2 = 2.0 * m;
            double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 / fix(1.0 + aa * d);
            c = fix(1.0 + aa / c);
            h *= d * c;
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 / fix(1.0 + aa * d);
            c = fix(1.0 + aa / c);
            double del = d * c;
            h *= del;
            if (Math.abs(del - 1.0) <= BETACF_EPS) {
                return h;
            }
        }
        throw new IllegalStateException("incomplete beta continued fraction did not converge");
    }

    /**
     * Regularized incomplete beta {@code I_x(a, b)} (API_SPEC 10.4).
     *
     * @param a first shape, &gt; 0
     * @param b second shape, &gt; 0
     * @param x argument in [0, 1]
     * @return {@code I_x(a, b)}
     * @throws IllegalStateException if the continued fraction does not converge
     */
    public static double betaincReg(double a, double b, double x) {
        require(Double.isFinite(a) && a > 0.0, "a must be finite and > 0");
        require(Double.isFinite(b) && b > 0.0, "b must be finite and > 0");
        require(Double.isFinite(x) && x >= 0.0 && x <= 1.0, "x must be in [0, 1]");
        if (x == 0.0) {
            return 0.0;
        }
        if (x == 1.0) {
            return 1.0;
        }
        double lnBt = lnGamma(a + b) - lnGamma(a) - lnGamma(b) + a * Math.log(x)
                + b * Math.log1p(-x);
        double bt = Math.exp(lnBt);
        if (x < (a + 1.0) / (a + b + 2.0)) {
            return bt * betacf(a, b, x) / a;
        }
        return 1.0 - bt * betacf(b, a, 1.0 - x) / b;
    }

    /**
     * Student-t CDF with {@code df > 0} degrees of freedom (non-integer allowed).
     *
     * @param t argument, finite
     * @param df degrees of freedom, &gt; 0
     * @return {@code P(T <= t)}
     */
    public static double studentTCdf(double t, double df) {
        require(Double.isFinite(t), "t must be finite");
        require(Double.isFinite(df) && df > 0.0, "df must be finite and > 0");
        double tail = 0.5 * betaincReg(0.5 * df, 0.5, df / (df + t * t));
        return t >= 0.0 ? 1.0 - tail : tail;
    }

    private static double[] meanVar(double[] xs, String name) {
        require(xs.length >= 2, name + " must have at least 2 values");
        double s = 0.0;
        for (double v : xs) {
            require(Double.isFinite(v), name + " must be finite");
            s += v;
        }
        double mean = s / xs.length;
        double ss = 0.0;
        for (double v : xs) {
            ss += (v - mean) * (v - mean);
        }
        return new double[] {mean, ss / (xs.length - 1)};
    }

    /**
     * Welch's t-test on per-order costs of strategies A and B (API_SPEC 10.4).
     *
     * @param a costs of strategy A (at least 2, finite)
     * @param b costs of strategy B (at least 2, finite)
     * @return difference of means, t statistic, degrees of freedom and two-sided p-value
     */
    public static WelchResult welchTTest(double[] a, double[] b) {
        double[] sa = meanVar(a, "a");
        double[] sb = meanVar(b, "b");
        double na = a.length;
        double nb = b.length;
        double ea = sa[1] / na;
        double eb = sb[1] / nb;
        double se2 = ea + eb;
        require(se2 > 0.0, "samples have zero variance");
        double diff = sa[0] - sb[0];
        double t = diff / Math.sqrt(se2);
        double df = se2 * se2 / (ea * ea / (na - 1.0) + eb * eb / (nb - 1.0));
        return new WelchResult(diff, t, df, betaincReg(0.5 * df, 0.5, df / (df + t * t)));
    }
}
