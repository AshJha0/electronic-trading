package io.github.ashjha0.etrade;

/**
 * Per-path arrays and aggregate statistics of a market-making simulation.
 *
 * <p>Per-path arrays (length {@code nPaths}): terminal PnL, the per-path time
 * average of {@code |q|}, the per-path maximum {@code |q|}, the per-path
 * {@code sqrt(time average of q^2)} (the "inventory std" measure used in the
 * fixed-spread comparison), and the total number of fills, both sides counted
 * (a step where the bid and the ask both fill contributes two).
 * Aggregates are means/stds over paths; standard errors: SE of a
 * per-path-mean statistic = sample std of the per-path values / sqrt(M);
 * {@code SE(stdPnl) = stdPnl/sqrt(2(M-1))} (normal approximation).
 */
public final class ASSimResult {

    /** Aggregate statistics with a standard error (see {@link #se(Stat)}). */
    public enum Stat {
        /** Mean terminal PnL. */
        MEAN_PNL("mean_pnl"),
        /** Sample std of terminal PnL. */
        STD_PNL("std_pnl"),
        /** Mean of the per-path time average of {@code |q|}. */
        MEAN_ABS_Q("mean_abs_q"),
        /** Mean of the per-path maximum {@code |q|}. */
        MEAN_MAX_ABS_Q("mean_max_abs_q"),
        /** Mean of the per-path {@code sqrt(time average of q^2)}. */
        RMS_Q("rms_q"),
        /** Mean of the per-path total fills. */
        MEAN_FILLS("mean_fills");

        private final String wireName;

        Stat(String wireName) {
            this.wireName = wireName;
        }

        /** @return the spec's snake_case name, e.g. {@code "mean_pnl"} */
        public String wireName() {
            return wireName;
        }

        /**
         * Looks up a statistic by its spec name.
         *
         * @param name e.g. {@code "rms_q"}
         * @return the statistic
         * @throws IllegalArgumentException for an unknown name, including null
         */
        public static Stat fromWireName(String name) {
            for (Stat s : values()) {
                if (s.wireName.equals(name)) {
                    return s;
                }
            }
            throw new IllegalArgumentException("unknown statistic '" + name + "'");
        }
    }

    private final double[] pnl;
    private final double[] meanAbsQPath;
    private final double[] maxAbsQPath;
    private final double[] rmsQPath;
    private final double[] fillsPath;
    private final double overallMaxAbsQ;
    private final boolean capViolated;
    private final double meanPnl;
    private final double stdPnl;
    private final double meanAbsQ;
    private final double meanMaxAbsQ;
    private final double rmsQ;
    private final double meanFills;
    private final double[] se = new double[Stat.values().length];

    ASSimResult(double[] pnl, double[] meanAbsQPath, double[] maxAbsQPath,
                double[] rmsQPath, double[] fillsPath, double overallMaxAbsQ,
                boolean capViolated) {
        this.pnl = pnl;
        this.meanAbsQPath = meanAbsQPath;
        this.maxAbsQPath = maxAbsQPath;
        this.rmsQPath = rmsQPath;
        this.fillsPath = fillsPath;
        this.overallMaxAbsQ = overallMaxAbsQ;
        this.capViolated = capViolated;
        this.meanPnl = mean(pnl);
        this.stdPnl = std(pnl, meanPnl);
        this.meanAbsQ = mean(meanAbsQPath);
        this.meanMaxAbsQ = mean(maxAbsQPath);
        this.rmsQ = mean(rmsQPath);
        this.meanFills = mean(fillsPath);
        int m = pnl.length;
        double rootM = StrictMath.sqrt(m);
        se[Stat.MEAN_PNL.ordinal()] = stdPnl / rootM;
        se[Stat.STD_PNL.ordinal()] = stdPnl / StrictMath.sqrt(2.0 * (m - 1));
        se[Stat.MEAN_ABS_Q.ordinal()] = std(meanAbsQPath, meanAbsQ) / rootM;
        se[Stat.MEAN_MAX_ABS_Q.ordinal()] = std(maxAbsQPath, meanMaxAbsQ) / rootM;
        se[Stat.RMS_Q.ordinal()] = std(rmsQPath, rmsQ) / rootM;
        se[Stat.MEAN_FILLS.ordinal()] = std(fillsPath, meanFills) / rootM;
    }

    /** Number of simulated paths M. */
    public int nPaths() {
        return pnl.length;
    }

    /** Per-path terminal PnL (defensive copy). */
    public double[] pnlPerPath() {
        return pnl.clone();
    }

    /** Per-path time average of {@code |q|} (defensive copy). */
    public double[] meanAbsQPerPath() {
        return meanAbsQPath.clone();
    }

    /** Per-path maximum {@code |q|} (defensive copy). */
    public double[] maxAbsQPerPath() {
        return maxAbsQPath.clone();
    }

    /** Per-path {@code sqrt(time average of q^2)} (defensive copy). */
    public double[] rmsQPerPath() {
        return rmsQPath.clone();
    }

    /** Per-path total fills, both sides counted (defensive copy). */
    public double[] fillsPerPath() {
        return fillsPath.clone();
    }

    /** Mean terminal PnL over paths. */
    public double meanPnl() {
        return meanPnl;
    }

    /** Sample standard deviation of terminal PnL (ddof = 1). */
    public double stdPnl() {
        return stdPnl;
    }

    /** Sharpe-like ratio mean/std of terminal PnL ({@code +inf} when std = 0). */
    public double sharpe() {
        return stdPnl == 0.0 ? Double.POSITIVE_INFINITY : meanPnl / stdPnl;
    }

    /** Mean over paths of the per-path time average of {@code |q|}. */
    public double meanAbsQ() {
        return meanAbsQ;
    }

    /** Mean over paths of the per-path maximum {@code |q|}. */
    public double meanMaxAbsQ() {
        return meanMaxAbsQ;
    }

    /** Mean over paths of the per-path {@code sqrt(time average of q^2)}. */
    public double rmsQ() {
        return rmsQ;
    }

    /** Mean over paths of the per-path total fills (both sides counted). */
    public double meanFills() {
        return meanFills;
    }

    /** Largest {@code |q|} observed across all paths and steps. */
    public double overallMaxAbsQ() {
        return overallMaxAbsQ;
    }

    /** True if any step ended with {@code |q| > qMax} (never happens by construction). */
    public boolean capViolated() {
        return capViolated;
    }

    /**
     * Standard error of an aggregate statistic.
     *
     * @param stat the statistic
     * @return the standard error
     * @throws IllegalArgumentException when {@code stat} is null
     */
    public double se(Stat stat) {
        Validation.require(stat != null, "stat must not be null");
        return se[stat.ordinal()];
    }

    /**
     * Standard error by spec name.
     *
     * @param stat one of {@code "mean_pnl"}, {@code "std_pnl"}, {@code "mean_abs_q"},
     *             {@code "mean_max_abs_q"}, {@code "rms_q"}, {@code "mean_fills"}
     * @return the standard error
     * @throws IllegalArgumentException for an unknown statistic name
     * @deprecated use {@link #se(Stat)}; this overload will be removed in the next release
     */
    @Deprecated
    public double se(String stat) {
        return se(Stat.fromWireName(stat));
    }

    @Override
    public String toString() {
        return "ASSimResult[nPaths=" + nPaths() + ", meanPnl=" + meanPnl + ", stdPnl=" + stdPnl
                + ", meanAbsQ=" + meanAbsQ + ", meanMaxAbsQ=" + meanMaxAbsQ + ", rmsQ=" + rmsQ
                + ", meanFills=" + meanFills + ", overallMaxAbsQ=" + overallMaxAbsQ
                + ", capViolated=" + capViolated + "]";
    }

    private static double mean(double[] a) {
        double s = 0.0;
        for (double v : a) {
            s += v;
        }
        return s / a.length;
    }

    private static double std(double[] a, double m) {
        double ss = 0.0;
        for (double v : a) {
            double d = v - m;
            ss += d * d;
        }
        return StrictMath.sqrt(ss / (a.length - 1));
    }
}
