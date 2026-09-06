package com.quant.etrade;

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

    private final double[] pnl;
    private final double[] meanAbsQPath;
    private final double[] maxAbsQPath;
    private final double[] rmsQPath;
    private final double[] fillsPath;
    private final double overallMaxAbsQ;
    private final boolean capViolated;

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
    }

    /** Number of simulated paths M. */
    public int nPaths() {
        return pnl.length;
    }

    /** Per-path terminal PnL (defensive copy). */
    public double[] pnlPerPath() {
        return pnl.clone();
    }

    /** Mean terminal PnL over paths. */
    public double meanPnl() {
        return mean(pnl);
    }

    /** Sample standard deviation of terminal PnL (ddof = 1). */
    public double stdPnl() {
        return std(pnl);
    }

    /** Sharpe-like ratio mean/std of terminal PnL ({@code +inf} when std = 0). */
    public double sharpe() {
        double s = stdPnl();
        return s == 0.0 ? Double.POSITIVE_INFINITY : meanPnl() / s;
    }

    /** Mean over paths of the per-path time average of {@code |q|}. */
    public double meanAbsQ() {
        return mean(meanAbsQPath);
    }

    /** Mean over paths of the per-path maximum {@code |q|}. */
    public double meanMaxAbsQ() {
        return mean(maxAbsQPath);
    }

    /** Mean over paths of the per-path {@code sqrt(time average of q^2)}. */
    public double rmsQ() {
        return mean(rmsQPath);
    }

    /** Mean over paths of the per-path total fills (both sides counted). */
    public double meanFills() {
        return mean(fillsPath);
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
     * @param stat one of {@code "mean_pnl"}, {@code "std_pnl"}, {@code "mean_abs_q"},
     *             {@code "mean_max_abs_q"}, {@code "rms_q"}, {@code "mean_fills"}
     * @return the standard error
     * @throws IllegalArgumentException for an unknown statistic name
     */
    public double se(String stat) {
        int m = nPaths();
        return switch (stat) {
            case "std_pnl" -> stdPnl() / Math.sqrt(2.0 * (m - 1));
            case "mean_pnl" -> std(pnl) / Math.sqrt(m);
            case "mean_abs_q" -> std(meanAbsQPath) / Math.sqrt(m);
            case "mean_max_abs_q" -> std(maxAbsQPath) / Math.sqrt(m);
            case "rms_q" -> std(rmsQPath) / Math.sqrt(m);
            case "mean_fills" -> std(fillsPath) / Math.sqrt(m);
            default -> throw new IllegalArgumentException("unknown statistic '" + stat + "'");
        };
    }

    private static double mean(double[] a) {
        double s = 0.0;
        for (double v : a) {
            s += v;
        }
        return s / a.length;
    }

    private static double std(double[] a) {
        double m = mean(a);
        double ss = 0.0;
        for (double v : a) {
            double d = v - m;
            ss += d * d;
        }
        return Math.sqrt(ss / (a.length - 1));
    }
}
