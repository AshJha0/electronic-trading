package io.github.ashjha0.etrade;

import static io.github.ashjha0.etrade.Validation.require;

import java.util.Arrays;

/**
 * Power-law impact execution: square-root costs, volume curves, alpha-aware (drift)
 * schedules (API_SPEC Section 8).
 *
 * <p>Temporary cost per interval is {@code eta_j * n^(1+alpha)} with
 * {@code eta_j = eta * w_j^(-alpha)} ({@code w_j} = fraction of daily volume in the
 * interval). {@code alpha = 1} with uniform weights is exactly Almgren-Chriss;
 * {@code alpha = 1/2} is the square-root law. The optimal schedule is found by a pinned
 * Newton method (tridiagonal Hessian, Armijo line search) with a bisection over horizons
 * when an adverse drift makes it optimal to finish early, verified by a KKT check.
 */
public final class Impact {

    /** Upper bound on intervals for the power-law solver (pinned). */
    public static final int MAX_IMPACT_INTERVALS = 10_000;
    /** Newton iteration cap per solve (pinned). */
    public static final int NEWTON_MAX_ITER = 100;
    /** Convergence: Newton step below this times X in every coordinate (pinned). */
    public static final double NEWTON_STEP_TOL = 1e-12;
    /** Convergence: Newton decrement below this times max(1, |U|) (pinned). */
    public static final double NEWTON_DECREMENT_TOL = 1e-14;
    /** Armijo sufficient-decrease constant (pinned). */
    public static final double ARMIJO_C = 1e-4;
    /** Smallest line-search step, 2^-40 (pinned). */
    public static final double MIN_STEP = 0x1.0p-40;

    private static final double[] US_EQUITY_VOLUME_CURVE_13 = {
        0.12, 0.09, 0.075, 0.065, 0.06, 0.055, 0.05, 0.055, 0.06, 0.065, 0.075, 0.09, 0.14};

    private Impact() {
    }

    /**
     * Illustrative US-equity intraday volume curve: thirteen 30-minute fractions of daily
     * volume (U-shaped, sums to 1; not fitted to data).
     *
     * @return a fresh copy of the curve
     */
    public static double[] usEquityVolumeCurve13() {
        return US_EQUITY_VOLUME_CURVE_13.clone();
    }

    /** Power-law impact execution problem (API_SPEC 8.1); immutable, validated. */
    public static final class Params {
        private final double totalShares;
        private final int nIntervals;
        private final double tau;
        private final double sigma;
        private final double eta;
        private final double alpha;
        private final double gammaP;
        private final double lambdaRa;
        private final double drift;
        private final double[] weights;
        private final double[] etas;

        /**
         * Validated constructor.
         *
         * @param totalShares shares to sell, &ge; 0
         * @param nIntervals number of intervals in [1, MAX_IMPACT_INTERVALS]
         * @param tau interval length, &gt; 0
         * @param sigma arithmetic volatility, &ge; 0
         * @param eta temporary-impact coefficient, &gt; 0
         * @param alpha impact exponent in (0, 1]
         * @param gammaP permanent-impact coefficient, &ge; 0
         * @param lambdaRa risk aversion, &ge; 0
         * @param drift adverse drift per share per time unit, &ge; 0
         * @param weights volume fractions per interval, or null for {@code tau} each
         * @throws IllegalArgumentException on invalid input, including the convexity condition
         */
        public Params(double totalShares, int nIntervals, double tau, double sigma, double eta,
                      double alpha, double gammaP, double lambdaRa, double drift,
                      double[] weights) {
            require(Double.isFinite(totalShares) && totalShares >= 0.0,
                    "total_shares must be finite and >= 0");
            require(nIntervals >= 1 && nIntervals <= MAX_IMPACT_INTERVALS,
                    "n_intervals must be an integer in [1, 10000]");
            require(Double.isFinite(tau) && tau > 0.0, "tau must be > 0");
            require(Double.isFinite(sigma) && sigma >= 0.0, "sigma must be >= 0");
            require(Double.isFinite(eta) && eta > 0.0, "eta must be > 0");
            require(Double.isFinite(alpha) && alpha > 0.0 && alpha <= 1.0,
                    "alpha must be in (0, 1]");
            require(Double.isFinite(gammaP) && gammaP >= 0.0, "gamma_p must be >= 0");
            require(Double.isFinite(lambdaRa) && lambdaRa >= 0.0, "lambda_ra must be >= 0");
            require(Double.isFinite(drift) && drift >= 0.0, "drift must be >= 0");
            double[] w;
            if (weights == null) {
                w = new double[nIntervals];
                Arrays.fill(w, tau);
            } else {
                require(weights.length == nIntervals, "weights must have length n_intervals");
                w = weights.clone();
                for (double v : w) {
                    require(Double.isFinite(v) && v > 0.0, "weights must be finite and > 0");
                }
            }
            double[] e = new double[nIntervals];
            for (int j = 0; j < nIntervals; j++) {
                e[j] = eta * Math.pow(w[j], -alpha);
            }
            if (totalShares > 0.0) {
                double xPow = Math.pow(totalShares, alpha - 1.0);
                for (double etaJ : e) {
                    require(alpha * (1.0 + alpha) * etaJ * xPow > gammaP,
                            "model invalid: temporary impact not convex enough against "
                            + "permanent impact (alpha*(1+alpha)*eta_j*X**(alpha-1) <= gamma_p)");
                }
            }
            this.totalShares = totalShares;
            this.nIntervals = nIntervals;
            this.tau = tau;
            this.sigma = sigma;
            this.eta = eta;
            this.alpha = alpha;
            this.gammaP = gammaP;
            this.lambdaRa = lambdaRa;
            this.drift = drift;
            this.weights = w;
            this.etas = e;
        }

        /** @return shares to sell */
        public double totalShares() {
            return totalShares;
        }

        /** @return number of intervals */
        public int nIntervals() {
            return nIntervals;
        }

        /** @return interval length */
        public double tau() {
            return tau;
        }

        /** @return arithmetic volatility */
        public double sigma() {
            return sigma;
        }

        /** @return temporary-impact coefficient */
        public double eta() {
            return eta;
        }

        /** @return impact exponent */
        public double alpha() {
            return alpha;
        }

        /** @return permanent-impact coefficient */
        public double gammaP() {
            return gammaP;
        }

        /** @return risk aversion */
        public double lambdaRa() {
            return lambdaRa;
        }

        /** @return adverse drift */
        public double drift() {
            return drift;
        }

        /** @return a copy of the volume weights */
        public double[] weights() {
            return weights.clone();
        }

        /** @return a copy of the per-interval coefficients {@code eta * w_j^(-alpha)} */
        public double[] etas() {
            return etas.clone();
        }
    }

    /**
     * Optimal schedule.
     *
     * @param holdings holdings path, length N + 1
     * @param horizon horizon m actually used ({@code x_k = 0} for {@code k >= m})
     * @param iterations total Newton iterations
     */
    public record ScheduleResult(double[] holdings, int horizon, int iterations) {
        /** Defensive copy. */
        public ScheduleResult {
            holdings = holdings.clone();
        }

        /** @return a copy of the holdings */
        @Override
        public double[] holdings() {
            return holdings.clone();
        }
    }

    private static void checkHoldings(Params p, double[] x) {
        require(x != null && x.length == p.nIntervals + 1,
                "holdings must have length n_intervals + 1");
        for (double v : x) {
            require(Double.isFinite(v), "holdings must be finite");
        }
        double scale = Math.max(1.0, Math.abs(x[0]));
        require(Math.abs(x[x.length - 1]) <= 1e-9 * scale,
                "holdings path must fully liquidate (x_N = 0)");
        for (int j = 1; j < x.length; j++) {
            require(x[j] - x[j - 1] <= 1e-12 * scale,
                    "holdings must be non-increasing (a sell program)");
        }
    }

    /**
     * {@code E[IS]} of a non-increasing, fully liquidating schedule (API_SPEC 8.1).
     *
     * @param p parameters
     * @param x holdings, length N + 1
     * @return expected implementation shortfall
     */
    public static double expectedCost(Params p, double[] x) {
        checkHoldings(p, x);
        double a1 = 1.0 + p.alpha;
        double temp = 0.0;
        for (int j = 1; j < x.length; j++) {
            double n = Math.max(x[j - 1] - x[j], 0.0);
            temp += p.etas[j - 1] * Math.pow(n, a1) - 0.5 * p.gammaP * n * n;
        }
        double hold = 0.0;
        for (int k = 1; k < x.length - 1; k++) {
            hold += x[k];
        }
        return 0.5 * p.gammaP * x[0] * x[0] + temp + p.drift * p.tau * hold;
    }

    /**
     * {@code Var[IS] = sigma^2 * tau * sum_{j>=1} x_j^2}.
     *
     * @param p parameters
     * @param x holdings, length N + 1
     * @return variance of implementation shortfall
     */
    public static double varianceCost(Params p, double[] x) {
        checkHoldings(p, x);
        double sq = 0.0;
        for (int k = 1; k < x.length; k++) {
            sq += x[k] * x[k];
        }
        return (p.sigma * p.sigma) * p.tau * sq;
    }

    /**
     * Volume-weighted schedule: trade {@code X * w_j / sum(w)}; {@code x_N = 0} exactly.
     *
     * @param totalShares shares, &ge; 0
     * @param weights positive weights
     * @return holdings, length {@code weights.length + 1}
     */
    public static double[] vwapHoldings(double totalShares, double[] weights) {
        require(Double.isFinite(totalShares) && totalShares >= 0.0,
                "total_shares must be finite and >= 0");
        require(weights != null && weights.length > 0, "weights must be a non-empty 1-D sequence");
        double sum = 0.0;
        for (double w : weights) {
            require(Double.isFinite(w) && w > 0.0, "weights must be finite and > 0");
            sum += w;
        }
        double[] x = new double[weights.length + 1];
        x[0] = totalShares;
        double cum = 0.0;
        for (int j = 0; j < weights.length; j++) {
            cum += weights[j];
            x[j + 1] = totalShares * (1.0 - cum / sum);
        }
        x[weights.length] = 0.0;
        return x;
    }

    private static double[] thomas(double[] diag, double[] off, double[] rhs) {
        int m = diag.length;
        double[] c = new double[m];
        double[] d = new double[m];
        c[0] = m > 1 ? off[0] / diag[0] : 0.0;
        d[0] = rhs[0] / diag[0];
        for (int k = 1; k < m; k++) {
            double denom = diag[k] - off[k - 1] * c[k - 1];
            c[k] = k < m - 1 ? off[k] / denom : 0.0;
            d[k] = (rhs[k] - off[k - 1] * d[k - 1]) / denom;
        }
        double[] out = new double[m];
        out[m - 1] = d[m - 1];
        for (int k = m - 2; k >= 0; k--) {
            out[k] = d[k] - c[k] * out[k + 1];
        }
        return out;
    }

    private static double objective(Params p, double[] x) {
        double bigX = x[0];
        double total = 0.5 * p.gammaP * bigX * bigX;
        double a1 = 1.0 + p.alpha;
        for (int j = 1; j < x.length; j++) {
            double n = x[j - 1] - x[j];
            total += p.etas[j - 1] * Math.pow(n, a1) - 0.5 * p.gammaP * n * n;
        }
        double hold = 0.0;
        for (int k = 1; k < x.length - 1; k++) {
            hold += x[k];
        }
        double sq = 0.0;
        for (int k = 1; k < x.length; k++) {
            sq += x[k] * x[k];
        }
        return total + p.drift * p.tau * hold + p.lambdaRa * (p.sigma * p.sigma) * p.tau * sq;
    }

    /** Interior optimum on the first m intervals, or null when infeasible. */
    private static ScheduleResult solveHorizon(Params p, int m) {
        double bigX = p.totalShares;
        if (m == 1) {
            return new ScheduleResult(new double[] {bigX, 0.0}, 1, 0);
        }
        double a = p.alpha;
        double a1 = 1.0 + p.alpha;
        double g2 = p.gammaP;
        double v2 = 2.0 * p.lambdaRa * (p.sigma * p.sigma) * p.tau;
        double dt = p.drift * p.tau;
        double[] x = new double[m + 1];
        for (int k = 0; k <= m; k++) {
            x[k] = bigX * (m - k) / m;
        }
        double[] n = new double[m];
        double[] c1 = new double[m];
        double[] c2 = new double[m];
        double[] grad = new double[m - 1];
        double[] diag = new double[m - 1];
        double[] off = new double[m - 1];
        double[] rhs = new double[m - 1];
        for (int it = 1; it <= NEWTON_MAX_ITER; it++) {
            for (int j = 0; j < m; j++) {
                n[j] = x[j] - x[j + 1];
            }
            for (int j = 0; j < m; j++) {
                c1[j] = a1 * p.etas[j] * Math.pow(n[j], a) - g2 * n[j];
                c2[j] = a * a1 * p.etas[j] * Math.pow(n[j], a - 1.0) - g2;
            }
            for (int k = 1; k < m; k++) {
                grad[k - 1] = c1[k] - c1[k - 1] + v2 * x[k] + dt;
                diag[k - 1] = c2[k - 1] + c2[k] + v2;
                off[k - 1] = -c2[k];
                rhs[k - 1] = -grad[k - 1];
            }
            double[] step = thomas(diag, off, rhs);
            double u0 = objective(p, x);
            double slope = 0.0;
            double maxStep = 0.0;
            for (int k = 0; k < m - 1; k++) {
                slope += grad[k] * step[k];
                maxStep = Math.max(maxStep, Math.abs(step[k]));
            }
            if (maxStep <= NEWTON_STEP_TOL * bigX
                    || -slope <= NEWTON_DECREMENT_TOL * Math.max(1.0, Math.abs(u0))) {
                return new ScheduleResult(x, m, it);
            }
            double t = 1.0;
            double[] accepted = null;
            while (t >= MIN_STEP) {
                double[] xn = new double[m + 1];
                xn[0] = x[0];
                for (int k = 1; k < m; k++) {
                    xn[k] = x[k] + t * step[k - 1];
                }
                xn[m] = 0.0;
                boolean positive = true;
                for (int j = 1; j <= m; j++) {
                    positive = positive && (xn[j - 1] - xn[j] > 0.0);
                }
                if (positive && objective(p, xn) <= u0 + ARMIJO_C * t * slope) {
                    accepted = xn;
                    break;
                }
                t *= 0.5;
            }
            if (accepted == null) {
                return null;
            }
            x = accepted;
        }
        return null;
    }

    /**
     * Minimises {@code E + lambdaRa*Var} over non-increasing liquidating schedules
     * (API_SPEC 8.2).
     *
     * @param p parameters
     * @return the optimal schedule
     * @throws IllegalStateException if the solver fails to find a certified optimum
     */
    public static ScheduleResult optimalSchedule(Params p) {
        int nInt = p.nIntervals;
        double[] holdings = new double[nInt + 1];
        if (p.totalShares == 0.0) {
            return new ScheduleResult(holdings, nInt, 0);
        }
        ScheduleResult full = solveHorizon(p, nInt);
        if (full != null) {
            return full;
        }
        if (!(p.drift > 0.0)) {
            throw new IllegalStateException("optimal schedule: Newton solve did not converge");
        }
        int lo = 1;
        int hi = nInt;
        int iters = 0;
        ScheduleResult best = solveHorizon(p, 1);
        while (hi - lo > 1) {
            int mid = (lo + hi) / 2;
            ScheduleResult r = solveHorizon(p, mid);
            if (r != null) {
                lo = mid;
                best = r;
                iters += r.iterations();
            } else {
                hi = mid;
            }
        }
        iters += best.iterations();
        double[] xm = best.holdings();
        double nLast = xm[lo - 1] - xm[lo];
        double marginal = (1.0 + p.alpha) * p.etas[lo - 1] * Math.pow(nLast, p.alpha)
                - p.gammaP * nLast;
        if (!(marginal <= p.drift * p.tau * (1.0 + 1e-9) + 1e-12)) {
            throw new IllegalStateException(
                    "optimal schedule: KKT check failed for the early-finish horizon");
        }
        System.arraycopy(xm, 0, holdings, 0, xm.length);
        return new ScheduleResult(holdings, lo, iters);
    }

    /**
     * Square-root-law estimate {@code 1e4 * y * dailyVol * sqrt(Q / adv)} in bps.
     *
     * @param dailyVol daily volatility as a fraction, &gt; 0
     * @param orderShares order size, &ge; 0
     * @param adv average daily volume, &gt; 0
     * @param y prefactor, &gt; 0
     * @return basis points
     */
    public static double sqrtLawCostBps(double dailyVol, double orderShares, double adv,
                                        double y) {
        require(Double.isFinite(dailyVol) && dailyVol > 0.0, "daily_vol must be finite and > 0");
        require(Double.isFinite(orderShares) && orderShares >= 0.0, "order_shares must be >= 0");
        require(Double.isFinite(adv) && adv > 0.0, "adv must be finite and > 0");
        require(Double.isFinite(y) && y > 0.0, "y must be finite and > 0");
        return 1e4 * y * dailyVol * Math.sqrt(orderShares / adv);
    }
}
