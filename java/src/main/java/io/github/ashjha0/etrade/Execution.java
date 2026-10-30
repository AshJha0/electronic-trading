package io.github.ashjha0.etrade;

import static io.github.ashjha0.etrade.Validation.require;
import static io.github.ashjha0.etrade.Validation.requireNonNull;


/**
 * Execution extensions: transient impact, adaptive re-planning and internalisation
 * (API_SPEC Section 12).
 *
 * <p><b>Transient impact (12.1).</b> Sell {@code X} over {@code N} intervals of length
 * {@code tau}; trade {@code n_j} (signed) at the start of interval {@code j}. The book
 * displacement obeys {@code D_1 = 0}, {@code D_{j+1} = a*(D_j + kappa*n_j)},
 * {@code a = exp(-rho*tau)}. A trade pays the displacement already present plus half its
 * own, {@code n*D_j + kappa*n^2/2}; with permanent impact {@code gammaP} booked as in
 * Section 1, {@code E[IS] = gammaP X^2/2 + n'An/2} with {@code A = kappa K - gammaP I},
 * {@code K_ij = a^|i-j|} (the Kac-Murdock-Szego matrix, whose inverse is tridiagonal). The
 * static optimum is found exactly by one tridiagonal (Thomas) solve. With {@code gammaP = 0}
 * the solution is proportional to {@code [1, 1-a, ..., 1-a, 1]}: equal blocks at both ends
 * and a constant rate between (Obizhaeva and Wang 2013, Proposition 1 in discrete time;
 * Alfonsi, Fruth and Schied 2010). As {@code rho -> infinity} the cost is Section 1's linear
 * temporary impact with {@code kappa = 2 eta / tau}.
 *
 * <p><b>Adaptive re-planning (12.2).</b> Section 1 dynamics plus an observable AR(1) drift
 * signal and random partial fills. The adaptive trader re-solves the remaining mean-variance
 * problem every interval; the static trader sends the time-0 plan. Without a signal and with
 * full fills the two coincide path by path (the Almgren-Chriss schedule is time consistent);
 * re-planning helps only when it uses information the plan did not have. Almgren and Lorenz
 * (2007) describe a separate source of gain (a global mean-variance criterion) that is not
 * modelled here.
 *
 * <p><b>Internalisation (12.3).</b> Opposite parent orders are crossed at the mid; only the
 * net residual goes to the market. Each side is costed with the Section 8 model as one
 * volume-weighted program and the saving is attributed pro rata. Crossing never increases
 * total impact cost.
 *
 * <p>References: Obizhaeva, A. A. and Wang, J. (2013), Optimal trading strategy and
 * supply/demand dynamics, J. Financial Markets 16(1), 1-32. Alfonsi, A., Fruth, A. and
 * Schied, A. (2010), Optimal execution strategies in limit order books with general shape
 * functions, Quantitative Finance 10(2), 143-157. Almgren, R. and Chriss, N. (2000), Optimal
 * execution of portfolio transactions, J. Risk 3(2), 5-39. Almgren, R. and Lorenz, J. (2007),
 * Adaptive arrival price, Trading 2007(1), 59-66.
 */
public final class Execution {

    /** Upper bound on intervals for every Section 12 entry point (pinned). */
    public static final int MAX_EXEC_INTERVALS = 1_000;
    /** Upper bound on the number of parent orders per side in {@code internalise}. */
    public static final int MAX_ORDERS = 10_000;

    private static final String N_INTERVALS_MSG =
            "n_intervals must be an integer in [1, " + MAX_EXEC_INTERVALS + "]";

    private Execution() {
    }

    /** Solve a symmetric tridiagonal system ({@code off[k]} couples k and k+1). */
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

    // ------------------------------------------------------------------ transient

    /**
     * Transient-impact problem (API_SPEC 12.1); validated at construction.
     *
     * @param totalShares shares to sell, finite and &ge; 0
     * @param nIntervals number of intervals in [1, MAX_EXEC_INTERVALS]
     * @param tau interval length, &gt; 0
     * @param kappa price displacement per share (inverse book depth), &gt; 0
     * @param rho resilience rate per time unit, finite and &gt; 0
     * @param gammaP permanent impact, &ge; 0 and {@code < kappa*(1-a)/(1+a)} (convexity)
     */
    public record TransientParams(double totalShares, int nIntervals, double tau, double kappa,
                                  double rho, double gammaP) {

        /** Validates all fields; throws {@link IllegalArgumentException}. */
        public TransientParams {
            require(Double.isFinite(totalShares) && totalShares >= 0.0,
                    "total_shares must be finite and >= 0");
            require(nIntervals >= 1 && nIntervals <= MAX_EXEC_INTERVALS, N_INTERVALS_MSG);
            require(Double.isFinite(tau) && tau > 0.0, "tau must be > 0");
            require(Double.isFinite(kappa) && kappa > 0.0, "kappa must be > 0");
            require(Double.isFinite(rho) && rho > 0.0, "rho must be finite and > 0");
            require(Double.isFinite(gammaP) && gammaP >= 0.0, "gamma_p must be >= 0");
            double a = StrictMath.exp(-rho * tau);
            require(gammaP < kappa * (1.0 - a) / (1.0 + a),
                    "model invalid: gamma_p >= kappa*(1-a)/(1+a) (cost not convex in the trades)");
        }

        /**
         * Parameters without permanent impact.
         *
         * @param totalShares shares to sell
         * @param nIntervals number of intervals
         * @param tau interval length
         * @param kappa inverse book depth
         * @param rho resilience rate
         */
        public TransientParams(double totalShares, int nIntervals, double tau, double kappa,
                               double rho) {
            this(totalShares, nIntervals, tau, kappa, rho, 0.0);
        }

        /** @return per-interval displacement decay {@code a = exp(-rho*tau)} */
        public double decay() {
            return StrictMath.exp(-rho * tau);
        }
    }

    private static void checkPath(TransientParams p, double[] x) {
        require(x != null && x.length == p.nIntervals() + 1,
                "holdings must have length n_intervals + 1");
        for (double v : x) {
            require(Double.isFinite(v), "holdings must be finite");
        }
        require(Math.abs(x[x.length - 1]) <= 1e-9 * Math.max(1.0, Math.abs(x[0])),
                "holdings path must fully liquidate (x_N = 0)");
    }

    /**
     * {@code E[IS]} of any fully liquidating schedule by the displacement recursion
     * (API_SPEC 12.1). Trades may have either sign; {@code X} in the permanent term is the
     * schedule's {@code x_0}.
     *
     * @param p parameters
     * @param holdings holdings path, length N + 1, {@code x_N = 0}
     * @return expected implementation shortfall
     * @throws IllegalArgumentException on an invalid path
     */
    public static double transientExpectedCost(TransientParams p, double[] holdings) {
        requireNonNull(p, "p");
        requireNonNull(holdings, "holdings");
        checkPath(p, holdings);
        double a = p.decay();
        double half = 0.5 * (p.kappa() - p.gammaP());
        double d = 0.0;
        double total = 0.0;
        for (int j = 1; j < holdings.length; j++) {
            double n = holdings[j - 1] - holdings[j];
            total += half * n * n + n * d;
            d = a * (d + p.kappa() * n);
        }
        double x0 = holdings[0];
        return 0.5 * p.gammaP() * x0 * x0 + total;
    }

    /**
     * Static schedule minimising {@code E[IS]} (API_SPEC 12.1): solves the tridiagonal
     * system {@code ((1-a^2) kappa I - gammaP T) y = T 1} with {@code T = (1-a^2) K^-1}.
     *
     * @param p parameters
     * @return holdings, length N + 1, with {@code x_0 = X} and {@code x_N = 0} exactly
     */
    public static double[] transientOptimalSchedule(TransientParams p) {
        requireNonNull(p, "p");
        int nInt = p.nIntervals();
        double bigX = p.totalShares();
        double[] x = new double[nInt + 1];
        x[0] = bigX;
        if (bigX == 0.0 || nInt == 1) {
            return x;
        }
        double a = p.decay();
        double b = -StrictMath.expm1(-p.rho() * p.tau());    // 1 - a without cancellation
        double ka = b * (1.0 + a) * p.kappa();          // (1 - a^2) * kappa
        double g = p.gammaP();
        double[] diag = new double[nInt];
        java.util.Arrays.fill(diag, ka - g * (1.0 + a * a));
        diag[0] = ka - g;
        diag[nInt - 1] = ka - g;
        double[] off = new double[nInt - 1];
        java.util.Arrays.fill(off, g * a);
        double[] rhs = new double[nInt];
        java.util.Arrays.fill(rhs, b * b);
        rhs[0] = b;
        rhs[nInt - 1] = b;
        double[] y = thomas(diag, off, rhs);
        double total = 0.0;
        for (double v : y) {
            total += v;
        }
        for (int j = 1; j < nInt; j++) {
            x[j] = x[j - 1] - bigX * y[j - 1] / total;
        }
        x[nInt] = 0.0;
        return x;
    }

    // ------------------------------------------------------------------ adaptive

    /**
     * Observable AR(1) drift signal and fill risk (API_SPEC 12.2):
     * {@code s_j = phi s_{j-1} + sqrt(1-phi^2) signalSd eps_j}, {@code s_0 = 0}; the price
     * drifts by {@code s_{j-1} tau} over interval {@code j}. A child order in interval
     * {@code j < N} fills the fraction {@code 1 - fillRisk*u_j}; the last always fills.
     *
     * @param phi signal persistence in [0, 1)
     * @param signalSd stationary signal standard deviation, &ge; 0
     * @param fillRisk maximum unfilled fraction in [0, 1)
     */
    public record SignalParams(double phi, double signalSd, double fillRisk) {

        /** Validates all fields; throws {@link IllegalArgumentException}. */
        public SignalParams {
            require(Double.isFinite(phi) && phi >= 0.0 && phi < 1.0, "phi must be in [0, 1)");
            require(Double.isFinite(signalSd) && signalSd >= 0.0, "signal_sd must be >= 0");
            require(Double.isFinite(fillRisk) && fillRisk >= 0.0 && fillRisk < 1.0,
                    "fill_risk must be in [0, 1)");
        }

        /** No signal and full fills. */
        public SignalParams() {
            this(0.0, 0.0, 0.0);
        }
    }

    private static void checkAc(ACParams p) {
        require(p.nIntervals() <= MAX_EXEC_INTERVALS, N_INTERVALS_MSG);
    }

    private static double[][] replanSystem(ACParams p, int m) {
        double e = p.etaTilde() / p.tau();
        double[] diag = new double[m - 1];
        java.util.Arrays.fill(diag, 2.0 * e + p.lambdaRa() * (p.sigma() * p.sigma()) * p.tau());
        double[] off = new double[m - 1];
        java.util.Arrays.fill(off, -e);
        return new double[][] {diag, off};
    }

    /**
     * Optimal remaining holdings {@code [x_k, x_{k+1}, .., x_N = 0]} from interval
     * {@code k} (0-based) given the current holding and signal (API_SPEC 12.2).
     *
     * @param p Almgren-Chriss parameters (Almgren and Chriss 2000), N &le; MAX_EXEC_INTERVALS
     * @param sig signal parameters
     * @param holding current holding, finite
     * @param signal current signal, finite
     * @param k intervals already traded, in [0, N)
     * @return remaining holdings, length {@code N - k + 1}
     * @throws IllegalArgumentException on invalid inputs
     */
    public static double[] replan(ACParams p, SignalParams sig, double holding, double signal,
                                  int k) {
        requireNonNull(p, "p");
        requireNonNull(sig, "sig");
        checkAc(p);
        require(k >= 0 && k < p.nIntervals(), "k must be an integer in [0, n_intervals)");
        require(Double.isFinite(holding), "holding must be finite");
        require(Double.isFinite(signal), "signal must be finite");
        int m = p.nIntervals() - k;
        if (m == 1) {
            return new double[] {holding, 0.0};
        }
        double[][] sys = replanSystem(p, m);
        double e = p.etaTilde() / p.tau();
        double[] rhs = new double[m - 1];
        for (int i = 0; i < m - 1; i++) {
            rhs[i] = 0.5 * p.tau() * StrictMath.pow(sig.phi(), i) * signal;
        }
        rhs[0] += e * holding;
        double[] y = thomas(sys[0], sys[1], rhs);
        double[] out = new double[m + 1];
        out[0] = holding;
        System.arraycopy(y, 0, out, 1, m - 1);
        out[m] = 0.0;
        return out;
    }

    /**
     * Feedback form of {@link #replan}: with {@code m} intervals left the next child order is
     * {@code c[m]*holding - d[m]*signal} (index 0 unused). The re-plan is linear in
     * {@code (holding, signal)}, so {@code c} and {@code d} are the first-step responses to
     * the two right-hand sides (API_SPEC 12.2).
     *
     * @param p Almgren-Chriss parameters, N &le; MAX_EXEC_INTERVALS
     * @param sig signal parameters
     * @return {@code {c, d}}, each of length N + 1
     */
    public static double[][] replanCoefficients(ACParams p, SignalParams sig) {
        requireNonNull(p, "p");
        requireNonNull(sig, "sig");
        checkAc(p);
        int nInt = p.nIntervals();
        double[] c = new double[nInt + 1];
        double[] d = new double[nInt + 1];
        if (nInt >= 1) {
            c[1] = 1.0;
        }
        double e = p.etaTilde() / p.tau();
        for (int m = 2; m <= nInt; m++) {
            double[][] sys = replanSystem(p, m);
            double[] rHold = new double[m - 1];
            rHold[0] = e;
            double[] rSig = new double[m - 1];
            for (int i = 0; i < m - 1; i++) {
                rSig[i] = 0.5 * p.tau() * StrictMath.pow(sig.phi(), i);
            }
            c[m] = 1.0 - thomas(sys[0], sys[1], rHold)[0];
            d[m] = thomas(sys[0], sys[1], rSig)[0];
        }
        return new double[][] {c, d};
    }

    /**
     * Static vs adaptive implementation shortfall over {@code nPaths} paths. {@code saving}
     * is the paired per-path difference {@code static - adaptive} (positive = adaptive
     * cheaper); SEs are sample std (ddof 1) / sqrt(M).
     *
     * @param staticMean mean static IS
     * @param staticSe its standard error
     * @param staticStd sample std of static IS
     * @param adaptiveMean mean adaptive IS
     * @param adaptiveSe its standard error
     * @param adaptiveStd sample std of adaptive IS
     * @param savingMean mean paired saving
     * @param savingSe its standard error
     * @param nPaths number of paths
     */
    public record AdaptiveSimResult(double staticMean, double staticSe, double staticStd,
                                    double adaptiveMean, double adaptiveSe, double adaptiveStd,
                                    double savingMean, double savingSe, int nPaths) {
    }

    private static double mean(double[] v) {
        double s = 0.0;
        for (double x : v) {
            s += x;
        }
        return s / v.length;
    }

    private static double sampleStd(double[] v, double mu) {
        double s = 0.0;
        for (double x : v) {
            s += (x - mu) * (x - mu);
        }
        return StrictMath.sqrt(s / (v.length - 1));
    }

    /**
     * Monte Carlo of static vs adaptive execution (API_SPEC 12.2); both strategies see the
     * same draws. Per interval {@code j = 1..N}: child order {@code q} (static: the time-0
     * plan's {@code n_j}; adaptive: {@code c[m] x - d[m] s}, {@code m = N - j + 1}; at
     * {@code j = N} both send the remainder); draw {@code u}, {@code xi}, {@code eps} (in
     * that order, per path); fill {@code e = f q}, {@code f = 1 - fillRisk u} ({@code 1} at
     * {@code j = N}); proceeds {@code += e (S - eta e / tau)}; then
     * {@code S += s tau + sigma sqrt(tau) xi - gammaP e},
     * {@code s = phi s + sqrt(1 - phi^2) signalSd eps}. {@code IS = -proceeds}.
     * Paths are the outer loop and intervals the inner loop; draws come from
     * {@code SplittableRandom} uniforms with Marsaglia polar normals (statistically, not bitwise, equal to the reference).
     *
     * @param p Almgren-Chriss parameters, N &le; MAX_EXEC_INTERVALS
     * @param sig signal parameters
     * @param nPaths number of paths in [2, AlmgrenChriss.MAX_PATHS]
     * @param seed RNG seed
     * @return the comparison
     * @throws IllegalArgumentException on invalid inputs
     */
    public static AdaptiveSimResult simulateAdaptive(ACParams p, SignalParams sig, int nPaths,
                                                     long seed) {
        requireNonNull(p, "p");
        requireNonNull(sig, "sig");
        checkAc(p);
        require(nPaths >= 2 && nPaths <= AlmgrenChriss.MAX_PATHS,
                "n_paths must be an integer in [2, " + AlmgrenChriss.MAX_PATHS + "]");
        int nInt = p.nIntervals();
        double[] plan = replan(p, sig, p.totalShares(), 0.0, 0);
        double[][] cd = replanCoefficients(p, sig);
        double[] c = cd[0];
        double[] d = cd[1];
        NormalRng rng = new NormalRng(seed);
        double srt = p.sigma() * StrictMath.sqrt(p.tau());
        double innov = StrictMath.sqrt(1.0 - sig.phi() * sig.phi()) * sig.signalSd();
        double[] isSt = new double[nPaths];
        double[] isAd = new double[nPaths];
        double[] sav = new double[nPaths];
        for (int path = 0; path < nPaths; path++) {
            double xSt = p.totalShares();
            double xAd = p.totalShares();
            double pxSt = 0.0;
            double pxAd = 0.0;
            double prSt = 0.0;
            double prAd = 0.0;
            double sgn = 0.0;
            for (int j = 1; j <= nInt; j++) {
                int m = nInt - j + 1;
                double qSt;
                double qAd;
                if (j == nInt) {
                    qSt = xSt;
                    qAd = xAd;
                } else {
                    qSt = plan[j - 1] - plan[j];
                    qAd = c[m] * xAd - d[m] * sgn;
                }
                double u = rng.nextDouble();
                double xi = rng.nextNormal();
                double eps = rng.nextNormal();
                double f = j == nInt ? 1.0 : 1.0 - sig.fillRisk() * u;
                double eSt = f * qSt;
                double eAd = f * qAd;
                prSt += eSt * (pxSt - p.eta() * eSt / p.tau());
                prAd += eAd * (pxAd - p.eta() * eAd / p.tau());
                double common = sgn * p.tau() + srt * xi;
                pxSt = pxSt + common - p.gammaP() * eSt;
                pxAd = pxAd + common - p.gammaP() * eAd;
                sgn = sig.phi() * sgn + innov * eps;
                xSt = xSt - eSt;
                xAd = xAd - eAd;
            }
            isSt[path] = -prSt;
            isAd[path] = -prAd;
            sav[path] = isSt[path] - isAd[path];
        }
        double mSt = mean(isSt);
        double mAd = mean(isAd);
        double mSav = mean(sav);
        double sdSt = sampleStd(isSt, mSt);
        double sdAd = sampleStd(isAd, mAd);
        double sdSav = sampleStd(sav, mSav);
        double root = StrictMath.sqrt(nPaths);
        return new AdaptiveSimResult(mSt, sdSt / root, sdSt, mAd, sdAd / root, sdAd,
                mSav, sdSav / root, nPaths);
    }

    // ------------------------------------------------------------------ internalisation

    /**
     * Crossing of opposite parent orders (API_SPEC 12.3). Costs are currency;
     * {@code *SavedBps} are per order in bps of its own notional and {@code savedBps} is in
     * bps of the gross notional {@code (B + S) * price}.
     *
     * @param crossedShares shares crossed at the mid, {@code min(B, S)}
     * @param residualShares net shares sent to the market, {@code |B - S|}
     * @param residualSide "buy", "sell" or "none"
     * @param baselineCost cost of both sides sent to the market separately
     * @param internalCost cost of the residual
     * @param savedCost baseline minus internal cost
     * @param savedBps saving in bps of the gross notional
     * @param buySavedBps per-buy-order saving in bps of its notional
     * @param sellSavedBps per-sell-order saving in bps of its notional
     */
    public record InternalisationResult(double crossedShares, double residualShares,
                                        String residualSide, double baselineCost,
                                        double internalCost, double savedCost, double savedBps,
                                        double[] buySavedBps, double[] sellSavedBps) {

        /** Defensive copies of the per-order arrays. */
        public InternalisationResult {
            buySavedBps = buySavedBps.clone();
            sellSavedBps = sellSavedBps.clone();
        }

        /** @return a copy of the per-buy-order savings */
        @Override
        public double[] buySavedBps() {
            return buySavedBps.clone();
        }

        /** @return a copy of the per-sell-order savings */
        @Override
        public double[] sellSavedBps() {
            return sellSavedBps.clone();
        }
    }

    private static double[] orders(String name, double[] q) {
        require(q != null, name + " must not be null");
        require(q.length <= MAX_ORDERS, name + " must have at most " + MAX_ORDERS + " orders");
        for (double v : q) {
            require(Double.isFinite(v) && v > 0.0, name + " must be finite and > 0");
        }
        return q.clone();
    }

    /**
     * {@link #internalise(double[], double[], double, double, double, double, int, double,
     * double, double, double, double[])} with alpha 0.5, Y 1, permFrac 0.10 and uniform
     * weights.
     *
     * @param buys buy parent order sizes, each finite and &gt; 0
     * @param sells sell parent order sizes, each finite and &gt; 0
     * @param price price
     * @param spread full quoted spread
     * @param adv average daily volume
     * @param dailyVol daily volatility as a fraction
     * @param nIntervals number of intervals in [1, MAX_EXEC_INTERVALS]
     * @param tau interval length in trading days
     * @return the crossing result
     */
    public static InternalisationResult internalise(double[] buys, double[] sells, double price,
                                                    double spread, double adv, double dailyVol,
                                                    int nIntervals, double tau) {
        requireNonNull(buys, "buys");
        requireNonNull(sells, "sells");
        return internalise(buys, sells, price, spread, adv, dailyVol, nIntervals, tau, 0.5, 1.0,
                Market.DEFAULT_PERM_FRAC, null);
    }

    /**
     * Cross buy and sell parent orders at the mid, send the residual to the market and report
     * the Section 8 impact cost saved (API_SPEC 12.3).
     *
     * @param buys buy parent order sizes, each finite and &gt; 0
     * @param sells sell parent order sizes, each finite and &gt; 0
     * @param price price
     * @param spread full quoted spread
     * @param adv average daily volume
     * @param dailyVol daily volatility as a fraction
     * @param nIntervals number of intervals in [1, MAX_EXEC_INTERVALS]
     * @param tau interval length in trading days
     * @param alpha impact exponent in (0, 1]
     * @param y square-root-law prefactor
     * @param permFrac permanent fraction in (0, 1]
     * @param weights volume fractions per interval, or null for uniform
     * @return the crossing result
     * @throws IllegalArgumentException on invalid inputs
     */
    public static InternalisationResult internalise(double[] buys, double[] sells, double price,
                                                    double spread, double adv, double dailyVol,
                                                    int nIntervals, double tau, double alpha,
                                                    double y, double permFrac, double[] weights) {
        requireNonNull(buys, "buys");
        requireNonNull(sells, "sells");
        double[] bQ = orders("buys", buys);
        double[] sQ = orders("sells", sells);
        require(bQ.length + sQ.length >= 1, "at least one order is required");
        require(nIntervals >= 1 && nIntervals <= MAX_EXEC_INTERVALS, N_INTERVALS_MSG);
        Market.ImpactMarket base = new Market.ImpactMarket(price, spread, adv, dailyVol, 0.0,
                nIntervals, tau, 0.0, alpha, y, permFrac, 0.0, weights);
        double bigB = 0.0;
        for (double v : bQ) {
            bigB += v;
        }
        double bigS = 0.0;
        for (double v : sQ) {
            bigS += v;
        }
        double crossed = Math.min(bigB, bigS);
        double residual = Math.abs(bigB - bigS);
        String side = bigB > bigS ? "buy" : bigS > bigB ? "sell" : "none";
        double cB = cost(base, bigB);
        double cS = cost(base, bigS);
        double cR = cost(base, residual);
        double baseline = cB + cS;
        double saved = baseline - cR;
        return new InternalisationResult(crossed, residual, side, baseline, cR, saved,
                Market.bps(saved, (bigB + bigS) * price),
                perOrder(bQ, bigB, cB, cR, "buy".equals(side), price),
                perOrder(sQ, bigS, cS, cR, "sell".equals(side), price));
    }

    private static double cost(Market.ImpactMarket base, double q) {
        if (q == 0.0) {
            return 0.0;
        }
        Impact.Params ip = Market.impactParamsFromMarket(base.withTotalShares(q));
        return Impact.expectedCost(ip, Impact.vwapHoldings(q, ip.weights()));
    }

    private static double[] perOrder(double[] qs, double total, double cSide, double cR,
                                     boolean majority, double price) {
        double[] out = new double[qs.length];
        for (int i = 0; i < qs.length; i++) {
            double q = qs[i];
            double baseI = cSide * q / total;
            double intI = majority ? cR * q / total : 0.0;
            out[i] = Market.bps(baseI - intI, q * price);
        }
        return out;
    }
}
