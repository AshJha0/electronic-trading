package io.github.ashjha0.etrade;

import static io.github.ashjha0.etrade.Validation.require;
import static io.github.ashjha0.etrade.Validation.requireNonNull;


/**
 * Almgren-Chriss discrete-time optimal execution (Almgren &amp; Chriss, 2000).
 *
 * <p>Sell {@code X} shares over {@code N} intervals of length {@code tau}
 * (horizon {@code T = N*tau}, times {@code t_j = j*tau}).  Holdings
 * {@code x_0 = X, ..., x_N = 0}; trade list {@code n_j = x_(j-1) - x_j}.
 * The mid price follows an arithmetic random walk with linear permanent
 * impact {@code g(v) = gammaP*v}; the trade in interval {@code j} executes at
 * {@code S_(j-1) - eta*n_j/tau} (linear temporary impact {@code h(v) = eta*v}).
 * Implementation shortfall: {@code IS = X*S_0 - sum_j n_j * S~_j}
 * (independent of {@code S_0}; the simulation uses {@code S_0 = 0}).
 *
 * <p>Exact discrete moments for any fully liquidating schedule (pinned in
 * API_SPEC.md; the permanent-impact cross terms collapse via
 * {@code sum_j n_j * sum_(i<j) n_i = (X^2 - sum_j n_j^2)/2}):
 *
 * <pre>
 *   E[IS]   = (1/2)*gammaP*X^2 + (etaTilde/tau) * sum_j n_j^2
 *   Var[IS] = sigma^2 * tau * sum_(j=1..N) x_j^2
 * </pre>
 *
 * The sums coincide with the paper's hyperbolic closed forms (with the
 * discrete kappa) to machine precision; they are used because they also cover
 * TWAP/POV/arbitrary schedules.  The paper's fixed-cost term
 * {@code epsilon*sum|n_j|} is omitted (schedule-independent for a one-sided
 * program).
 *
 * <p>The mean-variance objective {@code E + lambdaRa*Var} is minimised by the
 * sinh trajectory {@code x_j = X*sinh(kappa*(T-t_j))/sinh(kappa*T)} where the
 * discrete rate {@code kappa} solves
 * {@code (2/tau^2)*(cosh(kappa*tau)-1) = kappaTilde^2} with
 * {@code kappaTilde^2 = lambdaRa*sigma^2/etaTilde}.  Pinned evaluation
 * (API_SPEC 1.4): {@code kappa = log1p(x + sqrt(x*(2+x)))/tau} with
 * {@code x = kappaTilde^2*tau^2/2} (no cancellation for small x), and
 * {@code x_j = X*exp(-kappa*t_j)*expm1(-2*kappa*(T-t_j))/expm1(-2*kappa*T)}
 * (no sinh overflow for large kappa*T).  As {@code lambdaRa -> 0} the
 * trajectory degenerates to the linear TWAP schedule; to avoid 0/0 the
 * implementation switches to the exact linear branch when
 * {@code kappa*T < KAPPA_T_LIMIT} (covers {@code lambdaRa == 0} and
 * {@code sigma == 0}); the branch is continuous at the threshold to well
 * below golden tolerance.
 *
 * <p>Every size (intervals, POV clips, paths) is bounded by a pinned constant
 * before any allocation, so bad input is an {@link IllegalArgumentException},
 * never an overflow or an out-of-memory error.  Prices may go negative under
 * the arithmetic walk — a documented model property, never an error.
 */
public final class AlmgrenChriss {

    /** Threshold on {@code kappa*T} below which the exact linear (TWAP) branch is used. */
    public static final double KAPPA_T_LIMIT = 1e-6;

    /** Upper bound on the number of AC trading intervals N (pinned, API_SPEC 0). */
    public static final int MAX_AC_INTERVALS = 10_000_000;

    /** Upper bound on the number of POV intervals {@code M = ceil(X/clip)} (pinned). */
    public static final int MAX_POV_INTERVALS = 10_000_000;

    /** Upper bound on Monte Carlo path counts (pinned, API_SPEC 0). */
    public static final int MAX_PATHS = 10_000_000;

    private AlmgrenChriss() {
    }

    /** One point on the efficient frontier of execution.
     *
     * @param lambdaRa risk aversion of the point
     * @param kappa    discrete urgency parameter
     * @param eCost    expected cost E[IS]
     * @param varCost  variance of cost Var[IS]
     * @param stdCost  sqrt(Var[IS])
     */
    public record FrontierPoint(double lambdaRa, double kappa, double eCost,
                                double varCost, double stdCost) {
    }

    /** (E, Var) pair of a deterministic execution cost.
     *
     * @param eCost   expected cost
     * @param varCost variance of cost
     */
    public record CostMoments(double eCost, double varCost) {
    }

    /** Monte Carlo implementation-shortfall summary.
     *
     * @param mean   sample mean of IS over paths
     * @param var    sample variance of IS (ddof = 1)
     * @param seMean standard error of the mean, sqrt(var/M)
     * @param seVar  standard error of the variance, var*sqrt(2/(M-1)) (normal approx)
     * @param nPaths number of simulated paths M
     */
    public record ISSimResult(double mean, double var, double seMean, double seVar,
                              int nPaths) {
    }

    /**
     * Discrete urgency parameter {@code kappa}: the root of
     * {@code (2/tau^2)*(cosh(kappa*tau)-1) = kappaTilde^2}, i.e.
     * {@code kappa = arccosh(1 + kappaTilde^2*tau^2/2)/tau}; {@code 0} in the
     * {@code lambdaRa*sigma^2 = 0} limit.
     *
     * @param params model parameters
     * @return kappa (per unit time), {@code >= 0}
     */
    public static double kappa(ACParams params) {
        requireNonNull(params, "params");
        double kt2 = params.lambdaRa() * params.sigma() * params.sigma() / params.etaTilde();
        if (kt2 == 0.0) {
            return 0.0;
        }
        // arccosh(1 + e) = log1p(e + sqrt(e*(2 + e))): accurate for small e too.
        double e = 0.5 * kt2 * params.tau() * params.tau();
        return StrictMath.log1p(e + StrictMath.sqrt(e * (2.0 + e))) / params.tau();
    }

    /**
     * Characteristic decay time {@code theta = 1/kappa} of the optimal trajectory
     * (the paper's e-folding time, not {@code ln(2)/kappa}).
     *
     * @param params model parameters
     * @return the half-life; {@link Double#POSITIVE_INFINITY} in the TWAP limit
     */
    public static double halfLife(ACParams params) {
        requireNonNull(params, "params");
        double k = kappa(params);
        return k == 0.0 ? Double.POSITIVE_INFINITY : 1.0 / k;
    }

    /**
     * Alias of {@link #halfLife} under the paper's name ({@code theta = 1/kappa}).
     *
     * @param params model parameters
     * @return {@code 1/kappa}, or {@link Double#POSITIVE_INFINITY} in the TWAP limit
     */
    public static double characteristicTime(ACParams params) {
        requireNonNull(params, "params");
        return halfLife(params);
    }

    /**
     * Optimal holdings path {@code x_0..x_N} (length N+1, {@code x_0 = X} exactly,
     * {@code x_N = 0} exactly, every entry finite).  Uses the sinh solution in the
     * overflow-free exp/expm1 ratio form; switches to the exact linear (TWAP)
     * branch when {@code kappa*T < KAPPA_T_LIMIT}.
     *
     * @param params model parameters
     * @return holdings array of length {@code nIntervals + 1}
     */
    public static double[] trajectory(ACParams params) {
        requireNonNull(params, "params");
        int n = params.nIntervals();
        double x0 = params.totalShares();
        double[] x = new double[n + 1];
        if (x0 == 0.0) {
            return x;
        }
        double k = kappa(params);
        double bigT = params.horizon();
        if (k * bigT < KAPPA_T_LIMIT) {
            for (int j = 0; j <= n; j++) {
                x[j] = x0 * (n - j) / n;    // exact TWAP limit, no 0/0
            }
        } else {
            // sinh(k(T-t))/sinh(kT) = exp(-k t) * expm1(-2k(T-t)) / expm1(-2kT):
            // overflow-free; ratio formed first so that x_0 == X exactly.
            double denom = StrictMath.expm1(-2.0 * k * bigT);
            for (int j = 0; j <= n; j++) {
                double tj = j * params.tau();
                double ratio = StrictMath.expm1(-2.0 * k * (bigT - tj)) / denom;
                x[j] = x0 * StrictMath.exp(-k * tj) * ratio;
            }
        }
        x[n] = 0.0;
        return x;
    }

    /**
     * Optimal trade list {@code n_1..n_N} (shares sold per interval; sums to X).
     *
     * @param params model parameters
     * @return trade list of length {@code nIntervals}
     */
    public static double[] tradeList(ACParams params) {
        requireNonNull(params, "params");
        return tradesFromHoldings(trajectory(params));
    }

    /** Validates a user-supplied holdings path: non-null, length {@code >= 2}, finite. */
    private static void requireHoldings(double[] holdings) {
        require(holdings != null && holdings.length >= 2,
                "holdings must be an array of length >= 2");
        for (double v : holdings) {
            require(Double.isFinite(v), "holdings must be finite");
        }
    }

    /** Pinned liquidation check {@code |x_N| <= 1e-9 * max(1, |x_0|)}. */
    private static void requireLiquidating(double[] holdings) {
        require(Math.abs(holdings[holdings.length - 1])
                        <= 1e-9 * Math.max(1.0, Math.abs(holdings[0])),
                "holdings path must fully liquidate (x_N = 0)");
    }

    /**
     * Trade list {@code n_j = x_(j-1) - x_j} from a finite holdings path.
     *
     * @param holdings holdings path, length {@code >= 2}, finite
     * @return trades array of length {@code holdings.length - 1}
     */
    public static double[] tradesFromHoldings(double[] holdings) {
        requireHoldings(holdings);
        double[] n = new double[holdings.length - 1];
        for (int j = 0; j < n.length; j++) {
            n[j] = holdings[j] - holdings[j + 1];
        }
        return n;
    }

    /**
     * Linear TWAP holdings path {@code x_j = X*(N-j)/N}.
     *
     * @param totalShares shares to sell, finite and {@code >= 0}
     * @param nIntervals  number of intervals, in {@code [1, MAX_AC_INTERVALS]}
     * @return holdings array of length {@code nIntervals + 1}
     */
    public static double[] twapHoldings(double totalShares, int nIntervals) {
        require(Double.isFinite(totalShares) && totalShares >= 0.0,
                "totalShares must be finite and >= 0");
        require(nIntervals >= 1 && nIntervals <= MAX_AC_INTERVALS,
                "nIntervals must be an integer in [1, " + AlmgrenChriss.MAX_AC_INTERVALS + "]");
        double[] x = new double[nIntervals + 1];
        for (int j = 0; j <= nIntervals; j++) {
            x[j] = totalShares * (nIntervals - j) / nIntervals;
        }
        return x;
    }

    /**
     * POV (percentage-of-volume) holdings path: sells
     * {@code participation * marketVolumeRate * tau} shares per interval until
     * done (last clip partial).  The path length is set by the order size,
     * with {@code M = ceil(X/clip - 1e-12)} intervals and {@code x_M = 0} exactly;
     * {@code X/clip <= MAX_POV_INTERVALS} is required (a clip too small for the
     * order is an input error, never a huge allocation).
     *
     * @param totalShares      shares to sell, finite and {@code >= 0}
     * @param tau              interval length, finite and {@code > 0}
     * @param marketVolumeRate market volume rate (shares per unit time), finite, {@code > 0}
     * @param participation    participation fraction, in the open interval (0, 1)
     * @return holdings array of length {@code M + 1}
     */
    public static double[] povHoldings(double totalShares, double tau,
                                       double marketVolumeRate, double participation) {
        require(Double.isFinite(totalShares) && totalShares >= 0.0,
                "totalShares must be finite and >= 0");
        require(Double.isFinite(tau) && tau > 0.0, "tau must be > 0");
        require(Double.isFinite(marketVolumeRate) && marketVolumeRate > 0.0,
                "marketVolumeRate must be > 0");
        require(Double.isFinite(participation) && participation > 0.0 && participation < 1.0,
                "participation must be in (0, 1)");
        double clip = participation * marketVolumeRate * tau;
        if (totalShares == 0.0) {
            return new double[2];
        }
        double ratio = totalShares / clip;   // bounded BEFORE any cast/allocation
        require(ratio <= MAX_POV_INTERVALS,
                "POV clip too small: would need > " + MAX_POV_INTERVALS + " intervals");
        int nInt = (int) Math.ceil(ratio - 1e-12);
        double[] x = new double[nInt + 1];
        for (int i = 0; i <= nInt; i++) {
            x[i] = Math.max(totalShares - i * clip, 0.0);
        }
        x[nInt] = 0.0;
        return x;
    }

    /**
     * Expected cost of the optimal trajectory (see
     * {@link #expectedCost(ACParams, double[])}).
     *
     * @param params model parameters
     * @return E[IS]
     */
    public static double expectedCost(ACParams params) {
        requireNonNull(params, "params");
        return expectedCost(params, trajectory(params));
    }

    /**
     * {@code E[IS] = 0.5*gammaP*X^2 + (etaTilde/tau)*sum n_j^2} (exact discrete sum).
     * The order size X in the permanent-impact term is the schedule's
     * {@code x_0}, not {@code params.totalShares()} (pinned).
     *
     * @param params   model parameters
     * @param holdings any finite, fully liquidating path ({@code |x_N| <= 1e-9*max(1, X)})
     * @return E[IS]
     */
    public static double expectedCost(ACParams params, double[] holdings) {
        requireNonNull(params, "params");
        requireHoldings(holdings);
        requireLiquidating(holdings);
        double x0 = holdings[0];
        double[] n = tradesFromHoldings(holdings);
        double sumN2 = 0.0;
        for (double nj : n) {
            sumN2 += nj * nj;
        }
        return 0.5 * params.gammaP() * x0 * x0 + params.etaTilde() / params.tau() * sumN2;
    }

    /**
     * Variance of cost of the optimal trajectory (see
     * {@link #varianceCost(ACParams, double[])}).
     *
     * @param params model parameters
     * @return Var[IS]
     */
    public static double varianceCost(ACParams params) {
        requireNonNull(params, "params");
        return varianceCost(params, trajectory(params));
    }

    /**
     * {@code Var[IS] = sigma^2 * tau * sum_(j=1..N) x_j^2} (exact discrete sum;
     * {@code x_N = 0} contributes nothing).
     *
     * @param params   model parameters
     * @param holdings holdings path, length {@code >= 2}, finite
     * @return Var[IS]
     */
    public static double varianceCost(ACParams params, double[] holdings) {
        requireNonNull(params, "params");
        requireHoldings(holdings);
        double sumX2 = 0.0;
        for (int j = 1; j < holdings.length; j++) {
            sumX2 += holdings[j] * holdings[j];
        }
        return params.sigma() * params.sigma() * params.tau() * sumX2;
    }

    /**
     * (E, Var) of instantaneous liquidation: sell all X in the first interval.
     * {@code E = 0.5*gammaP*X^2 + etaTilde*X^2/tau}, {@code Var = 0}
     * (no residual exposure).
     *
     * @param params model parameters
     * @return the cost moments
     */
    public static CostMoments instantaneousCost(ACParams params) {
        requireNonNull(params, "params");
        double x0 = params.totalShares();
        double e = 0.5 * params.gammaP() * x0 * x0 + params.etaTilde() * x0 * x0 / params.tau();
        return new CostMoments(e, 0.0);
    }

    /**
     * Sweeps a risk-aversion grid; each point carries
     * {@code (lambdaRa, kappa, E, Var, sqrt(Var))} of the optimal strategy.
     *
     * @param params  model parameters (its own {@code lambdaRa} is ignored)
     * @param lambdas non-empty grid of risk aversions, each finite and {@code >= 0}
     *                (duplicates and unsorted grids are reproduced in input order)
     * @return one frontier point per grid value, in input order
     */
    public static FrontierPoint[] efficientFrontier(ACParams params, double[] lambdas) {
        requireNonNull(params, "params");
        require(lambdas != null && lambdas.length > 0, "lambdas grid must be non-empty");
        FrontierPoint[] out = new FrontierPoint[lambdas.length];
        for (int i = 0; i < lambdas.length; i++) {
            ACParams p = params.withLambda(lambdas[i]);
            double e = expectedCost(p);
            double v = varianceCost(p);
            out[i] = new FrontierPoint(lambdas[i], kappa(p), e, v, StrictMath.sqrt(v));
        }
        return out;
    }

    /**
     * Monte Carlo IS of the optimal schedule (see
     * {@link #simulateIS(ACParams, double[], int, long)}).
     *
     * @param params model parameters
     * @param nPaths number of paths, {@code >= 2}
     * @param seed   RNG seed (deterministic per seed)
     * @return the MC summary
     */
    public static ISSimResult simulateIS(ACParams params, int nPaths, long seed) {
        requireNonNull(params, "params");
        return simulateIS(params, trajectory(params), nPaths, seed);
    }

    /**
     * Simulates IS of a schedule under the arithmetic random walk.  Per path,
     * loop {@code j = 1..N}: execute {@code n_j} at {@code S_(j-1) - eta*n_j/tau},
     * then update {@code S_j = S_(j-1) + sigma*sqrt(tau)*xi_j - gammaP*n_j}
     * (draw order: one N(0,1) per interval, in interval order per path).
     * {@code IS = X*S_0 - proceeds} with {@code S_0 = 0}.  The schedule must
     * fully liquidate (same check as {@link #expectedCost}).  The closed forms
     * {@link #expectedCost} / {@link #varianceCost} match within a few SE.
     *
     * @param params   model parameters
     * @param holdings the schedule to execute (any finite, fully liquidating path)
     * @param nPaths   number of paths, in {@code [2, MAX_PATHS]}
     * @param seed     RNG seed (deterministic per seed; {@code SplittableRandom} uniforms with Marsaglia polar normals,
     *                 not NumPy's stream)
     * @return sample mean/variance of IS with standard errors
     */
    public static ISSimResult simulateIS(ACParams params, double[] holdings,
                                         int nPaths, long seed) {
        requireNonNull(params, "params");
        require(nPaths >= 2 && nPaths <= MAX_PATHS,
                "nPaths must be an integer in [2, " + MAX_PATHS + "]");
        requireHoldings(holdings);
        requireLiquidating(holdings);
        double[] n = tradesFromHoldings(holdings);
        NormalRng rng = new NormalRng(seed);
        double srt = params.sigma() * StrictMath.sqrt(params.tau());
        double[] shortfall = new double[nPaths];
        for (int p = 0; p < nPaths; p++) {
            double sPrev = 0.0;    // S_0 = 0; IS is independent of S_0
            double proceeds = 0.0;
            for (double nj : n) {
                double execPrice = sPrev - params.eta() * nj / params.tau();
                proceeds += nj * execPrice;
                sPrev = sPrev + srt * rng.nextNormal() - params.gammaP() * nj;
            }
            shortfall[p] = -proceeds;
        }
        double mean = 0.0;
        for (double v : shortfall) {
            mean += v;
        }
        mean /= nPaths;
        double ss = 0.0;
        for (double v : shortfall) {
            double d = v - mean;
            ss += d * d;
        }
        double var = ss / (nPaths - 1);
        return new ISSimResult(mean, var, StrictMath.sqrt(var / nPaths),
                var * StrictMath.sqrt(2.0 / (nPaths - 1)), nPaths);
    }
}
