package io.github.ashjha0.etrade;

import static io.github.ashjha0.etrade.Validation.require;
import static io.github.ashjha0.etrade.Validation.requireNonNull;


/**
 * Limit-order placement for execution: queue-position fill probability, the
 * passive-or-aggressive choice for one child order, and liquidation with limit and market
 * orders (API_SPEC Section 13).
 *
 * <p><b>Fill probability (13.1).</b> A FIFO queue holds {@code qAhead} units ahead of our
 * order of {@code size} units. Unit market orders arrive as a Poisson process of rate
 * {@code mu}; each unit ahead cancels at rate {@code theta}; our units never cancel. The fill
 * time is {@code sum_{k=1}^{qAhead} Exp(mu + theta k) + sum_{i=1}^{size} Exp(mu)} (the
 * pure-death queue of Cont, Stoikov and Talreja 2010; Cont and de Larrard 2013 use the same
 * order flow), evaluated by pinned uniformisation.
 *
 * <p><b>Passive or aggressive (13.2).</b> Crossing pays the half spread plus the Section 8
 * impact; posting earns the half spread with probability {@code P} and otherwise crosses
 * after an adverse move. The impact is paid either way (the square-root law holds for both
 * order types, Toth et al. 2011), so posting wins exactly when
 * {@code P > adverseMove / (adverseMove + spread)}; Moallemi and Yuan (2016) value queue
 * position on the same terms.
 *
 * <p><b>Liquidation (13.3).</b> A discrete Cartea and Jaimungal (2015) policy by backward
 * induction: post one unit at depth {@code max(0, 1/kappa - g)} or send a market order, with
 * a seeded Monte Carlo against all-aggressive and all-passive-then-sweep.
 *
 * <p>References: Cont, Stoikov and Talreja, Operations Research 58(3), 2010; Cont and de
 * Larrard, SIAM Journal on Financial Mathematics 4(1), 2013; Moallemi and Yuan, working
 * paper, Columbia Business School, 2016; Cartea and Jaimungal, Quantitative Finance 15(8),
 * 2015; Toth et al., Physical Review X 1(2), 2011.
 */
public final class Placement {

    /** Upper bound on {@code qAhead + size} in {@link #queueFillProbability} (pinned). */
    public static final int MAX_QUEUE_UNITS = 5_000;
    /** Upper bound on uniformisation steps in {@link #queueFillProbability} (pinned). */
    public static final int MAX_UNIFORMISATION_STEPS = 200_000;
    /** Upper bound on {@code totalUnits} of a liquidation (pinned). */
    public static final int MAX_LIQ_UNITS = 1_000;
    /** Upper bound on {@code nPeriods} of a liquidation (pinned). */
    public static final int MAX_LIQ_PERIODS = 1_000;

    private static final String TOTAL_UNITS_MSG =
            "total_units must be an integer in [1, " + MAX_LIQ_UNITS + "]";
    private static final String N_PERIODS_MSG =
            "n_periods must be an integer in [1, " + MAX_LIQ_PERIODS + "]";

    private Placement() {
    }

    // ------------------------------------------------------------------ fill probability

    /**
     * Probability that a resting order of {@code size} units with {@code qAhead} units ahead
     * fills completely within {@code horizon} (API_SPEC 13.1).
     *
     * @param qAhead units ahead in the queue, &ge; 0
     * @param size our order size in units, &ge; 1, {@code qAhead + size <= MAX_QUEUE_UNITS}
     * @param mu market-order arrival rate (units per time unit), finite and &gt; 0
     * @param theta cancellation rate per unit ahead, finite and &ge; 0
     * @param horizon horizon, finite and &ge; 0
     * @return the complete-fill probability
     * @throws IllegalArgumentException on invalid inputs or a horizon needing more than
     *     {@code MAX_UNIFORMISATION_STEPS} steps
     */
    public static double queueFillProbability(int qAhead, int size, double mu, double theta,
                                              double horizon) {
        require(qAhead >= 0, "q_ahead must be an integer >= 0");
        require(size >= 1, "size must be an integer >= 1");
        require(qAhead <= MAX_QUEUE_UNITS - size,
                "q_ahead + size must be <= " + MAX_QUEUE_UNITS);
        require(Double.isFinite(mu) && mu > 0.0, "mu must be finite and > 0");
        require(Double.isFinite(theta) && theta >= 0.0, "theta must be finite and >= 0");
        require(Double.isFinite(horizon) && horizon >= 0.0, "horizon must be finite and >= 0");
        double lam = mu + theta * qAhead;
        double bigL = lam * horizon;
        require(Double.isFinite(bigL),
                "horizon too long: (mu + theta*q_ahead)*horizon overflows");
        if (bigL == 0.0) {
            return 0.0;
        }
        double mD = Math.floor(bigL);
        double half = Math.ceil(10.0 * StrictMath.sqrt(bigL) + 10.0);
        require(mD + half <= MAX_UNIFORMISATION_STEPS,
                "horizon too long: (mu + theta*q_ahead)*horizon needs more than "
                        + MAX_UNIFORMISATION_STEPS + " uniformisation steps");
        int m = (int) mD;
        int hf = (int) half;
        int kLo = Math.max(0, m - hf);
        int kHi = m + hf;
        int nSt = qAhead + size;
        double[] go = new double[nSt];
        double[] stay = new double[nSt];
        for (int s = 0; s < nSt; s++) {
            double r = s < qAhead ? mu + theta * (qAhead - s) : mu;
            go[s] = r / lam;
            stay[s] = 1.0 - go[s];
        }
        double[] w = new double[kHi + 1];
        w[m] = 1.0;
        for (int k = m + 1; k <= kHi; k++) {
            w[k] = w[k - 1] * bigL / k;
        }
        for (int k = m - 1; k >= kLo; k--) {
            w[k] = w[k + 1] * (k + 1) / bigL;
        }
        double[] p = new double[nSt];
        p[0] = 1.0;
        double absorbed = 0.0;
        double acc = 0.0;
        double totalW = 0.0;
        for (int k = 0; k <= kHi; k++) {
            if (k > 0) {
                double prevMove = 0.0;
                for (int s = 0; s < nSt; s++) {
                    double mv = p[s] * go[s];
                    p[s] = p[s] * stay[s] + prevMove;
                    prevMove = mv;
                }
                absorbed = absorbed + prevMove;
            }
            if (k >= kLo) {
                acc += w[k] * absorbed;
                totalW += w[k];
            }
        }
        return Math.min(1.0, Math.max(0.0, acc / totalW));
    }

    // ------------------------------------------------------------------ passive or aggressive

    /**
     * Expected cost per share (currency, relative to the mid) of posting at the touch and of
     * crossing now (API_SPEC 13.2).
     *
     * @param passiveCost expected cost of posting
     * @param aggressiveCost cost of crossing now
     * @param impactCost Section 8 temporary impact per share
     * @param breakevenFillProb fill probability above which posting is cheaper
     * @param passive true when posting is strictly cheaper (ties cross)
     */
    public record PlacementDecision(double passiveCost, double aggressiveCost, double impactCost,
                                    double breakevenFillProb, boolean passive) {
    }

    /**
     * Compares posting one sell child order at the touch with crossing now (API_SPEC 13.2).
     *
     * @param size child order size in shares, &gt; 0
     * @param spread quoted spread, &ge; 0
     * @param fillProb probability the posted order fills, in [0, 1]
     * @param adverseMove adverse mid move per share before an unfilled order completes, &ge; 0
     * @param eta Section 8 impact coefficient, &ge; 0
     * @param alpha impact exponent in (0, 1]
     * @param weight liquidity weight of the interval, &gt; 0
     * @return the two costs and the choice
     * @throws IllegalArgumentException on invalid inputs
     */
    public static PlacementDecision placeOrCross(double size, double spread, double fillProb,
                                                 double adverseMove, double eta, double alpha,
                                                 double weight) {
        require(Double.isFinite(size) && size > 0.0, "size must be finite and > 0");
        require(Double.isFinite(spread) && spread >= 0.0, "spread must be finite and >= 0");
        require(Double.isFinite(fillProb) && fillProb >= 0.0 && fillProb <= 1.0,
                "fill_prob must be in [0, 1]");
        require(Double.isFinite(adverseMove) && adverseMove >= 0.0,
                "adverse_move must be finite and >= 0");
        require(Double.isFinite(eta) && eta >= 0.0, "eta must be finite and >= 0");
        require(Double.isFinite(alpha) && alpha > 0.0 && alpha <= 1.0,
                "alpha must be in (0, 1]");
        require(Double.isFinite(weight) && weight > 0.0, "weight must be finite and > 0");
        double hs = 0.5 * spread;
        double impact = eta * StrictMath.pow(weight, -alpha) * StrictMath.pow(size, alpha);
        double aggressive = hs + impact;
        double passive = fillProb * (impact - hs)
                + (1.0 - fillProb) * (adverseMove + hs + impact);
        double denom = adverseMove + spread;
        double breakeven = denom > 0.0 ? adverseMove / denom : 1.0;
        return new PlacementDecision(passive, aggressive, impact, breakeven, passive < aggressive);
    }

    // ------------------------------------------------------------------ liquidation

    /**
     * Sell {@code totalUnits} over {@code nPeriods} with limit and market orders
     * (API_SPEC 13.3); all fields finite.
     *
     * @param totalUnits units to sell, in [1, MAX_LIQ_UNITS]
     * @param nPeriods number of periods, in [1, MAX_LIQ_PERIODS]
     * @param dt period length, &gt; 0
     * @param sigma mid volatility per sqrt time unit, &ge; 0
     * @param spread quoted spread, &ge; 0
     * @param kappa fill decay per unit of depth, &gt; 0
     * @param pTouch per-period fill probability at the touch, in (0, 1]
     * @param beta book walk per unit of a market order, &ge; 0
     * @param phi running inventory penalty, &ge; 0
     */
    public record LiquidationParams(int totalUnits, int nPeriods, double dt, double sigma,
                                    double spread, double kappa, double pTouch, double beta,
                                    double phi) {

        /** Validates all fields; throws {@link IllegalArgumentException}. */
        public LiquidationParams {
            require(totalUnits >= 1 && totalUnits <= MAX_LIQ_UNITS, TOTAL_UNITS_MSG);
            require(nPeriods >= 1 && nPeriods <= MAX_LIQ_PERIODS, N_PERIODS_MSG);
            require(Double.isFinite(dt) && dt > 0.0, "dt must be finite and > 0");
            require(Double.isFinite(sigma) && sigma >= 0.0, "sigma must be finite and >= 0");
            require(Double.isFinite(spread) && spread >= 0.0, "spread must be finite and >= 0");
            require(Double.isFinite(kappa) && kappa > 0.0, "kappa must be finite and > 0");
            require(Double.isFinite(pTouch) && pTouch > 0.0 && pTouch <= 1.0,
                    "p_touch must be in (0, 1]");
            require(Double.isFinite(beta) && beta >= 0.0, "beta must be finite and >= 0");
            require(Double.isFinite(phi) && phi >= 0.0, "phi must be finite and >= 0");
        }
    }

    /**
     * Backward-induction policy (API_SPEC 13.3): {@code cost[j][q]} is
     * {@code (N + 1) x (Q + 1)}; {@code depth[j][q]} and {@code market[j][q]} are
     * {@code N x (Q + 1)} ({@code depth} is 0 where {@code market} is true).
     *
     * @param cost optimal expected cost to go, including the running penalty
     * @param depth posting depth beyond the touch
     * @param market true where a market order is sent
     */
    public record LiquidationPolicy(double[][] cost, double[][] depth, boolean[][] market) {

        /** @return {@code C_0(Q)}: expected shortfall plus penalties */
        public double expectedCost() {
            return cost[0][cost[0].length - 1];
        }
    }

    /**
     * Backward induction of API_SPEC 13.3 ({@code j} from {@code N - 1} down to 0, {@code q}
     * from 1 up to {@code Q}).
     *
     * @param p parameters
     * @return the policy
     */
    public static LiquidationPolicy liquidationPolicy(LiquidationParams p) {
        requireNonNull(p, "p");
        int nPer = p.nPeriods();
        int bigQ = p.totalUnits();
        double hs = 0.5 * p.spread();
        double[][] cost = new double[nPer + 1][bigQ + 1];
        double[][] depth = new double[nPer][bigQ + 1];
        boolean[][] market = new boolean[nPer][bigQ + 1];
        for (int q = 0; q <= bigQ; q++) {
            cost[nPer][q] = q * (hs + p.beta() * q);
        }
        double invK = 1.0 / p.kappa();
        for (int j = nPer - 1; j >= 0; j--) {
            for (int q = 1; q <= bigQ; q++) {
                double cHold = cost[j + 1][q];
                double cLess = cost[j + 1][q - 1];
                double g = hs + (cHold - cLess);
                double d = Math.max(0.0, invK - g);
                double gain = p.pTouch() * StrictMath.exp(-p.kappa() * d) * (d + g);
                double post = cHold - gain;
                double mkt = hs + p.beta() + cLess;
                double base;
                if (mkt < post) {
                    market[j][q] = true;
                    base = mkt;
                } else {
                    depth[j][q] = d;
                    base = post;
                }
                cost[j][q] = p.phi() * q * q * p.dt() + base;
            }
        }
        return new LiquidationPolicy(cost, depth, market);
    }

    /**
     * Even market-order schedule {@code n_j = floor(Q(j+1)/N) - floor(Qj/N)}.
     *
     * @param totalUnits units, in [1, MAX_LIQ_UNITS]
     * @param nPeriods periods, in [1, MAX_LIQ_PERIODS]
     * @return units per period
     */
    public static int[] aggressiveSchedule(int totalUnits, int nPeriods) {
        require(totalUnits >= 1 && totalUnits <= MAX_LIQ_UNITS, TOTAL_UNITS_MSG);
        require(nPeriods >= 1 && nPeriods <= MAX_LIQ_PERIODS, N_PERIODS_MSG);
        int[] out = new int[nPeriods];
        for (int j = 0; j < nPeriods; j++) {
            out[j] = totalUnits * (j + 1) / nPeriods - totalUnits * j / nPeriods;
        }
        return out;
    }

    /**
     * Expected shortfall of the all-aggressive baseline,
     * {@code sum_j n_j (halfSpread + beta n_j)}.
     *
     * @param p parameters
     * @return the deterministic cost
     */
    public static double aggressiveCost(LiquidationParams p) {
        requireNonNull(p, "p");
        double hs = 0.5 * p.spread();
        double total = 0.0;
        for (int n : aggressiveSchedule(p.totalUnits(), p.nPeriods())) {
            total += n * (hs + p.beta() * n);
        }
        return total;
    }

    /**
     * Shortfall (currency, {@code S_0 = 0}) of the policy and the two baselines; savings are
     * paired per path (baseline - optimal, positive = policy cheaper); SEs are sample std
     * (ddof 1) / sqrt(M).
     *
     * @param optimalMean mean shortfall of the policy
     * @param optimalSe its standard error
     * @param optimalStd its sample standard deviation
     * @param aggressiveMean mean shortfall of all-aggressive
     * @param aggressiveSe its standard error
     * @param aggressiveStd its sample standard deviation
     * @param passiveMean mean shortfall of all-passive-then-sweep
     * @param passiveSe its standard error
     * @param passiveStd its sample standard deviation
     * @param savingVsAggressiveMean mean paired saving aggressive - optimal
     * @param savingVsAggressiveSe its standard error
     * @param savingVsPassiveMean mean paired saving passive - optimal
     * @param savingVsPassiveSe its standard error
     * @param nPaths number of paths
     */
    public record LiquidationSimResult(double optimalMean, double optimalSe, double optimalStd,
                                       double aggressiveMean, double aggressiveSe,
                                       double aggressiveStd, double passiveMean,
                                       double passiveSe, double passiveStd,
                                       double savingVsAggressiveMean,
                                       double savingVsAggressiveSe,
                                       double savingVsPassiveMean, double savingVsPassiveSe,
                                       int nPaths) {
    }

    private static double[] meanStd(double[] v) {
        double s = 0.0;
        for (double x : v) {
            s += x;
        }
        double mean = s / v.length;
        double ss = 0.0;
        for (double x : v) {
            ss += (x - mean) * (x - mean);
        }
        return new double[] {mean, StrictMath.sqrt(ss / (v.length - 1))};
    }

    /**
     * Seeded Monte Carlo of the policy against all-aggressive and all-passive-then-sweep
     * (API_SPEC 13.3). Per period: draw {@code u ~ U(0,1)} then {@code z ~ N(0,1)} (shared by
     * the three strategies); trade at the current mid; then {@code S += sigma sqrt(dt) z};
     * sweep the remainder at the end. Paths are the outer loop and periods the inner loop;
     * draws come from {@code SplittableRandom} uniforms with Marsaglia polar normals (statistically, not bitwise, equal to the
     * reference).
     *
     * @param p parameters
     * @param nPaths number of paths in [2, AlmgrenChriss.MAX_PATHS]
     * @param seed RNG seed
     * @return the comparison
     * @throws IllegalArgumentException on invalid inputs
     */
    public static LiquidationSimResult simulateLiquidation(LiquidationParams p, int nPaths,
                                                           long seed) {
        requireNonNull(p, "p");
        require(nPaths >= 2 && nPaths <= AlmgrenChriss.MAX_PATHS,
                "n_paths must be an integer in [2, " + AlmgrenChriss.MAX_PATHS + "]");
        LiquidationPolicy pol = liquidationPolicy(p);
        int[] sched = aggressiveSchedule(p.totalUnits(), p.nPeriods());
        double hs = 0.5 * p.spread();
        double srt = p.sigma() * StrictMath.sqrt(p.dt());
        int nPer = p.nPeriods();
        int bigQ = p.totalUnits();
        double[][] fillAt = new double[nPer][bigQ + 1];
        for (int j = 0; j < nPer; j++) {
            for (int q = 0; q <= bigQ; q++) {
                fillAt[j][q] = p.pTouch() * StrictMath.exp(-p.kappa() * pol.depth()[j][q]);
            }
        }
        NormalRng rng = new NormalRng(seed);
        double[] isOpt = new double[nPaths];
        double[] isAgg = new double[nPaths];
        double[] isPas = new double[nPaths];
        double[] savA = new double[nPaths];
        double[] savP = new double[nPaths];
        for (int i = 0; i < nPaths; i++) {
            double s = 0.0;
            int qOpt = bigQ;
            int qPas = bigQ;
            double prOpt = 0.0;
            double prAgg = 0.0;
            double prPas = 0.0;
            for (int j = 0; j < nPer; j++) {
                double u = rng.nextDouble();
                double z = rng.nextNormal();
                if (qOpt > 0) {
                    if (pol.market()[j][qOpt]) {
                        prOpt += s - hs - p.beta();
                        qOpt--;
                    } else if (u < fillAt[j][qOpt]) {
                        prOpt += s + hs + pol.depth()[j][qOpt];
                        qOpt--;
                    }
                }
                if (qPas > 0 && u < p.pTouch()) {
                    prPas += s + hs;
                    qPas--;
                }
                int nj = sched[j];
                if (nj > 0) {
                    prAgg += nj * (s - hs - p.beta() * nj);
                }
                s += srt * z;
            }
            prOpt += qOpt * (s - hs - p.beta() * qOpt);
            prPas += qPas * (s - hs - p.beta() * qPas);
            isOpt[i] = -prOpt;
            isAgg[i] = -prAgg;
            isPas[i] = -prPas;
            savA[i] = isAgg[i] - isOpt[i];
            savP[i] = isPas[i] - isOpt[i];
        }
        double root = StrictMath.sqrt(nPaths);
        double[] so = meanStd(isOpt);
        double[] sa = meanStd(isAgg);
        double[] sp = meanStd(isPas);
        double[] va = meanStd(savA);
        double[] vp = meanStd(savP);
        return new LiquidationSimResult(so[0], so[1] / root, so[1], sa[0], sa[1] / root, sa[1],
                sp[0], sp[1] / root, sp[1], va[0], va[1] / root, vp[0], vp[1] / root, nPaths);
    }
}
