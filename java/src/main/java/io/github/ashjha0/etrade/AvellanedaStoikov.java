package io.github.ashjha0.etrade;

import static io.github.ashjha0.etrade.Validation.require;

import java.util.SplittableRandom;

/**
 * Avellaneda-Stoikov market making (2008 paper, finite horizon).
 *
 * <p>Closed forms (pinned in API_SPEC.md):
 *
 * <pre>
 *   reservation price  r(s,q,t) = s - q*gamma*sigma^2*(T - t)
 *   total spread       spread(t) = gamma*sigma^2*(T - t) + (2/gamma)*ln(1 + gamma/k)
 *   quotes             ask = r + spread/2 ,  bid = r - spread/2
 *   arrival intensity  lambda(delta) = A*exp(-k*delta)
 *   per-step fill prob P(fill) = 1 - exp(-lambda(delta)*dt)   (in [0, 1))
 * </pre>
 *
 * A long maker ({@code q > 0}) shades quotes down to attract buyers of its
 * inventory; a short maker shades up.  {@code delta} is measured from the
 * current mid and may be negative (a quote through the mid) — the intensity
 * formula still applies.  The exact thinned fill probability is the pinned
 * choice; the paper's own simulation uses the first-order {@code lambda*dt},
 * which over-fills by ~12% at {@code lambda*dt ~ 0.28} — hence the mean PnL
 * here sits below the paper's Table 1 while the PnL standard deviations agree.
 *
 * <p>Simulation loop order (pinned): per step, (1) quote from the current
 * {@code s}, {@code q}, remaining time (inventory cap: bid quoted only if
 * {@code q < qMax}, ask only if {@code q > -qMax}); (2) draw {@code u_ask}
 * then {@code u_bid} uniforms (always both, so the stream does not depend on
 * the cap state); (3) apply fills (unit size, both sides may fill); (4) draw
 * {@code z ~ N(0,1)} and move the mid, with the realised volatility stressed
 * by {@code stressFactor} for {@code t >= stressTimeFrac*T} while the agent
 * keeps quoting with the base sigma (model-misspecification scenario).
 * Terminal: {@code PnL = cash + q*S_T - etaLiq*|q|}.
 */
public final class AvellanedaStoikov {

    /** Upper bound on the number of simulation steps {@code T/dt} (pinned, API_SPEC 0). */
    public static final int MAX_AS_STEPS = 10_000_000;

    /** Upper bound on Monte Carlo path counts (pinned, API_SPEC 0). */
    public static final int MAX_AS_PATHS = 10_000_000;

    /** Quoting agent for {@link #simulate}. */
    public enum Agent {
        /** Avellaneda-Stoikov: reservation price +/- half the optimal spread. */
        AS,
        /** Fixed symmetric spread around the mid, no inventory skew. */
        FIXED
    }

    /** A two-sided quote.
     *
     * @param bid bid price
     * @param ask ask price
     */
    public record Quotes(double bid, double ask) {
    }

    private AvellanedaStoikov() {
    }

    /**
     * Reservation price {@code r(s,q,t) = s - q*gamma*sigma^2*(T - t)}.
     *
     * @param params model parameters
     * @param s      current mid price (finite)
     * @param q      current inventory (units, finite)
     * @param t      current time, in {@code [0, T]}
     * @return the reservation price
     */
    public static double reservationPrice(ASParams params, double s, double q, double t) {
        require(Double.isFinite(s) && Double.isFinite(q), "s and q must be finite");
        require(t >= 0.0 && t <= params.horizon(), "t must be in [0, horizon]");
        return s - q * params.gamma() * params.sigma() * params.sigma()
                * (params.horizon() - t);
    }

    /**
     * Total spread {@code gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/k)}.
     * Strictly positive; decreases linearly in {@code t} to the terminal floor
     * {@code (2/gamma)*ln(1 + gamma/k)} at {@code t = T}.
     *
     * @param params model parameters
     * @param t      current time, in {@code [0, T]}
     * @return the total quoted spread
     */
    public static double optimalSpread(ASParams params, double t) {
        require(t >= 0.0 && t <= params.horizon(), "t must be in [0, horizon]");
        return params.gamma() * params.sigma() * params.sigma() * (params.horizon() - t)
                + 2.0 / params.gamma() * Math.log1p(params.gamma() / params.k());
    }

    /**
     * (bid, ask) = reservation price -/+ half the optimal spread.
     *
     * @param params model parameters
     * @param s      current mid price (finite)
     * @param q      current inventory (finite)
     * @param t      current time, in {@code [0, T]}
     * @return the two-sided quote
     */
    public static Quotes quotes(ASParams params, double s, double q, double t) {
        double r = reservationPrice(params, s, q, t);
        double half = 0.5 * optimalSpread(params, t);
        return new Quotes(r - half, r + half);
    }

    /**
     * Arrival intensity {@code lambda(delta) = A*exp(-k*delta)}.
     *
     * @param params model parameters
     * @param delta  signed distance of the quote from the mid (finite; may be negative)
     * @return the Poisson arrival intensity
     */
    public static double intensity(ASParams params, double delta) {
        require(Double.isFinite(delta), "delta must be finite");
        return params.aRate() * Math.exp(-params.k() * delta);
    }

    /**
     * Per-step fill probability {@code 1 - exp(-lambda(delta)*dt)}, in {@code [0, 1)}.
     *
     * @param params model parameters
     * @param delta  signed quote distance from the mid
     * @return the thinned-Poisson fill probability
     */
    public static double fillProbability(ASParams params, double delta) {
        return -Math.expm1(-intensity(params, delta) * params.dt());
    }

    /**
     * Time average over {@code [0, T]} of the A-S spread (the fixed agent's
     * default): {@code gamma*sigma^2*T/2 + (2/gamma)*ln(1 + gamma/k)}.
     *
     * @param params model parameters
     * @return the average spread
     */
    public static double averageAsSpread(ASParams params) {
        return 0.5 * params.gamma() * params.sigma() * params.sigma() * params.horizon()
                + 2.0 / params.gamma() * Math.log1p(params.gamma() / params.k());
    }

    /**
     * Simulates the A-S agent with default fixed spread and no stress.
     *
     * @param params model parameters
     * @param nPaths number of paths, {@code >= 2}
     * @param seed   RNG seed
     * @return the simulation result
     */
    public static ASSimResult simulate(ASParams params, int nPaths, long seed) {
        return simulate(params, nPaths, seed, Agent.AS, averageAsSpread(params), 1.0, 0.5);
    }

    /**
     * Simulates the given agent with default fixed spread and no stress.
     *
     * @param params model parameters
     * @param nPaths number of paths, {@code >= 2}
     * @param seed   RNG seed
     * @param agent  the quoting agent
     * @return the simulation result
     */
    public static ASSimResult simulate(ASParams params, int nPaths, long seed, Agent agent) {
        return simulate(params, nPaths, seed, agent, averageAsSpread(params), 1.0, 0.5);
    }

    /**
     * Simulates the market maker over {@code nPaths} seeded paths (loop order
     * pinned; see the class javadoc).  Deterministic per (seed, agent, nPaths);
     * uses {@link SplittableRandom} (Java's own stream), so sample values differ
     * from the Python reference while agreeing statistically.
     *
     * @param agent          {@link Agent#AS} quotes the reservation price +/- half the
     *                       optimal spread; {@link Agent#FIXED} quotes the mid +/- half
     *                       {@code fixedSpread}
     * @param params         model parameters
     * @param nPaths         number of paths, in {@code [2, MAX_AS_PATHS]}
     * @param seed           RNG seed
     * @param fixedSpread    the fixed agent's total spread, finite and {@code > 0}
     *                       (use {@link #averageAsSpread} for the paper's default;
     *                       there is no NaN sentinel in Java)
     * @param stressFactor   realised-volatility multiplier applied for
     *                       {@code t >= stressTimeFrac*T}, {@code > 0} (1 = off);
     *                       the agent keeps quoting with the base sigma
     * @param stressTimeFrac start of the stress window as a fraction of T, in [0, 1]
     * @return per-path arrays and aggregate statistics
     */
    public static ASSimResult simulate(ASParams params, int nPaths, long seed, Agent agent,
                                       double fixedSpread, double stressFactor,
                                       double stressTimeFrac) {
        require(nPaths >= 2 && nPaths <= MAX_AS_PATHS,
                "nPaths must be an integer in [2, 10000000]");
        require(agent != null, "agent must be 'as' or 'fixed'");
        require(Double.isFinite(stressFactor) && stressFactor > 0.0,
                "stressFactor must be > 0");
        require(Double.isFinite(stressTimeFrac) && stressTimeFrac >= 0.0
                        && stressTimeFrac <= 1.0,
                "stressTimeFrac must be in [0, 1]");
        require(Double.isFinite(fixedSpread) && fixedSpread > 0.0,
                "fixedSpread must be finite and > 0");

        SplittableRandom rng = new SplittableRandom(seed);
        int nSteps = params.nSteps();
        double g = params.gamma();
        double sig2 = params.sigma() * params.sigma();
        double bigT = params.horizon();
        double dt = params.dt();
        double markup = 2.0 / g * Math.log1p(g / params.k());
        double sqdt = Math.sqrt(dt);
        double stressStart = stressTimeFrac * bigT;

        double[] pnl = new double[nPaths];
        double[] meanAbsQPath = new double[nPaths];
        double[] maxAbsQPath = new double[nPaths];
        double[] rmsQPath = new double[nPaths];
        double[] fillsPath = new double[nPaths];
        double overallMaxAbsQ = 0.0;
        boolean capViolated = false;

        for (int p = 0; p < nPaths; p++) {
            double s = params.s0();
            double q = 0.0;
            double cash = 0.0;
            double sumAbsQ = 0.0;
            double sumQ2 = 0.0;
            double maxAbsQ = 0.0;
            int fills = 0;
            for (int i = 0; i < nSteps; i++) {
                double t = i * dt;
                // 1. quote from the current s, q, remaining time
                double r;
                double half;
                if (agent == Agent.AS) {
                    r = s - q * g * sig2 * (bigT - t);
                    half = 0.5 * (g * sig2 * (bigT - t) + markup);
                } else {
                    r = s;
                    half = 0.5 * fixedSpread;
                }
                double ask = r + half;
                double bid = r - half;
                boolean quoteAsk = q > -params.qMax();
                boolean quoteBid = q < params.qMax();
                // 2. fill draws (u_ask first, then u_bid; both always drawn)
                double uAsk = rng.nextDouble();
                double uBid = rng.nextDouble();
                double pAsk = -Math.expm1(-params.aRate()
                        * Math.exp(-params.k() * (ask - s)) * dt);
                double pBid = -Math.expm1(-params.aRate()
                        * Math.exp(-params.k() * (s - bid)) * dt);
                boolean fillAsk = quoteAsk && uAsk < pAsk;
                boolean fillBid = quoteBid && uBid < pBid;
                // 3. inventory / cash update (unit fill size; both sides may fill,
                //    and both count: fills is a total fill count, API_SPEC 2.2)
                if (fillAsk) {
                    q -= 1.0;
                    cash += ask;
                    fills++;
                }
                if (fillBid) {
                    q += 1.0;
                    cash -= bid;
                    fills++;
                }
                double absQ = Math.abs(q);
                if (absQ > params.qMax() + 1e-12) {
                    capViolated = true;
                }
                sumAbsQ += absQ;
                sumQ2 += q * q;
                maxAbsQ = Math.max(maxAbsQ, absQ);
                // 4. mid move (stressed vol only affects the realised path)
                double sigT = params.sigma() * (t >= stressStart ? stressFactor : 1.0);
                s += sigT * sqdt * rng.nextGaussian();
            }
            pnl[p] = cash + q * s - params.etaLiq() * Math.abs(q);
            meanAbsQPath[p] = sumAbsQ / nSteps;
            maxAbsQPath[p] = maxAbsQ;
            rmsQPath[p] = Math.sqrt(sumQ2 / nSteps);
            fillsPath[p] = fills;
            overallMaxAbsQ = Math.max(overallMaxAbsQ, maxAbsQ);
        }
        return new ASSimResult(pnl, meanAbsQPath, maxAbsQPath, rmsQPath, fillsPath,
                overallMaxAbsQ, capViolated);
    }
}
