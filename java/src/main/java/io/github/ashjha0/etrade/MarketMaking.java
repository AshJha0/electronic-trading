package io.github.ashjha0.etrade;

import static io.github.ashjha0.etrade.Validation.require;
import static io.github.ashjha0.etrade.Validation.requireNonNull;

import java.util.Arrays;
import java.util.EnumMap;
import java.util.Map;

/**
 * Market making that can lose money: informed flow, ticks, fees, latency,
 * fair-value signals, PnL attribution and markouts (API_SPEC Section 9).
 *
 * <p>The Section 2 simulator ({@link AvellanedaStoikov#simulate}) is the paper's
 * world: fills are independent of the next price move, quotes sit at any real
 * price, trading is free and quotes never go stale.  This class keeps the A-S
 * quoting logic and adds those frictions one switch at a time, plus the
 * accounting a desk uses to see where the money went.
 *
 * <p>Simulation loop order (pinned, API_SPEC 9.4): per path, per step
 * {@code i} ({@code t = i*dt}), five draws always, in this order:
 * {@code z ~ N(0,1)}, {@code u_sig}, {@code u_ask}, {@code u_bid},
 * {@code u_inf ~ U(0,1)}.  The signal predicts {@code sign(z)} when
 * {@code u_sig < signalAccuracy} (else {@code -sign(z)}) and shifts fair value
 * by {@code (2*accuracy - 1)*pred*sigma*sqrt(dt)*sqrt(2/pi)}.  Quotes come from
 * the mid {@code latencySteps} steps ago plus that shift, rounded away from fair
 * value to the tick grid when {@code tick > 0}.  Uninformed fills use the
 * thinned-Poisson probability of Section 2 with distances from the true current
 * mid; an informed trader trades in the direction of {@code z} against a quote
 * inside the coming move, only if that side did not already fill and the cap
 * allows it.  Every fill earns {@code makerFee}.  Then the mid moves.
 *
 * <p>Attribution, exact per path:
 * {@code pnl = cash + q_T*S_T - liquidation = spread + fees + inventory - liquidation}.
 * {@code markout} is the one-step markout of the fills (part of {@code inventory});
 * negative means adverse selection.
 */
public final class MarketMaking {

    private static final double SQRT_2_OVER_PI = StrictMath.sqrt(2.0 / Math.PI);

    private MarketMaking() {
    }

    /** Quoting agent for {@link #simulate}. */
    public enum Agent {
        /** Avellaneda-Stoikov: reservation price +/- half the optimal spread. */
        AS,
        /** Asymptotic Gueant-Lehalle-Fernandez-Tapia quotes (API_SPEC 9.3). */
        GLFT,
        /** Fixed symmetric spread around fair value, no inventory skew. */
        FIXED
    }

    /** Per-path statistics reported by {@link Result} (API_SPEC 9.5). */
    public enum Stat {
        /** Terminal PnL after liquidation. */
        PNL,
        /** Spread captured against the mid before the move. */
        SPREAD,
        /** Maker fees (positive = rebate received). */
        FEES,
        /** Inventory PnL: sum of post-fill inventory times the mid move. */
        INVENTORY,
        /** Terminal liquidation penalty. */
        LIQUIDATION,
        /** One-step markout of the fills (negative = adverse selection). */
        MARKOUT,
        /** Total fill count, both sides. */
        FILLS,
        /** Fills against informed traders. */
        INFORMED_FILLS,
        /** Root-mean-square inventory over the steps. */
        RMS_Q
    }

    /**
     * Frictions and agent for {@link #simulate} (API_SPEC 9.1); validated on construction.
     * Start from {@link #defaults()} and change fields with the {@code with...} methods.
     *
     * @param agent          quoting agent
     * @param tick           tick size, finite and {@code >= 0} (0 = continuous prices)
     * @param makerFee       per share per fill, finite (positive = rebate, negative = fee)
     * @param latencySteps   quote staleness in steps, {@code >= 0} (and {@code <= n_steps}
     *                       at simulation time)
     * @param informedProb   per-step probability of an informed trader, in {@code [0, 1]}
     * @param signalAccuracy fair-value signal accuracy, in {@code [0.5, 1]} (0.5 = no signal)
     * @param fixedSpread    the fixed agent's total spread, finite and {@code > 0}, or
     *                       {@code null} for the average A-S spread of Section 2.1
     */
    public record Config(Agent agent, double tick, double makerFee, int latencySteps,
                         double informedProb, double signalAccuracy, Double fixedSpread) {

        /** Validates all fields; see the record javadoc for the constraints. */
        public Config {
            require(agent != null, "agent must not be null");
            require(Double.isFinite(tick) && tick >= 0.0, "tick must be finite and >= 0");
            require(Double.isFinite(makerFee), "maker_fee must be finite");
            require(latencySteps >= 0, "latency_steps must be an integer >= 0");
            require(Double.isFinite(informedProb) && informedProb >= 0.0 && informedProb <= 1.0,
                    "informed_prob must be in [0, 1]");
            require(Double.isFinite(signalAccuracy) && signalAccuracy >= 0.5
                            && signalAccuracy <= 1.0,
                    "signal_accuracy must be in [0.5, 1]");
            require(fixedSpread == null || Double.isFinite(fixedSpread) && fixedSpread > 0.0,
                    "fixed_spread must be finite and > 0");
        }

        /**
         * The frictionless A-S configuration: no tick, fee, latency, informed flow or
         * signal; default fixed spread.
         *
         * @return the default configuration
         */
        public static Config defaults() {
            return new Config(Agent.AS, 0.0, 0.0, 0, 0.0, 0.5, null);
        }

        /**
         * @param v the agent
         * @return a copy with this agent
         */
        public Config withAgent(Agent v) {
            return new Config(v, tick, makerFee, latencySteps, informedProb, signalAccuracy,
                    fixedSpread);
        }

        /**
         * @param v the tick size
         * @return a copy with this tick size
         */
        public Config withTick(double v) {
            return new Config(agent, v, makerFee, latencySteps, informedProb, signalAccuracy,
                    fixedSpread);
        }

        /**
         * @param v the maker fee
         * @return a copy with this maker fee
         */
        public Config withMakerFee(double v) {
            return new Config(agent, tick, v, latencySteps, informedProb, signalAccuracy,
                    fixedSpread);
        }

        /**
         * @param v the latency in steps
         * @return a copy with this latency
         */
        public Config withLatencySteps(int v) {
            return new Config(agent, tick, makerFee, v, informedProb, signalAccuracy,
                    fixedSpread);
        }

        /**
         * @param v the informed-trader probability
         * @return a copy with this probability
         */
        public Config withInformedProb(double v) {
            return new Config(agent, tick, makerFee, latencySteps, v, signalAccuracy,
                    fixedSpread);
        }

        /**
         * @param v the signal accuracy
         * @return a copy with this accuracy
         */
        public Config withSignalAccuracy(double v) {
            return new Config(agent, tick, makerFee, latencySteps, informedProb, v,
                    fixedSpread);
        }

        /**
         * @param v the fixed agent's total spread, or {@code null} for the default
         * @return a copy with this fixed spread
         */
        public Config withFixedSpread(Double v) {
            return new Config(agent, tick, makerFee, latencySteps, informedProb, signalAccuracy,
                    v);
        }
    }

    /**
     * GLFT quote distances from fair value.
     *
     * @param deltaBid distance of the bid below fair value
     * @param deltaAsk distance of the ask above fair value
     */
    public record HalfSpreads(double deltaBid, double deltaAsk) {
    }

    /** Per-path arrays for every {@link Stat}, plus the largest attribution residual. */
    public static final class Result {
        private final Map<Stat, double[]> paths;
        private final double maxIdentityError;

        private Result(Map<Stat, double[]> paths, double maxIdentityError) {
            this.paths = paths;
            this.maxIdentityError = maxIdentityError;
        }

        /** @return the number of simulated paths */
        public int nPaths() {
            return paths.get(Stat.PNL).length;
        }

        /**
         * Per-path values of a statistic.
         *
         * @param stat the statistic
         * @return a fresh copy of the per-path array
         */
        public double[] path(Stat stat) {
            require(stat != null, "stat must not be null");
            return paths.get(stat).clone();
        }

        /**
         * Mean over paths of a statistic.
         *
         * @param stat the statistic
         * @return the mean
         */
        public double mean(Stat stat) {
            require(stat != null, "stat must not be null");
            double sum = 0.0;
            for (double v : paths.get(stat)) {
                sum += v;
            }
            return sum / nPaths();
        }

        /**
         * Standard error of the mean: sample std (ddof 1) / sqrt(nPaths).
         *
         * @param stat the statistic
         * @return the standard error
         */
        public double se(Stat stat) {
            double m = mean(stat);
            double ss = 0.0;
            for (double v : paths.get(stat)) {
                ss += (v - m) * (v - m);
            }
            int n = nPaths();
            return StrictMath.sqrt(ss / (n - 1)) / StrictMath.sqrt(n);
        }

        /**
         * Largest attribution residual {@code |spread + fees + inventory - liquidation - pnl|}
         * over paths, relative to {@code max(1, |pnl|)}.
         *
         * @return the residual
         */
        public double maxIdentityError() {
            return maxIdentityError;
        }
    }

    /**
     * Rounds a quote pair away from fair value to the tick grid (API_SPEC 9.2):
     * {@code bid -> floor(bid/tick + 1e-9)*tick}, {@code ask -> ceil(ask/tick - 1e-9)*tick};
     * {@code tick = 0} returns the inputs.
     *
     * @param bid  bid price, finite
     * @param ask  ask price, finite
     * @param tick tick size, finite and {@code >= 0}
     * @return the rounded quote
     */
    public static AvellanedaStoikov.Quotes roundToTick(double bid, double ask, double tick) {
        require(Double.isFinite(tick) && tick >= 0.0, "tick must be finite and >= 0");
        require(Double.isFinite(bid) && Double.isFinite(ask), "bid and ask must be finite");
        if (tick == 0.0) {
            return new AvellanedaStoikov.Quotes(bid, ask);
        }
        return new AvellanedaStoikov.Quotes(Math.floor(bid / tick + 1e-9) * tick,
                Math.ceil(ask / tick - 1e-9) * tick);
    }

    /**
     * Closed-form asymptotic GLFT distances (API_SPEC 9.3):
     * {@code c1 = log1p(gamma/k)/gamma},
     * {@code c2 = sqrt(sigma^2 gamma/(2 k A) (1 + gamma/k)^(1 + k/gamma))},
     * {@code delta_bid = c1 + (2q + 1)/2 c2}, {@code delta_ask = c1 - (2q - 1)/2 c2}.
     *
     * @param params model parameters
     * @param q      current inventory, finite
     * @return the bid and ask distances from fair value
     */
    public static HalfSpreads glftHalfSpreads(ASParams params, double q) {
        requireNonNull(params, "params");
        require(Double.isFinite(q), "q must be finite");
        double g = params.gamma();
        double k = params.k();
        double c1 = StrictMath.log1p(g / k) / g;
        double c2 = glftC2(params);
        return new HalfSpreads(c1 + (2.0 * q + 1.0) / 2.0 * c2, c1 - (2.0 * q - 1.0) / 2.0 * c2);
    }

    private static double glftC2(ASParams params) {
        double g = params.gamma();
        double k = params.k();
        double s2 = params.sigma() * params.sigma();
        return StrictMath.sqrt(s2 * g / (2.0 * k * params.aRate()) * StrictMath.pow(1.0 + g / k, 1.0 + k / g));
    }

    /**
     * Size-weighted mid {@code (bid*askSize + ask*bidSize)/(bidSize + askSize)}, a simple
     * proxy for the microprice (Stoikov 2018).
     *
     * @param bid     bid price, finite
     * @param ask     ask price, finite and {@code >= bid}
     * @param bidSize bid size, finite and {@code >= 0}
     * @param askSize ask size, finite and {@code >= 0}, with a positive sum
     * @return the weighted mid
     */
    public static double weightedMid(double bid, double ask, double bidSize, double askSize) {
        require(Double.isFinite(bid), "bid must be finite");
        require(Double.isFinite(ask), "ask must be finite");
        require(ask >= bid, "ask must be >= bid");
        require(Double.isFinite(bidSize) && bidSize >= 0.0 && Double.isFinite(askSize)
                && askSize >= 0.0, "sizes must be finite and >= 0");
        require(bidSize + askSize > 0.0, "bid_size + ask_size must be > 0");
        return (bid * askSize + ask * bidSize) / (bidSize + askSize);
    }

    /**
     * Per-update order-flow imbalance of Cont, Kukanov and Stoikov (2014) for consecutive
     * top-of-book states:
     * {@code e_n = 1[Pb_n >= Pb_n-1] Qb_n - 1[Pb_n <= Pb_n-1] Qb_n-1
     * - 1[Pa_n <= Pa_n-1] Qa_n + 1[Pa_n >= Pa_n-1] Qa_n-1}.
     *
     * @param bidPx bid prices
     * @param bidSz bid sizes, {@code >= 0}
     * @param askPx ask prices
     * @param askSz ask sizes, {@code >= 0}
     * @return the {@code length - 1} values {@code e_1..e_n}
     */
    public static double[] orderFlowImbalance(double[] bidPx, double[] bidSz, double[] askPx,
                                              double[] askSz) {
        require(bidPx != null && bidSz != null && askPx != null && askSz != null,
                "book series must not be null");
        int n = bidPx.length;
        require(n >= 2, "need at least 2 book states");
        require(bidSz.length == n && askPx.length == n && askSz.length == n,
                "book series must be 1-D and equal length");
        for (double[] a : new double[][] {bidPx, bidSz, askPx, askSz}) {
            for (double v : a) {
                require(Double.isFinite(v), "book series must be finite");
            }
        }
        for (int i = 0; i < n; i++) {
            require(bidSz[i] >= 0.0 && askSz[i] >= 0.0, "sizes must be >= 0");
        }
        double[] e = new double[n - 1];
        for (int i = 1; i < n; i++) {
            double upB = bidPx[i] >= bidPx[i - 1] ? 1.0 : 0.0;
            double dnB = bidPx[i] <= bidPx[i - 1] ? 1.0 : 0.0;
            double dnA = askPx[i] <= askPx[i - 1] ? 1.0 : 0.0;
            double upA = askPx[i] >= askPx[i - 1] ? 1.0 : 0.0;
            e[i - 1] = upB * bidSz[i] - dnB * bidSz[i - 1] - dnA * askSz[i] + upA * askSz[i - 1];
        }
        return e;
    }

    /**
     * Simulates the market maker with frictions (API_SPEC 9.4; loop order pinned, see the
     * class javadoc).  Deterministic per (seed, config, nPaths); uses
     * {@code SplittableRandom} uniforms with Marsaglia polar normals, so sample values differ from the Python reference while
     * agreeing statistically.
     *
     * @param params model parameters
     * @param nPaths number of paths, in {@code [2, MAX_AS_PATHS]}
     * @param seed   RNG seed
     * @param config frictions and agent ({@link Config#defaults()} for none)
     * @return per-path attribution arrays
     */
    public static Result simulate(ASParams params, int nPaths, long seed, Config config) {
        require(params != null, "params must not be null");
        require(config != null, "config must not be null");
        require(nPaths >= 2 && nPaths <= AvellanedaStoikov.MAX_AS_PATHS,
                "nPaths must be an integer in [2, " + AvellanedaStoikov.MAX_AS_PATHS + "]");
        int nSteps = params.nSteps();
        int lat = config.latencySteps();
        require(lat <= nSteps, "latency_steps must be <= n_steps");

        NormalRng rng = new NormalRng(seed);
        double g = params.gamma();
        double sig = params.sigma();
        double sig2 = sig * sig;
        double bigT = params.horizon();
        double dt = params.dt();
        double k = params.k();
        double aRate = params.aRate();
        double qMax = params.qMax();
        double sqdt = StrictMath.sqrt(dt);
        double markup = 2.0 / g * StrictMath.log1p(g / k);
        double fixedHalf = 0.5 * (config.fixedSpread() != null ? config.fixedSpread()
                : 0.5 * g * sig2 * bigT + markup);
        double c1 = StrictMath.log1p(g / k) / g;
        double c2 = config.agent() == Agent.GLFT ? glftC2(params) : 0.0;
        double shiftScale = (2.0 * config.signalAccuracy() - 1.0) * sig * sqdt * SQRT_2_OVER_PI;
        double tick = config.tick();
        double fee = config.makerFee();

        Map<Stat, double[]> out = new EnumMap<>(Stat.class);
        for (Stat st : Stat.values()) {
            out.put(st, new double[nPaths]);
        }
        double maxErr = 0.0;
        // ring buffer of the last lat + 1 mids; hist[head] is the mid lat steps ago
        double[] hist = new double[lat + 1];

        for (int p = 0; p < nPaths; p++) {
            double s = params.s0();
            Arrays.fill(hist, s);
            int head = 0;
            double q = 0.0;
            double cash = 0.0;
            double spread = 0.0;
            double fees = 0.0;
            double fills = 0.0;
            double informedFills = 0.0;
            double inventory = 0.0;
            double markout = 0.0;
            double sumQ2 = 0.0;
            for (int i = 0; i < nSteps; i++) {
                double t = i * dt;
                // 1. five draws, always, in the pinned order
                double z = rng.nextNormal();
                double uSig = rng.nextDouble();
                double uAsk = rng.nextDouble();
                double uBid = rng.nextDouble();
                double uInf = rng.nextDouble();
                // 2. fair value from the stale mid plus the signal shift
                double sz = Math.signum(z);
                double pred = uSig < config.signalAccuracy() ? sz : -sz;
                double fair = hist[head] + shiftScale * pred;
                // 3. quotes from fair value, then tick rounding
                double bid;
                double ask;
                if (config.agent() == Agent.AS) {
                    double r = fair - q * g * sig2 * (bigT - t);
                    double half = 0.5 * (g * sig2 * (bigT - t) + markup);
                    bid = r - half;
                    ask = r + half;
                } else if (config.agent() == Agent.GLFT) {
                    bid = fair - (c1 + (2.0 * q + 1.0) / 2.0 * c2);
                    ask = fair + (c1 - (2.0 * q - 1.0) / 2.0 * c2);
                } else {
                    bid = fair - fixedHalf;
                    ask = fair + fixedHalf;
                }
                if (tick > 0.0) {
                    bid = Math.floor(bid / tick + 1e-9) * tick;
                    ask = Math.ceil(ask / tick - 1e-9) * tick;
                }
                // 4. uninformed fills, distances from the true current mid
                boolean canAsk = q > -qMax;
                boolean canBid = q < qMax;
                double pAsk = -StrictMath.expm1(-aRate * StrictMath.exp(-k * (ask - s)) * dt);
                double pBid = -StrictMath.expm1(-aRate * StrictMath.exp(-k * (s - bid)) * dt);
                boolean fillAsk = canAsk && uAsk < pAsk;
                boolean fillBid = canBid && uBid < pBid;
                // 5. informed fills against quotes inside the coming move
                double move = sig * sqdt * z;
                boolean informed = uInf < config.informedProb();
                boolean infAsk = informed && z > 0.0 && canAsk && !fillAsk && ask < s + move;
                boolean infBid = informed && z < 0.0 && canBid && !fillBid && bid > s + move;
                // 6. updates
                boolean fa = fillAsk || infAsk;
                boolean fb = fillBid || infBid;
                double faF = fa ? 1.0 : 0.0;
                double fbF = fb ? 1.0 : 0.0;
                double nFill = faF + fbF;
                q = q + fbF - faF;
                cash = cash + (fa ? ask : 0.0) - (fb ? bid : 0.0) + fee * nFill;
                spread += (fa ? ask - s : 0.0) + (fb ? s - bid : 0.0);
                fees += fee * nFill;
                fills += nFill;
                informedFills += (infAsk ? 1.0 : 0.0) + (infBid ? 1.0 : 0.0);
                inventory += q * move;
                markout += (fbF - faF) * move;
                sumQ2 += q * q;
                // 7. mid move; push into the history, dropping the oldest
                s = s + move;
                hist[head] = s;
                head = head == lat ? 0 : head + 1;
            }
            double liq = params.etaLiq() * Math.abs(q);
            double pnl = cash + q * s - liq;
            out.get(Stat.PNL)[p] = pnl;
            out.get(Stat.SPREAD)[p] = spread;
            out.get(Stat.FEES)[p] = fees;
            out.get(Stat.INVENTORY)[p] = inventory;
            out.get(Stat.LIQUIDATION)[p] = liq;
            out.get(Stat.MARKOUT)[p] = markout;
            out.get(Stat.FILLS)[p] = fills;
            out.get(Stat.INFORMED_FILLS)[p] = informedFills;
            out.get(Stat.RMS_Q)[p] = StrictMath.sqrt(sumQ2 / nSteps);
            double recon = spread + fees + inventory - liq;
            maxErr = Math.max(maxErr, Math.abs(recon - pnl) / Math.max(1.0, Math.abs(pnl)));
        }
        return new Result(out, maxErr);
    }
}
