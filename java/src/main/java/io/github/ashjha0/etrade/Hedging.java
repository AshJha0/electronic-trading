package io.github.ashjha0.etrade;

import static io.github.ashjha0.etrade.Validation.require;

import io.github.ashjha0.etrade.MarketMaking.Agent;
import io.github.ashjha0.etrade.MarketMaking.Config;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Hedged market making and risk controls (API_SPEC Section 14).
 *
 * <p>The Section 9 maker ({@link MarketMaking#simulate}) can shed inventory only by
 * skewing its own quotes.  This class adds a correlated hedge instrument B and hard risk
 * controls (loss limit, position limit, quote throttle, kill switch) to the Section 9
 * loop and keeps the accounting exact.
 *
 * <p>Instrument B moves {@code sigma*sqrt(dt)*(rho*z + sqrt(1 - rho^2)*z_b)} per step,
 * quoted in A-equivalent units, so the minimum-variance hedge ratio is {@code rho}
 * (Ederington 1979).  The quoting agents skew on the risk-equivalent exposure
 * {@code x = q + rho*y} in place of {@code q} (Gueant 2017).
 *
 * <p>Simulation loop order (pinned, API_SPEC 14.4): per path, per step {@code i}, six
 * draws always, in this order: {@code z ~ N(0,1)}, {@code u_sig}, {@code u_ask},
 * {@code u_bid}, {@code u_inf ~ U(0,1)}, {@code z_b ~ N(0,1)}.  Then stops (kill switch,
 * loss limit), desired quotes, the throttle, Section 9 fills against the standing
 * quotes, the position limit, the hedge, accounting and the moves of A and B.
 *
 * <p>Attribution, exact per path:
 * {@code pnl = spread + fees + inventory - liquidation + hedge_pnl - hedge_costs
 * - risk_costs}.
 */
public final class Hedging {

    private static final double SQRT_2_OVER_PI = Math.sqrt(2.0 / Math.PI);

    private Hedging() {
    }

    /** Per-path statistics reported by {@link Result} (API_SPEC 14.5). */
    public enum Stat {
        /** Terminal PnL after liquidation and the hedge unwind. */
        PNL,
        /** Spread captured against the mid before the move. */
        SPREAD,
        /** Maker fees (positive = rebate received). */
        FEES,
        /** Inventory PnL of A: sum of post-fill inventory times the mid move. */
        INVENTORY,
        /** Terminal liquidation penalty of A. */
        LIQUIDATION,
        /** One-step markout of the fills (negative = adverse selection). */
        MARKOUT,
        /** Total fill count, both sides. */
        FILLS,
        /** Fills against informed traders. */
        INFORMED_FILLS,
        /** Root-mean-square inventory of A over the steps. */
        RMS_Q,
        /** PnL of the B position: sum of {@code y} times the B move. */
        HEDGE_PNL,
        /** Hedge cost paid on every B trade, including the terminal unwind. */
        HEDGE_COSTS,
        /** Number of B trades, including the terminal unwind. */
        HEDGE_TRADES,
        /** Flatten penalties paid on forced flattens of A. */
        RISK_COSTS,
        /** Root-mean-square risk-equivalent exposure {@code q + rho*y}. */
        RMS_X,
        /** Root-mean-square quote skew while active. */
        RMS_SKEW,
        /** Number of quote updates. */
        QUOTE_UPDATES,
        /** 1 if the path stopped on the loss limit, else 0. */
        LOSS_STOP,
        /** 1 if the path stopped on the kill switch, else 0. */
        KILL_STOP,
        /** Number of position-limit flattens. */
        POS_FLATTENS,
        /** How far the mark fell below the loss limit at the loss stop (0 if none). */
        OVERSHOOT,
        /** Change of the mark over the step of the loss stop (0 if none). */
        STOP_MOVE
    }

    /**
     * Hedge instrument and rule (API_SPEC 14.1); validated on construction.
     *
     * @param rho            correlation of B's move with A's, in {@code [-1, 1]}
     * @param hedgeRatio     B shares per A share, finite and {@code >= 0} (0 = no hedge)
     * @param hedgeCost      cost per B share traded, finite and {@code >= 0}
     * @param hedgeThreshold uncovered A shares tolerated, finite and {@code >= 0}
     * @param hedgeTarget    fraction of the uncovered shares left after a hedge, in
     *                       {@code [0, 1)}
     */
    public record HedgeConfig(double rho, double hedgeRatio, double hedgeCost,
                              double hedgeThreshold, double hedgeTarget) {

        /** Validates all fields; see the record javadoc for the constraints. */
        public HedgeConfig {
            require(Double.isFinite(rho) && rho >= -1.0 && rho <= 1.0, "rho must be in [-1, 1]");
            require(Double.isFinite(hedgeRatio) && hedgeRatio >= 0.0,
                    "hedge_ratio must be finite and >= 0");
            require(Double.isFinite(hedgeCost) && hedgeCost >= 0.0,
                    "hedge_cost must be finite and >= 0");
            require(Double.isFinite(hedgeThreshold) && hedgeThreshold >= 0.0,
                    "hedge_threshold must be finite and >= 0");
            require(Double.isFinite(hedgeTarget) && hedgeTarget >= 0.0 && hedgeTarget < 1.0,
                    "hedge_target must be in [0, 1)");
        }

        /**
         * No hedge: every field 0.
         *
         * @return the default configuration
         */
        public static HedgeConfig defaults() {
            return new HedgeConfig(0.0, 0.0, 0.0, 0.0, 0.0);
        }
    }

    /**
     * Risk controls (API_SPEC 14.1); every default is off; validated on construction.
     *
     * @param lossLimit       loss stop on the marked PnL, finite and {@code >= 0} (0 = off)
     * @param positionLimit   flatten A when {@code |q| >= positionLimit}, {@code >= 0}
     *                        (0 = off)
     * @param flattenPenalty  per share of A flattened by a control, finite and {@code >= 0}
     * @param throttleUpdates quote updates allowed per window, {@code >= 0} (0 = off)
     * @param throttleWindow  window in steps, {@code >= 1} (and {@code <= n_steps})
     * @param killStep        step at which the book is flattened and stops, {@code >= -1}
     *                        (-1 = off; and {@code <= n_steps})
     */
    public record RiskConfig(double lossLimit, int positionLimit, double flattenPenalty,
                             int throttleUpdates, int throttleWindow, int killStep) {

        /** Validates all fields; see the record javadoc for the constraints. */
        public RiskConfig {
            require(Double.isFinite(lossLimit) && lossLimit >= 0.0,
                    "loss_limit must be finite and >= 0");
            require(positionLimit >= 0, "position_limit must be an integer >= 0");
            require(Double.isFinite(flattenPenalty) && flattenPenalty >= 0.0,
                    "flatten_penalty must be finite and >= 0");
            require(throttleUpdates >= 0, "throttle_updates must be an integer >= 0");
            require(throttleWindow >= 1, "throttle_window must be an integer >= 1");
            require(killStep >= -1, "kill_step must be an integer >= -1");
        }

        /**
         * All controls off.
         *
         * @return the default configuration
         */
        public static RiskConfig defaults() {
            return new RiskConfig(0.0, 0, 0.0, 0, 1, -1);
        }
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

        private double[] arr(Stat stat) {
            require(stat != null, "stat must not be null");
            return paths.get(stat);
        }

        /**
         * Per-path values of a statistic.
         *
         * @param stat the statistic
         * @return a fresh copy of the per-path array
         */
        public double[] path(Stat stat) {
            return arr(stat).clone();
        }

        /**
         * Mean over paths of a statistic.
         *
         * @param stat the statistic
         * @return the mean
         */
        public double mean(Stat stat) {
            double sum = 0.0;
            for (double v : arr(stat)) {
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
            for (double v : arr(stat)) {
                ss += (v - m) * (v - m);
            }
            int n = nPaths();
            return Math.sqrt(ss / (n - 1)) / Math.sqrt(n);
        }

        /**
         * Population variance {@code m2 = mean((v - mean)^2)} (API_SPEC 14.5).
         *
         * @param stat the statistic
         * @return the variance
         */
        public double var(Stat stat) {
            double m = mean(stat);
            double ss = 0.0;
            for (double v : arr(stat)) {
                ss += (v - m) * (v - m);
            }
            return ss / nPaths();
        }

        /**
         * Delta-method standard error of {@link #var}: {@code sqrt(max(0, m4 - m2^2)/n)}.
         *
         * @param stat the statistic
         * @return the standard error of the variance
         */
        public double varSe(Stat stat) {
            double m = mean(stat);
            double s2 = 0.0;
            double s4 = 0.0;
            for (double v : arr(stat)) {
                double d2 = (v - m) * (v - m);
                s2 += d2;
                s4 += d2 * d2;
            }
            int n = nPaths();
            double m2 = s2 / n;
            double m4 = s4 / n;
            return Math.sqrt(Math.max(0.0, m4 - m2 * m2) / n);
        }

        /**
         * Pinned type-7 quantile ({@link Hedging#quantile}) of a statistic over paths.
         *
         * @param stat the statistic
         * @param p    probability, in {@code [0, 1]}
         * @return the quantile
         */
        public double quantile(Stat stat, double p) {
            return Hedging.quantile(arr(stat), p);
        }

        /**
         * Largest attribution residual over paths, relative to {@code max(1, |pnl|)}.
         *
         * @return the residual
         */
        public double maxIdentityError() {
            return maxIdentityError;
        }
    }

    /**
     * B shares to trade under the pinned hedging rule (API_SPEC 14.2): with
     * {@code u = q + y/hedgeRatio}, {@code dy = -hedgeRatio*(1 - hedgeTarget)*u} when
     * {@code |u| > hedgeThreshold}, else 0; {@code hedgeRatio = 0} returns 0.
     *
     * @param q              A position, finite
     * @param y              B position, finite
     * @param hedgeRatio     B shares per A share, finite and {@code >= 0}
     * @param hedgeThreshold finite and {@code >= 0}
     * @param hedgeTarget    in {@code [0, 1)}
     * @return the B trade
     */
    public static double hedgeTrade(double q, double y, double hedgeRatio, double hedgeThreshold,
                                    double hedgeTarget) {
        require(Double.isFinite(q) && Double.isFinite(y), "q and y must be finite");
        require(Double.isFinite(hedgeRatio) && hedgeRatio >= 0.0,
                "hedge_ratio must be finite and >= 0");
        require(Double.isFinite(hedgeThreshold) && hedgeThreshold >= 0.0,
                "hedge_threshold must be finite and >= 0");
        require(Double.isFinite(hedgeTarget) && hedgeTarget >= 0.0 && hedgeTarget < 1.0,
                "hedge_target must be in [0, 1)");
        if (hedgeRatio == 0.0) {
            return 0.0;
        }
        double u = q + y / hedgeRatio;
        if (Math.abs(u) <= hedgeThreshold) {
            return 0.0;
        }
        return -hedgeRatio * (1.0 - hedgeTarget) * u;
    }

    /**
     * Pinned sample quantile (API_SPEC 14.6), Hyndman and Fan (1996) type 7: sort to
     * {@code v}, {@code h = (n - 1)p}, {@code j = floor(h)};
     * {@code Q = v[j] + (h - j)(v[j+1] - v[j])}, or {@code v[n-1]} when {@code j >= n - 1}.
     *
     * @param values non-empty, finite
     * @param p      probability, in {@code [0, 1]}
     * @return the quantile
     */
    public static double quantile(double[] values, double p) {
        require(values != null && values.length >= 1, "values must be a non-empty 1-D array");
        double[] v = values.clone();
        for (double x : v) {
            require(Double.isFinite(x), "values must be finite");
        }
        require(Double.isFinite(p) && p >= 0.0 && p <= 1.0, "p must be in [0, 1]");
        Arrays.sort(v);
        int n = v.length;
        double h = (n - 1) * p;
        int j = (int) Math.floor(h);
        if (j >= n - 1) {
            return v[n - 1];
        }
        return v[j] + (h - j) * (v[j + 1] - v[j]);
    }

    /**
     * Simulates the Section 9 maker with a hedge and risk controls (API_SPEC 14.4; loop
     * order pinned, see the class javadoc).  Deterministic per (seed, configs, nPaths);
     * uses {@link SplittableRandom}, so sample values differ from the Python reference
     * while agreeing statistically.  With default hedge and risk configs the result equals
     * Section 9 in distribution.
     *
     * @param params model parameters
     * @param nPaths number of paths, in {@code [2, MAX_AS_PATHS]}
     * @param seed   RNG seed
     * @param config frictions and agent ({@link Config#defaults()} for none)
     * @param hedge  hedge configuration ({@link HedgeConfig#defaults()} for none)
     * @param risk   risk controls ({@link RiskConfig#defaults()} for none)
     * @return per-path attribution arrays
     */
    public static Result simulateHedged(ASParams params, int nPaths, long seed, Config config,
                                        HedgeConfig hedge, RiskConfig risk) {
        require(params != null, "params must not be null");
        require(config != null, "config must not be null");
        require(hedge != null, "hedge must not be null");
        require(risk != null, "risk must not be null");
        require(nPaths >= 2 && nPaths <= AvellanedaStoikov.MAX_AS_PATHS,
                "n_paths must be an integer in [2, 10000000]");
        int nSteps = params.nSteps();
        int lat = config.latencySteps();
        require(lat <= nSteps, "latency_steps must be <= n_steps");
        require(risk.killStep() <= nSteps, "kill_step must be <= n_steps");
        require(risk.throttleWindow() <= nSteps, "throttle_window must be <= n_steps");

        SplittableRandom rng = new SplittableRandom(seed);
        double g = params.gamma();
        double sig = params.sigma();
        double sig2 = sig * sig;
        double bigT = params.horizon();
        double dt = params.dt();
        double k = params.k();
        double aRate = params.aRate();
        double qMax = params.qMax();
        double s0 = params.s0();
        double sqdt = Math.sqrt(dt);
        double markup = 2.0 / g * Math.log1p(g / k);
        double fixedHalf = 0.5 * (config.fixedSpread() != null ? config.fixedSpread()
                : 0.5 * g * sig2 * bigT + markup);
        double c1 = Math.log1p(g / k) / g;
        double c2 = Math.sqrt(sig2 * g / (2.0 * k * aRate) * Math.pow(1.0 + g / k, 1.0 + k / g));
        double shiftScale = (2.0 * config.signalAccuracy() - 1.0) * sig * sqdt * SQRT_2_OVER_PI;
        double tick = config.tick();
        double fee = config.makerFee();
        Agent agent = config.agent();
        double h = hedge.hedgeRatio();
        double rho = hedge.rho();
        double rhoPerp = Math.sqrt(Math.max(0.0, 1.0 - rho * rho));
        double hCost = hedge.hedgeCost();
        int m = risk.throttleUpdates();
        int w = risk.throttleWindow();
        double pen = risk.flattenPenalty();
        double lossLimit = risk.lossLimit();
        int posLimit = risk.positionLimit();
        int killStep = risk.killStep();

        Map<Stat, double[]> out = new EnumMap<>(Stat.class);
        for (Stat st : Stat.values()) {
            out.put(st, new double[nPaths]);
        }
        double maxErr = 0.0;
        double[] hist = new double[lat + 1];
        // ring of the steps of the last m quote updates, initialised very negative
        int[] upd = new int[Math.max(m, 1)];

        for (int p = 0; p < nPaths; p++) {
            double s = s0;
            double b = s0;
            Arrays.fill(hist, s);
            int head = 0;
            Arrays.fill(upd, Integer.MIN_VALUE);
            int ptr = 0;
            double q = 0.0;
            double y = 0.0;
            double cash = 0.0;
            boolean active = true;
            double bidQ = s0;
            double askQ = s0;
            double markPrev = 0.0;
            double spread = 0.0;
            double fees = 0.0;
            double fills = 0.0;
            double informedFills = 0.0;
            double inventory = 0.0;
            double markout = 0.0;
            double hedgePnl = 0.0;
            double hedgeCosts = 0.0;
            double hedgeTrades = 0.0;
            double riskCosts = 0.0;
            double quoteUpdates = 0.0;
            double lossStop = 0.0;
            double killStop = 0.0;
            double posFlattens = 0.0;
            double overshoot = 0.0;
            double stopMove = 0.0;
            double sumQ2 = 0.0;
            double sumX2 = 0.0;
            double sumSkew2 = 0.0;
            for (int i = 0; i < nSteps; i++) {
                double t = i * dt;
                // 1. six draws, always, in the pinned order
                double z = rng.nextGaussian();
                double uSig = rng.nextDouble();
                double uAsk = rng.nextDouble();
                double uBid = rng.nextDouble();
                double uInf = rng.nextDouble();
                double zB = rng.nextGaussian();
                // 2. stops: kill switch first, then the loss limit on the marked PnL
                double mark = cash + q * s + y * b;
                boolean kill = active && i == killStep;
                boolean loss = active && !kill && lossLimit > 0.0 && mark < -lossLimit;
                if (kill) {
                    killStop += 1.0;
                }
                if (loss) {
                    lossStop += 1.0;
                    overshoot = Math.max(0.0, -lossLimit - mark);
                    stopMove = Math.abs(mark - markPrev);
                }
                if (kill || loss) {
                    double costA = pen * Math.abs(q);
                    cash += q * s - costA;
                    riskCosts += costA;
                    q = 0.0;
                    double costB = hCost * Math.abs(y);
                    cash += y * b - costB;
                    hedgeCosts += costB;
                    if (y != 0.0) {
                        hedgeTrades += 1.0;
                    }
                    y = 0.0;
                    active = false;
                }
                markPrev = mark;
                // 3. desired quotes, skewed on the risk-equivalent exposure x
                double x = q + rho * y;
                double sz = Math.signum(z);
                double pred = uSig < config.signalAccuracy() ? sz : -sz;
                double fair = hist[head] + shiftScale * pred;
                double skew;
                double bid;
                double ask;
                if (agent == Agent.AS) {
                    skew = x * g * sig2 * (bigT - t);
                    double half = 0.5 * (g * sig2 * (bigT - t) + markup);
                    bid = fair - skew - half;
                    ask = fair - skew + half;
                } else if (agent == Agent.GLFT) {
                    skew = x * c2;
                    bid = fair - (c1 + (2.0 * x + 1.0) / 2.0 * c2);
                    ask = fair + (c1 - (2.0 * x - 1.0) / 2.0 * c2);
                } else {
                    skew = 0.0;
                    bid = fair - fixedHalf;
                    ask = fair + fixedHalf;
                }
                if (tick > 0.0) {
                    bid = Math.floor(bid / tick + 1e-9) * tick;
                    ask = Math.ceil(ask / tick - 1e-9) * tick;
                }
                if (active) {
                    sumSkew2 += skew * skew;
                }
                // 4. throttle: at most m updates in any w consecutive steps
                boolean allow;
                if (m > 0) {
                    allow = active && upd[ptr] <= i - w;
                    if (allow) {
                        upd[ptr] = i;
                        ptr = (ptr + 1) % m;
                    }
                } else {
                    allow = active;
                }
                if (allow) {
                    bidQ = bid;
                    askQ = ask;
                    quoteUpdates += 1.0;
                }
                // 5. Section 9 fills against the standing quotes
                boolean canAsk = active && q > -qMax;
                boolean canBid = active && q < qMax;
                double pAsk = -Math.expm1(-aRate * Math.exp(-k * (askQ - s)) * dt);
                double pBid = -Math.expm1(-aRate * Math.exp(-k * (s - bidQ)) * dt);
                boolean fillAsk = canAsk && uAsk < pAsk;
                boolean fillBid = canBid && uBid < pBid;
                double move = sig * sqdt * z;
                boolean informed = uInf < config.informedProb();
                boolean infAsk = informed && z > 0.0 && canAsk && !fillAsk && askQ < s + move;
                boolean infBid = informed && z < 0.0 && canBid && !fillBid && bidQ > s + move;
                boolean fa = fillAsk || infAsk;
                boolean fb = fillBid || infBid;
                double faF = fa ? 1.0 : 0.0;
                double fbF = fb ? 1.0 : 0.0;
                double nFill = faF + fbF;
                q = q + fbF - faF;
                cash = cash + (fa ? askQ : 0.0) - (fb ? bidQ : 0.0) + fee * nFill;
                spread += (fa ? askQ - s : 0.0) + (fb ? s - bidQ : 0.0);
                fees += fee * nFill;
                fills += nFill;
                informedFills += (infAsk ? 1.0 : 0.0) + (infBid ? 1.0 : 0.0);
                // 6. position limit: flatten A at the mid less the penalty
                if (active && posLimit > 0 && Math.abs(q) >= posLimit) {
                    double cost = pen * Math.abs(q);
                    cash += q * s - cost;
                    riskCosts += cost;
                    posFlattens += 1.0;
                    q = 0.0;
                }
                // 7. hedge in B at its mid plus the hedge cost
                if (active && h > 0.0) {
                    double u = q + y / h;
                    if (Math.abs(u) > hedge.hedgeThreshold()) {
                        double dy = -h * (1.0 - hedge.hedgeTarget()) * u;
                        double cost = hCost * Math.abs(dy);
                        cash -= dy * b + cost;
                        hedgeCosts += cost;
                        hedgeTrades += 1.0;
                        y = y + dy;
                    }
                }
                // 8. accounting, then the moves of A and B
                double moveB = sig * sqdt * (rho * z + rhoPerp * zB);
                inventory += q * move;
                markout += (fbF - faF) * move;
                hedgePnl += y * moveB;
                double xe = q + rho * y;
                sumQ2 += q * q;
                sumX2 += xe * xe;
                s = s + move;
                b = b + moveB;
                hist[head] = s;
                head = head == lat ? 0 : head + 1;
            }
            double liq = params.etaLiq() * Math.abs(q);
            double unwind = hCost * Math.abs(y);
            hedgeCosts += unwind;
            if (y != 0.0) {
                hedgeTrades += 1.0;
            }
            double pnl = cash + q * s + y * b - liq - unwind;
            out.get(Stat.PNL)[p] = pnl;
            out.get(Stat.SPREAD)[p] = spread;
            out.get(Stat.FEES)[p] = fees;
            out.get(Stat.INVENTORY)[p] = inventory;
            out.get(Stat.LIQUIDATION)[p] = liq;
            out.get(Stat.MARKOUT)[p] = markout;
            out.get(Stat.FILLS)[p] = fills;
            out.get(Stat.INFORMED_FILLS)[p] = informedFills;
            out.get(Stat.RMS_Q)[p] = Math.sqrt(sumQ2 / nSteps);
            out.get(Stat.HEDGE_PNL)[p] = hedgePnl;
            out.get(Stat.HEDGE_COSTS)[p] = hedgeCosts;
            out.get(Stat.HEDGE_TRADES)[p] = hedgeTrades;
            out.get(Stat.RISK_COSTS)[p] = riskCosts;
            out.get(Stat.RMS_X)[p] = Math.sqrt(sumX2 / nSteps);
            out.get(Stat.RMS_SKEW)[p] = Math.sqrt(sumSkew2 / nSteps);
            out.get(Stat.QUOTE_UPDATES)[p] = quoteUpdates;
            out.get(Stat.LOSS_STOP)[p] = lossStop;
            out.get(Stat.KILL_STOP)[p] = killStop;
            out.get(Stat.POS_FLATTENS)[p] = posFlattens;
            out.get(Stat.OVERSHOOT)[p] = overshoot;
            out.get(Stat.STOP_MOVE)[p] = stopMove;
            double recon = spread + fees + inventory - liq + hedgePnl - hedgeCosts - riskCosts;
            maxErr = Math.max(maxErr, Math.abs(recon - pnl) / Math.max(1.0, Math.abs(pnl)));
        }
        return new Result(out, maxErr);
    }
}
