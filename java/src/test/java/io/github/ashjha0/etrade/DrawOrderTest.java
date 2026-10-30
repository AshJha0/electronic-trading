package io.github.ashjha0.etrade;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertFalse;

import java.util.Arrays;
import org.junit.Test;

/**
 * Draw-order guards: an in-test reference of the Section 2.2 and Section 9.4
 * loops (same the pinned {@code NormalRng} generator (SplittableRandom uniforms, Marsaglia polar normals), same per-path / per-step nesting)
 * reproduces the library bitwise, and deliberately wrong draw orders diverge.
 */
public final class DrawOrderTest {

    private static final long SEED = 2024L;
    private static final int N_PATHS = 4;

    /** Small case where the cap binds often (qMax = 1, high fill rate), 50 steps. */
    private static ASParams params() {
        return new ASParams(100.0, 2.0, 1.0, 0.02, 0.1, 1.5, 140.0, 1, 0.05);
    }

    private enum AsOrder { RIGHT, BID_FIRST, SKIP_SUPPRESSED }

    private enum MmOrder { RIGHT, SWAP_ASK_BID, Z_LAST }

    /** Reference Section 2.2 loop for the A-S agent, no stress. */
    private static double[] asReference(ASParams p, int nPaths, long seed, AsOrder order) {
        NormalRng rng = new NormalRng(seed);
        int nSteps = p.nSteps();
        double g = p.gamma();
        double sig2 = p.sigma() * p.sigma();
        double bigT = p.horizon();
        double dt = p.dt();
        double markup = 2.0 / g * StrictMath.log1p(g / p.k());
        double sqdt = StrictMath.sqrt(dt);
        double[] pnl = new double[nPaths];
        for (int path = 0; path < nPaths; path++) {
            double s = p.s0();
            double q = 0.0;
            double cash = 0.0;
            for (int i = 0; i < nSteps; i++) {
                double t = i * dt;
                double r = s - q * g * sig2 * (bigT - t);
                double half = 0.5 * (g * sig2 * (bigT - t) + markup);
                double ask = r + half;
                double bid = r - half;
                boolean quoteAsk = q > -p.qMax();
                boolean quoteBid = q < p.qMax();
                double uAsk;
                double uBid;
                if (order == AsOrder.BID_FIRST) {
                    uBid = rng.nextDouble();
                    uAsk = rng.nextDouble();
                } else if (order == AsOrder.SKIP_SUPPRESSED) {
                    uAsk = quoteAsk ? rng.nextDouble() : 1.0;
                    uBid = quoteBid ? rng.nextDouble() : 1.0;
                } else {
                    uAsk = rng.nextDouble();
                    uBid = rng.nextDouble();
                }
                double pAsk = -StrictMath.expm1(-p.aRate() * StrictMath.exp(-p.k() * (ask - s)) * dt);
                double pBid = -StrictMath.expm1(-p.aRate() * StrictMath.exp(-p.k() * (s - bid)) * dt);
                if (quoteAsk && uAsk < pAsk) {
                    q -= 1.0;
                    cash += ask;
                }
                if (quoteBid && uBid < pBid) {
                    q += 1.0;
                    cash -= bid;
                }
                double sigT = p.sigma() * 1.0;
                s += sigT * sqdt * rng.nextNormal();
            }
            pnl[path] = cash + q * s - p.etaLiq() * Math.abs(q);
        }
        return pnl;
    }

    /** Reference Section 9.4 loop (A-S quotes, tick rounding, latency, informed flow). */
    private static double[] mmReference(ASParams p, int nPaths, long seed,
                                        MarketMaking.Config c, MmOrder order) {
        NormalRng rng = new NormalRng(seed);
        int nSteps = p.nSteps();
        int lat = c.latencySteps();
        double g = p.gamma();
        double sig = p.sigma();
        double sig2 = sig * sig;
        double bigT = p.horizon();
        double dt = p.dt();
        double k = p.k();
        double aRate = p.aRate();
        double qMax = p.qMax();
        double sqdt = StrictMath.sqrt(dt);
        double markup = 2.0 / g * StrictMath.log1p(g / k);
        double shiftScale = (2.0 * c.signalAccuracy() - 1.0) * sig * sqdt
                * StrictMath.sqrt(2.0 / Math.PI);
        double tick = c.tick();
        double fee = c.makerFee();
        double[] pnl = new double[nPaths];
        double[] hist = new double[lat + 1];
        for (int path = 0; path < nPaths; path++) {
            double s = p.s0();
            Arrays.fill(hist, s);
            int head = 0;
            double q = 0.0;
            double cash = 0.0;
            for (int i = 0; i < nSteps; i++) {
                double t = i * dt;
                double z;
                double uSig;
                double uAsk;
                double uBid;
                double uInf;
                if (order == MmOrder.Z_LAST) {
                    uSig = rng.nextDouble();
                    uAsk = rng.nextDouble();
                    uBid = rng.nextDouble();
                    uInf = rng.nextDouble();
                    z = rng.nextNormal();
                } else {
                    z = rng.nextNormal();
                    uSig = rng.nextDouble();
                    if (order == MmOrder.SWAP_ASK_BID) {
                        uBid = rng.nextDouble();
                        uAsk = rng.nextDouble();
                    } else {
                        uAsk = rng.nextDouble();
                        uBid = rng.nextDouble();
                    }
                    uInf = rng.nextDouble();
                }
                double sz = Math.signum(z);
                double pred = uSig < c.signalAccuracy() ? sz : -sz;
                double fair = hist[head] + shiftScale * pred;
                double r = fair - q * g * sig2 * (bigT - t);
                double half = 0.5 * (g * sig2 * (bigT - t) + markup);
                double bid = r - half;
                double ask = r + half;
                if (tick > 0.0) {
                    bid = Math.floor(bid / tick + 1e-9) * tick;
                    ask = Math.ceil(ask / tick - 1e-9) * tick;
                }
                boolean canAsk = q > -qMax;
                boolean canBid = q < qMax;
                double pAsk = -StrictMath.expm1(-aRate * StrictMath.exp(-k * (ask - s)) * dt);
                double pBid = -StrictMath.expm1(-aRate * StrictMath.exp(-k * (s - bid)) * dt);
                boolean fillAsk = canAsk && uAsk < pAsk;
                boolean fillBid = canBid && uBid < pBid;
                double move = sig * sqdt * z;
                boolean informed = uInf < c.informedProb();
                boolean infAsk = informed && z > 0.0 && canAsk && !fillAsk && ask < s + move;
                boolean infBid = informed && z < 0.0 && canBid && !fillBid && bid > s + move;
                boolean fa = fillAsk || infAsk;
                boolean fb = fillBid || infBid;
                double faF = fa ? 1.0 : 0.0;
                double fbF = fb ? 1.0 : 0.0;
                double nFill = faF + fbF;
                q = q + fbF - faF;
                cash = cash + (fa ? ask : 0.0) - (fb ? bid : 0.0) + fee * nFill;
                s = s + move;
                hist[head] = s;
                head = head == lat ? 0 : head + 1;
            }
            double liq = p.etaLiq() * Math.abs(q);
            pnl[path] = cash + q * s - liq;
        }
        return pnl;
    }

    private static MarketMaking.Config mmConfig() {
        return MarketMaking.Config.defaults().withTick(0.01).withMakerFee(-0.001)
                .withLatencySteps(2).withInformedProb(0.3).withSignalAccuracy(0.8);
    }

    @Test
    public void asReferenceReproducesLibraryBitwise() {
        double[] lib = AvellanedaStoikov.simulate(params(), N_PATHS, SEED).pnlPerPath();
        assertArrayEquals(lib, asReference(params(), N_PATHS, SEED, AsOrder.RIGHT), 0.0);
    }

    @Test
    public void asWrongOrdersDiverge() {
        double[] lib = AvellanedaStoikov.simulate(params(), N_PATHS, SEED).pnlPerPath();
        assertFalse(Arrays.equals(lib,
                asReference(params(), N_PATHS, SEED, AsOrder.BID_FIRST)));
        assertFalse(Arrays.equals(lib,
                asReference(params(), N_PATHS, SEED, AsOrder.SKIP_SUPPRESSED)));
    }

    @Test
    public void mmReferenceReproducesLibraryBitwise() {
        double[] lib = MarketMaking.simulate(params(), N_PATHS, SEED, mmConfig())
                .path(MarketMaking.Stat.PNL);
        assertArrayEquals(lib,
                mmReference(params(), N_PATHS, SEED, mmConfig(), MmOrder.RIGHT), 0.0);
    }

    @Test
    public void mmWrongOrdersDiverge() {
        double[] lib = MarketMaking.simulate(params(), N_PATHS, SEED, mmConfig())
                .path(MarketMaking.Stat.PNL);
        assertFalse(Arrays.equals(lib,
                mmReference(params(), N_PATHS, SEED, mmConfig(), MmOrder.SWAP_ASK_BID)));
        assertFalse(Arrays.equals(lib,
                mmReference(params(), N_PATHS, SEED, mmConfig(), MmOrder.Z_LAST)));
    }
}
