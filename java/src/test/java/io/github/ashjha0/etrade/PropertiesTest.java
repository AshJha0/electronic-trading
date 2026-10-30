package io.github.ashjha0.etrade;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.Random;
import org.junit.Test;

/**
 * Seeded randomized property tests (fixed-seed {@link Random}, a few hundred
 * iterations each) plus the pinned product bounds of API_SPEC 0.
 */
public final class PropertiesTest {

    private static final int ITERS = 300;

    private static double uniform(Random r, double lo, double hi) {
        return lo + (hi - lo) * r.nextDouble();
    }

    private static double logUniform(Random r, double lo, double hi) {
        return Math.exp(uniform(r, Math.log(lo), Math.log(hi)));
    }

    private static void relClose(String what, double expected, double actual, double tol) {
        assertTrue(what + ": " + actual + " vs " + expected,
                Math.abs(actual - expected) <= tol * Math.max(1.0, Math.abs(expected)));
    }

    private static void checkLiquidating(String what, double[] x, double bigX) {
        assertEquals(what + " x_0", bigX, x[0], 1e-15 * bigX);
        assertEquals(what + " x_N", 0.0, x[x.length - 1], 0.0);
        double sum = 0.0;
        for (int j = 1; j < x.length; j++) {
            assertTrue(what + " non-increasing at " + j, x[j] <= x[j - 1]);
            sum += x[j - 1] - x[j];
        }
        relClose(what + " trades sum", bigX, sum, 1e-9);
    }

    private static ACParams randomAc(Random r) {
        double tau = logUniform(r, 0.01, 2.0);
        double eta = logUniform(r, 1e-8, 1e-4);
        double gammaP = r.nextDouble() * 1.9 * eta / tau;
        return new ACParams(logUniform(r, 1.0, 1e8), 1 + r.nextInt(200), tau,
                r.nextInt(10) == 0 ? 0.0 : logUniform(r, 1e-3, 10.0), eta, gammaP,
                r.nextInt(10) == 0 ? 0.0 : logUniform(r, 1e-9, 1e-2));
    }

    @Test
    public void acTwapAndImpactSchedulesLiquidate() {
        Random r = new Random(101L);
        for (int it = 0; it < ITERS; it++) {
            ACParams p = randomAc(r);
            checkLiquidating("ac " + p, AlmgrenChriss.trajectory(p), p.totalShares());
            checkLiquidating("twap", AlmgrenChriss.twapHoldings(p.totalShares(),
                    p.nIntervals()), p.totalShares());
        }
        for (int it = 0; it < 100; it++) {
            double bigX = logUniform(r, 1e3, 1e7);
            int n = 1 + r.nextInt(20);
            double alpha = uniform(r, 0.3, 1.0);
            double tau = logUniform(r, 0.01, 1.0);
            double eta = logUniform(r, 1e-4, 1.0);
            double[] w = null;
            if (r.nextBoolean()) {
                w = new double[n];
                for (int j = 0; j < n; j++) {
                    w[j] = uniform(r, 0.2, 2.0) * tau;
                }
            }
            double minEta = Double.POSITIVE_INFINITY;
            for (int j = 0; j < n; j++) {
                minEta = Math.min(minEta, eta * Math.pow(w == null ? tau : w[j], -alpha));
            }
            double bound = alpha * (1.0 + alpha) * minEta * Math.pow(bigX, alpha - 1.0);
            Impact.Params p = new Impact.Params(bigX, n, tau, logUniform(r, 1e-3, 5.0), eta,
                    alpha, r.nextDouble() * 0.9 * bound,
                    r.nextBoolean() ? 0.0 : logUniform(r, 1e-9, 1e-4),
                    r.nextBoolean() ? 0.0 : logUniform(r, 1e-4, 1.0), w);
            checkLiquidating("impact", Impact.optimalSchedule(p).holdings(), bigX);
        }
    }

    private static Analytics.ISDecomposition decompose(String side, double pd, double p0,
                                                       double pt, double bigX, double[] n,
                                                       double[] px) {
        return Analytics.decomposeShortfall(side, pd, p0, pt, bigX, n, px);
    }

    @Test
    public void isDecompositionIdentityAndShiftInvariance() {
        Random r = new Random(202L);
        for (int it = 0; it < ITERS; it++) {
            String side = r.nextBoolean() ? "buy" : "sell";
            double base = logUniform(r, 1.0, 1e4);
            double pd = base * uniform(r, 0.95, 1.05);
            double p0 = base * uniform(r, 0.95, 1.05);
            double pt = base * uniform(r, 0.95, 1.05);
            double bigX = logUniform(r, 1.0, 1e6);
            int m = r.nextInt(6);
            double[] n = new double[m];
            double[] px = new double[m];
            double left = bigX;
            for (int j = 0; j < m; j++) {
                n[j] = left * r.nextDouble() * 0.5;
                left -= n[j];
                px[j] = base * uniform(r, 0.95, 1.05);
            }
            Analytics.ISDecomposition d = decompose(side, pd, p0, pt, bigX, n, px);
            double scale = bigX * base * 1.1;
            double tol = 1e-9 * Math.max(1.0, scale);
            assertTrue(Math.abs(d.total() - (d.delay() + d.trading() + d.opportunity())) <= tol);
            double sign = "sell".equals(side) ? 1.0 : -1.0;
            double sumNp = 0.0;
            for (int j = 0; j < m; j++) {
                sumNp += n[j] * px[j];
            }
            double identity = sign * (bigX * pd - sumNp - d.remainingQty() * pt);
            assertTrue(Math.abs(d.total() - identity) <= tol);

            double c = uniform(r, -0.5, 2.0) * base;
            double[] pxc = new double[m];
            for (int j = 0; j < m; j++) {
                pxc[j] = px[j] + c;
            }
            Analytics.ISDecomposition e = decompose(side, pd + c, p0 + c, pt + c, bigX, n, pxc);
            double stol = 1e-9 * Math.max(1.0, bigX * (base + Math.abs(c)) * 1.1);
            assertTrue(Math.abs(d.delay() - e.delay()) <= stol);
            assertTrue(Math.abs(d.trading() - e.trading()) <= stol);
            assertTrue(Math.abs(d.opportunity() - e.opportunity()) <= stol);
            assertTrue(Math.abs(d.total() - e.total()) <= stol);
        }
    }

    @Test
    public void impactCostNonDecreasingInSize() {
        Random r = new Random(303L);
        for (int it = 0; it < ITERS; it++) {
            double x1 = logUniform(r, 1.0, 1e7);
            double x2 = x1 * uniform(r, 1.0, 10.0);
            double vol = logUniform(r, 1e-3, 0.1);
            double adv = logUniform(r, 1e4, 1e8);
            double y = uniform(r, 0.1, 2.0);
            assertTrue(Impact.sqrtLawCostBps(vol, x1, adv, y) * x1
                    <= Impact.sqrtLawCostBps(vol, x2, adv, y) * x2);

            ACParams p = randomAc(r);
            ACParams p1 = new ACParams(x1, p.nIntervals(), p.tau(), p.sigma(), p.eta(),
                    p.gammaP(), p.lambdaRa());
            ACParams p2 = new ACParams(x2, p.nIntervals(), p.tau(), p.sigma(), p.eta(),
                    p.gammaP(), p.lambdaRa());
            assertTrue(AlmgrenChriss.instantaneousCost(p1).eCost()
                    <= AlmgrenChriss.instantaneousCost(p2).eCost());
            assertTrue(AlmgrenChriss.expectedCost(p1, AlmgrenChriss.twapHoldings(x1,
                    p.nIntervals())) <= AlmgrenChriss.expectedCost(p2,
                    AlmgrenChriss.twapHoldings(x2, p.nIntervals())));
        }
    }

    @Test
    public void bpsIsLinear() {
        Random r = new Random(404L);
        for (int it = 0; it < ITERS; it++) {
            double a = uniform(r, -1e6, 1e6);
            double b = uniform(r, -1e6, 1e6);
            double n = logUniform(r, 1.0, 1e9);
            double k = uniform(r, -100.0, 100.0);
            double ba = Market.bps(a, n);
            double bb = Market.bps(b, n);
            assertTrue(Math.abs(Market.bps(a + b, n) - (ba + bb))
                    <= 1e-12 * (Math.abs(ba) + Math.abs(bb)));
            relClose("k*a", k * ba, Market.bps(k * a, n), 1e-12);
        }
    }

    @Test
    public void marketConversions() {
        Random r = new Random(505L);
        double[] dtChoices = {1.0, 5.0, 10.0, 30.0, 60.0};
        for (int it = 0; it < ITERS; it++) {
            double price = logUniform(r, 1.0, 1e4);
            double spread = price * logUniform(r, 1e-5, 1e-2);
            double adv = logUniform(r, 1e4, 1e8);
            double dv = logUniform(r, 1e-3, 0.1);
            double tau = logUniform(r, 0.01, 1.0);
            double tempFrac = uniform(r, 0.005, 0.05);
            double permFrac = uniform(r, 0.05, 1.0);
            ACParams ac = Market.acParamsFromMarket(new Market.ACMarket(price, spread, adv, dv,
                    logUniform(r, 1.0, 1e6), 1 + r.nextInt(50), tau, 1e-6, tempFrac, permFrac));
            relClose("eta", spread / (tempFrac * adv), ac.eta(), 1e-12);
            relClose("gammaP", spread / (permFrac * adv), ac.gammaP(), 1e-12);
            relClose("sigma", price * dv, ac.sigma(), 1e-12);

            double tick = price * logUniform(r, 1e-5, 1e-3);
            double touch = logUniform(r, 10.0, 1e4);
            double halving = uniform(r, 0.5, 10.0);
            double skew = uniform(r, 0.1, 5.0);
            ASParams as = Market.asParamsFromMarket(new Market.ASMarket(price, tick, dv, touch,
                    halving, skew, 1 + r.nextInt(20), uniform(r, 0.0, 3.0),
                    dtChoices[r.nextInt(dtChoices.length)], 23_400.0, 1 + r.nextInt(3)));
            relClose("touch", touch, AvellanedaStoikov.intensity(as, tick / 2.0), 1e-9);
            double d = uniform(r, 0.0, 5.0) * tick;
            relClose("halving", 0.5 * AvellanedaStoikov.intensity(as, d),
                    AvellanedaStoikov.intensity(as, d + halving * tick), 1e-9);
            double s = as.s0();
            double skewGot = s - AvellanedaStoikov.reservationPrice(as, s, 1.0, 0.0);
            assertTrue("skew", Math.abs(skewGot - skew * tick) <= 1e-12 * price);
        }
    }

    private static final double[] DOUBLES = {
        Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 1e308, -1e308,
        -1.0, 0.0, 1e-300, 1e-6, 0.5, 1.0, 100.0, 1e6};
    private static final int[] INTS = {-1, 0, 1, 5, 13, Integer.MAX_VALUE, 10_000_001};

    @FunctionalInterface
    private interface Call {
        void run();
    }

    private static void allowedOutcome(String what, Call c) {
        try {
            c.run();
        } catch (IllegalArgumentException e) {
            // documented input error
        } catch (Throwable t) {
            fail(what + " threw " + t);
        }
    }

    @Test
    public void validationNeverCrashes() {
        Random r = new Random(606L);
        for (int it = 0; it < 400; it++) {
            double[] d = new double[13];
            for (int i = 0; i < d.length; i++) {
                d[i] = r.nextInt(4) == 0 ? uniform(r, -10.0, 1e3)
                        : DOUBLES[r.nextInt(DOUBLES.length)];
            }
            int ni = INTS[r.nextInt(INTS.length)];
            int qm = INTS[r.nextInt(INTS.length)];
            allowedOutcome("ACParams", () -> new ACParams(d[0], ni, d[1], d[2], d[3], d[4],
                    d[5]));
            allowedOutcome("ASParams", () -> new ASParams(d[0], d[1], d[2], d[3], d[4], d[5],
                    d[6], qm, d[7]));
            allowedOutcome("povHoldings", () -> AlmgrenChriss.povHoldings(d[0], d[1], d[2],
                    d[3]));
            allowedOutcome("twapHoldings", () -> AlmgrenChriss.twapHoldings(d[0], ni));
            allowedOutcome("decomposeShortfall", () -> Analytics.decomposeShortfall(
                    r.nextBoolean() ? "buy" : "sell", d[0], d[1], d[2], d[3],
                    new double[] {d[4], d[5]}, new double[] {d[6], d[7]}));
            allowedOutcome("bps", () -> Market.bps(d[0], d[1]));
            allowedOutcome("acParamsFromMarket", () -> Market.acParamsFromMarket(
                    new Market.ACMarket(d[0], d[1], d[2], d[3], d[4], ni, d[5], d[6], d[7],
                            d[8])));
            allowedOutcome("asParamsFromMarket", () -> Market.asParamsFromMarket(
                    new Market.ASMarket(d[0], d[1], d[2], d[3], d[4], d[5], qm, d[6], d[7],
                            d[8], d[9])));
            allowedOutcome("Impact.Params", () -> new Impact.Params(d[0], ni, d[1], d[2],
                    d[3], d[4], d[5], d[6], d[7], null));
            allowedOutcome("impactParamsFromMarket", () -> Market.impactParamsFromMarket(
                    new Market.ImpactMarket(d[0], d[1], d[2], d[3], d[4], ni, d[5], d[6],
                            d[7], d[8], d[9], d[10], null)));
        }
    }

    // ---- product bounds (API_SPEC 0) ----

    @Test
    public void productBounds() {
        assertEquals(1_000_000_000L, AlmgrenChriss.MAX_MC_WORK);
        assertEquals(10_000_000_000L, AvellanedaStoikov.MAX_AS_WORK);

        ACParams ac = new ACParams(1000.0, 101, 1.0, 0.5, 1e-4, 1e-6, 0.0);
        double[] h = AlmgrenChriss.twapHoldings(1000.0, 101);   // 1e7 * 101 > 1e9
        try {
            AlmgrenChriss.simulateIS(ac, h, AlmgrenChriss.MAX_PATHS, 1L);
            fail("expected product bound rejection");
        } catch (IllegalArgumentException e) {
            assertEquals("n_paths * n_intervals must be <= 1000000000", e.getMessage());
        }
        assertEquals(10, AlmgrenChriss.simulateIS(ac, h, 10, 1L).nPaths());

        ASParams as = new ASParams(100.0, 1.0, 1001.0, 1.0, 0.1, 1.5, 1.0, 2, 0.0);
        try {   // 1e7 * 1001 > 1e10
            AvellanedaStoikov.simulate(as, AvellanedaStoikov.MAX_AS_PATHS, 1L);
            fail("expected product bound rejection");
        } catch (IllegalArgumentException e) {
            assertEquals("n_paths * n_steps must be <= 10000000000", e.getMessage());
        }
        assertEquals(3, AvellanedaStoikov.simulate(as, 3, 1L).nPaths());
    }
}
