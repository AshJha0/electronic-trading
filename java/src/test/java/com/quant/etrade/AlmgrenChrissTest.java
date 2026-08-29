package com.quant.etrade;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/** Unit tests for the Almgren-Chriss module (properties, edge cases, errors). */
public final class AlmgrenChrissTest {

    private static final ACParams LIQUID =
            new ACParams(1_000_000.0, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6);

    @Test
    public void trajectoryMonotoneDecreasingAndSumsToX() {
        // property-style: whole risk-aversion grid
        for (double lam : new double[] {0.0, 1e-8, 5e-7, 2e-6, 1e-5, 1e-3}) {
            ACParams p = LIQUID.withLambda(lam);
            double[] x = AlmgrenChriss.trajectory(p);
            assertEquals(p.nIntervals() + 1, x.length);
            // x_0 = X up to one rounding of the sinh-ratio product (the Python
            // reference evaluates the same expression order)
            assertEquals(p.totalShares(), x[0], 1e-9 * p.totalShares());
            assertEquals(0.0, x[x.length - 1], 0.0);
            for (int j = 1; j < x.length; j++) {
                assertTrue("holdings must decrease (sell program), lambda=" + lam,
                        x[j] <= x[j - 1] + 1e-12);
            }
            double[] n = AlmgrenChriss.tradeList(p);
            double sum = 0.0;
            for (double nj : n) {
                sum += nj;
            }
            assertEquals(p.totalShares(), sum, 1e-12 * p.totalShares());
        }
    }

    @Test
    public void higherRiskAversionFrontLoadsLowersVarRaisesE() {
        ACParams lo = LIQUID.withLambda(5e-7);
        ACParams hi = LIQUID.withLambda(5e-6);
        assertTrue(AlmgrenChriss.tradeList(hi)[0] > AlmgrenChriss.tradeList(lo)[0]);
        assertTrue(AlmgrenChriss.varianceCost(hi) < AlmgrenChriss.varianceCost(lo));
        assertTrue(AlmgrenChriss.expectedCost(hi) > AlmgrenChriss.expectedCost(lo));
    }

    @Test
    public void tradeListSatisfiesCoshIdentity() {
        double[] n = AlmgrenChriss.tradeList(LIQUID);
        double kappa = AlmgrenChriss.kappa(LIQUID);
        double bigT = LIQUID.horizon();
        double tau = LIQUID.tau();
        double pre = 2.0 * LIQUID.totalShares() * Math.sinh(0.5 * kappa * tau)
                / Math.sinh(kappa * bigT);
        for (int j = 1; j <= n.length; j++) {
            double expected = pre * Math.cosh(kappa * (bigT - (j - 0.5) * tau));
            assertEquals("n_" + j, expected, n[j - 1],
                    1e-10 * Math.max(1.0, Math.abs(expected)));
        }
    }

    @Test
    public void kappaLimitContinuityNearThreshold() {
        // lambda tiny enough that kappa*T straddles KAPPA_T_LIMIT: both branches
        // must agree with TWAP to well below golden tolerance.
        double[] twap = AlmgrenChriss.twapHoldings(LIQUID.totalShares(),
                LIQUID.nIntervals());
        for (double lam : new double[] {5e-20, 2e-19}) {   // below / above threshold
            double[] x = AlmgrenChriss.trajectory(LIQUID.withLambda(lam));
            for (int j = 0; j < x.length; j++) {
                assertEquals(twap[j], x[j], 1e-4);   // shares, on a 1e6-share order
            }
        }
        double kBelow = AlmgrenChriss.kappa(LIQUID.withLambda(5e-20));
        double kAbove = AlmgrenChriss.kappa(LIQUID.withLambda(2e-19));
        assertTrue(kBelow * LIQUID.horizon() < AlmgrenChriss.KAPPA_T_LIMIT);
        assertTrue(kAbove * LIQUID.horizon() >= AlmgrenChriss.KAPPA_T_LIMIT);
    }

    @Test
    public void lambdaZeroGivesTwapAndInfiniteHalfLife() {
        ACParams p = LIQUID.withLambda(0.0);
        assertEquals(0.0, AlmgrenChriss.kappa(p), 0.0);
        assertTrue(Double.isInfinite(AlmgrenChriss.halfLife(p)));
        double[] twap = AlmgrenChriss.twapHoldings(p.totalShares(), p.nIntervals());
        double[] x = AlmgrenChriss.trajectory(p);
        for (int j = 0; j < x.length; j++) {
            assertEquals(twap[j], x[j], 0.0);
        }
    }

    @Test
    public void zeroSharesAndSingleClipEdgeCases() {
        ACParams empty = new ACParams(0.0, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6);
        double[] x = AlmgrenChriss.trajectory(empty);
        for (double v : x) {
            assertEquals(0.0, v, 0.0);
        }
        assertEquals(0.0, AlmgrenChriss.expectedCost(empty), 0.0);
        assertEquals(0.0, AlmgrenChriss.varianceCost(empty), 0.0);

        ACParams single = new ACParams(1000.0, 1, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6);
        double[] xs = AlmgrenChriss.trajectory(single);
        assertEquals(2, xs.length);
        assertEquals(1000.0, xs[0], 0.0);
        assertEquals(0.0, xs[1], 0.0);
    }

    @Test
    public void instantaneousCostMatchesFormulaAndVarZero() {
        AlmgrenChriss.CostMoments cm = AlmgrenChriss.instantaneousCost(LIQUID);
        double x0 = LIQUID.totalShares();
        double expected = 0.5 * LIQUID.gammaP() * x0 * x0
                + LIQUID.etaTilde() * x0 * x0 / LIQUID.tau();
        assertEquals(expected, cm.eCost(), 1e-10 * expected);
        assertEquals(0.0, cm.varCost(), 0.0);
        // instant liquidation costs more than any spread-out schedule
        assertTrue(cm.eCost() > AlmgrenChriss.expectedCost(LIQUID));
    }

    @Test
    public void povScheduleShape() {
        double[] pov = AlmgrenChriss.povHoldings(1_000_000.0, 1.0, 2_500_000.0, 0.15);
        assertEquals(4, pov.length);            // 3 intervals used
        assertEquals(1_000_000.0, pov[0], 0.0);
        assertEquals(625_000.0, pov[1], 1e-9);
        assertEquals(250_000.0, pov[2], 1e-9);
        assertEquals(0.0, pov[3], 0.0);
    }

    @Test
    public void closedFormsMatchMonteCarloWithinBand() {
        // 3-4 SE band on both the optimal and the TWAP schedule
        AlmgrenChriss.ISSimResult mc = AlmgrenChriss.simulateIS(LIQUID, 4000, 7);
        double e = AlmgrenChriss.expectedCost(LIQUID);
        double v = AlmgrenChriss.varianceCost(LIQUID);
        assertTrue(Math.abs(mc.mean() - e) <= 4.0 * mc.seMean());
        assertTrue(Math.abs(mc.var() - v) <= 4.0 * mc.seVar());

        double[] twap = AlmgrenChriss.twapHoldings(LIQUID.totalShares(),
                LIQUID.nIntervals());
        AlmgrenChriss.ISSimResult mcT = AlmgrenChriss.simulateIS(LIQUID, twap, 4000, 11);
        assertTrue(Math.abs(mcT.mean() - AlmgrenChriss.expectedCost(LIQUID, twap))
                <= 4.0 * mcT.seMean());
        assertTrue(Math.abs(mcT.var() - AlmgrenChriss.varianceCost(LIQUID, twap))
                <= 4.0 * mcT.seVar());
    }

    @Test
    public void simulationIsDeterministicPerSeed() {
        AlmgrenChriss.ISSimResult a = AlmgrenChriss.simulateIS(LIQUID, 500, 99);
        AlmgrenChriss.ISSimResult b = AlmgrenChriss.simulateIS(LIQUID, 500, 99);
        assertEquals(a.mean(), b.mean(), 0.0);
        assertEquals(a.var(), b.var(), 0.0);
    }

    @Test
    public void frontierIsMonotoneInLambda() {
        AlmgrenChriss.FrontierPoint[] fps = AlmgrenChriss.efficientFrontier(LIQUID,
                new double[] {1e-8, 1e-7, 1e-6, 1e-5});
        for (int i = 1; i < fps.length; i++) {
            assertTrue(fps[i].eCost() >= fps[i - 1].eCost());
            assertTrue(fps[i].varCost() <= fps[i - 1].varCost());
            assertEquals(Math.sqrt(fps[i].varCost()), fps[i].stdCost(), 1e-12);
        }
    }

    @Test
    public void invalidInputsThrow() {
        // model validity: gamma_p*tau >= 2*eta
        assertThrows(IllegalArgumentException.class,
                () -> new ACParams(1e6, 5, 1.0, 0.95, 2.5e-6, 5.0e-6, 2e-6));
        assertThrows(IllegalArgumentException.class,
                () -> new ACParams(-1.0, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6));
        assertThrows(IllegalArgumentException.class,
                () -> new ACParams(1e6, 0, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6));
        assertThrows(IllegalArgumentException.class,
                () -> new ACParams(1e6, 5, 0.0, 0.95, 2.5e-6, 2.5e-7, 2e-6));
        assertThrows(IllegalArgumentException.class,
                () -> new ACParams(1e6, 5, 1.0, -0.1, 2.5e-6, 2.5e-7, 2e-6));
        assertThrows(IllegalArgumentException.class,
                () -> new ACParams(1e6, 5, 1.0, 0.95, 0.0, 2.5e-7, 2e-6));
        assertThrows(IllegalArgumentException.class,
                () -> new ACParams(1e6, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, -1e-6));
        // frontier grid must be non-empty
        assertThrows(IllegalArgumentException.class,
                () -> AlmgrenChriss.efficientFrontier(LIQUID, new double[0]));
        // POV participation open interval, positive volume
        assertThrows(IllegalArgumentException.class,
                () -> AlmgrenChriss.povHoldings(1e6, 1.0, 2.5e6, 0.0));
        assertThrows(IllegalArgumentException.class,
                () -> AlmgrenChriss.povHoldings(1e6, 1.0, 2.5e6, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> AlmgrenChriss.povHoldings(1e6, 1.0, 0.0, 0.15));
        // schedules must liquidate
        assertThrows(IllegalArgumentException.class,
                () -> AlmgrenChriss.expectedCost(LIQUID, new double[] {1e6, 5e5, 1e3}));
        // nPaths >= 2
        assertThrows(IllegalArgumentException.class,
                () -> AlmgrenChriss.simulateIS(LIQUID, 1, 0));
    }
}
