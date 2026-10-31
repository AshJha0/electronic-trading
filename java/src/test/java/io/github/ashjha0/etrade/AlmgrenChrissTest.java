package io.github.ashjha0.etrade;

import static io.github.ashjha0.etrade.Rejects.assertRejects;
import static io.github.ashjha0.etrade.Rejects.bad;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/** Unit tests for the Almgren-Chriss module (properties, edge cases, errors). */
final class AlmgrenChrissTest {

    private static final ACParams LIQUID =
            new ACParams(1_000_000.0, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6);

    @Test
    void trajectoryMonotoneDecreasingAndSumsToX() {
        // property-style: whole risk-aversion grid
        for (double lam : new double[] {0.0, 1e-8, 5e-7, 2e-6, 1e-5, 1e-3}) {
            ACParams p = LIQUID.withLambda(lam);
            double[] x = AlmgrenChriss.trajectory(p);
            assertEquals(p.nIntervals() + 1, x.length);
            // x_0 == X exactly: the expm1 ratio is formed before the product
            assertEquals(p.totalShares(), x[0], 0.0);
            assertEquals(0.0, x[x.length - 1], 0.0);
            for (int j = 1; j < x.length; j++) {
                assertTrue(x[j] <= x[j - 1] + 1e-12, "holdings must decrease (sell program), lambda=" + lam);
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
    void higherRiskAversionFrontLoadsLowersVarRaisesE() {
        ACParams lo = LIQUID.withLambda(5e-7);
        ACParams hi = LIQUID.withLambda(5e-6);
        assertTrue(AlmgrenChriss.tradeList(hi)[0] > AlmgrenChriss.tradeList(lo)[0]);
        assertTrue(AlmgrenChriss.varianceCost(hi) < AlmgrenChriss.varianceCost(lo));
        assertTrue(AlmgrenChriss.expectedCost(hi) > AlmgrenChriss.expectedCost(lo));
    }

    @Test
    void tradeListSatisfiesCoshIdentity() {
        double[] n = AlmgrenChriss.tradeList(LIQUID);
        double kappa = AlmgrenChriss.kappa(LIQUID);
        double bigT = LIQUID.horizon();
        double tau = LIQUID.tau();
        double pre = 2.0 * LIQUID.totalShares() * Math.sinh(0.5 * kappa * tau)
                / Math.sinh(kappa * bigT);
        for (int j = 1; j <= n.length; j++) {
            double expected = pre * Math.cosh(kappa * (bigT - (j - 0.5) * tau));
            assertEquals(expected, n[j - 1],
                    1e-10 * Math.max(1.0, Math.abs(expected)), "n_" + j);
        }
    }

    @Test
    void kappaLimitContinuityNearThreshold() {
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
    void lambdaZeroGivesTwapAndInfiniteHalfLife() {
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
    void zeroSharesAndSingleClipEdgeCases() {
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
    void instantaneousCostMatchesFormulaAndVarZero() {
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
    void povScheduleShape() {
        double[] pov = AlmgrenChriss.povHoldings(1_000_000.0, 1.0, 2_500_000.0, 0.15);
        assertEquals(4, pov.length);            // 3 intervals used
        assertEquals(1_000_000.0, pov[0], 0.0);
        assertEquals(625_000.0, pov[1], 1e-9);
        assertEquals(250_000.0, pov[2], 1e-9);
        assertEquals(0.0, pov[3], 0.0);
    }

    @Test
    void closedFormsMatchMonteCarloWithinBand() {
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
    void simulationIsDeterministicPerSeed() {
        AlmgrenChriss.ISSimResult a = AlmgrenChriss.simulateIS(LIQUID, 500, 99);
        AlmgrenChriss.ISSimResult b = AlmgrenChriss.simulateIS(LIQUID, 500, 99);
        assertEquals(a.mean(), b.mean(), 0.0);
        assertEquals(a.var(), b.var(), 0.0);
    }

    @Test
    void frontierIsMonotoneInLambda() {
        AlmgrenChriss.FrontierPoint[] fps = AlmgrenChriss.efficientFrontier(LIQUID,
                new double[] {1e-8, 1e-7, 1e-6, 1e-5});
        for (int i = 1; i < fps.length; i++) {
            assertTrue(fps[i].eCost() >= fps[i - 1].eCost());
            assertTrue(fps[i].varCost() <= fps[i - 1].varCost());
            assertEquals(Math.sqrt(fps[i].varCost()), fps[i].stdCost(), 1e-12);
        }
    }

    @ParameterizedTest(name = "[{index}] {0}")
    @MethodSource("invalidInputsThrowCases")
    void invalidInputsThrow(String fragment, Executable call) {
        assertRejects(fragment, call);
    }

    static List<Arguments> invalidInputsThrowCases() {
        List<Arguments> c = new ArrayList<>();
        // model validity: gamma_p*tau >= 2*eta
        c.add(bad("gammaP*tau",
                () -> new ACParams(1e6, 5, 1.0, 0.95, 2.5e-6, 5.0e-6, 2e-6)));
        c.add(bad("totalShares",
                () -> new ACParams(-1.0, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6)));
        c.add(bad("nIntervals",
                () -> new ACParams(1e6, 0, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6)));
        c.add(bad("tau",
                () -> new ACParams(1e6, 5, 0.0, 0.95, 2.5e-6, 2.5e-7, 2e-6)));
        c.add(bad("sigma",
                () -> new ACParams(1e6, 5, 1.0, -0.1, 2.5e-6, 2.5e-7, 2e-6)));
        c.add(bad("eta",
                () -> new ACParams(1e6, 5, 1.0, 0.95, 0.0, 2.5e-7, 2e-6)));
        c.add(bad("lambdaRa",
                () -> new ACParams(1e6, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, -1e-6)));
        // frontier grid must be non-empty
        c.add(bad("lambdas",
                () -> AlmgrenChriss.efficientFrontier(LIQUID, new double[0])));
        // POV participation open interval, positive volume
        c.add(bad("participation",
                () -> AlmgrenChriss.povHoldings(1e6, 1.0, 2.5e6, 0.0)));
        c.add(bad("participation",
                () -> AlmgrenChriss.povHoldings(1e6, 1.0, 2.5e6, 1.0)));
        c.add(bad("marketVolumeRate",
                () -> AlmgrenChriss.povHoldings(1e6, 1.0, 0.0, 0.15)));
        // schedules must liquidate
        c.add(bad("holdings",
                () -> AlmgrenChriss.expectedCost(LIQUID, new double[] {1e6, 5e5, 1e3})));
        // nPaths >= 2
        c.add(bad("nPaths",
                () -> AlmgrenChriss.simulateIS(LIQUID, 1, 0)));
        return c;
    }

    // -----------------------------------------------------------------------
    // Robustness and literature tests: overflow-free trajectory, log1p kappa, POV size bounds, liquidation checks, paper closed forms
    // -----------------------------------------------------------------------

    private static ACParams fxSet(int nIntervals) {
        return new ACParams(5e7, nIntervals, 0.1, 0.006, 2e-9, 5e-11, 1e-4);
    }

    private static void expectFiniteLargeKappaT(ACParams p, int nInt) {
        double kappa = AlmgrenChriss.kappa(p);
        assertTrue(kappa * p.horizon() > 710.0, "sinh would overflow here");
        double[] x = AlmgrenChriss.trajectory(p);
        assertEquals(nInt + 1, x.length);
        assertEquals(p.totalShares(), x[0], 0.0);
        assertEquals(0.0, x[nInt], 0.0);
        double sumN = 0.0;
        int normal = 0;
        for (int j = 0; j < x.length; j++) {
            assertTrue(Double.isFinite(x[j]), "x_" + j + " must be finite");
            if (j > 0) {
                assertTrue(x[j] <= x[j - 1]);
                sumN += x[j - 1] - x[j];
                if (x[j] > Double.MIN_NORMAL * x[0]) {
                    assertTrue(x[j] < x[j - 1], "strictly decreasing while normal, j=" + j);
                    normal++;
                }
            }
        }
        assertTrue(normal >= 800);
        assertEquals(p.totalShares(), sumN, 1e-12 * p.totalShares());
        double r = Math.exp(-kappa * p.tau());
        assertEquals(r, x[1] / x[0], 1e-9 * r);
        assertTrue(Double.isFinite(AlmgrenChriss.expectedCost(p)));
        assertTrue(Double.isFinite(AlmgrenChriss.varianceCost(p)));
        for (AlmgrenChriss.FrontierPoint fp : AlmgrenChriss.efficientFrontier(p,
                new double[] {1e-8, 1e-6, 1e-4, 1.0})) {
            assertTrue(Double.isFinite(fp.eCost()) && Double.isFinite(fp.stdCost())
                    && Double.isFinite(fp.kappa()));
        }
    }

    @Test
    void trajectoryFiniteForLargeKappaT() {
        // kappa*T ~ 846 and ~2683: NaN with the textbook sinh ratio.
        expectFiniteLargeKappaT(new ACParams(1e6, 1000, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6), 1000);
        expectFiniteLargeKappaT(fxSet(20000), 20000);
    }

    @Test
    void paperClosedFormsMatchDiscreteSums() {
        // Almgren-Chriss (2000): closed-form E and V of the optimal strategy
        // (with the discrete kappa) equal the exact sums to 1e-12.
        ACParams illiquid = new ACParams(5e5, 10, 0.5, 1.5, 1e-5, 5e-7, 5e-6);
        for (ACParams p : new ACParams[] {LIQUID, illiquid, fxSet(10)}) {
            double k = AlmgrenChriss.kappa(p);
            double bigT = p.horizon();
            double tau = p.tau();
            double x = p.totalShares();
            double sh = Math.sinh(k * bigT);
            double ePaper = 0.5 * p.gammaP() * x * x
                    + p.etaTilde() * x * x * Math.tanh(0.5 * k * tau)
                    * (tau * Math.sinh(2 * k * bigT) + 2 * bigT * Math.sinh(k * tau))
                    / (2 * tau * tau * sh * sh);
            double vPaper = 0.5 * p.sigma() * p.sigma() * x * x
                    * (tau * sh * Math.cosh(k * (bigT - tau)) - bigT * Math.sinh(k * tau))
                    / (sh * sh * Math.sinh(k * tau));
            assertEquals(ePaper, AlmgrenChriss.expectedCost(p), 1e-12 * ePaper);
            assertEquals(vPaper, AlmgrenChriss.varianceCost(p), 1e-12 * vPaper);
        }
    }

    @Test
    void kappaSmallTauAccuracy() {
        // kappa = kappa_tilde*(1 - kappa_tilde^2 tau^2/24 + O(tau^4)).
        for (double lam : new double[] {1e-6, 1e-12}) {
            ACParams p = new ACParams(1e6, 5000, 1e-3, 0.95, 2.5e-6, 2.5e-7, lam);
            double kt2 = lam * p.sigma() * p.sigma() / p.etaTilde();
            double series = Math.sqrt(kt2) * (1.0 - kt2 * p.tau() * p.tau() / 24.0);
            assertEquals(series, AlmgrenChriss.kappa(p), 1e-9 * series);
            assertEquals(1.0 / series, AlmgrenChriss.halfLife(p), 1e-9 / series);
            assertEquals(AlmgrenChriss.halfLife(p), AlmgrenChriss.characteristicTime(p), 0.0);
        }
    }

    @Test
    void povClipTooSmallIsInputError() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> AlmgrenChriss.povHoldings(1e6, 1.0, 1.0, 1e-9));
        assertTrue(e.getMessage().contains("POV clip too small"));
        assertThrows(IllegalArgumentException.class,
                () -> AlmgrenChriss.povHoldings(AlmgrenChriss.MAX_POV_INTERVALS + 1.0,
                        1.0, 1.0, 0.5));
    }

    @Test
    void povBoundaryIntervalCounts() {
        double clip = 0.15 * 2.5e6;   // 375,000
        double[] x2 = AlmgrenChriss.povHoldings(750_000.0, 1.0, 2.5e6, 0.15);
        assertEquals(3, x2.length);    // X/clip = 2 exactly -> 2 intervals
        assertEquals(375_000.0, x2[1], 0.0);
        assertEquals(0.0, x2[2], 0.0);
        double[] x3 = AlmgrenChriss.povHoldings(750_000.001, 1.0, 2.5e6, 0.15);
        assertEquals(4, x3.length);    // one share-thousandth more -> 3 intervals
        double[] n3 = AlmgrenChriss.tradesFromHoldings(x3);
        assertEquals(clip, n3[0], 0.0);
        assertEquals(clip, n3[1], 0.0);
        assertEquals(0.001, n3[2], 1e-6);
        assertEquals(0.0, x3[3], 0.0);
    }

    @Test
    void simulateIsRejectsNonLiquidatingSchedule() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> AlmgrenChriss.simulateIS(LIQUID, new double[] {1e6, 5e5, 1e3}, 10, 0L));
        assertTrue(e.getMessage().contains("liquidate"));
        assertThrows(IllegalArgumentException.class,
                () -> AlmgrenChriss.simulateIS(LIQUID, new double[] {1e6, Double.NaN, 0.0},
                        10, 0L));
        assertThrows(IllegalArgumentException.class,
                () -> AlmgrenChriss.simulateIS(LIQUID, new double[] {1e6}, 10, 0L));
        assertThrows(IllegalArgumentException.class,
                () -> AlmgrenChriss.simulateIS(LIQUID, AlmgrenChriss.MAX_PATHS + 1, 0L));
        assertThrows(IllegalArgumentException.class,
                () -> AlmgrenChriss.simulateIS(LIQUID, Integer.MAX_VALUE, 0L));
    }

    @Test
    void frontierRejectsNonFiniteLambdaAndKeepsOrder() {
        assertThrows(IllegalArgumentException.class,
                () -> AlmgrenChriss.efficientFrontier(LIQUID, new double[] {1e-7, Double.NaN}));
        assertThrows(IllegalArgumentException.class,
                () -> AlmgrenChriss.efficientFrontier(LIQUID,
                        new double[] {Double.POSITIVE_INFINITY}));
        assertThrows(IllegalArgumentException.class,
                () -> AlmgrenChriss.efficientFrontier(LIQUID, new double[] {1e-7, -1e-7}));
        AlmgrenChriss.FrontierPoint[] f = AlmgrenChriss.efficientFrontier(LIQUID,
                new double[] {2e-6, 1e-8, 2e-6});
        assertEquals(3, f.length);
        assertEquals(2e-6, f[0].lambdaRa(), 0.0);
        assertEquals(1e-8, f[1].lambdaRa(), 0.0);
        assertEquals(f[0], f[2]);
        assertTrue(f[1].eCost() < f[0].eCost());
    }

    @Test
    void twapAndPovRejectNonFinite() {
        double nan = Double.NaN;
        double inf = Double.POSITIVE_INFINITY;
        assertThrows(IllegalArgumentException.class, () -> AlmgrenChriss.twapHoldings(inf, 5));
        assertThrows(IllegalArgumentException.class, () -> AlmgrenChriss.twapHoldings(nan, 5));
        assertThrows(IllegalArgumentException.class,
                () -> AlmgrenChriss.povHoldings(nan, 1.0, 1e6, 0.1));
        assertThrows(IllegalArgumentException.class,
                () -> AlmgrenChriss.povHoldings(inf, 1.0, 1e6, 0.1));
        assertThrows(IllegalArgumentException.class,
                () -> AlmgrenChriss.povHoldings(1e6, inf, 1e6, 0.1));
        assertThrows(IllegalArgumentException.class,
                () -> AlmgrenChriss.povHoldings(1e6, 1.0, inf, 0.1));
        assertThrows(IllegalArgumentException.class,
                () -> AlmgrenChriss.povHoldings(1e6, 1.0, 1e6, nan));
    }

    @Test
    void sizeBoundsAreInputErrors() {
        assertThrows(IllegalArgumentException.class,
                () -> new ACParams(1e6, AlmgrenChriss.MAX_AC_INTERVALS + 1, 1.0, 0.95,
                        2.5e-6, 2.5e-7, 2e-6));
        assertThrows(IllegalArgumentException.class,
                () -> new ACParams(1e6, Integer.MAX_VALUE, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6));
        assertThrows(IllegalArgumentException.class,
                () -> AlmgrenChriss.twapHoldings(1e6, AlmgrenChriss.MAX_AC_INTERVALS + 1));
        assertThrows(IllegalArgumentException.class,
                () -> AlmgrenChriss.twapHoldings(1e6, Integer.MAX_VALUE));
        assertEquals(AlmgrenChriss.MAX_AC_INTERVALS + 1,
                AlmgrenChriss.twapHoldings(1.0, AlmgrenChriss.MAX_AC_INTERVALS).length);
    }

    @Test
    void costFunctionsValidateHoldings() {
        assertThrows(IllegalArgumentException.class,
                () -> AlmgrenChriss.varianceCost(LIQUID, new double[] {1e6}));
        assertThrows(IllegalArgumentException.class,
                () -> AlmgrenChriss.varianceCost(LIQUID,
                        new double[] {1e6, Double.POSITIVE_INFINITY, 0.0}));
        assertThrows(IllegalArgumentException.class,
                () -> AlmgrenChriss.expectedCost(LIQUID, new double[] {1e6, Double.NaN, 0.0}));
        assertThrows(IllegalArgumentException.class,
                () -> AlmgrenChriss.tradesFromHoldings(null));
        // X is taken from the schedule (x_0), not from params (pinned).
        double[] half = AlmgrenChriss.twapHoldings(5e5, 5);
        double eHalf = 0.5 * LIQUID.gammaP() * 5e5 * 5e5 + LIQUID.etaTilde() * 5e5 * 5e5 / 5.0;
        assertEquals(eHalf, AlmgrenChriss.expectedCost(LIQUID, half), 1e-12 * eHalf);
    }

    @Test
    void trajectoryX0ExactAndSinhIdentityModerate() {
        for (double lam : new double[] {1e-8, 5e-7, 2e-6, 1e-4}) {
            ACParams p = LIQUID.withLambda(lam);
            double k = AlmgrenChriss.kappa(p);
            double bigT = p.horizon();
            double[] x = AlmgrenChriss.trajectory(p);
            assertEquals(p.totalShares(), x[0], 0.0);
            for (int j = 1; j < p.nIntervals(); j++) {
                double ref = p.totalShares() * Math.sinh(k * (bigT - j * p.tau()))
                        / Math.sinh(k * bigT);
                assertEquals(ref, x[j], 1e-14 * ref, "j=" + j);
            }
        }
    }

    @Test
    void povOverflowingClipAndTinyRemainder() {
        assertThrows(IllegalArgumentException.class,
                () -> AlmgrenChriss.povHoldings(1.0, 1e200, 1e200, 0.5));
        double[] x = AlmgrenChriss.povHoldings(1e-13, 1.0, 1.0, 0.5);
        assertEquals(2, x.length);
        assertEquals(1e-13, x[0], 0.0);
        assertEquals(0.0, x[1], 0.0);
    }
}
