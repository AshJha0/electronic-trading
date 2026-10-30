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

/** Unit tests for the execution-analytics module. */
final class AnalyticsTest {

    @Test
    void decompositionIdentityHoldsForBothSides() {
        double[] qtys = {300.0, 400.0, 200.0};
        // property-style: both sides x several price scenarios
        double[][] scenarios = {
            {100.0, 99.5, 98.8}, {100.0, 100.4, 101.3}, {50.0, 50.0, 50.0},
        };
        for (Side side : new Side[] {Side.SELL, Side.BUY}) {
            for (double[] sc : scenarios) {
                double[] prices = {sc[1] - 0.1, sc[1] + 0.2, sc[1]};
                Analytics.ISDecomposition d = Analytics.decomposeShortfall(side,
                        sc[0], sc[1], sc[2], 1000.0, qtys, prices);
                // exact identity: total == delay + trading + opportunity
                assertEquals(d.total(), d.delay() + d.trading() + d.opportunity(), 0.0);
                // alternative closed form sign*(X*Pd - sum n*p - R*PT)
                double sign = side == Side.SELL ? 1.0 : -1.0;
                double sumNp = 0.0;
                for (int j = 0; j < qtys.length; j++) {
                    sumNp += qtys[j] * prices[j];
                }
                double alt = sign * (1000.0 * sc[0] - sumNp - d.remainingQty() * sc[2]);
                assertEquals(alt, d.total(), 1e-10 * Math.max(1.0, Math.abs(alt)));
                assertEquals(900.0, d.executedQty(), 0.0);
                assertEquals(100.0, d.remainingQty(), 0.0);
            }
        }
    }

    @Test
    void sellDecompositionComponents() {
        Analytics.ISDecomposition d = Analytics.decomposeShortfall(Side.SELL, 100.0, 99.5,
                98.8, 1000.0, new double[] {300.0, 400.0, 200.0},
                new double[] {99.4, 99.2, 99.1});
        assertEquals(450.0, d.delay(), 1e-10);       // 900 * (100 - 99.5)
        assertEquals(230.0, d.trading(), 1e-9);      // slippage vs arrival
        assertEquals(120.0, d.opportunity(), 1e-9);  // 100 * (100 - 98.8)
        assertEquals(800.0, d.total(), 1e-9);
    }

    @Test
    void fullyExecutedOrderHasNoOpportunityCost() {
        Analytics.ISDecomposition d = Analytics.decomposeShortfall(Side.BUY, 10.0, 10.0,
                12.0, 500.0, new double[] {500.0}, new double[] {10.5});
        assertEquals(0.0, d.opportunity(), 0.0);
        assertEquals(0.0, d.remainingQty(), 0.0);
        assertEquals(250.0, d.trading(), 1e-10);
    }

    @Test
    void participationRateBasics() {
        assertEquals(0.08, Analytics.participationRate(120_000.0, 1_500_000.0), 1e-15);
        assertEquals(0.0, Analytics.participationRate(0.0, 1.0), 0.0);
    }

    @Test
    void olsRecoversExactLine() {
        double[] x = {0.0, 1.0, 2.0, 3.0, 4.0};
        double[] y = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = 2.5 - 1.25 * x[i];
        }
        Analytics.OLSResult r = Analytics.ols(x, y);
        assertEquals(-1.25, r.slope(), 1e-12);
        assertEquals(2.5, r.intercept(), 1e-12);
        assertEquals(1.0, r.rSquared(), 1e-12);
        assertEquals(5, r.nObs());
    }

    @Test
    void olsConstantYHasRSquaredOne() {
        // SST = 0 and the fit is exact -> R^2 defined as 1
        Analytics.OLSResult r = Analytics.ols(new double[] {1.0, 2.0, 3.0},
                new double[] {4.0, 4.0, 4.0});
        assertEquals(0.0, r.slope(), 1e-15);
        assertEquals(4.0, r.intercept(), 1e-15);
        assertEquals(1.0, r.rSquared(), 0.0);
    }

    @Test
    void syntheticFillsPinnedConstruction() {
        double[][] f = Analytics.syntheticImpactFills(25, 0.5, 8.0, 0.05);
        assertEquals(25, f[0].length);
        assertEquals(0.02, f[0][0], 1e-15);
        assertEquals(0.20, f[0][24], 1e-15);
        assertEquals(0.5 + 8.0 * 0.02 + 0.05 * Math.sin(0.3), f[1][0], 1e-15);
        // regression on the pinned fills recovers slope near the true value
        Analytics.OLSResult r = Analytics.estimateImpact(f[0], f[1]);
        assertTrue(Math.abs(r.slope() - 8.0) < 0.5);
        assertTrue(r.rSquared() > 0.98);
    }

    @ParameterizedTest(name = "[{index}] {0}")
    @MethodSource("invalidInputsThrowCases")
    void invalidInputsThrow(String fragment, Executable call) {
        assertRejects(fragment, call);
    }

    @SuppressWarnings("deprecation")   // also checks the deprecated String overload
    static List<Arguments> invalidInputsThrowCases() {
        List<Arguments> c = new ArrayList<>();
        double[] q = {100.0};
        double[] p = {10.0};
        c.add(bad("side",       // bad side
                () -> Analytics.decomposeShortfall("short", 10, 10, 10, 100, q, p)));
        c.add(bad("fillQtys and fillPrices",       // mismatched fill arrays
                () -> Analytics.decomposeShortfall(Side.SELL, 10, 10, 10, 100, q,
                        new double[] {10.0, 11.0})));
        c.add(bad("totalQty",       // over-execution
                () -> Analytics.decomposeShortfall(Side.SELL, 10, 10, 10, 50, q, p)));
        c.add(bad("fillQtys",       // negative fill qty
                () -> Analytics.decomposeShortfall(Side.SELL, 10, 10, 10, 100,
                        new double[] {-1.0}, p)));
        c.add(bad("decisionPrice",       // non-finite price
                () -> Analytics.decomposeShortfall(Side.SELL, Double.NaN, 10, 10, 100, q, p)));
        c.add(bad("executedVolume",       // negative executed volume
                () -> Analytics.participationRate(-1.0, 100.0)));
        c.add(bad("marketVolume",       // zero market volume
                () -> Analytics.participationRate(1.0, 0.0)));
        c.add(bad("x and y",       // n < 2
                () -> Analytics.ols(new double[] {1.0}, new double[] {1.0})));
        c.add(bad("x is",       // degenerate x
                () -> Analytics.ols(new double[] {2.0, 2.0}, new double[] {1.0, 2.0})));
        c.add(bad("y values",       // non-finite y
                () -> Analytics.ols(new double[] {1.0, 2.0},
                        new double[] {1.0, Double.POSITIVE_INFINITY})));
        c.add(bad("nObs",       // nObs < 2
                () -> Analytics.syntheticImpactFills(1, 0.5, 8.0, 0.05)));
        return c;
    }

    // -----------------------------------------------------------------------
    // Robustness tests: price-shift invariance, empty fills, participation above 1
    // -----------------------------------------------------------------------

    @Test
    void decompositionPriceShiftInvariance() {
        // Adding c to every price leaves every component unchanged (Q + R = X).
        double c = 37.5;
        double[] qty = {300.0, 400.0, 200.0};
        double[] px = {99.4, 99.2, 99.1};
        double[] pxC = {99.4 + c, 99.2 + c, 99.1 + c};
        for (Side side : new Side[] {Side.SELL, Side.BUY}) {
            Analytics.ISDecomposition d0 = Analytics.decomposeShortfall(side, 100.0, 99.5,
                    98.8, 1000.0, qty, px);
            Analytics.ISDecomposition d1 = Analytics.decomposeShortfall(side, 100.0 + c,
                    99.5 + c, 98.8 + c, 1000.0, qty, pxC);
            assertEquals(d0.total(), d1.total(), 1e-9);
            assertEquals(d0.delay(), d1.delay(), 1e-9);
            assertEquals(d0.trading(), d1.trading(), 1e-9);
            assertEquals(d0.opportunity(), d1.opportunity(), 1e-9);
        }
    }

    @Test
    void decompositionEmptyFills() {
        Analytics.ISDecomposition d = Analytics.decomposeShortfall(Side.SELL, 100.0, 99.0, 98.0,
                500.0, new double[0], new double[0]);
        assertEquals(0.0, d.delay(), 0.0);
        assertEquals(0.0, d.trading(), 0.0);
        assertEquals(1000.0, d.opportunity(), 1e-12);   // 500 * (100 - 98)
        assertEquals(1000.0, d.total(), 1e-12);
        assertEquals(0.0, d.executedQty(), 0.0);
        assertEquals(500.0, d.remainingQty(), 0.0);
        Analytics.ISDecomposition b = Analytics.decomposeShortfall(Side.BUY, 100.0, 99.0, 98.0,
                500.0, new double[0], new double[0]);
        assertEquals(-1000.0, b.opportunity(), 1e-12);  // price fell: a gain for a buyer
    }

    @Test
    void participationRateAboveOneIsError() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> Analytics.participationRate(2e6, 1.5e6));
        assertTrue(e.getMessage().contains("exceeds market volume"));
        assertEquals(1.0, Analytics.participationRate(1.5e6, 1.5e6), 0.0);   // boundary
        assertThrows(IllegalArgumentException.class,
                () -> Analytics.participationRate(Double.NaN, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> Analytics.participationRate(1.0, Double.POSITIVE_INFINITY));
    }
}
