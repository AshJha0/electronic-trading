package com.quant.etrade;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/** Unit tests for the execution-analytics module. */
public final class AnalyticsTest {

    @Test
    public void decompositionIdentityHoldsForBothSides() {
        double[] qtys = {300.0, 400.0, 200.0};
        // property-style: both sides x several price scenarios
        double[][] scenarios = {
            {100.0, 99.5, 98.8}, {100.0, 100.4, 101.3}, {50.0, 50.0, 50.0},
        };
        for (String side : new String[] {"sell", "buy"}) {
            for (double[] sc : scenarios) {
                double[] prices = {sc[1] - 0.1, sc[1] + 0.2, sc[1]};
                Analytics.ISDecomposition d = Analytics.decomposeShortfall(side,
                        sc[0], sc[1], sc[2], 1000.0, qtys, prices);
                // exact identity: total == delay + trading + opportunity
                assertEquals(d.total(), d.delay() + d.trading() + d.opportunity(), 0.0);
                // alternative closed form sign*(X*Pd - sum n*p - R*PT)
                double sign = "sell".equals(side) ? 1.0 : -1.0;
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
    public void sellDecompositionComponents() {
        Analytics.ISDecomposition d = Analytics.decomposeShortfall("sell", 100.0, 99.5,
                98.8, 1000.0, new double[] {300.0, 400.0, 200.0},
                new double[] {99.4, 99.2, 99.1});
        assertEquals(450.0, d.delay(), 1e-10);       // 900 * (100 - 99.5)
        assertEquals(230.0, d.trading(), 1e-9);      // slippage vs arrival
        assertEquals(120.0, d.opportunity(), 1e-9);  // 100 * (100 - 98.8)
        assertEquals(800.0, d.total(), 1e-9);
    }

    @Test
    public void fullyExecutedOrderHasNoOpportunityCost() {
        Analytics.ISDecomposition d = Analytics.decomposeShortfall("buy", 10.0, 10.0,
                12.0, 500.0, new double[] {500.0}, new double[] {10.5});
        assertEquals(0.0, d.opportunity(), 0.0);
        assertEquals(0.0, d.remainingQty(), 0.0);
        assertEquals(250.0, d.trading(), 1e-10);
    }

    @Test
    public void participationRateBasics() {
        assertEquals(0.08, Analytics.participationRate(120_000.0, 1_500_000.0), 1e-15);
        assertEquals(0.0, Analytics.participationRate(0.0, 1.0), 0.0);
    }

    @Test
    public void olsRecoversExactLine() {
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
    public void olsConstantYHasRSquaredOne() {
        // SST = 0 and the fit is exact -> R^2 defined as 1
        Analytics.OLSResult r = Analytics.ols(new double[] {1.0, 2.0, 3.0},
                new double[] {4.0, 4.0, 4.0});
        assertEquals(0.0, r.slope(), 1e-15);
        assertEquals(4.0, r.intercept(), 1e-15);
        assertEquals(1.0, r.rSquared(), 0.0);
    }

    @Test
    public void syntheticFillsPinnedConstruction() {
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

    @Test
    public void invalidInputsThrow() {
        double[] q = {100.0};
        double[] p = {10.0};
        assertThrows(IllegalArgumentException.class,       // bad side
                () -> Analytics.decomposeShortfall("short", 10, 10, 10, 100, q, p));
        assertThrows(IllegalArgumentException.class,       // mismatched fill arrays
                () -> Analytics.decomposeShortfall("sell", 10, 10, 10, 100, q,
                        new double[] {10.0, 11.0}));
        assertThrows(IllegalArgumentException.class,       // over-execution
                () -> Analytics.decomposeShortfall("sell", 10, 10, 10, 50, q, p));
        assertThrows(IllegalArgumentException.class,       // negative fill qty
                () -> Analytics.decomposeShortfall("sell", 10, 10, 10, 100,
                        new double[] {-1.0}, p));
        assertThrows(IllegalArgumentException.class,       // non-finite price
                () -> Analytics.decomposeShortfall("sell", Double.NaN, 10, 10, 100, q, p));
        assertThrows(IllegalArgumentException.class,       // negative executed volume
                () -> Analytics.participationRate(-1.0, 100.0));
        assertThrows(IllegalArgumentException.class,       // zero market volume
                () -> Analytics.participationRate(1.0, 0.0));
        assertThrows(IllegalArgumentException.class,       // n < 2
                () -> Analytics.ols(new double[] {1.0}, new double[] {1.0}));
        assertThrows(IllegalArgumentException.class,       // degenerate x
                () -> Analytics.ols(new double[] {2.0, 2.0}, new double[] {1.0, 2.0}));
        assertThrows(IllegalArgumentException.class,       // non-finite y
                () -> Analytics.ols(new double[] {1.0, 2.0},
                        new double[] {1.0, Double.POSITIVE_INFINITY}));
        assertThrows(IllegalArgumentException.class,       // nObs < 2
                () -> Analytics.syntheticImpactFills(1, 0.5, 8.0, 0.05));
    }
}
