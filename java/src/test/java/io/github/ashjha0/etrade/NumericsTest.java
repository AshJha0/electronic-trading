package io.github.ashjha0.etrade;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * Standard-error formulas checked numerically (Sections 1, 2 and 9), plus
 * closed-form deterministic cases where every SE must be exactly zero.
 */
public final class NumericsTest {

    private static final String[] AS_STATS = {
        "mean_pnl", "std_pnl", "mean_abs_q", "mean_max_abs_q", "rms_q", "mean_fills"};

    private static double mean(double[] a) {
        double s = 0.0;
        for (double v : a) {
            s += v;
        }
        return s / a.length;
    }

    private static double sampleStd(double[] a) {
        double m = mean(a);
        double ss = 0.0;
        for (double v : a) {
            ss += (v - m) * (v - m);
        }
        return Math.sqrt(ss / (a.length - 1));
    }

    private static void rel(String what, double expected, double actual, double tol) {
        assertTrue(what + ": " + actual + " vs " + expected,
                Math.abs(actual - expected) <= tol * Math.max(Math.abs(expected), 1e-300));
    }

    // ---- Section 1 Monte Carlo ----

    @Test
    public void acSeFormulas() {
        ACParams p = new ACParams(1e6, 10, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6);
        int m = 400;
        AlmgrenChriss.ISSimResult r = AlmgrenChriss.simulateIS(p, m, 7L);
        assertEquals(m, r.nPaths());
        assertTrue(r.var() > 0.0);
        rel("seMean", Math.sqrt(r.var() / m), r.seMean(), 1e-15);
        rel("seVar", r.var() * Math.sqrt(2.0 / (m - 1)), r.seVar(), 1e-15);
        // the MC moments agree with the closed forms within a few SE
        double e = AlmgrenChriss.expectedCost(p);
        double v = AlmgrenChriss.varianceCost(p);
        assertTrue(Math.abs(r.mean() - e) <= 5.0 * r.seMean());
        assertTrue(Math.abs(r.var() - v) <= 5.0 * r.seVar());
    }

    @Test
    public void acZeroSigmaIsDeterministic() {
        ACParams p = new ACParams(1e6, 10, 1.0, 0.0, 2.5e-6, 2.5e-7, 0.0);
        double[] twap = AlmgrenChriss.twapHoldings(1e6, 10);
        for (double[] h : new double[][] {AlmgrenChriss.trajectory(p), twap}) {
            AlmgrenChriss.ISSimResult r = AlmgrenChriss.simulateIS(p, h, 50, 3L);
            rel("mean == E[IS]", AlmgrenChriss.expectedCost(p, h), r.mean(), 1e-12);
            assertEquals(0.0, r.var(), 0.0);
            assertEquals(0.0, r.seMean(), 0.0);
            assertEquals(0.0, r.seVar(), 0.0);
        }
    }

    // ---- Section 2 simulation ----

    @Test
    public void asSeFromPerPathPnl() {
        ASParams p = new ASParams(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, 5, 0.0);
        ASSimResult r = AvellanedaStoikov.simulate(p, 300, 11L);
        double[] pnl = r.pnlPerPath();
        int m = pnl.length;
        rel("mean_pnl", mean(pnl), r.meanPnl(), 1e-12);
        rel("std_pnl", sampleStd(pnl), r.stdPnl(), 1e-12);
        rel("se(mean_pnl)", sampleStd(pnl) / Math.sqrt(m), r.se("mean_pnl"), 1e-12);
        assertEquals(r.stdPnl() / Math.sqrt(2.0 * (m - 1)), r.se("std_pnl"), 0.0);
        for (String s : AS_STATS) {
            assertTrue(s, r.se(s) > 0.0 && Double.isFinite(r.se(s)));
        }
    }

    @Test
    public void asNoQuotingGivesZeroSe() {
        ASParams p = new ASParams(100.0, 2.0, 1.0, 0.01, 0.1, 1.5, 140.0, 0, 0.5);
        ASSimResult r = AvellanedaStoikov.simulate(p, 20, 5L);
        for (double v : r.pnlPerPath()) {
            assertEquals(0.0, v, 0.0);
        }
        assertEquals(0.0, r.meanFills(), 0.0);
        for (String s : AS_STATS) {
            assertEquals(s, 0.0, r.se(s), 0.0);
        }
    }

    @Test
    public void asSaturatedFillsAreDeterministic() {
        // sigma = 0, A huge: both sides fill every step (P == 1 > u), q stays 0,
        // pnl = n_steps * (ask - bid) = n_steps * (2/gamma) ln(1 + gamma/k).
        double gamma = 0.1;
        double k = 1.5;
        ASParams p = new ASParams(100.0, 0.0, 1.0, 0.01, gamma, k, 1e6, 10, 0.0);
        ASSimResult r = AvellanedaStoikov.simulate(p, 30, 9L);
        int n = p.nSteps();
        assertEquals(2.0 * n, r.meanFills(), 0.0);
        rel("pnl", n * 2.0 / gamma * Math.log1p(gamma / k), r.meanPnl(), 1e-9);
        double[] pnl = r.pnlPerPath();
        for (double v : pnl) {
            assertEquals(pnl[0], v, 0.0);   // identical per-path values
        }
        // identical values: SE is zero up to rounding of the mean
        for (String s : AS_STATS) {
            assertTrue(s, r.se(s) <= 1e-14 * Math.max(1.0, Math.abs(pnl[0])));
        }
    }

    // ---- Section 9 simulation ----

    @Test
    public void mmSeFromPerPathArrays() {
        ASParams p = new ASParams(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, 5, 0.01);
        MarketMaking.Config cfg = MarketMaking.Config.defaults().withTick(0.01)
                .withMakerFee(0.002).withLatencySteps(2).withInformedProb(0.2)
                .withSignalAccuracy(0.7);
        MarketMaking.Result r = MarketMaking.simulate(p, 200, 13L, cfg);
        int m = r.nPaths();
        for (MarketMaking.Stat st : MarketMaking.Stat.values()) {
            double[] a = r.path(st);
            assertEquals(m, a.length);
            rel("mean " + st, mean(a), r.mean(st), 1e-12);
            double expected = sampleStd(a) / Math.sqrt(m);
            if (expected == 0.0) {
                assertEquals(st.name(), 0.0, r.se(st), 0.0);
            } else {
                rel("se " + st, expected, r.se(st), 1e-12);
            }
        }
        assertTrue(r.se(MarketMaking.Stat.PNL) > 0.0);
    }

    @Test
    public void mmNoQuotingGivesZeroSe() {
        ASParams p = new ASParams(100.0, 2.0, 1.0, 0.01, 0.1, 1.5, 140.0, 0, 0.5);
        MarketMaking.Config cfg = MarketMaking.Config.defaults().withInformedProb(0.5);
        MarketMaking.Result r = MarketMaking.simulate(p, 20, 5L, cfg);
        for (MarketMaking.Stat st : MarketMaking.Stat.values()) {
            if (st != MarketMaking.Stat.INVENTORY && st != MarketMaking.Stat.MARKOUT) {
                assertEquals(st.name(), 0.0, r.mean(st), 0.0);
            }
            assertEquals(st.name(), 0.0, r.se(st), 0.0);
        }
    }

    @Test
    public void mmSaturatedFills() {
        ASParams p = new ASParams(100.0, 0.0, 1.0, 0.01, 0.1, 1.5, 1e6, 10, 0.0);
        MarketMaking.Result r = MarketMaking.simulate(p, 30, 9L, MarketMaking.Config.defaults());
        int n = p.nSteps();
        for (double f : r.path(MarketMaking.Stat.FILLS)) {
            assertEquals(2.0 * n, f, 0.0);
        }
        assertEquals(0.0, r.se(MarketMaking.Stat.FILLS), 0.0);
        double[] pnl = r.path(MarketMaking.Stat.PNL);
        for (double v : pnl) {
            assertEquals(pnl[0], v, 0.0);
        }
        assertTrue(r.se(MarketMaking.Stat.PNL) <= 1e-14 * Math.max(1.0, Math.abs(pnl[0])));
    }
}
