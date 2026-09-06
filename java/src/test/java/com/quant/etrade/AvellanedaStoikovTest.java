package com.quant.etrade;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/** Unit tests for the Avellaneda-Stoikov module (properties, edge cases, errors). */
public final class AvellanedaStoikovTest {

    private static final ASParams LIQUID =
            new ASParams(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, 20, 0.5);

    @Test
    public void spreadPositiveAndDecreasingToTerminalFloor() {
        double floor = 2.0 / LIQUID.gamma() * Math.log1p(LIQUID.gamma() / LIQUID.k());
        double prev = Double.POSITIVE_INFINITY;
        // property-style: time grid down to the terminal time
        for (double t = 0.0; t <= 1.0 + 1e-12; t += 0.1) {
            double sp = AvellanedaStoikov.optimalSpread(LIQUID, Math.min(t, 1.0));
            assertTrue("spread must be > 0", sp > 0.0);
            assertTrue("spread must decrease in t", sp <= prev + 1e-12);
            assertTrue("spread must stay above the terminal floor", sp >= floor - 1e-12);
            prev = sp;
        }
        assertEquals(floor, AvellanedaStoikov.optimalSpread(LIQUID, LIQUID.horizon()),
                1e-15);
    }

    @Test
    public void reservationPriceSignConventions() {
        double s = 100.0;
        // long maker (q > 0) shades down: selling pressure
        assertTrue(AvellanedaStoikov.reservationPrice(LIQUID, s, 3.0, 0.3) < s);
        // short maker shades up
        assertTrue(AvellanedaStoikov.reservationPrice(LIQUID, s, -3.0, 0.3) > s);
        // flat maker quotes the mid
        assertEquals(s, AvellanedaStoikov.reservationPrice(LIQUID, s, 0.0, 0.3), 0.0);
        // at the terminal time r = s regardless of inventory
        assertEquals(s, AvellanedaStoikov.reservationPrice(LIQUID, s, 5.0,
                LIQUID.horizon()), 0.0);
    }

    @Test
    public void quotesAreSymmetricAroundReservation() {
        AvellanedaStoikov.Quotes q = AvellanedaStoikov.quotes(LIQUID, 100.0, 3.0, 0.3);
        double r = AvellanedaStoikov.reservationPrice(LIQUID, 100.0, 3.0, 0.3);
        double sp = AvellanedaStoikov.optimalSpread(LIQUID, 0.3);
        assertEquals(r, 0.5 * (q.bid() + q.ask()), 1e-12);
        assertEquals(sp, q.ask() - q.bid(), 1e-12);
    }

    @Test
    public void fillProbabilityRangeAndMonotonicity() {
        double prev = 1.0;
        for (double delta = -0.5; delta <= 2.0; delta += 0.25) {
            double p = AvellanedaStoikov.fillProbability(LIQUID, delta);
            assertTrue("P(fill) in [0,1)", p >= 0.0 && p < 1.0);
            assertTrue("P(fill) decreasing in delta", p <= prev + 1e-15);
            prev = p;
        }
        // consistency with the intensity closed form
        double delta = 0.6;
        double lam = AvellanedaStoikov.intensity(LIQUID, delta);
        assertEquals(140.0 * Math.exp(-1.5 * 0.6), lam, 1e-12);
        assertEquals(-Math.expm1(-lam * LIQUID.dt()),
                AvellanedaStoikov.fillProbability(LIQUID, delta), 1e-15);
    }

    @Test
    public void averageSpreadFormula() {
        double expected = 0.5 * LIQUID.gamma() * LIQUID.sigma() * LIQUID.sigma()
                * LIQUID.horizon()
                + 2.0 / LIQUID.gamma() * Math.log1p(LIQUID.gamma() / LIQUID.k());
        assertEquals(expected, AvellanedaStoikov.averageAsSpread(LIQUID), 1e-15);
        // equals the time average of the endpoints of the linear-in-t spread
        double mid = 0.5 * (AvellanedaStoikov.optimalSpread(LIQUID, 0.0)
                + AvellanedaStoikov.optimalSpread(LIQUID, LIQUID.horizon()));
        assertEquals(mid, AvellanedaStoikov.averageAsSpread(LIQUID), 1e-12);
    }

    @Test
    public void inventoryCapRespectedOnEveryStep() {
        ASParams tight = new ASParams(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, 3, 0.5);
        ASSimResult res = AvellanedaStoikov.simulate(tight, 200, 1);
        assertFalse(res.capViolated());
        assertTrue(res.overallMaxAbsQ() <= tight.qMax() + 1e-12);
    }

    @Test
    public void qMaxZeroQuotesNothing() {
        ASParams noQuote = new ASParams(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, 0, 0.5);
        ASSimResult res = AvellanedaStoikov.simulate(noQuote, 50, 3);
        assertEquals(0.0, res.meanFills(), 0.0);
        assertEquals(0.0, res.meanPnl(), 0.0);
        assertEquals(0.0, res.stdPnl(), 0.0);
        assertTrue(Double.isInfinite(res.sharpe()));
    }

    @Test
    public void asInventoryTighterThanFixedSpread() {
        // statistical boolean with margin: A-S skew controls inventory
        ASSimResult resAs = AvellanedaStoikov.simulate(LIQUID, 800, 5,
                AvellanedaStoikov.Agent.AS);
        ASSimResult resFx = AvellanedaStoikov.simulate(LIQUID, 800, 5,
                AvellanedaStoikov.Agent.FIXED);
        assertTrue("A-S rms q should be well below fixed-spread rms q",
                resAs.rmsQ() < 0.75 * resFx.rmsQ());
    }

    @Test
    public void stressRaisesPnlVolatility() {
        ASSimResult base = AvellanedaStoikov.simulate(LIQUID, 800, 17);
        ASSimResult stress = AvellanedaStoikov.simulate(LIQUID, 800, 17,
                AvellanedaStoikov.Agent.AS, AvellanedaStoikov.averageAsSpread(LIQUID),
                2.0, 0.5);
        assertTrue(stress.stdPnl() > base.stdPnl());
    }

    @Test
    public void simulationDeterministicPerSeed() {
        ASSimResult a = AvellanedaStoikov.simulate(LIQUID, 100, 42);
        ASSimResult b = AvellanedaStoikov.simulate(LIQUID, 100, 42);
        assertEquals(a.meanPnl(), b.meanPnl(), 0.0);
        assertEquals(a.stdPnl(), b.stdPnl(), 0.0);
        assertEquals(a.meanFills(), b.meanFills(), 0.0);
        assertEquals(a.rmsQ(), b.rmsQ(), 0.0);
        // and a different seed gives a different sample
        ASSimResult c = AvellanedaStoikov.simulate(LIQUID, 100, 43);
        assertTrue(a.meanPnl() != c.meanPnl());
    }

    @Test
    public void invalidInputsThrow() {
        assertThrows(IllegalArgumentException.class,       // gamma = 0 outside model
                () -> new ASParams(100.0, 2.0, 1.0, 0.005, 0.0, 1.5, 140.0, 20, 0.5));
        assertThrows(IllegalArgumentException.class,       // k <= 0
                () -> new ASParams(100.0, 2.0, 1.0, 0.005, 0.1, 0.0, 140.0, 20, 0.5));
        assertThrows(IllegalArgumentException.class,       // A <= 0
                () -> new ASParams(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 0.0, 20, 0.5));
        assertThrows(IllegalArgumentException.class,       // sigma < 0
                () -> new ASParams(100.0, -1.0, 1.0, 0.005, 0.1, 1.5, 140.0, 20, 0.5));
        assertThrows(IllegalArgumentException.class,       // dt > T
                () -> new ASParams(100.0, 2.0, 1.0, 2.0, 0.1, 1.5, 140.0, 20, 0.5));
        assertThrows(IllegalArgumentException.class,       // T/dt not integral
                () -> new ASParams(100.0, 2.0, 1.0, 0.003, 0.1, 1.5, 140.0, 20, 0.5));
        assertThrows(IllegalArgumentException.class,       // q_max < 0
                () -> new ASParams(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, -1, 0.5));
        assertThrows(IllegalArgumentException.class,       // eta_liq < 0
                () -> new ASParams(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, 20, -0.5));
        assertThrows(IllegalArgumentException.class,       // t outside [0, T]
                () -> AvellanedaStoikov.reservationPrice(LIQUID, 100.0, 0.0, 1.5));
        assertThrows(IllegalArgumentException.class,
                () -> AvellanedaStoikov.optimalSpread(LIQUID, -0.1));
        assertThrows(IllegalArgumentException.class,       // nPaths < 2
                () -> AvellanedaStoikov.simulate(LIQUID, 1, 0));
        assertThrows(IllegalArgumentException.class,       // fixedSpread <= 0
                () -> AvellanedaStoikov.simulate(LIQUID, 10, 0,
                        AvellanedaStoikov.Agent.FIXED, 0.0, 1.0, 0.5));
        assertThrows(IllegalArgumentException.class,       // stressFactor <= 0
                () -> AvellanedaStoikov.simulate(LIQUID, 10, 0,
                        AvellanedaStoikov.Agent.AS, 1.0, 0.0, 0.5));
        assertThrows(IllegalArgumentException.class,       // stressTimeFrac outside [0,1]
                () -> AvellanedaStoikov.simulate(LIQUID, 10, 0,
                        AvellanedaStoikov.Agent.AS, 1.0, 1.0, 1.5));
        assertThrows(IllegalArgumentException.class,       // unknown statistic name
                () -> AvellanedaStoikov.simulate(LIQUID, 10, 0).se("nope"));
    }

    // -----------------------------------------------------------------------
    // Robustness tests (review findings CRIT-1, MAJ-8, MIN-16, T-8, T-11, T-14)
    // -----------------------------------------------------------------------

    @Test
    public void meanFillsCountsBothSides() {
        // Saturated book: P(fill) rounds to 1 on both sides every step, so the
        // fill count is exactly 2*nSteps (400), not the step count (200).
        ASParams sat = new ASParams(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 1e6, 10_000, 0.0);
        ASSimResult r = AvellanedaStoikov.simulate(sat, 50, 1);
        assertEquals(2.0 * sat.nSteps(), r.meanFills(), 0.0);
        // Liquid set: bounded by 2 per step, integral, >= peak inventory.
        ASSimResult l = AvellanedaStoikov.simulate(LIQUID, 400, 8);
        assertTrue(l.meanFills() <= 2.0 * LIQUID.nSteps());
        assertTrue(l.meanFills() >= l.meanMaxAbsQ());
        assertTrue(l.se("mean_fills") > 0.0);
    }

    @Test
    public void fixedSpreadMustBeFiniteAndPositive() {
        for (double bad : new double[] {Double.POSITIVE_INFINITY, Double.NaN, 0.0, -1.0}) {
            assertThrows(IllegalArgumentException.class,
                    () -> AvellanedaStoikov.simulate(LIQUID, 10, 0,
                            AvellanedaStoikov.Agent.FIXED, bad, 1.0, 0.5));
        }
        double avg = AvellanedaStoikov.averageAsSpread(LIQUID);
        assertThrows(IllegalArgumentException.class,          // stressTimeFrac NaN
                () -> AvellanedaStoikov.simulate(LIQUID, 10, 0,
                        AvellanedaStoikov.Agent.AS, avg, 1.0, Double.NaN));
        assertThrows(IllegalArgumentException.class,          // stressFactor inf
                () -> AvellanedaStoikov.simulate(LIQUID, 10, 0,
                        AvellanedaStoikov.Agent.AS, avg, Double.POSITIVE_INFINITY, 0.5));
        assertThrows(IllegalArgumentException.class,
                () -> AvellanedaStoikov.simulate(LIQUID, AvellanedaStoikov.MAX_AS_PATHS + 1, 0));
    }

    @Test
    public void stressEdgeFractions() {
        double avg = AvellanedaStoikov.averageAsSpread(LIQUID);
        ASSimResult base = AvellanedaStoikov.simulate(LIQUID, 300, 9);
        // frac = 1.0: no step has t >= T, so bitwise identical to unstressed.
        ASSimResult late = AvellanedaStoikov.simulate(LIQUID, 300, 9,
                AvellanedaStoikov.Agent.AS, avg, 5.0, 1.0);
        // factor = 1.0 at any fraction: identical.
        ASSimResult unit = AvellanedaStoikov.simulate(LIQUID, 300, 9,
                AvellanedaStoikov.Agent.AS, avg, 1.0, 0.0);
        double[] b = base.pnlPerPath();
        double[] lp = late.pnlPerPath();
        double[] up = unit.pnlPerPath();
        for (int i = 0; i < b.length; i++) {
            assertEquals(b[i], lp[i], 0.0);
            assertEquals(b[i], up[i], 0.0);
        }
        // frac = 0.0 with factor 5: every step stressed, dispersion up.
        ASSimResult early = AvellanedaStoikov.simulate(LIQUID, 300, 9,
                AvellanedaStoikov.Agent.AS, avg, 5.0, 0.0);
        assertTrue(early.stdPnl() > base.stdPnl());
    }

    @Test
    public void quotesRejectNonFiniteState() {
        double nan = Double.NaN;
        double inf = Double.POSITIVE_INFINITY;
        assertThrows(IllegalArgumentException.class,
                () -> AvellanedaStoikov.reservationPrice(LIQUID, nan, 0.0, 0.0));
        assertThrows(IllegalArgumentException.class,
                () -> AvellanedaStoikov.reservationPrice(LIQUID, 100.0, inf, 0.0));
        assertThrows(IllegalArgumentException.class,
                () -> AvellanedaStoikov.quotes(LIQUID, 100.0, inf, 0.0));
        assertThrows(IllegalArgumentException.class,
                () -> AvellanedaStoikov.quotes(LIQUID, -inf, 0.0, 0.0));
        assertThrows(IllegalArgumentException.class,
                () -> AvellanedaStoikov.reservationPrice(LIQUID, 100.0, 0.0, nan));
        assertThrows(IllegalArgumentException.class,
                () -> AvellanedaStoikov.fillProbability(LIQUID, nan));
    }

    @Test
    public void nStepsIntegralityFloat() {
        ASParams p3 = new ASParams(100.0, 2.0, 1.0, 1.0 / 3.0, 0.1, 1.5, 140.0, 20, 0.5);
        assertEquals(3, p3.nSteps());
        assertThrows(IllegalArgumentException.class,
                () -> new ASParams(100.0, 2.0, 1.0, 0.3, 0.1, 1.5, 140.0, 20, 0.5));
        ASParams p1 = new ASParams(100.0, 2.0, 1.0, 1.0, 0.1, 1.5, 140.0, 20, 0.5); // dt = T
        assertEquals(1, p1.nSteps());
        ASSimResult r = AvellanedaStoikov.simulate(p1, 200, 4);
        // one step: the time average of |q| is the single post-update value
        assertEquals(r.meanMaxAbsQ(), r.meanAbsQ(), 0.0);
        assertEquals(r.meanMaxAbsQ(), r.rmsQ(), 0.0);
        assertTrue(r.meanFills() <= 2.0);
        // step-count bound
        assertThrows(IllegalArgumentException.class,
                () -> new ASParams(100.0, 2.0, 1.0, 1.0 / (10.0 * AvellanedaStoikov.MAX_AS_STEPS),
                        0.1, 1.5, 140.0, 20, 0.5));
    }
}
