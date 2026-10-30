package io.github.ashjha0.etrade;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/** Enum APIs, deprecated overloads, result accessors and the pinned normal generator. */
final class ApiHardeningTest {

    private static final ASParams LIQUID =
            new ASParams(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, 20, 0.5);

    private static double mean(double[] a) {
        double s = 0.0;
        for (double v : a) {
            s += v;
        }
        return s / a.length;
    }

    private static double std(double[] a) {
        double m = mean(a);
        double ss = 0.0;
        for (double v : a) {
            ss += (v - m) * (v - m);
        }
        return Math.sqrt(ss / (a.length - 1));
    }

    @Test
    void perPathArraysAreDefensiveCopiesMatchingTheAggregates() {
        ASSimResult r = AvellanedaStoikov.simulate(LIQUID, 200, 11);
        double[] fills = r.fillsPerPath();
        assertNotSame(fills, r.fillsPerPath());
        fills[0] = -1.0;
        assertTrue(r.fillsPerPath()[0] >= 0.0);
        assertEquals(200, r.meanAbsQPerPath().length);
        assertEquals(mean(r.pnlPerPath()), r.meanPnl(), 0.0);
        assertEquals(mean(r.meanAbsQPerPath()), r.meanAbsQ(), 0.0);
        assertEquals(mean(r.maxAbsQPerPath()), r.meanMaxAbsQ(), 0.0);
        assertEquals(mean(r.rmsQPerPath()), r.rmsQ(), 0.0);
        assertEquals(mean(r.fillsPerPath()), r.meanFills(), 0.0);
        double rootM = Math.sqrt(200);
        assertEquals(std(r.rmsQPerPath()) / rootM, r.se(ASSimResult.Stat.RMS_Q), 1e-15);
        assertEquals(r.stdPnl() / Math.sqrt(2.0 * 199), r.se(ASSimResult.Stat.STD_PNL), 0.0);
        assertTrue(r.toString().startsWith("ASSimResult[nPaths=200, meanPnl="));
    }

    @ParameterizedTest
    @EnumSource(ASSimResult.Stat.class)
    @SuppressWarnings("deprecation")
    void deprecatedStringStatMatchesEnum(ASSimResult.Stat stat) {
        ASSimResult r = AvellanedaStoikov.simulate(LIQUID, 50, 3);
        assertEquals(stat, ASSimResult.Stat.fromWireName(stat.wireName()));
        assertEquals(r.se(stat), r.se(stat.wireName()), 0.0);
    }

    @ParameterizedTest
    @EnumSource(Side.class)
    @SuppressWarnings("deprecation")
    void deprecatedStringSideMatchesEnum(Side side) {
        double[] q = {300.0, 400.0};
        double[] px = {99.4, 99.2};
        Analytics.ISDecomposition a = Analytics.decomposeShortfall(side, 100.0, 99.5, 98.8,
                1000.0, q, px);
        Analytics.ISDecomposition b = Analytics.decomposeShortfall(
                side.name().toLowerCase(java.util.Locale.ROOT), 100.0, 99.5, 98.8, 1000.0, q, px);
        assertEquals(a, b);
        assertEquals(side, Side.parse(side.name().toLowerCase(java.util.Locale.ROOT)));
    }

    @Test
    void nullArgumentsAreNamed() {
        Rejects.assertRejects("side", () -> Analytics.decomposeShortfall((Side) null, 1, 1, 1,
                1, new double[0], new double[0]));
        Rejects.assertRejects("params", () -> AlmgrenChriss.trajectory(null));
        Rejects.assertRejects("stat", () -> AvellanedaStoikov.simulate(LIQUID, 2, 0)
                .se((ASSimResult.Stat) null));
        Rejects.assertRejects("statistic", () -> ASSimResult.Stat.fromWireName(null));
    }

    @Test
    void normalGeneratorIsPinnedAndStandard() {
        NormalRng a = new NormalRng(42);
        NormalRng b = new NormalRng(42);
        int n = 200_000;
        double s = 0.0;
        double s2 = 0.0;
        for (int i = 0; i < n; i++) {
            double z = a.nextNormal();
            assertEquals(Double.doubleToRawLongBits(z), Double.doubleToRawLongBits(b.nextNormal()));
            s += z;
            s2 += z * z;
        }
        assertEquals(0.0, s / n, 4.0 / Math.sqrt(n));
        assertEquals(1.0, s2 / n, 4.0 * Math.sqrt(2.0 / n));
        // first pair reproduces the documented polar recipe on SplittableRandom uniforms
        java.util.SplittableRandom u = new java.util.SplittableRandom(7);
        double x;
        double y;
        double q;
        do {
            x = 2.0 * u.nextDouble() - 1.0;
            y = 2.0 * u.nextDouble() - 1.0;
            q = x * x + y * y;
        } while (q >= 1.0 || q == 0.0);
        double f = StrictMath.sqrt(-2.0 * StrictMath.log(q) / q);
        NormalRng c = new NormalRng(7);
        assertArrayEquals(new double[] {x * f, y * f}, new double[] {c.nextNormal(), c.nextNormal()});
        assertThrows(IllegalArgumentException.class, () -> Side.parse("hold"));
    }
}
