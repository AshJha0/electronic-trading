package io.github.ashjha0.etrade;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import io.github.ashjha0.etrade.Placement.LiquidationParams;
import io.github.ashjha0.etrade.Placement.LiquidationPolicy;
import io.github.ashjha0.etrade.Placement.LiquidationSimResult;
import io.github.ashjha0.etrade.Placement.PlacementDecision;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.Test;

/**
 * Limit-order placement (API_SPEC Section 13): every case of {@code placement_golden.json},
 * the queue fill probability against the Poisson tail and its monotonicity, the
 * passive-or-aggressive rule, the liquidation policy against its Monte Carlo, validation.
 */
public final class PlacementTest {

    private static final Map<String, Object> GOLDEN = load();

    @SuppressWarnings("unchecked")
    private static Map<String, Object> load() {
        try {
            return (Map<String, Object>) Json.parse(
                    Files.readString(Path.of("../data/golden/placement_golden.json")));
        } catch (IOException e) {
            throw new IllegalStateException("cannot load placement_golden.json", e);
        }
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> cases() {
        return (List<Map<String, Object>>) (List<?>) GOLDEN.get("cases");
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> obj(Map<String, Object> m, String key) {
        return (Map<String, Object>) m.get(key);
    }

    private static double num(Map<String, Object> m, String key) {
        Object v = m.get(key);
        if (v == null) {
            throw new IllegalStateException("missing key " + key);
        }
        return (Double) v;
    }

    private static LiquidationParams liqFrom(Map<String, Object> in) {
        return new LiquidationParams((int) num(in, "total_units"), (int) num(in, "n_periods"),
                num(in, "dt"), num(in, "sigma"), num(in, "spread"), num(in, "kappa"),
                num(in, "p_touch"), num(in, "beta"), num(in, "phi"));
    }

    private static LiquidationParams patient(double sigma, int q, int n) {
        return new LiquidationParams(q, n, 0.02, sigma, 0.02, 30.0, 0.3, 0.005, 0.0);
    }

    /** Actual values and (for liquidation_sim) their own standard errors. */
    private static Map<String, Double> actual(Map<String, Object> c, Map<String, Double> se) {
        String kind = (String) c.get("kind");
        Map<String, Object> in = obj(c, "inputs");
        Map<String, Double> out = new LinkedHashMap<>();
        if ("fill_probability".equals(kind) || "placement_decision".equals(kind)) {
            double fp = Placement.queueFillProbability((int) num(in, "q_ahead"),
                    (int) num(in, "size"), num(in, "mu"), num(in, "theta"), num(in, "horizon"));
            out.put("fill_prob", fp);
            if ("placement_decision".equals(kind)) {
                PlacementDecision d = Placement.placeOrCross(num(in, "child_shares"),
                        num(in, "spread"), fp, num(in, "adverse_move"), num(in, "eta"),
                        num(in, "alpha"), num(in, "weight"));
                out.put("passive_cost", d.passiveCost());
                out.put("aggressive_cost", d.aggressiveCost());
                out.put("impact_cost", d.impactCost());
                out.put("breakeven_fill_prob", d.breakevenFillProb());
                out.put("passive", d.passive() ? 1.0 : 0.0);
            }
            return out;
        }
        LiquidationParams p = liqFrom(in);
        if ("liquidation_policy".equals(kind)) {
            LiquidationPolicy pol = Placement.liquidationPolicy(p);
            int nMarket = 0;
            for (boolean[] row : pol.market()) {
                for (int q = 1; q < row.length; q++) {
                    nMarket += row[q] ? 1 : 0;
                }
            }
            out.put("dp_cost", pol.expectedCost());
            out.put("depth_first", pol.depth()[0][p.totalUnits()]);
            out.put("depth_last", pol.depth()[p.nPeriods() - 1][1]);
            out.put("n_market", (double) nMarket);
            out.put("aggressive_cost", Placement.aggressiveCost(p));
            return out;
        }
        LiquidationSimResult r = Placement.simulateLiquidation(p, (int) num(in, "n_paths_ref"),
                (long) num(in, "seed"));
        out.put("optimal_mean", r.optimalMean());
        out.put("aggressive_mean", r.aggressiveMean());
        out.put("passive_mean", r.passiveMean());
        out.put("saving_vs_aggressive_mean", r.savingVsAggressiveMean());
        out.put("saving_vs_passive_mean", r.savingVsPassiveMean());
        se.put("optimal_mean", r.optimalSe());
        se.put("aggressive_mean", r.aggressiveSe());
        se.put("passive_mean", r.passiveSe());
        se.put("saving_vs_aggressive_mean", r.savingVsAggressiveSe());
        se.put("saving_vs_passive_mean", r.savingVsPassiveSe());
        return out;
    }

    @Test
    public void everyGoldenCaseEveryKey() {
        assertEquals(11, cases().size());
        List<String> failures = new ArrayList<>();
        for (Map<String, Object> c : cases()) {
            String name = (String) c.get("name");
            Map<String, Object> expect = obj(c, "expect");
            Map<String, Double> se = new LinkedHashMap<>();
            Map<String, Double> got = actual(c, se);
            double tol = num(c, "tol");
            if ("liquidation_sim".equals(c.get("kind"))) {
                int checked = 0;
                for (String key : expect.keySet()) {
                    if (key.endsWith("_se")) {
                        continue;
                    }
                    checked++;
                    assertTrue(name + "." + key + " computed", got.containsKey(key));
                    double ref = num(expect, key);
                    double seRef = num(expect, key + "_se");
                    double band = tol * Math.sqrt(seRef * seRef + se.get(key) * se.get(key));
                    double err = Math.abs(got.get(key) - ref);
                    if (err > band) {
                        failures.add(name + "." + key + ": got " + got.get(key) + " ref " + ref
                                + " err " + err + " > band " + band);
                    }
                }
                assertEquals(name + ": every statistical key", got.size(), checked);
                continue;
            }
            assertEquals(name + ": every expect key asserted", expect.keySet(), got.keySet());
            for (Map.Entry<String, Double> e : got.entrySet()) {
                double want = num(expect, e.getKey());
                if (Math.abs(e.getValue() - want) > tol * Math.max(1.0, Math.abs(want))) {
                    failures.add(name + "." + e.getKey() + ": got " + e.getValue() + " want "
                            + want);
                }
            }
        }
        assertTrue(String.join("\n", failures), failures.isEmpty());
    }

    // ------------------------------------------------------------ fill probability

    /** {@code P(Poisson(lam) >= n)} by direct summation of the lower tail (small n). */
    private static double poissonSf(int n, double lam) {
        double term = Math.exp(-lam);
        double cdf = 0.0;
        for (int k = 0; k < n; k++) {
            cdf += term;
            term *= lam / (k + 1);
        }
        return 1.0 - cdf;
    }

    @Test
    public void noCancellationsIsThePoissonTail() {
        double v = Placement.queueFillProbability(30, 5, 10.0, 0.0, 2.0);
        assertEquals(1.0, v / poissonSf(35, 20.0), 1e-10);
        assertEquals(1.0 - Math.exp(-1.2), Placement.queueFillProbability(0, 1, 3.0, 0.0, 0.4),
                1e-14);
    }

    @Test
    public void fillProbabilityDecreasesWithQueuePositionAndSize() {
        double prev = 2.0;
        for (int q = 0; q < 120; q += 5) {
            double v = Placement.queueFillProbability(q, 10, 20.0, 0.05, 3.0);
            assertTrue("q=" + q, v < prev);
            prev = v;
        }
        prev = 2.0;
        for (int n = 1; n < 40; n += 3) {
            double v = Placement.queueFillProbability(20, n, 20.0, 0.05, 3.0);
            assertTrue("n=" + n, v < prev);
            prev = v;
        }
    }

    @Test
    public void fillProbabilityIncreasesWithHorizonAndCancellations() {
        double prev = -1.0;
        for (int i = 0; i < 30; i++) {
            double h = 1.0 + 3.0 * i / 29.0;
            double v = Placement.queueFillProbability(50, 10, 20.0, 0.05, h);
            assertTrue("h=" + h, v > prev);
            prev = v;
        }
        prev = -1.0;
        for (double t : new double[] {0.0, 0.05, 0.2, 1.0}) {
            double v = Placement.queueFillProbability(50, 10, 20.0, t, 2.0);
            assertTrue("theta=" + t, v > prev);
            prev = v;
        }
    }

    @Test
    public void fillProbabilityEdges() {
        assertEquals(0.0, Placement.queueFillProbability(10, 3, 5.0, 0.2, 0.0), 0.0);
        assertEquals(1.0, Placement.queueFillProbability(0, 1, 1e3, 0.0, 100.0), 0.0);
        assertTrue(Placement.queueFillProbability(4999, 1, 1.0, 0.0, 1.0) <= 1e-300);
    }

    // ------------------------------------------------------------ passive or aggressive

    @Test
    public void decisionMatchesSection8AndClosedFormDifference() {
        PlacementDecision d = Placement.placeOrCross(2000.0, 0.02, 0.7, 0.01, 1e-4, 0.5,
                1.0 / 13.0);
        Impact.Params ip = new Impact.Params(2000.0, 1, 1.0 / 13.0, 0.0, 1e-4, 0.5, 0.0, 0.0,
                0.0, null);
        double e = Impact.expectedCost(ip, new double[] {2000.0, 0.0});
        assertEquals(1.0, d.impactCost() / (e / 2000.0), 1e-12);
        assertEquals(0.3 * 0.01 - 2.0 * 0.7 * 0.01, d.passiveCost() - d.aggressiveCost(), 1e-15);
        assertTrue(d.passive());
    }

    @Test
    public void aggressiveChosenWhenSpreadIsNegligible() {
        for (double fp : new double[] {0.0, 0.2, 0.5, 0.9, 1.0}) {
            for (double adv : new double[] {0.0, 0.004, 0.05}) {
                assertFalse(fp + " " + adv,
                        Placement.placeOrCross(500.0, 0.0, fp, adv, 1e-4, 0.5, 0.1).passive());
            }
        }
    }

    @Test
    public void breakevenFillProbabilitySeparatesTheChoices() {
        double adv = 0.015;
        double spread = 0.01;
        for (int i = 0; i <= 40; i++) {
            double fp = i / 40.0;
            PlacementDecision d = Placement.placeOrCross(100.0, spread, fp, adv, 1e-4, 0.6, 0.2);
            assertEquals(adv / (adv + spread), d.breakevenFillProb(), 1e-15);
            if (Math.abs(fp - d.breakevenFillProb()) > 1e-9) {
                assertEquals("fp=" + fp, fp > d.breakevenFillProb(), d.passive());
            }
        }
        assertEquals(1.0,
                Placement.placeOrCross(100.0, 0.0, 0.5, 0.0, 1e-4, 0.6, 0.2).breakevenFillProb(),
                0.0);
    }

    // ------------------------------------------------------------ liquidation

    @Test
    public void policyClosedFormOnePeriod() {
        LiquidationPolicy pol = Placement.liquidationPolicy(patient(0.5, 1, 1));
        double g = 0.01 + 0.015;
        double d = 1.0 / 30.0 - g;
        assertEquals(d, pol.depth()[0][1], 1e-15);
        assertEquals(0.015 - 0.3 * Math.exp(-30.0 * d) / 30.0, pol.expectedCost(), 1e-15);
        assertFalse(pol.market()[0][1]);
    }

    @Test
    public void depthFallsWithInventory() {
        LiquidationPolicy pol = Placement.liquidationPolicy(patient(0.5, 10, 50));
        for (int j = 0; j < pol.depth().length; j++) {
            double prev = Double.MAX_VALUE;
            for (int q = 1; q < pol.depth()[j].length; q++) {
                if (pol.market()[j][q]) {
                    continue;
                }
                assertTrue(j + " " + q, pol.depth()[j][q] <= prev + 1e-15);
                prev = pol.depth()[j][q];
            }
        }
    }

    @Test
    public void monteCarloMatchesDpAndBeatsBaselinesWithoutPenalty() {
        LiquidationParams p = patient(0.5, 10, 50);
        LiquidationSimResult r = Placement.simulateLiquidation(p, 20_000, 5);
        assertTrue(Math.abs(r.optimalMean() - Placement.liquidationPolicy(p).expectedCost())
                <= 4.0 * r.optimalSe());
        assertTrue(Math.abs(r.aggressiveMean() - Placement.aggressiveCost(p))
                <= 4.0 * r.aggressiveSe());
        assertTrue(r.savingVsAggressiveMean() > 4.0 * r.savingVsAggressiveSe());
        assertTrue(r.savingVsPassiveMean() > 4.0 * r.savingVsPassiveSe());
    }

    @Test
    public void deterministicWithoutVolatility() {
        LiquidationParams p = patient(0.0, 4, 4);
        LiquidationSimResult r = Placement.simulateLiquidation(p, 50, 1);
        assertTrue(r.aggressiveStd() <= 1e-15);
        assertEquals(Placement.aggressiveCost(p), r.aggressiveMean(), 1e-15);
    }

    @Test
    public void simulationIsDeterministic() {
        assertEquals(Placement.simulateLiquidation(patient(0.5, 10, 50), 200, 9),
                Placement.simulateLiquidation(patient(0.5, 10, 50), 200, 9));
    }

    @Test
    public void aggressiveScheduleSplitsEvenly() {
        assertArrayEquals(new int[] {2, 3, 2, 3}, Placement.aggressiveSchedule(10, 4));
        assertArrayEquals(new int[] {0, 1, 0, 1, 1}, Placement.aggressiveSchedule(3, 5));
        int total = 0;
        for (int n : Placement.aggressiveSchedule(997, 50)) {
            total += n;
        }
        assertEquals(997, total);
    }

    // ------------------------------------------------------------ validation

    private static void expectThrow(String fragment, Runnable r) {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, r::run);
        assertTrue(e.getMessage() + " should mention " + fragment,
                e.getMessage().contains(fragment));
    }

    @Test
    public void fillProbabilityValidation() {
        double inf = Double.POSITIVE_INFINITY;
        expectThrow("q_ahead", () -> Placement.queueFillProbability(-1, 1, 1.0, 0.0, 1.0));
        expectThrow("size", () -> Placement.queueFillProbability(0, 0, 1.0, 0.0, 1.0));
        expectThrow("q_ahead + size",
                () -> Placement.queueFillProbability(4999, 2, 1.0, 0.0, 1.0));
        expectThrow("mu", () -> Placement.queueFillProbability(0, 1, 0.0, 0.0, 1.0));
        expectThrow("mu", () -> Placement.queueFillProbability(0, 1, inf, 0.0, 1.0));
        expectThrow("theta", () -> Placement.queueFillProbability(0, 1, 1.0, -0.1, 1.0));
        expectThrow("theta", () -> Placement.queueFillProbability(0, 1, 1.0, Double.NaN, 1.0));
        expectThrow("horizon", () -> Placement.queueFillProbability(0, 1, 1.0, 0.0, -1.0));
        expectThrow("horizon", () -> Placement.queueFillProbability(0, 1, 1.0, 0.0, inf));
        expectThrow("horizon too long",
                () -> Placement.queueFillProbability(0, 1, 1e6, 0.0, 1.0));
        expectThrow("horizon too long",
                () -> Placement.queueFillProbability(0, 1, 1e300, 0.0, 1e300));
    }

    @Test
    public void decisionValidation() {
        Object[][] bad = {{0, 0.0, "size"}, {1, -0.01, "spread"}, {2, 1.1, "fill_prob"},
            {2, Double.NaN, "fill_prob"}, {3, -1.0, "adverse_move"}, {4, -1.0, "eta"},
            {5, 0.0, "alpha"}, {5, 1.5, "alpha"}, {6, 0.0, "weight"}};
        for (Object[] b : bad) {
            double[] a = {100.0, 0.01, 0.5, 0.01, 1e-4, 0.5, 0.1};
            a[(Integer) b[0]] = (Double) b[1];
            expectThrow((String) b[2],
                    () -> Placement.placeOrCross(a[0], a[1], a[2], a[3], a[4], a[5], a[6]));
        }
    }

    @Test
    public void liquidationParamsValidation() {
        expectThrow("total_units",
                () -> new LiquidationParams(0, 50, 0.02, 0.5, 0.02, 30.0, 0.3, 0.005, 0.0));
        expectThrow("total_units",
                () -> new LiquidationParams(1001, 50, 0.02, 0.5, 0.02, 30.0, 0.3, 0.005, 0.0));
        expectThrow("n_periods",
                () -> new LiquidationParams(10, 0, 0.02, 0.5, 0.02, 30.0, 0.3, 0.005, 0.0));
        expectThrow("n_periods",
                () -> new LiquidationParams(10, 1001, 0.02, 0.5, 0.02, 30.0, 0.3, 0.005, 0.0));
        expectThrow("dt",
                () -> new LiquidationParams(10, 50, 0.0, 0.5, 0.02, 30.0, 0.3, 0.005, 0.0));
        expectThrow("sigma",
                () -> new LiquidationParams(10, 50, 0.02, -1.0, 0.02, 30.0, 0.3, 0.005, 0.0));
        expectThrow("spread",
                () -> new LiquidationParams(10, 50, 0.02, 0.5, Double.NaN, 30.0, 0.3, 0.005, 0.0));
        expectThrow("kappa",
                () -> new LiquidationParams(10, 50, 0.02, 0.5, 0.02, 0.0, 0.3, 0.005, 0.0));
        expectThrow("p_touch",
                () -> new LiquidationParams(10, 50, 0.02, 0.5, 0.02, 30.0, 0.0, 0.005, 0.0));
        expectThrow("p_touch",
                () -> new LiquidationParams(10, 50, 0.02, 0.5, 0.02, 30.0, 1.01, 0.005, 0.0));
        expectThrow("beta",
                () -> new LiquidationParams(10, 50, 0.02, 0.5, 0.02, 30.0, 0.3, -1.0, 0.0));
        expectThrow("phi", () -> new LiquidationParams(10, 50, 0.02, 0.5, 0.02, 30.0, 0.3, 0.005,
                Double.POSITIVE_INFINITY));
        expectThrow("n_paths", () -> Placement.simulateLiquidation(patient(0.5, 10, 50), 1, 0));
        expectThrow("total_units", () -> Placement.aggressiveSchedule(0, 5));
        expectThrow("n_periods", () -> Placement.aggressiveSchedule(5, 0));
    }
}
