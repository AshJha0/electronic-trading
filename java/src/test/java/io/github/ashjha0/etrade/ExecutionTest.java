package io.github.ashjha0.etrade;

import static io.github.ashjha0.etrade.Rejects.assertRejects;
import static io.github.ashjha0.etrade.Rejects.bad;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.ashjha0.etrade.Execution.AdaptiveSimResult;
import io.github.ashjha0.etrade.Execution.InternalisationResult;
import io.github.ashjha0.etrade.Execution.SignalParams;
import io.github.ashjha0.etrade.Execution.TransientParams;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Execution extensions (API_SPEC Section 12): every case of {@code execution_golden.json},
 * transient impact and its reduction to the linear case, the Obizhaeva-Wang shape, adaptive
 * re-planning, internalisation and validation.
 */
final class ExecutionTest {

    private static final Map<String, Object> GOLDEN = load();

    @SuppressWarnings("unchecked")
    private static Map<String, Object> load() {
        try {
            return (Map<String, Object>) TestData.load("golden/execution_golden.json");
        } catch (IOException e) {
            throw new IllegalStateException("cannot load execution_golden.json", e);
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

    private static double numOr(Map<String, Object> m, String key, double d) {
        Object v = m.get(key);
        return v == null ? d : (Double) v;
    }

    private static double[] list(Map<String, Object> in, String prefix, String count) {
        int n = (int) num(in, count);
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = num(in, prefix + i);
        }
        return out;
    }

    /** Actual values and (for adaptive_sim) their own standard errors. */
    private static Map<String, Double> actual(Map<String, Object> c, Map<String, Double> se) {
        String kind = (String) c.get("kind");
        Map<String, Object> in = obj(c, "inputs");
        Map<String, Double> out = new LinkedHashMap<>();
        if ("transient_schedule".equals(kind)) {
            TransientParams p = new TransientParams(num(in, "total_shares"),
                    (int) num(in, "n_intervals"), num(in, "tau"), num(in, "kappa"),
                    num(in, "rho"), num(in, "gamma_p"));
            double[] x = Execution.transientOptimalSchedule(p);
            for (int j = 1; j < p.nIntervals(); j++) {
                out.put(String.format("xfrac_%02d", j), x[j] / p.totalShares());
            }
            out.put("e_cost", Execution.transientExpectedCost(p, x));
            out.put("twap_e_cost", Execution.transientExpectedCost(p,
                    AlmgrenChriss.twapHoldings(p.totalShares(), p.nIntervals())));
            return out;
        }
        if ("internalise".equals(kind)) {
            InternalisationResult r = Execution.internalise(list(in, "buy_", "n_buys"),
                    list(in, "sell_", "n_sells"), num(in, "price"), num(in, "spread"),
                    num(in, "adv"), num(in, "daily_vol"), (int) num(in, "n_intervals"),
                    num(in, "tau"), numOr(in, "alpha", 0.5), numOr(in, "y", 1.0),
                    numOr(in, "perm_frac", Market.DEFAULT_PERM_FRAC),
                    num(in, "volume_curve") == 1.0 ? Impact.usEquityVolumeCurve13() : null);
            out.put("crossed_shares", r.crossedShares());
            out.put("residual_shares", r.residualShares());
            out.put("baseline_cost", r.baselineCost());
            out.put("internal_cost", r.internalCost());
            out.put("saved_cost", r.savedCost());
            out.put("saved_bps", r.savedBps());
            double[] b = r.buySavedBps();
            for (int i = 0; i < b.length; i++) {
                out.put("buy" + i + "_saved_bps", b[i]);
            }
            double[] s = r.sellSavedBps();
            for (int i = 0; i < s.length; i++) {
                out.put("sell" + i + "_saved_bps", s[i]);
            }
            return out;
        }
        ACParams ac = new ACParams(num(in, "ac_total_shares"), (int) num(in, "ac_n_intervals"),
                num(in, "ac_tau"), num(in, "ac_sigma"), num(in, "ac_eta"), num(in, "ac_gamma_p"),
                num(in, "ac_lambda_ra"));
        SignalParams sig = new SignalParams(num(in, "phi"), num(in, "signal_sd"),
                num(in, "fill_risk"));
        AdaptiveSimResult r = Execution.simulateAdaptive(ac, sig, (int) num(in, "n_paths_ref"),
                (long) num(in, "seed"));
        out.put("static_mean", r.staticMean());
        out.put("adaptive_mean", r.adaptiveMean());
        out.put("saving_mean", r.savingMean());
        se.put("static_mean", r.staticSe());
        se.put("adaptive_mean", r.adaptiveSe());
        se.put("saving_mean", r.savingSe());
        return out;
    }

    @Test
    void everyGoldenCaseEveryKey() {
        assertEquals(8, cases().size());
        List<String> failures = new ArrayList<>();
        for (Map<String, Object> c : cases()) {
            String name = (String) c.get("name");
            Map<String, Object> expect = obj(c, "expect");
            Map<String, Double> se = new LinkedHashMap<>();
            Map<String, Double> got = actual(c, se);
            double tol = num(c, "tol");
            if ("adaptive_sim".equals(c.get("kind"))) {
                int checked = 0;
                for (String key : expect.keySet()) {
                    if (key.endsWith("_se")) {
                        continue;
                    }
                    checked++;
                    assertTrue(got.containsKey(key), name + "." + key + " computed");
                    double ref = num(expect, key);
                    double seRef = num(expect, key + "_se");
                    double band = tol * Math.sqrt(seRef * seRef + se.get(key) * se.get(key));
                    double err = Math.abs(got.get(key) - ref);
                    if (err > band) {
                        failures.add(name + "." + key + ": got " + got.get(key) + " ref " + ref
                                + " err " + err + " > band " + band);
                    }
                }
                assertTrue(checked > 0, name + ": no statistical keys");
                continue;
            }
            assertEquals(expect.keySet(), got.keySet(), name + ": every expect key asserted");
            for (Map.Entry<String, Double> e : got.entrySet()) {
                double want = num(expect, e.getKey());
                if (Math.abs(e.getValue() - want) > tol * Math.max(1.0, Math.abs(want))) {
                    failures.add(name + "." + e.getKey() + ": got " + e.getValue() + " want "
                            + want);
                }
            }
        }
        assertTrue(failures.isEmpty(), String.join("\n", failures));
    }

    // ------------------------------------------------------------ transient impact

    private static double[][] denseA(TransientParams p, int n) {
        double a = p.decay();
        double[][] m = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                m[i][j] = p.kappa() * Math.pow(a, Math.abs(i - j)) - (i == j ? p.gammaP() : 0.0);
            }
        }
        return m;
    }

    /** Gaussian elimination with partial pivoting. */
    private static double[] solve(double[][] a0, double[] b0) {
        int n = b0.length;
        double[][] a = new double[n][];
        for (int i = 0; i < n; i++) {
            a[i] = a0[i].clone();
        }
        double[] b = b0.clone();
        for (int k = 0; k < n; k++) {
            int piv = k;
            for (int i = k + 1; i < n; i++) {
                if (Math.abs(a[i][k]) > Math.abs(a[piv][k])) {
                    piv = i;
                }
            }
            double[] t = a[k];
            a[k] = a[piv];
            a[piv] = t;
            double tb = b[k];
            b[k] = b[piv];
            b[piv] = tb;
            for (int i = k + 1; i < n; i++) {
                double f = a[i][k] / a[k][k];
                for (int j = k; j < n; j++) {
                    a[i][j] -= f * a[k][j];
                }
                b[i] -= f * b[k];
            }
        }
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double s = b[i];
            for (int j = i + 1; j < n; j++) {
                s -= a[i][j] * x[j];
            }
            x[i] = s / a[i][i];
        }
        return x;
    }

    private static double[] trades(double[] x) {
        double[] n = new double[x.length - 1];
        for (int j = 0; j < n.length; j++) {
            n[j] = x[j] - x[j + 1];
        }
        return n;
    }

    @Test
    void costRecursionEqualsQuadraticForm() {
        TransientParams p = new TransientParams(1e6, 8, 0.25, 3e-6, 1.5, 4e-7);
        SplittableRandom rng = new SplittableRandom(1);
        double[] n = new double[8];
        double sum = 0.0;
        for (int j = 0; j < 8; j++) {
            n[j] = rng.nextGaussian() * 1e5 + 1.25e5;
            sum += n[j];
        }
        double[] x = new double[9];
        x[0] = sum;
        for (int j = 1; j <= 8; j++) {
            x[j] = x[j - 1] - n[j - 1];
        }
        x[8] = 0.0;
        n = trades(x);
        double[][] a = denseA(p, 8);
        double quad = 0.0;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                quad += n[i] * a[i][j] * n[j];
            }
        }
        double want = 0.5 * p.gammaP() * x[0] * x[0] + 0.5 * quad;
        assertEquals(1.0, Execution.transientExpectedCost(p, x) / want, 1e-12);
    }

    @Test
    void optimumSolvesTheDenseSystem() {
        for (double g : new double[] {0.0, 4e-7}) {
            TransientParams p = new TransientParams(1e6, 12, 0.25, 3e-6, 1.5, g);
            double[] ones = new double[12];
            java.util.Arrays.fill(ones, 1.0);
            double[] y = solve(denseA(p, 12), ones);
            double s = 0.0;
            for (double v : y) {
                s += v;
            }
            double[] got = trades(Execution.transientOptimalSchedule(p));
            for (int j = 0; j < 12; j++) {
                assertEquals(p.totalShares() * y[j] / s, got[j], 1e-9 * p.totalShares());
            }
        }
    }

    @Test
    void obizhaevaWangShape() {
        TransientParams p = new TransientParams(1e6, 10, 0.1, 2e-6, 5.0, 0.0);
        double[] n = trades(Execution.transientOptimalSchedule(p));
        assertEquals(1.0, n[9] / n[0], 1e-12);
        for (int j = 1; j < 9; j++) {
            assertEquals(n[1], n[j], 1e-12 * p.totalShares());
        }
        assertEquals(1.0 - p.decay(), n[1] / n[0], 1e-12 * (1.0 - p.decay()));
        for (double v : n) {
            assertTrue(v > 0.0);
        }
    }

    @Test
    void optimumBeatsTwapAndPerturbations() {
        TransientParams p = new TransientParams(2.5e5, 26, 1.0 / 26, 4e-6, 2.0, 1e-8);
        double[] x = Execution.transientOptimalSchedule(p);
        double eOpt = Execution.transientExpectedCost(p, x);
        assertTrue(eOpt < Execution.transientExpectedCost(p,
                AlmgrenChriss.twapHoldings(2.5e5, 26)));
        SplittableRandom rng = new SplittableRandom(3);
        for (int t = 0; t < 100; t++) {
            double[] z = x.clone();
            for (int j = 1; j < 26; j++) {
                z[j] += rng.nextGaussian() * 1e3;
            }
            assertTrue(Execution.transientExpectedCost(p, z) >= eOpt);
        }
    }

    @Test
    void fastResilienceReducesToAlmgrenChriss() {
        ACParams ac = new ACParams(1e6, 10, 0.5, 0.95, 2.5e-6, 2.5e-7, 0.0);
        double kappa = 2.0 * ac.eta() / ac.tau();
        TransientParams fast = new TransientParams(ac.totalShares(), ac.nIntervals(), ac.tau(),
                kappa, 1e3, ac.gammaP());
        assertTrue(fast.decay() < 1e-200);
        double[] xOw = Execution.transientOptimalSchedule(fast);
        double[] xAc = AlmgrenChriss.trajectory(ac);
        for (int j = 0; j < xOw.length; j++) {
            assertEquals(xAc[j], xOw[j], 1e-9 * ac.totalShares());
        }
        double[] twap = AlmgrenChriss.twapHoldings(ac.totalShares(), ac.nIntervals());
        double[] block = new double[11];
        block[0] = ac.totalShares();
        for (double[] x : new double[][] {xOw, twap, block}) {
            assertEquals(1.0, Execution.transientExpectedCost(fast, x)
                    / AlmgrenChriss.expectedCost(ac, x), 1e-12);
        }
        assertEquals(1.0, Execution.transientExpectedCost(fast, xOw)
                / AlmgrenChriss.expectedCost(ac), 1e-12);
        double prev = Double.POSITIVE_INFINITY;
        for (double rho : new double[] {0.5, 2.0, 8.0}) {
            TransientParams p = new TransientParams(ac.totalShares(), ac.nIntervals(), ac.tau(),
                    kappa, rho, ac.gammaP());
            double gap = Execution.transientExpectedCost(p, twap)
                    - AlmgrenChriss.expectedCost(ac, twap);
            assertTrue(gap < prev && gap > 0.0);
            prev = gap;
        }
        Impact.Params ip = new Impact.Params(ac.totalShares(), ac.nIntervals(), ac.tau(),
                ac.sigma(), ac.eta(), 1.0, ac.gammaP(), 0.0, 0.0, null);
        assertEquals(1.0, Execution.transientExpectedCost(fast, twap)
                / Impact.expectedCost(ip, twap), 1e-12);
    }

    @Test
    void singleIntervalAndZeroShares() {
        TransientParams p = new TransientParams(5.0, 1, 1.0, 1.0, 1.0);
        double[] x = Execution.transientOptimalSchedule(p);
        assertEquals(5.0, x[0], 0.0);
        assertEquals(0.0, x[1], 0.0);
        assertEquals(12.5, Execution.transientExpectedCost(p, new double[] {5.0, 0.0}), 1e-12);
        for (double v : Execution.transientOptimalSchedule(
                new TransientParams(0.0, 4, 1.0, 1.0, 1.0))) {
            assertEquals(0.0, v, 0.0);
        }
    }

    // ------------------------------------------------------------ adaptive

    private static ACParams ac10() {
        return new ACParams(1e6, 10, 0.5, 0.95, 2.5e-6, 2.5e-7, 2e-6);
    }

    @Test
    void staticPlanIsTheAlmgrenChrissTrajectory() {
        ACParams ac = ac10();
        double[] plan = Execution.replan(ac, new SignalParams(), ac.totalShares(), 0.0, 0);
        double[] x = AlmgrenChriss.trajectory(ac);
        assertEquals(x.length, plan.length);
        for (int j = 0; j < x.length; j++) {
            assertEquals(x[j], plan[j], 1e-9 * ac.totalShares());
        }
    }

    @Test
    void replanIsTimeConsistentWithoutSignal() {
        ACParams ac = ac10();
        SignalParams sig = new SignalParams();
        double[] plan = Execution.replan(ac, sig, ac.totalShares(), 0.0, 0);
        for (int k = 1; k < 10; k++) {
            double[] tail = Execution.replan(ac, sig, plan[k], 0.0, k);
            for (int i = 0; i < tail.length; i++) {
                assertEquals(plan[k + i], tail[i], 1e-9 * ac.totalShares());
            }
        }
    }

    @Test
    void feedbackCoefficientsMatchReplan() {
        ACParams ac = ac10();
        SignalParams sig = new SignalParams(0.6, 0.4, 0.0);
        double[][] cd = Execution.replanCoefficients(ac, sig);
        for (int k = 0; k < 10; k++) {
            int m = 10 - k;
            double[] x = Execution.replan(ac, sig, 3e5, 0.7, k);
            double want = cd[0][m] * 3e5 - cd[1][m] * 0.7;
            assertEquals(1.0, (x[0] - x[1]) / want, 1e-12);
        }
    }

    @Test
    void signalMakesTheTraderHoldWhenPriceIsExpectedToRise() {
        ACParams ac = ac10();
        SignalParams sig = new SignalParams(0.7, 0.5, 0.0);
        assertTrue(Execution.replan(ac, sig, 5e5, 1.0, 3)[1]
                > Execution.replan(ac, sig, 5e5, 0.0, 3)[1]);
    }

    @Test
    void adaptiveEqualsStaticWithoutSignalOrFillRisk() {
        AdaptiveSimResult r = Execution.simulateAdaptive(ac10(), new SignalParams(), 4000, 7);
        assertTrue(Math.abs(r.savingMean()) <= 1e-9 * r.staticMean());
        assertEquals(1.0, r.adaptiveStd() / r.staticStd(), 1e-9);
        assertTrue(Math.abs(r.staticMean() - AlmgrenChriss.expectedCost(ac10()))
                <= 4.0 * r.staticSe());
    }

    @Test
    void adaptiveHelpsWithSignalOrFillRisk() {
        for (SignalParams sig : new SignalParams[] {new SignalParams(0.7, 0.5, 0.0),
            new SignalParams(0.0, 0.0, 0.5)}) {
            AdaptiveSimResult r = Execution.simulateAdaptive(ac10(), sig, 4000, 8);
            assertTrue(r.savingMean() > 4.0 * r.savingSe(), sig + ": " + r);
        }
    }

    @Test
    void adaptiveIsDeterministic() {
        SignalParams sig = new SignalParams(0.5, 0.3, 0.2);
        assertEquals(Execution.simulateAdaptive(ac10(), sig, 100, 5),
                Execution.simulateAdaptive(ac10(), sig, 100, 5));
    }

    // ------------------------------------------------------------ internalisation

    private static InternalisationResult market(double[] buys, double[] sells) {
        return Execution.internalise(buys, sells, 100.0, 0.01, 5e6, 0.015, 13, 1.0 / 13.0);
    }

    @Test
    void crossingNeverIncreasesTotalCost() {
        SplittableRandom rng = new SplittableRandom(11);
        for (int t = 0; t < 200; t++) {
            double[] buys = new double[rng.nextInt(0, 4)];
            for (int i = 0; i < buys.length; i++) {
                buys[i] = rng.nextDouble(1e3, 3e5);
            }
            double[] sells = new double[rng.nextInt(1, 4)];
            for (int i = 0; i < sells.length; i++) {
                sells[i] = rng.nextDouble(1e3, 3e5);
            }
            InternalisationResult r = market(buys, sells);
            assertTrue(r.internalCost() <= r.baselineCost());
            assertTrue(r.savedCost() >= 0.0);
            for (double v : r.buySavedBps()) {
                assertTrue(v >= 0.0);
            }
            for (double v : r.sellSavedBps()) {
                assertTrue(v >= 0.0);
            }
        }
    }

    @Test
    void internalisationAccounting() {
        InternalisationResult r = market(new double[] {3e5, 1e5}, new double[] {2.5e5});
        assertEquals(2.5e5, r.crossedShares(), 0.0);
        assertEquals(1.5e5, r.residualShares(), 0.0);
        assertEquals("buy", r.residualSide());
        double[] q = {3e5, 1e5, 2.5e5};
        double[] per = {r.buySavedBps()[0], r.buySavedBps()[1], r.sellSavedBps()[0]};
        double total = 0.0;
        for (int i = 0; i < 3; i++) {
            total += per[i] * q[i] * 100.0 / 1e4;
        }
        assertEquals(1.0, total / r.savedCost(), 1e-12);
        assertEquals(1.0, r.savedBps() / (1e4 * r.savedCost() / (6.5e5 * 100.0)), 1e-15);
        InternalisationResult one = market(new double[0], new double[] {1e5, 2e5});
        assertEquals(0.0, one.savedCost(), 0.0);
        assertEquals("sell", one.residualSide());
        InternalisationResult even = market(new double[] {2e5}, new double[] {1e5, 1e5});
        assertEquals(0.0, even.internalCost(), 0.0);
        assertEquals("none", even.residualSide());
    }

    // ------------------------------------------------------------ validation

    @ParameterizedTest(name = "[{index}] {0}")
    @MethodSource("transientParamsValidationCases")
    void transientParamsValidation(String fragment, Executable call) {
        assertRejects(fragment, call);
    }

    static List<Arguments> transientParamsValidationCases() {
        List<Arguments> c = new ArrayList<>();
        c.add(bad("total_shares", () -> new TransientParams(-1.0, 10, 0.1, 2e-6, 5.0, 0.0)));
        c.add(bad("n_intervals", () -> new TransientParams(1e6, 0, 0.1, 2e-6, 5.0, 0.0)));
        c.add(bad("n_intervals", () -> new TransientParams(1e6,
                Execution.MAX_EXEC_INTERVALS + 1, 0.1, 2e-6, 5.0, 0.0)));
        c.add(bad("tau", () -> new TransientParams(1e6, 10, 0.0, 2e-6, 5.0, 0.0)));
        c.add(bad("kappa", () -> new TransientParams(1e6, 10, 0.1, 0.0, 5.0, 0.0)));
        c.add(bad("rho", () -> new TransientParams(1e6, 10, 0.1, 2e-6, 0.0, 0.0)));
        c.add(bad("rho", () -> new TransientParams(1e6, 10, 0.1, 2e-6,
                Double.POSITIVE_INFINITY, 0.0)));
        c.add(bad("gamma_p", () -> new TransientParams(1e6, 10, 0.1, 2e-6, 5.0, -1.0)));
        c.add(bad("gamma_p", () -> new TransientParams(1e6, 10, 0.1, 2e-6, 5.0, 1e-6)));
        return c;
    }

    @ParameterizedTest(name = "[{index}] {0}")
    @MethodSource("transientCostValidatesHoldingsCases")
    void transientCostValidatesHoldings(String fragment, Executable call) {
        assertRejects(fragment, call);
    }

    static List<Arguments> transientCostValidatesHoldingsCases() {
        List<Arguments> c = new ArrayList<>();
        TransientParams p = new TransientParams(1e6, 4, 0.25, 2e-6, 5.0);
        c.add(bad("length", () -> Execution.transientExpectedCost(p, new double[] {1.0, 0.0})));
        c.add(bad("holdings", () -> Execution.transientExpectedCost(p, null)));
        c.add(bad("liquidate", () -> Execution.transientExpectedCost(p,
                new double[] {1.0, 0.8, 0.6, 0.4, 0.2})));
        c.add(bad("finite", () -> Execution.transientExpectedCost(p,
                new double[] {1.0, Double.NaN, 0.6, 0.4, 0.0})));
        return c;
    }

    @ParameterizedTest(name = "[{index}] {0}")
    @MethodSource("signalParamsValidationCases")
    void signalParamsValidation(String fragment, Executable call) {
        assertRejects(fragment, call);
    }

    static List<Arguments> signalParamsValidationCases() {
        List<Arguments> c = new ArrayList<>();
        c.add(bad("phi", () -> new SignalParams(1.0, 0.0, 0.0)));
        c.add(bad("phi", () -> new SignalParams(-0.1, 0.0, 0.0)));
        c.add(bad("signal_sd", () -> new SignalParams(0.0, -1.0, 0.0)));
        c.add(bad("fill_risk", () -> new SignalParams(0.0, 0.0, 1.0)));
        c.add(bad("fill_risk", () -> new SignalParams(0.0, 0.0, Double.NaN)));
        return c;
    }

    @ParameterizedTest(name = "[{index}] {0}")
    @MethodSource("adaptiveValidationCases")
    void adaptiveValidation(String fragment, Executable call) {
        assertRejects(fragment, call);
    }

    static List<Arguments> adaptiveValidationCases() {
        List<Arguments> c = new ArrayList<>();
        ACParams ac = ac10();
        SignalParams sig = new SignalParams();
        c.add(bad("n_paths", () -> Execution.simulateAdaptive(ac, sig, 1, 0)));
        c.add(bad("k must", () -> Execution.replan(ac, sig, 1.0, 0.0, 10)));
        c.add(bad("k must", () -> Execution.replan(ac, sig, 1.0, 0.0, -1)));
        c.add(bad("holding", () -> Execution.replan(ac, sig, Double.POSITIVE_INFINITY, 0.0,
                0)));
        c.add(bad("signal", () -> Execution.replan(ac, sig, 1.0, Double.NaN, 0)));
        ACParams big = new ACParams(1.0, 1001, 1.0, 1.0, 1.0, 0.0, 0.0);
        c.add(bad("n_intervals", () -> Execution.simulateAdaptive(big, sig, 100, 0)));
        c.add(bad("n_intervals", () -> Execution.replanCoefficients(big, sig)));
        return c;
    }

    @ParameterizedTest(name = "[{index}] {0}")
    @MethodSource("internaliseValidationCases")
    void internaliseValidation(String fragment, Executable call) {
        assertRejects(fragment, call);
    }

    static List<Arguments> internaliseValidationCases() {
        List<Arguments> c = new ArrayList<>();
        c.add(bad("buys", () -> market(new double[] {0.0}, new double[] {1.0})));
        c.add(bad("sells", () -> market(new double[] {1.0}, new double[] {Double.NaN})));
        c.add(bad("buys", () -> market(null, new double[] {1.0})));
        c.add(bad("buys", () -> market(new double[Execution.MAX_ORDERS + 1],
                new double[] {1.0})));
        c.add(bad("at least one", () -> market(new double[0], new double[0])));
        c.add(bad("n_intervals", () -> Execution.internalise(new double[] {1.0},
                new double[] {2.0}, 100.0, 0.01, 5e6, 0.015, 0, 1.0)));
        c.add(bad("spread", () -> Execution.internalise(new double[] {1.0},
                new double[] {2.0}, 100.0, 0.0, 5e6, 0.015, 13, 1.0 / 13.0)));
        return c;
    }
}
