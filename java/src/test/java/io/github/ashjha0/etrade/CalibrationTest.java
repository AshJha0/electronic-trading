package io.github.ashjha0.etrade;

import static io.github.ashjha0.etrade.Rejects.assertRejects;
import static io.github.ashjha0.etrade.Rejects.bad;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.ashjha0.etrade.Calibration.FillIntensity;
import io.github.ashjha0.etrade.Calibration.SpreadStats;
import io.github.ashjha0.etrade.Calibration.VolatilityCurve;
import io.github.ashjha0.etrade.Market.ASMarket;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Calibration estimators (API_SPEC Section 11): every case of {@code calibration_golden.json},
 * defining properties, the round trip into the market-unit layer, and validation.
 */
final class CalibrationTest {

    private static final Map<String, Object> GOLDEN = load("golden/calibration_golden.json");

    @SuppressWarnings("unchecked")
    private static Map<String, Object> load(String rel) {
        try {
            return (Map<String, Object>) TestData.load(rel);
        } catch (IOException e) {
            throw new IllegalStateException("cannot load " + rel, e);
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> obj(Map<String, Object> m, String key) {
        return (Map<String, Object>) m.get(key);
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> cases() {
        return (List<Map<String, Object>>) (List<?>) GOLDEN.get("cases");
    }

    private static double num(Map<String, Object> m, String key) {
        Object v = m.get(key);
        if (v == null) {
            throw new IllegalStateException("missing key " + key);
        }
        return (Double) v;
    }

    private static double[] array(Map<String, Object> in, String name) {
        int n = (int) num(in, "n");
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = num(in, name + "_" + i);
        }
        return out;
    }

    private static String idx(String prefix, int j) {
        return String.format("%s_%02d", prefix, j);
    }

    private static Map<String, Double> actual(Map<String, Object> c) {
        Map<String, Object> in = obj(c, "inputs");
        Map<String, Double> out = new LinkedHashMap<>();
        String kind = (String) c.get("kind");
        switch (kind) {
            case "cal_volume_curve" -> {
                double[] w = Calibration.volumeCurve(array(in, "times"), array(in, "sizes"),
                        num(in, "session_start"), num(in, "session_end"),
                        (int) num(in, "n_bins"));
                for (int j = 0; j < w.length; j++) {
                    out.put(idx("frac", j), w[j]);
                }
            }
            case "cal_volatility" -> {
                VolatilityCurve v = Calibration.volatilityCurve(array(in, "times"),
                        array(in, "mids"), num(in, "session_start"), num(in, "session_end"),
                        (int) num(in, "n_bins"));
                out.put("daily_vol", v.dailyVol());
                out.put("n_returns", (double) v.nReturns());
                for (int j = 0; j < v.binVol().length; j++) {
                    out.put(idx("bin_vol", j), v.binVol()[j]);
                }
            }
            case "cal_spread" -> {
                SpreadStats s = Calibration.spreadStats(array(in, "spreads"), array(in, "mids"),
                        num(in, "tick"));
                out.put("mean_ticks", s.meanTicks());
                out.put("median_ticks", s.medianTicks());
                out.put("mean_bps", s.meanBps());
                out.put("median_bps", s.medianBps());
                out.put("n", (double) s.n());
            }
            case "cal_fill_intensity" -> {
                FillIntensity f = Calibration.fillIntensity(array(in, "distances"),
                        num(in, "tick"), (int) num(in, "n_levels"), num(in, "days"),
                        num(in, "sides"));
                out.put("a_rate", f.aRate());
                out.put("k", f.k());
                out.put("touch_fills_per_day", f.touchFillsPerDay());
                out.put("halving_ticks", f.halvingTicks());
                out.put("n_buckets", (double) f.nBuckets());
            }
            case "cal_fitted_curve" -> {
                for (int j = 0; j < Calibration.FITTED_VOLUME_CURVE_13.size(); j++) {
                    out.put(idx("w", j), Calibration.FITTED_VOLUME_CURVE_13.get(j));
                }
            }
            default -> throw new IllegalStateException("unknown kind " + kind);
        }
        return out;
    }

    @Test
    void everyGoldenCaseEveryKey() {
        List<String> failures = new ArrayList<>();
        assertEquals(9, cases().size());
        for (Map<String, Object> c : cases()) {
            String name = (String) c.get("name");
            Map<String, Object> expect = obj(c, "expect");
            Map<String, Double> got = actual(c);
            assertEquals(expect.keySet(), got.keySet(), name + ": every expect key asserted");
            double tol = num(c, "tol");
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

    @Test
    void fittedCurveSumsToOneAndIsImmutable() {
        double[] w = Calibration.fittedVolumeCurve13();
        assertEquals(13, w.length);
        double sum = 0.0;
        for (double v : w) {
            sum += v;
        }
        assertEquals(1.0, sum, 1e-12);
        w[0] = 99.0;
        assertEquals(0.1987, Calibration.fittedVolumeCurve13()[0], 0.0);
        assertThrows(UnsupportedOperationException.class,
                () -> Calibration.FITTED_VOLUME_CURVE_13.set(0, 1.0));
        double[] x = Impact.vwapHoldings(1000.0, Calibration.fittedVolumeCurve13());
        assertEquals(1000.0, x[0], 0.0);
        assertEquals(0.0, x[x.length - 1], 0.0);
    }

    @Test
    void volumeCurveIsADistribution() {
        double[] w = Calibration.volumeCurve(new double[] {1.0, 2.0, 3.0, 7.0},
                new double[] {10.0, 20.0, 30.0, 40.0}, 0.0, 8.0, 2);
        assertArrayEquals(new double[] {0.6, 0.4}, w, 1e-15);
    }

    @Test
    void volatilityOfAConstantReturn() {
        double r = 0.001;
        double[] t = new double[11];
        double[] mids = new double[11];
        for (int i = 0; i < 11; i++) {
            t[i] = i;
            mids[i] = 100.0 * Math.exp(r * i);
        }
        VolatilityCurve v = Calibration.volatilityCurve(t, mids, 0.0, 10.0, 1);
        assertEquals(1.0, v.dailyVol() / (r * Math.sqrt(10.0)), 1e-9);
        assertEquals(10, v.nReturns());
    }

    private static double[] exponentialDistances(double tick) {
        int[] counts = {1024, 512, 256, 128, 64};
        List<Double> dist = new ArrayList<>();
        for (int b = 0; b < counts.length; b++) {
            int nxt = b + 1 < counts.length ? counts[b + 1] : 0;
            for (int i = 0; i < counts[b] - nxt; i++) {
                dist.add((b + 0.5) * tick);
            }
        }
        return dist.stream().mapToDouble(Double::doubleValue).toArray();
    }

    @Test
    void fillIntensityRecoversAnExactExponential() {
        double tick = 0.01;
        FillIntensity f = Calibration.fillIntensity(exponentialDistances(tick), tick, 5);
        assertEquals(1.0, f.k() / (-Math.log(0.5) / tick), 1e-12);
        assertEquals(1.0, f.halvingTicks(), 1e-12);
        assertEquals(1.0, f.touchFillsPerDay() / 512.0, 1e-12);
        assertEquals(5, f.nBuckets());
    }

    @Test
    void fillIntensityRoundTripsThroughTheMarketLayer() {
        Map<String, Object> in = null;
        for (Map<String, Object> c : cases()) {
            if ("cal_fill_intensity".equals(c.get("kind"))) {
                in = obj(c, "inputs");
                break;
            }
        }
        double tick = num(in, "tick");
        FillIntensity f = Calibration.fillIntensity(array(in, "distances"), tick,
                (int) num(in, "n_levels"));
        ASParams a = Market.asParamsFromMarket(new ASMarket(100.0, tick, 0.01,
                f.touchFillsPerDay(), f.halvingTicks(), 1.0, 10, 1.0, 5.0,
                Market.DEFAULT_SESSION_SECONDS, 1.0));
        assertEquals(1.0, a.k() / f.k(), 1e-12);
        assertEquals(1.0, a.aRate() / f.aRate(), 1e-12);
    }

    private static double[] a(double... v) {
        return v;
    }

    @ParameterizedTest(name = "[{index}] {0}")
    @MethodSource("validationMessagesCases")
    void validationMessages(String fragment, Executable call) {
        assertRejects(fragment, call);
    }

    static List<Arguments> validationMessagesCases() {
        List<Arguments> c = new ArrayList<>();
        c.add(bad("session_end must be finite and > session_start",
                () -> Calibration.volumeCurve(a(0.0), a(1.0), 0.0, 0.0, 1)));
        c.add(bad("n_bins must be an integer",
                () -> Calibration.volumeCurve(a(0.0), a(1.0), 0.0, 1.0, 0)));
        c.add(bad("times and sizes must have the same length",
                () -> Calibration.volumeCurve(a(0.0), a(1.0, 2.0), 0.0, 1.0, 1)));
        c.add(bad("sizes must be finite and >= 0",
                () -> Calibration.volumeCurve(a(0.0), a(-1.0), 0.0, 1.0, 1)));
        c.add(bad("no volume inside the session",
                () -> Calibration.volumeCurve(a(5.0), a(1.0), 0.0, 1.0, 1)));
        c.add(bad("mids must be finite and > 0",
                () -> Calibration.volatilityCurve(a(0.0, 1.0), a(1.0, 0.0), 0.0, 1.0, 1)));
        c.add(bad("times must be non-decreasing",
                () -> Calibration.volatilityCurve(a(1.0, 0.0), a(1.0, 1.0), 0.0, 1.0, 1)));
        c.add(bad("no return inside the session",
                () -> Calibration.volatilityCurve(a(0.0), a(1.0), 0.0, 1.0, 1)));
        c.add(bad("spreads must not be empty",
                () -> Calibration.spreadStats(a(), a(), 0.01)));
        c.add(bad("tick must be finite and > 0",
                () -> Calibration.spreadStats(a(0.01), a(100.0), 0.0)));
        c.add(bad("spreads must be finite and >= 0",
                () -> Calibration.spreadStats(a(-0.01), a(100.0), 0.01)));
        c.add(bad("n_levels must be an integer",
                () -> Calibration.fillIntensity(a(0.005), 0.01, 1)));
        c.add(bad("days must be finite and > 0",
                () -> Calibration.fillIntensity(a(0.005), 0.01, 3, 0.0, 2.0)));
        c.add(bad("sides must be finite and > 0",
                () -> Calibration.fillIntensity(a(0.005), 0.01, 3, 1.0, 0.0)));
        c.add(bad("distances must be finite and >= 0",
                () -> Calibration.fillIntensity(a(-1.0), 0.01, 3)));
        c.add(bad("fewer than two distance buckets have fills",
                () -> Calibration.fillIntensity(a(0.005), 0.01, 3)));
        c.add(bad("fitted fill intensity does not decay with distance",
                () -> Calibration.fillIntensity(a(0.015, 0.015), 0.01, 3)));
        return c;
    }
}
