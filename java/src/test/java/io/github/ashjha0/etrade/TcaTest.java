package io.github.ashjha0.etrade;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import io.github.ashjha0.etrade.Tca.MarkoutResult;
import io.github.ashjha0.etrade.Tca.ParticipationResult;
import io.github.ashjha0.etrade.Tca.SlippageReport;
import io.github.ashjha0.etrade.Tca.WelchResult;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import org.junit.function.ThrowingRunnable;

/**
 * Post-trade TCA (API_SPEC Section 10): every case of {@code tca_golden.json}, sign
 * conventions, invariances and validation.
 */
public final class TcaTest {

    private static final Map<String, Object> GOLDEN = load("golden/tca_golden.json");

    private static final double[] VOLS = {1000.0, 2000.0, 1500.0, 500.0};
    private static final double[] VWAPS = {100.00, 100.05, 100.12, 100.08};
    private static final int[] BINS = {0, 1, 1, 2};
    private static final double[] QTYS = {100.0, 150.0, 50.0, 200.0};
    private static final double[] PXS = {100.02, 100.06, 100.07, 100.15};

    @SuppressWarnings("unchecked")
    private static Map<String, Object> load(String rel) {
        try {
            return (Map<String, Object>) Json.parse(Files.readString(Path.of("../data", rel)));
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

    private static double[] arr(Map<String, Object> in, String prefix, String nKey) {
        double[] out = new double[(int) num(in, nKey)];
        for (int i = 0; i < out.length; i++) {
            out[i] = num(in, prefix + "_" + i);
        }
        return out;
    }

    private static int[] ints(Map<String, Object> in, String prefix, String nKey) {
        double[] v = arr(in, prefix, nKey);
        int[] out = new int[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = (int) v[i];
        }
        return out;
    }

    private static String side(Map<String, Object> in) {
        return num(in, "side_buy") == 1.0 ? "buy" : "sell";
    }

    private static void putIndexed(Map<String, Double> out, String prefix, double[] xs) {
        for (int i = 0; i < xs.length; i++) {
            out.put(prefix + "_" + i, xs[i]);
        }
    }

    private static Map<String, Double> actual(Map<String, Object> c) {
        Map<String, Object> in = obj(c, "inputs");
        Map<String, Double> out = new LinkedHashMap<>();
        switch ((String) c.get("kind")) {
            case "slippage": {
                SlippageReport r = Tca.slippage(side(in), num(in, "arrival_price"),
                        num(in, "close_price"), ints(in, "bin", "n_fills"),
                        arr(in, "qty", "n_fills"), arr(in, "px", "n_fills"),
                        arr(in, "vol", "n_bins"), arr(in, "vwap", "n_bins"));
                out.put("executed_qty", r.executedQty());
                out.put("avg_price", r.avgPrice());
                out.put("market_vwap", r.marketVwap());
                out.put("interval_vwap", r.intervalVwap());
                out.put("arrival_bps", r.arrivalBps());
                out.put("vwap_bps", r.vwapBps());
                out.put("interval_vwap_bps", r.intervalVwapBps());
                out.put("close_bps", r.closeBps());
                break;
            }
            case "markout": {
                MarkoutResult m = Tca.markouts(side(in), ints(in, "step", "n_fills"),
                        arr(in, "qty", "n_fills"), arr(in, "mid", "n_mids"),
                        ints(in, "h", "n_horizons"));
                putIndexed(out, "per_fill", m.perFill());
                putIndexed(out, "per_share", m.perShare());
                break;
            }
            case "participation": {
                ParticipationResult p = Tca.participation(ints(in, "bin", "n_fills"),
                        arr(in, "qty", "n_fills"), arr(in, "vol", "n_bins"));
                putIndexed(out, "rate", p.perBin());
                out.put("overall", p.overall());
                break;
            }
            case "welch": {
                WelchResult w = Tca.welchTTest(arr(in, "a", "n_a"), arr(in, "b", "n_b"));
                out.put("mean_diff", w.meanDiff());
                out.put("t_stat", w.tStat());
                out.put("df", w.df());
                out.put("p_value", w.pValue());
                break;
            }
            default:
                out.put("cdf", Tca.studentTCdf(num(in, "t"), num(in, "df")));
        }
        return out;
    }

    @Test
    public void everyGoldenCaseEveryKey() {
        List<String> failures = new ArrayList<>();
        assertEquals(11, cases().size());
        for (Map<String, Object> c : cases()) {
            String name = (String) c.get("name");
            Map<String, Object> expect = obj(c, "expect");
            Map<String, Double> got = actual(c);
            assertEquals(name + ": every expect key asserted", expect.keySet(), got.keySet());
            double tol = num(c, "tol");
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

    @Test
    public void buySellSymmetry() {
        SlippageReport b = Tca.slippage("buy", 100.01, 100.09, BINS, QTYS, PXS, VOLS, VWAPS);
        SlippageReport s = Tca.slippage("sell", 100.01, 100.09, BINS, QTYS, PXS, VOLS, VWAPS);
        assertEquals(-s.arrivalBps(), b.arrivalBps(), 1e-12);
        assertEquals(-s.vwapBps(), b.vwapBps(), 1e-12);
        assertEquals(-s.intervalVwapBps(), b.intervalVwapBps(), 1e-12);
        assertEquals(-s.closeBps(), b.closeBps(), 1e-12);
    }

    @Test
    public void priceScaleAndShiftInvariance() {
        double c = 3.7;
        double k = 25.0;
        double[] pxC = new double[PXS.length];
        double[] pxK = new double[PXS.length];
        for (int i = 0; i < PXS.length; i++) {
            pxC[i] = PXS[i] * c;
            pxK[i] = PXS[i] + k;
        }
        double[] vwC = new double[VWAPS.length];
        double[] vwK = new double[VWAPS.length];
        for (int i = 0; i < VWAPS.length; i++) {
            vwC[i] = VWAPS[i] * c;
            vwK[i] = VWAPS[i] + k;
        }
        SlippageReport r1 = Tca.slippage("buy", 100.01, 100.09, BINS, QTYS, PXS, VOLS, VWAPS);
        SlippageReport r2 = Tca.slippage("buy", 100.01 * c, 100.09 * c, BINS, QTYS, pxC, VOLS,
                vwC);
        assertEquals(r1.arrivalBps(), r2.arrivalBps(), 1e-10 * Math.abs(r1.arrivalBps()));
        assertEquals(r1.vwapBps(), r2.vwapBps(), 1e-10 * Math.abs(r1.vwapBps()));
        SlippageReport r3 = Tca.slippage("buy", 100.01 + k, 100.09 + k, BINS, QTYS, pxK, VOLS,
                vwK);
        assertEquals(r1.arrivalBps() * 100.01, r3.arrivalBps() * (100.01 + k), 1e-5);
        assertEquals(r1.vwapBps() * r1.marketVwap(), r3.vwapBps() * r3.marketVwap(), 1e-5);
        assertEquals(r1.closeBps() * 100.09, r3.closeBps() * (100.09 + k), 1e-5);
    }

    @Test
    public void arrivalSlippageIsPeroldTradingCost() {
        SlippageReport r = Tca.slippage("buy", 100.01, 100.09, BINS, QTYS, PXS, VOLS, VWAPS);
        Analytics.ISDecomposition d = Analytics.decomposeShortfall("buy", 100.01, 100.01,
                100.09, 500.0, QTYS, PXS);
        assertEquals(d.trading(), r.arrivalBps() * 100.01 * r.executedQty() / 1e4, 1e-9);
    }

    @Test
    public void markoutSign() {
        double[] mids = {10.0, 10.5, 9.0};
        MarkoutResult buy = Tca.markouts("buy", new int[] {0, 1}, new double[] {1.0, 3.0}, mids,
                new int[] {0, 1});
        assertEquals(0.0, buy.perFill()[0], 0.0);
        assertEquals(-0.5, buy.perFill()[1], 1e-15);
        assertEquals(-1.0, buy.perShare()[1], 1e-15);
        MarkoutResult sell = Tca.markouts("sell", new int[] {0, 1}, new double[] {1.0, 3.0},
                mids, new int[] {0, 1});
        assertEquals(0.5, sell.perFill()[1], 1e-15);
    }

    @Test
    public void knownTValues() {
        assertEquals(0.75, Tca.studentTCdf(1.0, 1.0), 1e-14);
        assertEquals(0.5, Tca.studentTCdf(0.0, 4.0), 0.0);
        assertEquals(0.5 + 1.5 / (2.0 * Math.sqrt(4.25)), Tca.studentTCdf(1.5, 2.0), 1e-14);
        assertEquals(0.5 * Math.log(Math.PI), Tca.lnGamma(0.5), 1e-14);
        assertEquals(Math.log(362880.0), Tca.lnGamma(10.0), 1e-13);
    }

    private static void invalid(String fragment, ThrowingRunnable r) {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, r);
        assertTrue(e.getMessage(), e.getMessage().contains(fragment));
    }

    @Test
    public void validationNamesTheArgument() {
        invalid("side", () -> Tca.slippage("hold", 1, 1, BINS, QTYS, PXS, VOLS, VWAPS));
        invalid("arrival_price", () -> Tca.slippage("buy", 0, 1, BINS, QTYS, PXS, VOLS, VWAPS));
        invalid("fill_bins must be integers in [0, 4)", () -> Tca.slippage("buy", 1, 1,
                new int[] {0, 1, 1, 4}, QTYS, PXS, VOLS, VWAPS));
        invalid("fill_bins", () -> Tca.slippage("buy", 1, 1, new int[] {0, 1, 1, -1}, QTYS,
                PXS, VOLS, VWAPS));
        invalid("executed quantity", () -> Tca.slippage("buy", 1, 1, BINS, new double[4], PXS,
                VOLS, VWAPS));
        invalid("traded bins", () -> Tca.slippage("buy", 1, 1, BINS, QTYS, PXS,
                new double[] {0, 0, 0, 1}, VWAPS));
        invalid("non-empty", () -> Tca.slippage("buy", 1, 1, new int[0], new double[0],
                new double[0], VOLS, VWAPS));
        invalid("exceeds mid series", () -> Tca.markouts("buy", new int[] {0, 1},
                new double[] {1, 1}, new double[] {1, 1}, new int[] {1}));
        invalid("horizons", () -> Tca.markouts("buy", new int[] {0}, new double[] {1},
                new double[] {1, 1}, new int[] {-1}));
        invalid("exceeds market volume", () -> Tca.participation(new int[] {0},
                new double[] {1001}, VOLS));
        invalid("total market volume", () -> Tca.participation(new int[] {0},
                new double[] {0}, new double[] {0}));
        invalid("a must have at least 2", () -> Tca.welchTTest(new double[] {1},
                new double[] {1, 2}));
        invalid("zero variance", () -> Tca.welchTTest(new double[] {1, 1},
                new double[] {2, 2}));
        invalid("df", () -> Tca.studentTCdf(1.0, 0.0));
        invalid("x must be in", () -> Tca.betaincReg(1.0, 1.0, 1.5));
        invalid("x must be finite", () -> Tca.lnGamma(-1.0));
    }
}
