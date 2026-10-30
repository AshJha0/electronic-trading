package io.github.ashjha0.etrade;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import io.github.ashjha0.etrade.Market.ImpactMarket;
import io.github.ashjha0.etrade.Market.PretradeReport;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.Test;

/**
 * Power-law impact execution (API_SPEC Section 8): every case of
 * {@code impact_golden.json}, reduction to Almgren-Chriss, the square-root law, drift and
 * early finish, VWAP, the report and validation.
 */
public final class ImpactTest {

    private static final Map<String, Object> GOLDEN = load();

    @SuppressWarnings("unchecked")
    private static Map<String, Object> load() {
        try {
            return (Map<String, Object>) Json.parse(
                    Files.readString(Path.of("../data/golden/impact_golden.json")));
        } catch (IOException e) {
            throw new IllegalStateException("cannot load impact_golden.json", e);
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

    private static ImpactMarket marketFrom(Map<String, Object> in) {
        return new ImpactMarket(num(in, "price"), num(in, "spread"), num(in, "adv"),
                num(in, "daily_vol"), num(in, "total_shares"), (int) num(in, "n_intervals"),
                num(in, "tau"), numOr(in, "lambda_ra", 0.0), numOr(in, "alpha", 0.5),
                numOr(in, "y", 1.0), numOr(in, "perm_frac", Market.DEFAULT_PERM_FRAC),
                numOr(in, "drift_bps_per_day", 0.0),
                num(in, "volume_curve") == 1.0 ? Impact.usEquityVolumeCurve13() : null);
    }

    private static Map<String, Double> actual(Map<String, Object> c) {
        String kind = (String) c.get("kind");
        Map<String, Object> in = obj(c, "inputs");
        Map<String, Double> out = new LinkedHashMap<>();
        if ("sqrt_law".equals(kind)) {
            out.put("bps", Impact.sqrtLawCostBps(num(in, "daily_vol"), num(in, "order_shares"),
                    num(in, "adv"), num(in, "y")));
            return out;
        }
        ImpactMarket m = marketFrom(in);
        if ("pretrade".equals(kind)) {
            int nl = (int) num(in, "n_lambdas");
            double[] lambdas = new double[nl];
            for (int i = 0; i < nl; i++) {
                lambdas[i] = num(in, "lambda_" + i);
            }
            PretradeReport rep = Market.pretradeReport(m, lambdas);
            out.put("sqrt_law_bps", rep.sqrtLawBps());
            out.put("half_spread_bps", rep.halfSpreadBps());
            for (int i = 0; i < rep.rows().size(); i++) {
                out.put("row" + i + "_impact_bps", rep.rows().get(i).impactBps());
                out.put("row" + i + "_sd_bps", rep.rows().get(i).sdBps());
                out.put("row" + i + "_horizon", (double) rep.rows().get(i).horizon());
            }
            return out;
        }
        Impact.Params p = Market.impactParamsFromMarket(m);
        if ("impact_vwap".equals(kind)) {
            double[] x = Impact.vwapHoldings(p.totalShares(), p.weights());
            out.put("e_cost", Impact.expectedCost(p, x));
            out.put("var_cost", Impact.varianceCost(p, x));
            return out;
        }
        Impact.ScheduleResult r = Impact.optimalSchedule(p);
        double[] x = r.holdings();
        for (int j = 1; j < p.nIntervals(); j++) {
            out.put(String.format("xfrac_%02d", j), x[j] / p.totalShares());
        }
        out.put("e_cost", Impact.expectedCost(p, x));
        out.put("var_cost", Impact.varianceCost(p, x));
        out.put("horizon", (double) r.horizon());
        return out;
    }

    private static ImpactMarket largeCap() {
        return ImpactMarket.of(100.0, 0.01, 5e6, 0.015, 250_000.0, 13, 1.0 / 13.0);
    }

    @Test
    public void everyGoldenCaseEveryKey() {
        assertEquals(8, cases().size());
        List<String> failures = new ArrayList<>();
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
    public void alphaOneReproducesAlmgrenChriss() {
        ACParams ac = new ACParams(1e6, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6);
        Impact.Params ip = new Impact.Params(1e6, 5, 1.0, 0.95, 2.5e-6, 1.0, 2.5e-7, 2e-6, 0.0,
                null);
        Impact.ScheduleResult r = Impact.optimalSchedule(ip);
        double[] x = AlmgrenChriss.trajectory(ac);
        double[] h = r.holdings();
        for (int j = 0; j < x.length; j++) {
            assertEquals(x[j], h[j], 1e-10 * 1e6);
        }
        assertEquals(1.0, Impact.expectedCost(ip, h) / AlmgrenChriss.expectedCost(ac), 1e-12);
        assertTrue(r.iterations() <= 3);
    }

    @Test
    public void oneDayTwapIsTheSquareRootLaw() {
        ImpactMarket m = largeCap();
        Impact.Params p = Market.impactParamsFromMarket(m);
        double[] x = AlmgrenChriss.twapHoldings(m.totalShares(), m.nIntervals());
        double perm = 0.5 * p.gammaP() * m.totalShares() * m.totalShares() * (1.0 - 1.0 / 13.0);
        double law = Market.bps(Impact.expectedCost(p, x) - perm, m.totalShares() * m.price());
        assertEquals(1.0, law / Impact.sqrtLawCostBps(0.015, 250_000.0, 5e6, 1.0), 1e-12);
    }

    @Test
    public void driftShortensTheHorizon() {
        int prev = Integer.MAX_VALUE;
        for (double drift : new double[] {0.0, 50.0, 200.0, 1000.0, 5000.0}) {
            Impact.ScheduleResult r = Impact.optimalSchedule(
                    Market.impactParamsFromMarket(largeCap().withDriftBpsPerDay(drift)));
            assertTrue(r.horizon() <= prev);
            prev = r.horizon();
            double[] h = r.holdings();
            for (int k = r.horizon(); k < h.length; k++) {
                assertEquals(0.0, h[k], 0.0);
            }
        }
        assertTrue(prev < 13);
    }

    @Test
    public void vwapBeatsTwapUnderTheCurve() {
        double[] xu = Impact.vwapHoldings(1e6, new double[] {1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
        double[] xt = AlmgrenChriss.twapHoldings(1e6, 10);
        for (int j = 0; j < xu.length; j++) {
            assertEquals(xt[j], xu[j], 1e-6);
        }
        ImpactMarket m = largeCap().withWeights(Impact.usEquityVolumeCurve13());
        Impact.Params p = Market.impactParamsFromMarket(m);
        assertTrue(Impact.expectedCost(p, Impact.vwapHoldings(m.totalShares(), m.weights()))
                < Impact.expectedCost(p, AlmgrenChriss.twapHoldings(m.totalShares(), 13)));
    }

    @Test
    public void pretradeReportStructure() {
        PretradeReport rep = Market.pretradeReport(largeCap(), new double[] {0.0, 1e-6, 1e-5});
        assertEquals(5, rep.rows().size());
        assertEquals("twap", rep.rows().get(0).label());
        assertEquals("vwap", rep.rows().get(1).label());
        assertEquals("optimal", rep.rows().get(4).label());
        for (Market.PretradeRow r : rep.rows()) {
            assertEquals(r.impactBps() + rep.halfSpreadBps(), r.totalBps(), 0.0);
        }
        assertTrue(rep.rows().get(2).sdBps() > rep.rows().get(3).sdBps());
        assertTrue(rep.rows().get(3).sdBps() > rep.rows().get(4).sdBps());
        assertThrows(IllegalArgumentException.class,
                () -> Market.pretradeReport(largeCap(), new double[0]));
    }

    @Test
    public void validation() {
        assertThrows(IllegalArgumentException.class,
                () -> new Impact.Params(1e5, 13, 1.0 / 13, 1.5, 1e-3, 0.0, 2e-8, 0.0, 0.0, null));
        assertThrows(IllegalArgumentException.class,
                () -> new Impact.Params(1e5, 13, 1.0 / 13, 1.5, 1e-3, 1.5, 2e-8, 0.0, 0.0, null));
        assertThrows(IllegalArgumentException.class,
                () -> new Impact.Params(1e5, 13, 1.0 / 13, 1.5, 0.0, 0.5, 2e-8, 0.0, 0.0, null));
        assertThrows(IllegalArgumentException.class,
                () -> new Impact.Params(1e5, 13, 1.0 / 13, 1.5, 1e-3, 0.5, 2e-8, 0.0, -1.0, null));
        assertThrows(IllegalArgumentException.class,
                () -> new Impact.Params(1e5, 10_001, 1e-4, 1.5, 1e-3, 0.5, 2e-8, 0.0, 0.0, null));
        assertThrows(IllegalArgumentException.class,
                () -> new Impact.Params(1e5, 13, 1.0 / 13, 1.5, 1e-3, 0.5, 2e-8, 0.0, 0.0,
                        new double[12]));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> new Impact.Params(1e8, 13, 1.0 / 13, 1.5, 1e-6, 0.5, 1e-3, 0.0, 0.0, null));
        assertTrue(e.getMessage().contains("convex"));
        Impact.Params p = Market.impactParamsFromMarket(largeCap());
        assertThrows(IllegalArgumentException.class,
                () -> Impact.expectedCost(p, new double[] {1.0, 0.0}));
        double[] up = AlmgrenChriss.twapHoldings(250_000.0, 13);
        up[3] = up[1] + 1.0;
        assertThrows(IllegalArgumentException.class, () -> Impact.expectedCost(p, up));
        assertThrows(IllegalArgumentException.class,
                () -> Impact.sqrtLawCostBps(0.0, 1.0, 1.0, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> Impact.vwapHoldings(1.0, new double[0]));
        assertThrows(IllegalArgumentException.class,
                () -> Market.impactParamsFromMarket(largeCap().withDriftBpsPerDay(-1.0)));
    }

    @Test
    public void zeroSharesIsTheZeroSchedule() {
        Impact.ScheduleResult r = Impact.optimalSchedule(Market.impactParamsFromMarket(
                largeCap().withTotalShares(0.0).withLambdaRa(1e-6)));
        for (double v : r.holdings()) {
            assertEquals(0.0, v, 0.0);
        }
        assertEquals(0, r.iterations());
    }
}
