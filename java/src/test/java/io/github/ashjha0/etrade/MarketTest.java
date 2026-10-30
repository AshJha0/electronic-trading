package io.github.ashjha0.etrade;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import io.github.ashjha0.etrade.Market.ACMarket;
import io.github.ashjha0.etrade.Market.ASMarket;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.UnaryOperator;
import org.junit.Test;

/**
 * Market-unit layer (API_SPEC Section 7): every case of {@code market_golden.json}, the
 * paper identity, the defining properties of the A-S conversion, and validation.
 */
public final class MarketTest {

    private static final Map<String, Object> GOLDEN = load("golden/market_golden.json");
    private static final Map<String, Object> PRESETS = load("presets.json");

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

    private static double numOr(Map<String, Object> m, String key, double fallback) {
        Object v = m.get(key);
        return v == null ? fallback : (Double) v;
    }

    private static ACMarket acFrom(Map<String, Object> v) {
        return new ACMarket(num(v, "price"), num(v, "spread"), num(v, "adv"),
                num(v, "daily_vol"), num(v, "total_shares"), (int) num(v, "n_intervals"),
                num(v, "tau"), num(v, "lambda_ra"),
                numOr(v, "temp_frac", Market.DEFAULT_TEMP_FRAC),
                numOr(v, "perm_frac", Market.DEFAULT_PERM_FRAC));
    }

    private static ASMarket asFrom(Map<String, Object> v) {
        return new ASMarket(num(v, "price"), num(v, "tick"), num(v, "daily_vol"),
                num(v, "touch_fills_per_day"), num(v, "halving_ticks"), num(v, "skew_ticks"),
                (int) num(v, "q_max"), num(v, "liq_ticks"), num(v, "dt_seconds"),
                numOr(v, "session_seconds", Market.DEFAULT_SESSION_SECONDS),
                numOr(v, "horizon_days", 1.0));
    }

    private static Map<String, Double> actual(Map<String, Object> c) {
        Map<String, Object> in = obj(c, "inputs");
        Map<String, Double> out = new LinkedHashMap<>();
        if ("market_ac".equals(c.get("kind"))) {
            ACMarket m = acFrom(in);
            ACParams p = Market.acParamsFromMarket(m);
            double notional = m.totalShares() * m.price();
            out.put("sigma", p.sigma());
            out.put("eta", p.eta());
            out.put("gamma_p", p.gammaP());
            out.put("kappa", AlmgrenChriss.kappa(p));
            out.put("e_cost_bps", Market.bps(AlmgrenChriss.expectedCost(p), notional));
            out.put("sd_cost_bps",
                    Market.bps(Math.sqrt(AlmgrenChriss.varianceCost(p)), notional));
            out.put("spread_cost_bps",
                    Market.bps(Market.halfSpreadCost(m.spread(), m.totalShares()), notional));
        } else {
            ASMarket m = asFrom(in);
            ASParams a = Market.asParamsFromMarket(m);
            out.put("sigma", a.sigma());
            out.put("k", a.k());
            out.put("a_rate", a.aRate());
            out.put("gamma", a.gamma());
            out.put("eta_liq", a.etaLiq());
            out.put("dt", a.dt());
            out.put("n_steps", (double) a.nSteps());
            out.put("spread_open_bps",
                    Market.bps(AvellanedaStoikov.optimalSpread(a, 0.0), m.price()));
            out.put("spread_close_bps",
                    Market.bps(AvellanedaStoikov.optimalSpread(a, a.horizon()), m.price()));
        }
        return out;
    }

    @Test
    public void everyGoldenCaseEveryKey() {
        List<String> failures = new ArrayList<>();
        assertEquals(9, cases().size());
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
    public void presetsCovered() {
        Set<String> names = new HashSet<>();
        for (Map<String, Object> c : cases()) {
            names.add((String) c.get("name"));
        }
        for (String preset : PRESETS.keySet()) {
            Map<String, Object> body = obj(PRESETS, preset);
            for (String side : new String[] {"ac", "as"}) {
                if (body.containsKey(side)) {
                    assertTrue(preset, names.contains("market_" + side + "_" + preset));
                }
            }
        }
    }

    @Test
    public void paperCalibrationReproducesLiquidSet() {
        ACParams p = Market.acParamsFromMarket(
                acFrom(obj(obj(PRESETS, "paper_almgren_chriss_2000"), "ac")));
        assertEquals(2.5e-6, p.eta(), 1e-21);
        assertEquals(2.5e-7, p.gammaP(), 1e-22);
        assertEquals(0.95, p.sigma(), 1e-15);
    }

    @Test
    public void asConversionDefiningProperties() {
        ASMarket m = asFrom(obj(obj(PRESETS, "us_large_cap"), "as"));
        ASParams a = Market.asParamsFromMarket(m);
        double d0 = m.tick() / 2.0;
        double touch = AvellanedaStoikov.intensity(a, d0);
        assertEquals(1.0, touch / m.touchFillsPerDay(), 1e-12);
        assertEquals(0.5, AvellanedaStoikov.intensity(a, d0 + m.halvingTicks() * m.tick()) / touch,
                1e-12);
        double skew = a.s0() - AvellanedaStoikov.reservationPrice(a, a.s0(), 1.0, 0.0);
        assertEquals(1.0, skew / (m.skewTicks() * m.tick()), 1e-12);
        assertEquals(4680, a.nSteps());
    }

    @Test
    public void reportingHelpers() {
        assertEquals(5.0, Market.bps(5.0, 1e4), 0.0);
        assertEquals(5.0, Market.halfSpreadCost(0.01, 1000.0), 1e-15);
        assertThrows(IllegalArgumentException.class, () -> Market.bps(1.0, 0.0));
        assertThrows(IllegalArgumentException.class, () -> Market.bps(Double.NaN, 1.0));
        assertThrows(IllegalArgumentException.class, () -> Market.halfSpreadCost(0.0, 1.0));
    }

    private static void expectAcError(String field, UnaryOperator<ACMarket> mutate) {
        ACMarket base = acFrom(obj(obj(PRESETS, "us_large_cap"), "ac"));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> Market.acParamsFromMarket(mutate.apply(base)));
        assertTrue(field + ": " + e.getMessage(), e.getMessage().contains(field));
    }

    private static void expectAsError(String field, UnaryOperator<ASMarket> mutate) {
        ASMarket base = asFrom(obj(obj(PRESETS, "us_large_cap"), "as"));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> Market.asParamsFromMarket(mutate.apply(base)));
        assertTrue(field + ": " + e.getMessage(), e.getMessage().contains(field));
    }

    @Test
    public void acValidationNamesTheField() {
        expectAcError("price", b -> new ACMarket(0.0, b.spread(), b.adv(), b.dailyVol(),
                b.totalShares(), b.nIntervals(), b.tau(), b.lambdaRa(), b.tempFrac(),
                b.permFrac()));
        expectAcError("spread", b -> new ACMarket(b.price(), -0.01, b.adv(), b.dailyVol(),
                b.totalShares(), b.nIntervals(), b.tau(), b.lambdaRa(), b.tempFrac(),
                b.permFrac()));
        expectAcError("adv", b -> new ACMarket(b.price(), b.spread(), Double.POSITIVE_INFINITY,
                b.dailyVol(), b.totalShares(), b.nIntervals(), b.tau(), b.lambdaRa(),
                b.tempFrac(), b.permFrac()));
        expectAcError("daily_vol", b -> new ACMarket(b.price(), b.spread(), b.adv(), Double.NaN,
                b.totalShares(), b.nIntervals(), b.tau(), b.lambdaRa(), b.tempFrac(),
                b.permFrac()));
        expectAcError("temp_frac", b -> new ACMarket(b.price(), b.spread(), b.adv(),
                b.dailyVol(), b.totalShares(), b.nIntervals(), b.tau(), b.lambdaRa(), 1.5,
                b.permFrac()));
        expectAcError("perm_frac", b -> new ACMarket(b.price(), b.spread(), b.adv(),
                b.dailyVol(), b.totalShares(), b.nIntervals(), b.tau(), b.lambdaRa(),
                b.tempFrac(), 0.0));
    }

    @Test
    public void asValidationNamesTheField() {
        expectAsError("tick", b -> new ASMarket(b.price(), 0.0, b.dailyVol(),
                b.touchFillsPerDay(), b.halvingTicks(), b.skewTicks(), b.qMax(), b.liqTicks(),
                b.dtSeconds(), b.sessionSeconds(), b.horizonDays()));
        expectAsError("touch_fills_per_day", b -> new ASMarket(b.price(), b.tick(), b.dailyVol(),
                0.0, b.halvingTicks(), b.skewTicks(), b.qMax(), b.liqTicks(), b.dtSeconds(),
                b.sessionSeconds(), b.horizonDays()));
        expectAsError("halving_ticks", b -> new ASMarket(b.price(), b.tick(), b.dailyVol(),
                b.touchFillsPerDay(), Double.POSITIVE_INFINITY, b.skewTicks(), b.qMax(),
                b.liqTicks(), b.dtSeconds(), b.sessionSeconds(), b.horizonDays()));
        expectAsError("skew_ticks", b -> new ASMarket(b.price(), b.tick(), b.dailyVol(),
                b.touchFillsPerDay(), b.halvingTicks(), 0.0, b.qMax(), b.liqTicks(),
                b.dtSeconds(), b.sessionSeconds(), b.horizonDays()));
        expectAsError("liq_ticks", b -> new ASMarket(b.price(), b.tick(), b.dailyVol(),
                b.touchFillsPerDay(), b.halvingTicks(), b.skewTicks(), b.qMax(), -1.0,
                b.dtSeconds(), b.sessionSeconds(), b.horizonDays()));
        expectAsError("session_seconds", b -> new ASMarket(b.price(), b.tick(), b.dailyVol(),
                b.touchFillsPerDay(), b.halvingTicks(), b.skewTicks(), b.qMax(), b.liqTicks(),
                b.dtSeconds(), Double.NaN, b.horizonDays()));
        expectAsError("multiple of dt", b -> new ASMarket(b.price(), b.tick(), b.dailyVol(),
                b.touchFillsPerDay(), b.halvingTicks(), b.skewTicks(), b.qMax(), b.liqTicks(),
                7.0, b.sessionSeconds(), b.horizonDays()));
    }
}
