package io.github.ashjha0.etrade;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import io.github.ashjha0.etrade.Hedging.HedgeConfig;
import io.github.ashjha0.etrade.Hedging.Result;
import io.github.ashjha0.etrade.Hedging.RiskConfig;
import io.github.ashjha0.etrade.Hedging.Stat;
import io.github.ashjha0.etrade.Market.ASMarket;
import io.github.ashjha0.etrade.MarketMaking.Agent;
import io.github.ashjha0.etrade.MarketMaking.Config;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.SplittableRandom;
import org.junit.Test;

/**
 * Hedged market making and risk controls (API_SPEC Section 14): every case of
 * {@code hedging_golden.json}, the attribution identity on every path, no hedge equals
 * Section 9 in distribution, the loss-limit overshoot bound, the effect of each control,
 * determinism and validation.
 */
public final class HedgingTest {


    /** Paths per statistical case (the spec asks for at least 2000). */
    private static final int N_TEST = 2000;

    private static final ASParams LARGE = Market.asParamsFromMarket(new ASMarket(
            100.0, 0.01, 0.015, 1170.0, 0.462, 2.25, 50, 1.0, 5.0, 23400.0, 1.0));

    private static final Config BOOK = Config.defaults().withTick(0.01).withMakerFee(0.002)
            .withInformedProb(0.15);
    private static final Config FIXED = BOOK.withAgent(Agent.FIXED);
    private static final HedgeConfig HEDGE = new HedgeConfig(0.9, 0.9, 0.005, 0.0, 0.0);
    private static final HedgeConfig NO_HEDGE = HedgeConfig.defaults();
    private static final RiskConfig NO_RISK = RiskConfig.defaults();

    @SuppressWarnings("unchecked")
    private static Map<String, Object> load() {
        try {
            return (Map<String, Object>) Json.parse(
                    Files.readString(Path.of("../data/golden/hedging_golden.json")));
        } catch (IOException e) {
            throw new IllegalStateException("cannot load hedging_golden.json", e);
        }
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> cases() {
        return (List<Map<String, Object>>) (List<?>) load().get("cases");
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

    private static ASParams paramsFrom(Map<String, Object> in) {
        return Market.asParamsFromMarket(new ASMarket(num(in, "price"), num(in, "tick"),
                num(in, "daily_vol"), num(in, "touch_fills_per_day"), num(in, "halving_ticks"),
                num(in, "skew_ticks"), (int) num(in, "q_max"), num(in, "liq_ticks"),
                num(in, "dt_seconds"), num(in, "session_seconds"), num(in, "horizon_days")));
    }

    private static Result sim(Config cfg, HedgeConfig hedge, RiskConfig risk, int n, long seed) {
        return Hedging.simulateHedged(LARGE, n, seed, cfg, hedge, risk);
    }

    private static Result sim(Config cfg, HedgeConfig hedge, RiskConfig risk) {
        return sim(cfg, hedge, risk, 600, 11);
    }

    private static void assertIdentityOnEveryPath(Result r) {
        double[] pnl = r.path(Stat.PNL);
        double[] sp = r.path(Stat.SPREAD);
        double[] fe = r.path(Stat.FEES);
        double[] inv = r.path(Stat.INVENTORY);
        double[] liq = r.path(Stat.LIQUIDATION);
        double[] hp = r.path(Stat.HEDGE_PNL);
        double[] hc = r.path(Stat.HEDGE_COSTS);
        double[] rc = r.path(Stat.RISK_COSTS);
        for (int i = 0; i < pnl.length; i++) {
            double recon = sp[i] + fe[i] + inv[i] - liq[i] + hp[i] - hc[i] - rc[i];
            assertTrue("identity on path " + i,
                    Math.abs(recon - pnl[i]) <= 1e-9 * Math.max(1.0, Math.abs(pnl[i])));
            assertTrue(hc[i] >= 0.0 && rc[i] >= 0.0);
        }
        assertTrue(r.maxIdentityError() <= 1e-9);
    }

    private static void exactCase(String name, Map<String, Object> c, String key, double act) {
        Map<String, Object> exp = obj(c, "expect");
        assertEquals(name + " keys", 1, exp.size());
        double e = num(exp, key);
        assertEquals(name + "." + key, e, act, num(c, "tol") * Math.max(1.0, Math.abs(e)));
    }

    private static void simCase(String name, Map<String, Object> c) {
        Map<String, Object> in = obj(c, "inputs");
        Map<String, Object> exp = obj(c, "expect");
        Agent agent = Agent.values()[(int) num(in, "agent_code")];
        Config cfg = Config.defaults().withAgent(agent).withTick(num(in, "quote_tick"))
                .withMakerFee(num(in, "maker_fee"))
                .withLatencySteps((int) num(in, "latency_steps"))
                .withInformedProb(num(in, "informed_prob"))
                .withSignalAccuracy(num(in, "signal_accuracy"));
        HedgeConfig hedge = new HedgeConfig(num(in, "rho"), num(in, "hedge_ratio"),
                num(in, "hedge_cost"), num(in, "hedge_threshold"), num(in, "hedge_target"));
        RiskConfig risk = new RiskConfig(num(in, "loss_limit"), (int) num(in, "position_limit"),
                num(in, "flatten_penalty"), (int) num(in, "throttle_updates"),
                (int) num(in, "throttle_window"), (int) num(in, "kill_step"));
        Result r = Hedging.simulateHedged(paramsFrom(in), N_TEST, (long) num(in, "seed") + 1,
                cfg, hedge, risk);
        assertIdentityOnEveryPath(r);
        double mult = num(c, "tol");
        int checked = 0;
        for (String key : exp.keySet()) {
            if (key.endsWith("_se")) {
                continue;
            }
            double got;
            double seOwn;
            if (key.equals("pnl_var")) {
                got = r.var(Stat.PNL);
                seOwn = r.varSe(Stat.PNL);
            } else {
                Stat st = Stat.valueOf(key.toUpperCase(Locale.ROOT));
                got = r.mean(st);
                seOwn = r.se(st);
            }
            double ref = num(exp, key);
            double band = mult * Math.hypot(num(exp, key + "_se"), seOwn) + 1e-12;
            assertTrue(name + "." + key + ": " + got + " vs " + ref,
                    Math.abs(got - ref) <= band);
            checked += 2;
        }
        assertEquals(name + " every key", exp.size(), checked);
        assertEquals(name + " key count", 32, checked);
    }

    @Test
    public void everyGoldenCase() {
        List<Map<String, Object>> cs = cases();
        assertEquals(18, cs.size());
        int exact = 0;
        int stat = 0;
        for (Map<String, Object> c : cs) {
            String name = (String) c.get("name");
            String kind = (String) c.get("kind");
            Map<String, Object> in = obj(c, "inputs");
            switch (kind) {
                case "hedge_trade":
                    exactCase(name, c, "dy", Hedging.hedgeTrade(num(in, "q"), num(in, "y"),
                            num(in, "hedge_ratio"), num(in, "hedge_threshold"),
                            num(in, "hedge_target")));
                    exact++;
                    break;
                case "hedge_quantile": {
                    double[] v = new double[10];
                    for (int i = 0; i < 10; i++) {
                        v[i] = num(in, "v_" + i);
                    }
                    exactCase(name, c, "q", Hedging.quantile(v, num(in, "p")));
                    exact++;
                    break;
                }
                case "hedge_sim":
                    simCase(name, c);
                    stat++;
                    break;
                default:
                    throw new IllegalStateException("unknown kind " + kind);
            }
        }
        assertEquals(8, exact);
        assertEquals(10, stat);
    }

    @Test
    public void identityHoldsOnEveryPath() {
        assertIdentityOnEveryPath(sim(Config.defaults(), NO_HEDGE, NO_RISK, 300, 9));
        assertIdentityOnEveryPath(sim(BOOK, HEDGE, NO_RISK, 300, 9));
        assertIdentityOnEveryPath(sim(
                Config.defaults().withAgent(Agent.GLFT).withTick(0.01).withLatencySteps(2),
                HEDGE, new RiskConfig(0.0, 3, 0.02, 0, 1, -1), 300, 9));
        assertIdentityOnEveryPath(sim(FIXED, new HedgeConfig(-0.5, 0.4, 0.01, 2.0, 0.5),
                new RiskConfig(2.0, 0, 0.01, 2, 5, 3000), 300, 9));
    }

    @Test
    public void noHedgeNoControlsIsSection9InDistribution() {
        for (Config cfg : new Config[] {BOOK, FIXED}) {
            Result a = Hedging.simulateHedged(LARGE, N_TEST, 31, cfg, NO_HEDGE, NO_RISK);
            MarketMaking.Result b = MarketMaking.simulate(LARGE, N_TEST, 32, cfg);
            for (MarketMaking.Stat ms : MarketMaking.Stat.values()) {
                Stat st = Stat.valueOf(ms.name());
                double band = 4.0 * Math.hypot(a.se(st), b.se(ms)) + 1e-12;
                assertTrue(ms.name(), Math.abs(a.mean(st) - b.mean(ms)) <= band);
            }
            for (Stat st : new Stat[] {Stat.HEDGE_PNL, Stat.HEDGE_COSTS, Stat.RISK_COSTS,
                Stat.LOSS_STOP, Stat.KILL_STOP}) {
                for (double v : a.path(st)) {
                    assertEquals(st.name(), 0.0, v, 0.0);
                }
            }
            assertArrayEquals(a.path(Stat.RMS_Q), a.path(Stat.RMS_X), 0.0);
            for (double v : a.path(Stat.QUOTE_UPDATES)) {
                assertEquals(LARGE.nSteps(), v, 0.0);
            }
        }
    }

    @Test
    public void lossLimitOvershootIsAtMostOneStepMove() {
        Result r = sim(FIXED, NO_HEDGE, new RiskConfig(3.0, 0, 0.01, 0, 1, -1), 800, 5);
        double[] stop = r.path(Stat.LOSS_STOP);
        double[] over = r.path(Stat.OVERSHOOT);
        double[] move = r.path(Stat.STOP_MOVE);
        double[] liq = r.path(Stat.LIQUIDATION);
        double[] upd = r.path(Stat.QUOTE_UPDATES);
        int stopped = 0;
        for (int i = 0; i < stop.length; i++) {
            assertTrue(over[i] <= move[i] + 1e-9);
            if (stop[i] == 1.0) {
                stopped++;
                assertEquals(0.0, liq[i], 0.0);
                assertTrue(upd[i] < 4680.0);
            } else {
                assertEquals(0.0, over[i], 0.0);
            }
        }
        assertTrue("stopped " + stopped, stopped > 100);
    }

    @Test
    public void hedgingCutsVarianceAndSkewAtACost() {
        Result fixedUn = sim(FIXED, NO_HEDGE, NO_RISK);
        Result fixedH = sim(FIXED, HEDGE, NO_RISK);
        assertTrue(fixedH.var(Stat.PNL) < 0.5 * fixedUn.var(Stat.PNL));
        assertTrue(fixedH.mean(Stat.RMS_X) < 0.5 * fixedH.mean(Stat.RMS_Q));
        assertTrue(fixedH.mean(Stat.HEDGE_COSTS) > 0.0);
        assertTrue(fixedH.mean(Stat.HEDGE_TRADES) > 0.0);
        Result asUn = sim(BOOK, NO_HEDGE, NO_RISK);
        Result asH = sim(BOOK, HEDGE, NO_RISK);
        assertTrue(asH.mean(Stat.RMS_SKEW) < 0.6 * asUn.mean(Stat.RMS_SKEW));
        assertTrue(asH.var(Stat.PNL) < asUn.var(Stat.PNL));
    }

    @Test
    public void eachControlTriggersAndCutsTheTail() {
        Result base = sim(FIXED, NO_HEDGE, NO_RISK);
        Result pos = sim(FIXED, NO_HEDGE, new RiskConfig(0.0, 10, 0.01, 0, 1, -1));
        assertTrue(pos.mean(Stat.POS_FLATTENS) > 1.0 && pos.mean(Stat.RISK_COSTS) > 0.0);
        assertTrue(pos.quantile(Stat.PNL, 0.05) > base.quantile(Stat.PNL, 0.05));
        Result loss = sim(FIXED, NO_HEDGE, new RiskConfig(5.0, 0, 0.01, 0, 1, -1));
        assertTrue(loss.mean(Stat.LOSS_STOP) > 0.2);
        assertTrue(loss.quantile(Stat.PNL, 0.05) > base.quantile(Stat.PNL, 0.05));
        Result kill = sim(FIXED, NO_HEDGE, new RiskConfig(0.0, 0, 0.0, 0, 1, 1000));
        for (double v : kill.path(Stat.KILL_STOP)) {
            assertEquals(1.0, v, 0.0);
        }
        for (double v : kill.path(Stat.QUOTE_UPDATES)) {
            assertEquals(1000.0, v, 0.0);
        }
        Result thr = sim(BOOK, NO_HEDGE, new RiskConfig(0.0, 0, 0.0, 1, 3, -1));
        for (double v : thr.path(Stat.QUOTE_UPDATES)) {
            assertEquals(1560.0, v, 0.0); // ceil(4680 / 3)
        }
        assertTrue(thr.mean(Stat.PNL) < sim(BOOK, NO_HEDGE, NO_RISK).mean(Stat.PNL));
    }

    @Test
    public void deterministicPerSeed() {
        RiskConfig risk = new RiskConfig(4.0, 0, 0.0, 2, 3, -1);
        Result a = sim(FIXED, HEDGE, risk, 100, 11);
        Result b = sim(FIXED, HEDGE, risk, 100, 11);
        for (Stat st : Stat.values()) {
            assertArrayEquals(a.path(st), b.path(st), 0.0);
        }
        a.path(Stat.PNL)[0] = 1e9;
        assertEquals(b.path(Stat.PNL)[0], a.path(Stat.PNL)[0], 0.0);
    }

    @Test
    public void hedgeTradeLeavesTargetFractionUncovered() {
        double q = 9.0;
        double y = -2.0;
        double h = 0.8;
        double u = q + y / h;
        double dy = Hedging.hedgeTrade(q, y, h, 1.0, 0.25);
        assertEquals(0.25 * u, q + (y + dy) / h, 1e-14 * Math.abs(u));
        assertEquals(0.0, Hedging.hedgeTrade(1.0, 0.0, 0.9, 1.0, 0.0), 0.0);
        assertEquals(0.0, Hedging.hedgeTrade(5.0, 0.0, 0.0, 0.0, 0.0), 0.0);
    }

    @Test
    public void quantileIsType7() {
        SplittableRandom rng = new SplittableRandom(3);
        double[] v = new double[57];
        for (int i = 0; i < v.length; i++) {
            v[i] = rng.nextGaussian();
        }
        double[] s = v.clone();
        java.util.Arrays.sort(s);
        assertEquals(s[0], Hedging.quantile(v, 0.0), 0.0);
        assertEquals(s[56], Hedging.quantile(v, 1.0), 0.0);
        assertEquals(s[28], Hedging.quantile(v, 0.5), 1e-15);
        // h = 56 * 0.33 = 18.48
        assertEquals(s[18] + 0.48 * (s[19] - s[18]), Hedging.quantile(v, 0.33), 1e-14);
        assertEquals(4.0, Hedging.quantile(new double[] {4.0}, 0.3), 0.0);
        assertEquals(2.5, Hedging.quantile(new double[] {4.0, 1.0, 3.0, 2.0}, 0.5), 1e-15);
    }

    @Test
    public void invalidInputsThrow() {
        assertThrows(IllegalArgumentException.class, () -> new HedgeConfig(1.5, 0, 0, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> new HedgeConfig(0, -0.1, 0, 0, 0));
        assertThrows(IllegalArgumentException.class,
                () -> new HedgeConfig(0, 0, Double.NaN, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> new HedgeConfig(0, 0, 0, -1.0, 0));
        assertThrows(IllegalArgumentException.class, () -> new HedgeConfig(0, 0, 0, 0, 1.0));
        assertThrows(IllegalArgumentException.class, () -> new RiskConfig(-1.0, 0, 0, 0, 1, -1));
        assertThrows(IllegalArgumentException.class,
                () -> new RiskConfig(Double.POSITIVE_INFINITY, 0, 0, 0, 1, -1));
        assertThrows(IllegalArgumentException.class, () -> new RiskConfig(0, -1, 0, 0, 1, -1));
        assertThrows(IllegalArgumentException.class,
                () -> new RiskConfig(0, 0, -0.01, 0, 1, -1));
        assertThrows(IllegalArgumentException.class, () -> new RiskConfig(0, 0, 0, -1, 1, -1));
        assertThrows(IllegalArgumentException.class, () -> new RiskConfig(0, 0, 0, 0, 0, -1));
        assertThrows(IllegalArgumentException.class, () -> new RiskConfig(0, 0, 0, 0, 1, -2));
        Config d = Config.defaults();
        assertThrows(IllegalArgumentException.class,
                () -> Hedging.simulateHedged(LARGE, 1, 0, d, NO_HEDGE, NO_RISK));
        assertThrows(IllegalArgumentException.class,
                () -> Hedging.simulateHedged(LARGE, 10, 0, d, NO_HEDGE,
                        new RiskConfig(0, 0, 0, 0, 1, 10_000)));
        assertThrows(IllegalArgumentException.class,
                () -> Hedging.simulateHedged(LARGE, 10, 0, d, NO_HEDGE,
                        new RiskConfig(0, 0, 0, 1, 10_000, -1)));
        assertThrows(IllegalArgumentException.class,
                () -> Hedging.simulateHedged(LARGE, 10, 0, null, NO_HEDGE, NO_RISK));
        Result r = Hedging.simulateHedged(LARGE, 10, 0, d, NO_HEDGE, NO_RISK);
        assertThrows(IllegalArgumentException.class, () -> r.mean(null));
        assertThrows(IllegalArgumentException.class, () -> Hedging.quantile(new double[0], 0.5));
        assertThrows(IllegalArgumentException.class,
                () -> Hedging.quantile(new double[] {1.0, 2.0}, 1.5));
        assertThrows(IllegalArgumentException.class,
                () -> Hedging.hedgeTrade(1.0, 0.0, 0.9, 0.0, 1.0));
    }
}
