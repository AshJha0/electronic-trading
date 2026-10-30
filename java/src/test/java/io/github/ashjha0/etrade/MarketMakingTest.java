package io.github.ashjha0.etrade;

import static io.github.ashjha0.etrade.Rejects.assertRejects;
import static io.github.ashjha0.etrade.Rejects.bad;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.ashjha0.etrade.Market.ASMarket;
import io.github.ashjha0.etrade.MarketMaking.Agent;
import io.github.ashjha0.etrade.MarketMaking.Config;
import io.github.ashjha0.etrade.MarketMaking.Result;
import io.github.ashjha0.etrade.MarketMaking.Stat;
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
 * Market making with frictions (API_SPEC Section 9): every case of
 * {@code mm_golden.json}, the attribution identity on every path, determinism and
 * validation.
 */
final class MarketMakingTest {

    private static final Map<String, Object> GOLDEN = load();

    /** Paths per statistical case (the spec asks for at least 2000). */
    private static final int N_PATHS = 4000;

    private static final ASParams LARGE = Market.asParamsFromMarket(new ASMarket(
            100.0, 0.01, 0.015, 1170.0, 0.462, 2.25, 50, 1.0, 5.0, 23400.0, 1.0));

    @SuppressWarnings("unchecked")
    private static Map<String, Object> load() {
        try {
            return (Map<String, Object>) TestData.load("golden/mm_golden.json");
        } catch (IOException e) {
            throw new IllegalStateException("cannot load mm_golden.json", e);
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

    private static ASParams paramsFrom(Map<String, Object> in) {
        return Market.asParamsFromMarket(new ASMarket(num(in, "price"), 0.01,
                num(in, "daily_vol"), num(in, "touch_fills_per_day"), num(in, "halving_ticks"),
                num(in, "skew_ticks"), (int) num(in, "q_max"), num(in, "liq_ticks"),
                num(in, "dt_seconds"), num(in, "session_seconds"), num(in, "horizon_days")));
    }

    private static void assertIdentityOnEveryPath(Result r) {
        double[] pnl = r.path(Stat.PNL);
        double[] sp = r.path(Stat.SPREAD);
        double[] fe = r.path(Stat.FEES);
        double[] inv = r.path(Stat.INVENTORY);
        double[] liq = r.path(Stat.LIQUIDATION);
        for (int i = 0; i < pnl.length; i++) {
            double recon = sp[i] + fe[i] + inv[i] - liq[i];
            assertTrue(Math.abs(recon - pnl[i]) <= 1e-9 * Math.max(1.0, Math.abs(pnl[i])), "identity on path " + i);
        }
        assertTrue(r.maxIdentityError() <= 1e-9);
    }

    private static void exactCase(String name, Map<String, Object> c, Map<String, Double> act) {
        Map<String, Object> exp = obj(c, "expect");
        double tol = num(c, "tol");
        assertEquals(exp.keySet(), act.keySet(), name + " keys");
        for (String key : exp.keySet()) {
            double e = num(exp, key);
            assertEquals(e, act.get(key), tol * Math.max(1.0, Math.abs(e)), name + "." + key);
        }
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
        Result r = MarketMaking.simulate(paramsFrom(in), N_PATHS, (long) num(in, "seed"), cfg);
        assertIdentityOnEveryPath(r);
        double mult = num(c, "tol");
        int checked = 0;
        for (String key : exp.keySet()) {
            if (key.endsWith("_se")) {
                continue;
            }
            Stat st = Stat.valueOf(key.toUpperCase(java.util.Locale.ROOT));
            double ref = num(exp, key);
            double seRef = num(exp, key + "_se");
            double seOwn = r.se(st);
            double diff = Math.abs(r.mean(st) - ref);
            assertTrue(diff <= mult * Math.sqrt(seRef * seRef + seOwn * seOwn), name + "." + key + ": " + r.mean(st) + " vs " + ref);
            checked += 2;
        }
        assertEquals(exp.size(), checked, name + " every key");
    }

    @Test
    void everyGoldenCase() {
        List<Map<String, Object>> cs = cases();
        assertEquals(14, cs.size());
        for (Map<String, Object> c : cs) {
            String name = (String) c.get("name");
            String kind = (String) c.get("kind");
            Map<String, Object> in = obj(c, "inputs");
            Map<String, Double> act = new LinkedHashMap<>();
            switch (kind) {
                case "mm_tick": {
                    AvellanedaStoikov.Quotes q = MarketMaking.roundToTick(num(in, "bid"),
                            num(in, "ask"), num(in, "tick"));
                    act.put("bid", q.bid());
                    act.put("ask", q.ask());
                    exactCase(name, c, act);
                    break;
                }
                case "mm_glft": {
                    MarketMaking.HalfSpreads h =
                            MarketMaking.glftHalfSpreads(paramsFrom(in), num(in, "q"));
                    act.put("delta_bid", h.deltaBid());
                    act.put("delta_ask", h.deltaAsk());
                    exactCase(name, c, act);
                    break;
                }
                case "mm_wmid":
                    act.put("wmid", MarketMaking.weightedMid(num(in, "bid"), num(in, "ask"),
                            num(in, "bid_size"), num(in, "ask_size")));
                    exactCase(name, c, act);
                    break;
                case "mm_ofi": {
                    double[][] s = new double[4][5];
                    String[] names = {"bid_px_", "bid_sz_", "ask_px_", "ask_sz_"};
                    for (int a = 0; a < 4; a++) {
                        for (int i = 0; i < 5; i++) {
                            s[a][i] = num(in, names[a] + i);
                        }
                    }
                    double[] e = MarketMaking.orderFlowImbalance(s[0], s[1], s[2], s[3]);
                    double sum = 0.0;
                    for (int i = 0; i < e.length; i++) {
                        act.put("e_" + (i + 1), e[i]);
                        sum += e[i];
                    }
                    act.put("ofi", sum);
                    exactCase(name, c, act);
                    break;
                }
                case "mm_sim":
                    simCase(name, c);
                    break;
                default:
                    throw new IllegalStateException("unknown kind " + kind);
            }
        }
    }

    @Test
    void identityHoldsForEveryAgentAndFriction() {
        Config base = Config.defaults().withTick(0.01).withMakerFee(-0.001)
                .withLatencySteps(2).withInformedProb(0.2).withSignalAccuracy(0.7);
        for (Agent a : Agent.values()) {
            Result r = MarketMaking.simulate(LARGE, 50, 7, base.withAgent(a));
            assertIdentityOnEveryPath(r);
            assertEquals(50, r.nPaths());
        }
        Result fx = MarketMaking.simulate(LARGE, 50, 7,
                base.withAgent(Agent.FIXED).withFixedSpread(0.05));
        assertIdentityOnEveryPath(fx);
    }

    @Test
    void deterministicPerSeed() {
        Config cfg = Config.defaults().withTick(0.01).withInformedProb(0.1);
        Result a = MarketMaking.simulate(LARGE, 20, 42, cfg);
        Result b = MarketMaking.simulate(LARGE, 20, 42, cfg);
        for (Stat st : Stat.values()) {
            assertArrayEquals(a.path(st), b.path(st), 0.0);
        }
        Result c = MarketMaking.simulate(LARGE, 20, 43, cfg);
        assertNotEquals(a.mean(Stat.PNL), c.mean(Stat.PNL), 0.0);
        // defensive copy
        a.path(Stat.PNL)[0] = 1e9;
        assertEquals(b.path(Stat.PNL)[0], a.path(Stat.PNL)[0], 0.0);
    }

    @Test
    void helpersAndFrictionsBehave() {
        // tick 0 is the identity; rounding is away from fair value
        AvellanedaStoikov.Quotes q = MarketMaking.roundToTick(99.995, 100.005, 0.01);
        assertTrue(q.bid() <= 99.995 && q.ask() >= 100.005);
        // GLFT: long inventory widens the bid, tightens the ask
        MarketMaking.HalfSpreads h = MarketMaking.glftHalfSpreads(LARGE, 2.0);
        assertTrue(h.deltaBid() > h.deltaAsk());
        // equal sizes give the plain mid
        assertEquals(100.0, MarketMaking.weightedMid(99.99, 100.01, 5.0, 5.0), 1e-12);
        // no informed flow means no informed fills
        Result r = MarketMaking.simulate(LARGE, 10, 1, Config.defaults());
        assertEquals(0.0, r.mean(Stat.INFORMED_FILLS), 0.0);
        assertEquals(0.0, r.mean(Stat.FEES), 0.0);
        assertTrue(r.se(Stat.FILLS) > 0.0);
    }

    @ParameterizedTest(name = "[{index}] {0}")
    @MethodSource("invalidInputsThrowCases")
    void invalidInputsThrow(String fragment, Executable call) {
        assertRejects(fragment, call);
    }

    static List<Arguments> invalidInputsThrowCases() {
        List<Arguments> c = new ArrayList<>();
        Config d = Config.defaults();
        c.add(bad("tick", () -> d.withTick(-0.01)));
        c.add(bad("tick", () -> d.withTick(Double.NaN)));
        c.add(bad("maker_fee", () -> d.withMakerFee(Double.NaN)));
        c.add(bad("latency_steps", () -> d.withLatencySteps(-1)));
        c.add(bad("informed_prob", () -> d.withInformedProb(1.5)));
        c.add(bad("signal_accuracy", () -> d.withSignalAccuracy(0.4)));
        c.add(bad("fixed_spread", () -> d.withFixedSpread(0.0)));
        c.add(bad("agent", () -> d.withAgent(null)));
        c.add(bad("nPaths",
                () -> MarketMaking.simulate(LARGE, 1, 0, d)));
        c.add(bad("config",
                () -> MarketMaking.simulate(LARGE, 10, 0, null)));
        c.add(bad("latency_steps",
                () -> MarketMaking.simulate(LARGE, 10, 0,
                        d.withLatencySteps(LARGE.nSteps() + 1))));
        c.add(bad("bid and ask",
                () -> MarketMaking.roundToTick(Double.NaN, 1.0, 0.01)));
        c.add(bad("tick",
                () -> MarketMaking.roundToTick(1.0, 2.0, -1.0)));
        c.add(bad("q must",
                () -> MarketMaking.glftHalfSpreads(LARGE, Double.NaN)));
        c.add(bad("ask must",
                () -> MarketMaking.weightedMid(100.0, 99.0, 1.0, 1.0)));
        c.add(bad("bid_size + ask_size",
                () -> MarketMaking.weightedMid(99.0, 100.0, 0.0, 0.0)));
        c.add(bad("sizes",
                () -> MarketMaking.weightedMid(99.0, 100.0, -1.0, 2.0)));
        double[] two = {1.0, 1.0};
        c.add(bad("book states",
                () -> MarketMaking.orderFlowImbalance(new double[] {1.0}, new double[] {1.0},
                        new double[] {1.0}, new double[] {1.0})));
        c.add(bad("book series",
                () -> MarketMaking.orderFlowImbalance(two, two, two, new double[] {1.0})));
        c.add(bad("sizes",
                () -> MarketMaking.orderFlowImbalance(two, new double[] {1.0, -1.0}, two, two)));
        c.add(bad("book series",
                () -> MarketMaking.orderFlowImbalance(two, two, new double[] {1.0, Double.NaN},
                        two)));
        return c;
    }
}
