package io.github.ashjha0.etrade;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;

/**
 * Golden-value suite: every case in ../data/golden/golden.json is recomputed
 * from its inputs and checked against the stored expectation.
 *
 * <p>Exact cases: |actual - expected| &lt;= tol * max(1, |expected|).
 * Statistical cases: the golden reference is the 20k-path Python run; this
 * suite re-simulates with Java's own RNG (&gt;= 2000 paths, own SEs) and asserts
 * |actual - ref| &lt;= tol * sqrt(se_ref^2 + se_own^2), tol = 4.
 */
final class GoldenTest {

    private static final int N_PATHS_TEST = 2000;
    private static final int AC_MC_PATHS_TEST = 5000;

    private static final Map<String, Map<String, Object>> CASES = loadCases();

    @SuppressWarnings("unchecked")
    private static Map<String, Map<String, Object>> loadCases() {
        try {
            Map<String, Object> root = (Map<String, Object>) TestData.load("golden/golden.json");
            Map<String, Map<String, Object>> byName = new LinkedHashMap<>();
            for (Object c : (List<Object>) root.get("cases")) {
                Map<String, Object> m = (Map<String, Object>) c;
                byName.put((String) m.get("name"), m);
            }
            return byName;
        } catch (IOException e) {
            throw new IllegalStateException("cannot load golden.json", e);
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> inputs(String name) {
        return (Map<String, Object>) CASES.get(name).get("inputs");
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> expect(String name) {
        return (Map<String, Object>) CASES.get(name).get("expect");
    }

    private static double tol(String name) {
        return (Double) CASES.get(name).get("tol");
    }

    private static double d(Map<String, Object> m, String key) {
        return (Double) m.get(key);
    }

    private static ACParams acFromCfg(Map<String, Object> in) {
        return new ACParams(d(in, "ac_total_shares"), (int) d(in, "ac_n_intervals"),
                d(in, "ac_tau"), d(in, "ac_sigma"), d(in, "ac_eta"),
                d(in, "ac_gamma_p"), d(in, "ac_lambda_ra"));
    }

    private static ASParams asFromCfg(Map<String, Object> in) {
        return new ASParams(d(in, "as_s0"), d(in, "as_sigma"), d(in, "as_horizon"),
                d(in, "as_dt"), d(in, "as_gamma"), d(in, "as_k"), d(in, "as_a_rate"),
                (int) d(in, "as_q_max"), d(in, "as_eta_liq"));
    }

    /** Exact check per API_SPEC section 5: relative for large, absolute near 0. */
    private static void checkExact(String name, Map<String, Double> actual) {
        double t = tol(name);
        for (Map.Entry<String, Object> e : expect(name).entrySet()) {
            String key = e.getKey();
            assertTrue(actual.containsKey(key), name + ": missing key " + key);
            double expected = (Double) e.getValue();
            double err = Math.abs(actual.get(key) - expected);
            assertTrue(err <= t * Math.max(1.0, Math.abs(expected)), name + "." + key + ": got " + actual.get(key) + ", want "
                            + expected + ", err " + err);
        }
    }

    /** Statistical check per API_SPEC section 4: 4*sqrt(se_ref^2 + se_own^2) band. */
    private static void checkStatistical(String name, Map<String, Double> actual,
                                         Map<String, Double> actualSe) {
        double mult = tol(name);
        Map<String, Object> exp = expect(name);
        for (Map.Entry<String, Object> e : exp.entrySet()) {
            String key = e.getKey();
            if (key.endsWith("_se")) {
                continue;
            }
            double ref = (Double) e.getValue();
            double seRef = (Double) exp.get(key + "_se");
            double band = mult * Math.sqrt(seRef * seRef
                    + actualSe.get(key) * actualSe.get(key));
            double err = Math.abs(actual.get(key) - ref);
            assertTrue(err <= band, name + "." + key + ": got " + actual.get(key) + ", ref " + ref
                    + ", err " + err + " > band " + band);
        }
    }

    private static Map<String, Double> acSolutionActual(Map<String, Object> in) {
        ACParams p = acFromCfg(in);
        double[] x = AlmgrenChriss.trajectory(p);
        double[] n = AlmgrenChriss.tradeList(p);
        Map<String, Double> a = new LinkedHashMap<>();
        a.put("kappa", AlmgrenChriss.kappa(p));
        a.put("half_life", AlmgrenChriss.halfLife(p));
        a.put("x_j", x[(int) d(in, "x_index")]);
        a.put("n_1", n[0]);
        a.put("e_cost", AlmgrenChriss.expectedCost(p));
        a.put("var_cost", AlmgrenChriss.varianceCost(p));
        return a;
    }

    @Test
    void acLiquidSolution() {
        checkExact("ac_liquid_solution", acSolutionActual(inputs("ac_liquid_solution")));
    }

    @Test
    void acIlliquidSolution() {
        checkExact("ac_illiquid_solution", acSolutionActual(inputs("ac_illiquid_solution")));
    }

    @Test
    void acTwap() {
        ACParams p = acFromCfg(inputs("ac_twap_liquid"));
        double[] twap = AlmgrenChriss.twapHoldings(p.totalShares(), p.nIntervals());
        checkExact("ac_twap_liquid", Map.of(
                "e_cost", AlmgrenChriss.expectedCost(p, twap),
                "var_cost", AlmgrenChriss.varianceCost(p, twap)));
    }

    @Test
    void acPov() {
        Map<String, Object> in = inputs("ac_pov_liquid");
        ACParams p = acFromCfg(in);
        double[] pov = AlmgrenChriss.povHoldings(p.totalShares(), p.tau(),
                d(in, "pov_market_volume_rate"), d(in, "pov_participation"));
        checkExact("ac_pov_liquid", Map.of(
                "e_cost", AlmgrenChriss.expectedCost(p, pov),
                "var_cost", AlmgrenChriss.varianceCost(p, pov),
                "n_intervals_used", (double) (pov.length - 1)));
    }

    @Test
    void acFrontierPoint() {
        ACParams p = acFromCfg(inputs("ac_frontier_point_liquid"));
        AlmgrenChriss.FrontierPoint fp =
                AlmgrenChriss.efficientFrontier(p, new double[] {p.lambdaRa()})[0];
        checkExact("ac_frontier_point_liquid", Map.of(
                "e_cost", fp.eCost(), "std_cost", fp.stdCost(), "kappa", fp.kappa()));
    }

    @Test
    void acKappaSmallTau() {
        // kappa_tilde^2*tau^2/2 ~ 2e-9: the log1p form matches; acosh(1+x)
        // would miss by ~1e-8 relative.
        ACParams p = acFromCfg(inputs("ac_kappa_small_tau"));
        assertEquals(5000, p.nIntervals());
        checkExact("ac_kappa_small_tau", Map.of(
                "kappa", AlmgrenChriss.kappa(p),
                "half_life", AlmgrenChriss.halfLife(p),
                "x_1", AlmgrenChriss.trajectory(p)[1]));
    }

    @Test
    void acLambdaZeroLimit() {
        ACParams p = acFromCfg(inputs("ac_lambda_zero_limit_liquid"));
        assertEquals(0.0, p.lambdaRa(), 0.0);
        double[] twap = AlmgrenChriss.twapHoldings(p.totalShares(), p.nIntervals());
        double[] x = AlmgrenChriss.trajectory(p);
        double diff = 0.0;
        for (int j = 0; j < x.length; j++) {
            diff = Math.max(diff, Math.abs(x[j] - twap[j]));
        }
        checkExact("ac_lambda_zero_limit_liquid", Map.of("max_abs_diff_vs_twap", diff));
    }

    @Test
    void acInstantLiquidation() {
        AlmgrenChriss.CostMoments cm =
                AlmgrenChriss.instantaneousCost(acFromCfg(inputs("ac_instant_liquidation_liquid")));
        checkExact("ac_instant_liquidation_liquid",
                Map.of("e_cost", cm.eCost(), "var_cost", cm.varCost()));
    }

    @Test
    void acMcStatistical() {
        Map<String, Object> in = inputs("ac_mc_liquid");
        ACParams p = acFromCfg(in);
        AlmgrenChriss.ISSimResult mc = AlmgrenChriss.simulateIS(p, AC_MC_PATHS_TEST,
                (long) d(in, "seed"));
        checkStatistical("ac_mc_liquid",
                Map.of("mean_cost", mc.mean(), "var_cost", mc.var()),
                Map.of("mean_cost", mc.seMean(), "var_cost", mc.seVar()));
    }

    @Test
    void asReservation() {
        Map<String, Object> in = inputs("as_reservation_liquid");
        ASParams ap = asFromCfg(in);
        checkExact("as_reservation_liquid", Map.of("reservation_price",
                AvellanedaStoikov.reservationPrice(ap, d(in, "s"), d(in, "q"), d(in, "t"))));
    }

    @Test
    void asSpread() {
        Map<String, Object> in = inputs("as_spread_liquid");
        ASParams ap = asFromCfg(in);
        checkExact("as_spread_liquid", Map.of(
                "spread", AvellanedaStoikov.optimalSpread(ap, d(in, "t")),
                "spread_terminal", AvellanedaStoikov.optimalSpread(ap, ap.horizon())));
    }

    @Test
    void asFillProb() {
        Map<String, Object> in = inputs("as_fill_prob_liquid");
        ASParams ap = asFromCfg(in);
        double delta = d(in, "delta");
        checkExact("as_fill_prob_liquid", Map.of(
                "intensity", AvellanedaStoikov.intensity(ap, delta),
                "fill_probability", AvellanedaStoikov.fillProbability(ap, delta)));
    }

    private void asSimCase(String name, String[] keys) {
        Map<String, Object> in = inputs(name);
        ASParams ap = asFromCfg(in);
        ASSimResult res = AvellanedaStoikov.simulate(ap, N_PATHS_TEST, (long) d(in, "seed"));
        assertTrue(!res.capViolated(), name + ": inventory cap violated");
        Map<String, Double> vals = new LinkedHashMap<>();
        vals.put("mean_pnl", res.meanPnl());
        vals.put("std_pnl", res.stdPnl());
        vals.put("mean_abs_q", res.meanAbsQ());
        vals.put("mean_max_abs_q", res.meanMaxAbsQ());
        vals.put("mean_fills", res.meanFills());
        Map<String, Double> actual = new LinkedHashMap<>();
        Map<String, Double> se = new LinkedHashMap<>();
        for (String k : keys) {
            actual.put(k, vals.get(k));
            se.put(k, res.se(ASSimResult.Stat.fromWireName(k)));
        }
        checkStatistical(name, actual, se);
    }

    @Test
    void asSimLiquid() {
        asSimCase("as_sim_liquid", new String[] {"mean_pnl", "std_pnl", "mean_abs_q",
            "mean_max_abs_q", "mean_fills"});
    }

    @Test
    void asSimFx() {
        asSimCase("as_sim_fx", new String[] {"mean_pnl", "std_pnl", "mean_abs_q"});
    }

    @Test
    void asFixedComparisonStatistical() {
        Map<String, Object> in = inputs("as_fixed_comparison_liquid");
        ASParams ap = asFromCfg(in);
        long seed = (long) d(in, "seed");
        ASSimResult resAs = AvellanedaStoikov.simulate(ap, N_PATHS_TEST, seed,
                AvellanedaStoikov.Agent.AS);
        ASSimResult resFx = AvellanedaStoikov.simulate(ap, N_PATHS_TEST, seed,
                AvellanedaStoikov.Agent.FIXED, d(in, "fixed_spread"), 1.0, 0.5);
        checkStatistical("as_fixed_comparison_liquid",
                Map.of("as_rms_q", resAs.rmsQ(), "fixed_rms_q", resFx.rmsQ()),
                Map.of("as_rms_q", resAs.se(ASSimResult.Stat.RMS_Q), "fixed_rms_q", resFx.se(ASSimResult.Stat.RMS_Q)));
        // the qualitative claim itself
        assertTrue(resAs.rmsQ() < resFx.rmsQ());
    }

    private void isDecompositionCase(String name, String side) {
        Map<String, Object> in = inputs(name);
        Analytics.ISDecomposition dec = Analytics.decomposeShortfall(Side.parse(side),
                d(in, "decision_price"), d(in, "arrival_price"), d(in, "close_price"),
                d(in, "total_qty"),
                new double[] {d(in, "q1"), d(in, "q2"), d(in, "q3")},
                new double[] {d(in, "p1"), d(in, "p2"), d(in, "p3")});
        checkExact(name, Map.of("total", dec.total(), "delay", dec.delay(),
                "trading", dec.trading(), "opportunity", dec.opportunity()));
    }

    @Test
    void isDecompositionSell() {
        isDecompositionCase("is_decomposition_sell", "sell");
    }

    @Test
    void isDecompositionBuy() {
        isDecompositionCase("is_decomposition_buy", "buy");
    }

    @Test
    void participationRateGolden() {
        Map<String, Object> in = inputs("participation_rate");
        checkExact("participation_rate", Map.of("participation",
                Analytics.participationRate(d(in, "executed_volume"),
                        d(in, "market_volume"))));
    }

    @Test
    void impactRegressionGolden() {
        Map<String, Object> in = inputs("impact_regression");
        double[][] fills = Analytics.syntheticImpactFills((int) d(in, "n_obs"),
                d(in, "true_intercept"), d(in, "true_slope"), d(in, "noise_amp"));
        Analytics.OLSResult r = Analytics.estimateImpact(fills[0], fills[1]);
        checkExact("impact_regression", Map.of("slope", r.slope(),
                "intercept", r.intercept(), "r_squared", r.rSquared()));
    }

    @Test
    void allCasesCovered() {
        Set<String> covered = new HashSet<>(Set.of(
                "ac_liquid_solution", "ac_illiquid_solution", "ac_twap_liquid",
                "ac_pov_liquid", "ac_frontier_point_liquid",
                "ac_lambda_zero_limit_liquid", "ac_instant_liquidation_liquid",
                "ac_kappa_small_tau", "ac_mc_liquid", "as_reservation_liquid",
                "as_spread_liquid", "as_fill_prob_liquid", "as_sim_liquid", "as_sim_fx",
                "as_fixed_comparison_liquid", "is_decomposition_sell",
                "is_decomposition_buy", "participation_rate", "impact_regression"));
        assertEquals(covered, CASES.keySet());
        assertEquals(19, CASES.size());
    }
}
