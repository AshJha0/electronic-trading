package io.github.ashjha0.etrade;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.Test;

/**
 * Whole-trajectory golden cases (schema v2, API_SPEC 5.1): every indexed key of
 * every case in ../data/golden/trajectory_golden.json is recomputed and compared
 * with {@code |a - e| <= tol * max(1, |e|)}.
 */
public final class TrajectoryGoldenTest {

    @SuppressWarnings("unchecked")
    private static Map<String, Object> load() {
        try {
            String text = Files.readString(Path.of("../data/golden/trajectory_golden.json"));
            return (Map<String, Object>) Json.parse(text);
        } catch (IOException e) {
            throw new IllegalStateException("cannot load trajectory_golden.json", e);
        }
    }

    private static double num(Map<String, Object> m, String key) {
        Object v = m.get(key);
        if (!(v instanceof Double)) {
            throw new AssertionError("missing numeric key " + key);
        }
        return (Double) v;
    }

    private static ACParams acParams(Map<String, Object> in) {
        return new ACParams(num(in, "ac_total_shares"), (int) num(in, "ac_n_intervals"),
                num(in, "ac_tau"), num(in, "ac_sigma"), num(in, "ac_eta"),
                num(in, "ac_gamma_p"), num(in, "ac_lambda_ra"));
    }

    private static void close(String name, String key, double actual, double expected,
                              double tol) {
        assertTrue(name + "." + key + ": " + actual + " vs " + expected,
                Math.abs(actual - expected) <= tol * Math.max(1.0, Math.abs(expected)));
    }

    private static int checkVector(String name, Map<String, Object> expect, String prefix,
                                   double[] actual, double tol) {
        int len = (int) num(expect, "len");
        assertEquals(name + ".len", len, actual.length);
        for (int j = 0; j < len; j++) {
            close(name, prefix + "_" + j, actual[j], num(expect, prefix + "_" + j), tol);
        }
        return len;
    }

    @Test
    @SuppressWarnings("unchecked")
    public void everyKeyOfEveryCase() {
        Map<String, Object> root = load();
        assertEquals(2.0, num(root, "schema"), 0.0);
        List<Object> cases = (List<Object>) root.get("cases");
        assertEquals(7, cases.size());
        int checkedKeys = 0;
        for (Object o : cases) {
            Map<String, Object> c = (Map<String, Object>) o;
            String name = (String) c.get("name");
            String kind = (String) c.get("kind");
            Map<String, Object> in = (Map<String, Object>) c.get("inputs");
            Map<String, Object> ex = (Map<String, Object>) c.get("expect");
            double tol = num(c, "tol");
            int keys;
            switch (kind) {
                case "ac_trajectory":
                    keys = checkVector(name, ex, "x", AlmgrenChriss.trajectory(acParams(in)), tol);
                    break;
                case "twap":
                    keys = checkVector(name, ex, "x", AlmgrenChriss.twapHoldings(
                            num(in, "ac_total_shares"), (int) num(in, "ac_n_intervals")), tol);
                    break;
                case "pov":
                    keys = checkVector(name, ex, "x", AlmgrenChriss.povHoldings(
                            num(in, "ac_total_shares"), num(in, "ac_tau"),
                            num(in, "pov_market_volume_rate"), num(in, "pov_participation")),
                            tol);
                    break;
                case "frontier": {
                    int n = (int) num(in, "n_lambdas");
                    double[] grid = new double[n];
                    for (int i = 0; i < n; i++) {
                        grid[i] = num(in, "lambda_" + i);
                    }
                    AlmgrenChriss.FrontierPoint[] fr =
                            AlmgrenChriss.efficientFrontier(acParams(in), grid);
                    assertEquals(name + ".len", (int) num(ex, "len"), fr.length);
                    for (int i = 0; i < n; i++) {
                        close(name, "kappa_" + i, fr[i].kappa(), num(ex, "kappa_" + i), tol);
                        close(name, "e_cost_" + i, fr[i].eCost(), num(ex, "e_cost_" + i), tol);
                        close(name, "std_cost_" + i, fr[i].stdCost(),
                                num(ex, "std_cost_" + i), tol);
                    }
                    keys = 3 * n;
                    break;
                }
                case "impact_schedule": {
                    double[] w = num(in, "volume_curve") == 1.0
                            ? Impact.usEquityVolumeCurve13() : null;
                    Market.ImpactMarket m = new Market.ImpactMarket(num(in, "price"),
                            num(in, "spread"), num(in, "adv"), num(in, "daily_vol"),
                            num(in, "total_shares"), (int) num(in, "n_intervals"),
                            num(in, "tau"), num(in, "lambda_ra"), num(in, "alpha"),
                            num(in, "y"), Market.DEFAULT_PERM_FRAC,
                            num(in, "drift_bps_per_day"), w);
                    Impact.Params p = Market.impactParamsFromMarket(m);
                    double[] h = Impact.optimalSchedule(p).holdings();
                    double[] frac = new double[h.length];
                    for (int j = 0; j < h.length; j++) {
                        frac[j] = h[j] / p.totalShares();
                    }
                    keys = checkVector(name, ex, "xfrac", frac, tol);
                    break;
                }
                default:
                    fail("unknown kind " + kind);
                    return;
            }
            // every expect key (indexed keys + len) was asserted
            assertEquals(name + ": expect key count", ex.size(), keys + 1);
            checkedKeys += keys;
        }
        assertTrue(checkedKeys > 0);
    }
}
