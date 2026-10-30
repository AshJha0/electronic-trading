package io.github.ashjha0.etrade;

import java.util.Locale;

/**
 * End-to-end demo of the etrade library: Almgren-Chriss optimal trajectories
 * and efficient frontier, TWAP/POV/instantaneous baselines with an MC
 * cross-check, Avellaneda-Stoikov market making (A-S vs fixed-spread agent,
 * liquid and FX-like parameter sets, vol-stress scenario), and an
 * implementation-shortfall decomposition with a market-impact regression.
 *
 * <p>Parameter values mirror {@code data/params_liquid.json},
 * {@code data/params_fx.json} and {@code data/parent_order.json}.
 * Run with {@code bash demo.sh} after {@code bash build.sh}.
 */
public final class Demo {

    private Demo() {
    }

    private static void hdr(String title) {
        System.out.println();
        System.out.println("=== " + title + " "
                + "=".repeat(Math.max(0, 66 - title.length())));
    }

    private static void acSection() {
        // mirrors data/params_liquid.json (ac_*) and data/parent_order.json
        ACParams p = new ACParams(1_000_000.0, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6);
        double povVolumeRate = 2_500_000.0;
        double povParticipation = 0.15;
        double[] lambdaGrid = {0.0, 1e-8, 1e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5};

        hdr("Almgren-Chriss optimal trajectories (sell 1.0e6 shares, N=5)");
        System.out.printf("%10s %9s %10s | holdings x_j (thousands)%n",
                "lambda_ra", "kappa", "half-life");
        for (double lam : new double[] {0.0, 5e-7, 2e-6, 1e-5}) {
            ACParams pl = p.withLambda(lam);
            double[] x = AlmgrenChriss.trajectory(pl);
            double hl = AlmgrenChriss.halfLife(pl);
            String hlS = Double.isFinite(hl)
                    ? String.format(Locale.ROOT, "%10.3f", hl)
                    : String.format(Locale.ROOT, "%10s", "inf");
            StringBuilder sb = new StringBuilder();
            for (double v : x) {
                sb.append(String.format(Locale.ROOT, "%9.1f", v / 1e3));
            }
            System.out.printf(Locale.ROOT, "%10.1e %9.4f %s |%s%n",
                    lam, AlmgrenChriss.kappa(pl), hlS, sb);
        }

        hdr("Efficient frontier of execution (E[cost] vs sqrt(Var[cost]))");
        System.out.printf("%10s %14s %14s %9s%n",
                "lambda_ra", "E[cost] $", "sqrt(Var) $", "kappa");
        for (AlmgrenChriss.FrontierPoint fp
                : AlmgrenChriss.efficientFrontier(p, lambdaGrid)) {
            System.out.printf(Locale.ROOT, "%10.1e %,14.0f %,14.0f %9.4f%n",
                    fp.lambdaRa(), fp.eCost(), fp.stdCost(), fp.kappa());
        }

        hdr("Baselines: optimal vs TWAP vs POV vs instantaneous");
        double[] twap = AlmgrenChriss.twapHoldings(p.totalShares(), p.nIntervals());
        double[] pov = AlmgrenChriss.povHoldings(p.totalShares(), p.tau(),
                povVolumeRate, povParticipation);
        AlmgrenChriss.CostMoments inst = AlmgrenChriss.instantaneousCost(p);
        String[] names = {"AC optimal", "TWAP", "POV 15%", "Instant"};
        double[] es = {AlmgrenChriss.expectedCost(p),
                AlmgrenChriss.expectedCost(p, twap),
                AlmgrenChriss.expectedCost(p, pov), inst.eCost()};
        double[] vs = {AlmgrenChriss.varianceCost(p),
                AlmgrenChriss.varianceCost(p, twap),
                AlmgrenChriss.varianceCost(p, pov), inst.varCost()};
        System.out.printf("%12s %14s %14s%n", "strategy", "E[cost] $", "sqrt(Var) $");
        for (int i = 0; i < names.length; i++) {
            System.out.printf(Locale.ROOT, "%12s %,14.0f %,14.0f%n",
                    names[i], es[i], Math.sqrt(vs[i]));
        }
        AlmgrenChriss.ISSimResult mc = AlmgrenChriss.simulateIS(p, 20_000, 123);
        System.out.printf(Locale.ROOT,
                "MC check (20k paths): E=%,.0f (closed %,.0f, se %,.0f); "
                        + "Var/1e9=%,.1f (closed %,.1f)%n",
                mc.mean(), AlmgrenChriss.expectedCost(p), mc.seMean(),
                mc.var() / 1e9, AlmgrenChriss.varianceCost(p) / 1e9);
    }

    private static void asSection(ASParams ap, String label, long seed) {
        hdr("Avellaneda-Stoikov market making (" + label + ")");
        AvellanedaStoikov.Quotes qt = AvellanedaStoikov.quotes(ap, ap.s0(), 3.0, 0.0);
        System.out.printf(Locale.ROOT,
                "params: s0=%s sigma=%s gamma=%s k=%s A=%s T=%s dt=%s q_max=%d%n",
                ap.s0(), ap.sigma(), ap.gamma(), ap.k(), ap.aRate(), ap.horizon(),
                ap.dt(), ap.qMax());
        System.out.printf(Locale.ROOT,
                "example quote at t=0, q=+3: bid=%.4f ask=%.4f (mid %s, reservation %.4f)%n",
                qt.bid(), qt.ask(), ap.s0(),
                AvellanedaStoikov.reservationPrice(ap, ap.s0(), 3.0, 0.0));
        int nPaths = 2000;
        ASSimResult resAs = AvellanedaStoikov.simulate(ap, nPaths, seed,
                AvellanedaStoikov.Agent.AS);
        ASSimResult resFx = AvellanedaStoikov.simulate(ap, nPaths, seed,
                AvellanedaStoikov.Agent.FIXED);
        System.out.printf("%14s %10s %9s %7s %8s %11s %7s %7s%n", "agent", "mean PnL",
                "std PnL", "Sharpe", "mean|q|", "mean max|q|", "rms q", "fills");
        System.out.printf(Locale.ROOT,
                "%14s %10.4f %9.4f %7.2f %8.3f %11.3f %7.3f %7.1f%n",
                "A-S", resAs.meanPnl(), resAs.stdPnl(), resAs.sharpe(),
                resAs.meanAbsQ(), resAs.meanMaxAbsQ(), resAs.rmsQ(), resAs.meanFills());
        System.out.printf(Locale.ROOT,
                "%14s %10.4f %9.4f %7.2f %8.3f %11.3f %7.3f %7.1f%n",
                "fixed-spread", resFx.meanPnl(), resFx.stdPnl(), resFx.sharpe(),
                resFx.meanAbsQ(), resFx.meanMaxAbsQ(), resFx.rmsQ(), resFx.meanFills());
        System.out.printf(Locale.ROOT,
                "inventory control: A-S rms q = %.3f vs fixed %.3f "
                        + "(similar PnL, much tighter inventory)%n",
                resAs.rmsQ(), resFx.rmsQ());
        ASSimResult stress = AvellanedaStoikov.simulate(ap, nPaths, seed,
                AvellanedaStoikov.Agent.AS, AvellanedaStoikov.averageAsSpread(ap),
                2.0, 0.5);
        System.out.printf(Locale.ROOT,
                "stress (vol x2 after T/2): std PnL %.4f -> %.4f%n",
                resAs.stdPnl(), stress.stdPnl());
    }

    private static void analyticsSection() {
        hdr("Implementation-shortfall decomposition (sell parent order)");
        // mirrors data/parent_order.json
        Analytics.ISDecomposition d = Analytics.decomposeShortfall(Side.SELL, 100.0, 99.5,
                98.8, 1000.0, new double[] {300.0, 400.0, 200.0},
                new double[] {99.4, 99.2, 99.1});
        System.out.printf(Locale.ROOT,
                "decision %.2f, arrival %.2f, close %.2f; executed %.0f/%.0f%n",
                100.0, 99.5, 98.8, d.executedQty(), 1000.0);
        System.out.printf(Locale.ROOT, "  %12s: %10.2f $%n", "delay", d.delay());
        System.out.printf(Locale.ROOT, "  %12s: %10.2f $%n", "trading", d.trading());
        System.out.printf(Locale.ROOT, "  %12s: %10.2f $%n", "opportunity", d.opportunity());
        System.out.printf(Locale.ROOT, "  %12s: %10.2f $%n", "TOTAL", d.total());
        double part = Analytics.participationRate(120_000.0, 1_500_000.0);
        System.out.printf(Locale.ROOT,
                "participation rate example: 120k / 1.5m = %.2f%%%n", 100.0 * part);

        hdr("Market-impact regression (cost/share vs participation, native OLS)");
        double[][] fills = Analytics.syntheticImpactFills(25, 0.5, 8.0, 0.05);
        Analytics.OLSResult reg = Analytics.estimateImpact(fills[0], fills[1]);
        System.out.printf(Locale.ROOT,
                "fitted: cost = %.4f + %.4f * participation (true 0.50 + 8.00), "
                        + "R^2 = %.4f, n = %d%n",
                reg.intercept(), reg.slope(), reg.rSquared(), reg.nObs());
    }

    /** Presets in market units, results in bps (mirrors data/presets.json). */
    private static void marketSection() {
        hdr("Market presets in bps (execution: sell order; market making: A-S spread)");
        String[] names = {"paper_almgren_chriss_2000", "us_large_cap", "us_small_cap", "us_etf",
            "eurusd"};
        Market.ACMarket[] ac = {
            Market.ACMarket.of(50.0, 0.125, 5e6, 0.019, 1e6, 5, 1.0, 2e-6),
            Market.ACMarket.of(100.0, 0.01, 5e6, 0.015, 250_000.0, 13, 1.0 / 13.0, 1e-7),
            Market.ACMarket.of(20.0, 0.05, 3e5, 0.035, 30_000.0, 13, 1.0 / 13.0, 3e-5),
            Market.ACMarket.of(500.0, 0.01, 6e7, 0.01, 600_000.0, 13, 1.0 / 13.0, 7e-10),
            Market.ACMarket.of(1.10, 5e-5, 2e9, 0.005, 1e8, 24, 1.0 / 24.0, 8e-8),
        };
        Market.ASMarket[] mm = {
            null,
            new Market.ASMarket(100.0, 0.01, 0.015, 1170.0, 0.462, 2.25, 50, 1.0, 5.0, 23_400.0,
                1.0),
            new Market.ASMarket(20.0, 0.01, 0.035, 195.0, 2.0, 5.0, 10, 5.0, 5.0, 23_400.0, 1.0),
            new Market.ASMarket(500.0, 0.01, 0.01, 11_700.0, 0.5, 1.0, 200, 1.0, 1.0, 23_400.0,
                1.0),
            new Market.ASMarket(1.10, 1e-5, 0.005, 8640.0, 2.0, 3.0, 50, 2.0, 5.0, 86_400.0, 1.0),
        };
        System.out.printf(Locale.ROOT, "%26s %10s %8s %8s %9s %22s%n", "preset", "order %ADV",
                "E[IS]", "sd", "half-sprd", "A-S spread open/close");
        for (int i = 0; i < names.length; i++) {
            ACParams p = Market.acParamsFromMarket(ac[i]);
            double notional = ac[i].totalShares() * ac[i].price();
            double e = Market.bps(AlmgrenChriss.expectedCost(p), notional);
            double sd = Market.bps(Math.sqrt(AlmgrenChriss.varianceCost(p)), notional);
            double hs = Market.bps(Market.halfSpreadCost(ac[i].spread(), ac[i].totalShares()),
                    notional);
            String spread = "-";
            if (mm[i] != null) {
                ASParams a = Market.asParamsFromMarket(mm[i]);
                spread = String.format(Locale.ROOT, "%8.2f / %6.2f bps",
                        Market.bps(AvellanedaStoikov.optimalSpread(a, 0.0), a.s0()),
                        Market.bps(AvellanedaStoikov.optimalSpread(a, a.horizon()), a.s0()));
            }
            System.out.printf(Locale.ROOT, "%26s %9.1f%% %6.2fbp %6.2fbp %7.2fbp %22s%n",
                    names[i], ac[i].totalShares() / ac[i].adv() * 100.0, e, sd, hs, spread);
        }
    }

    /** Pre-trade report on the power-law model (API_SPEC Section 8). */
    private static void pretradeSection() {
        hdr("Pre-trade report: US large cap, sell 5% ADV in a day (sqrt impact, U-curve)");
        Market.ImpactMarket m = Market.ImpactMarket.of(100.0, 0.01, 5e6, 0.015, 250_000.0, 13,
                1.0 / 13.0).withWeights(Impact.usEquityVolumeCurve13());
        Market.PretradeReport rep = Market.pretradeReport(m, new double[] {0.0, 1e-7, 1e-6, 1e-5});
        System.out.printf(Locale.ROOT, "square-root-law estimate %.2f bps; half spread %.2f bps%n",
                rep.sqrtLawBps(), rep.halfSpreadBps());
        System.out.printf(Locale.ROOT, "%10s %9s %9s %9s %9s %8s%n", "strategy", "lambda",
                "impact", "sd", "total", "horizon");
        for (Market.PretradeRow r : rep.rows()) {
            System.out.printf(Locale.ROOT, "%10s %9.1e %7.2fbp %7.2fbp %7.2fbp %8d%n", r.label(),
                    r.lambdaRa(), r.impactBps(), r.sdBps(), r.totalBps(), r.horizon());
        }
        Impact.Params p = Market.impactParamsFromMarket(m.withDriftBpsPerDay(50.0));
        double notional = m.totalShares() * m.price();
        double[] vwap = Impact.vwapHoldings(m.totalShares(), m.weights());
        Impact.ScheduleResult s = Impact.optimalSchedule(p);
        double eV = Market.bps(Impact.expectedCost(p, vwap), notional);
        double eO = Market.bps(Impact.expectedCost(p, s.holdings()), notional);
        System.out.printf(Locale.ROOT, "alpha-aware: with 50 bps/day adverse drift, VWAP costs "
                + "%.2f bps and the drift-aware schedule %.2f bps (saves %.2f bps)%n", eV, eO,
                eV - eO);
    }

    /**
     * Runs the demo.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        System.out.println("etrade demo - Almgren-Chriss execution & "
                + "Avellaneda-Stoikov market making");
        acSection();
        // mirrors data/params_liquid.json / data/params_fx.json (as_*)
        asSection(new ASParams(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, 20, 0.5),
                "liquid large-cap", 42);
        asSection(new ASParams(1.1, 0.004, 1.0, 0.005, 100.0, 2000.0, 500.0, 50, 2e-4),
                "FX-like: tight spread, high k", 7);
        analyticsSection();
        marketSection();
        pretradeSection();
        System.out.println();
        System.out.println("done.");
    }
}
