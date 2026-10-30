package io.github.ashjha0.etrade;

import java.util.Locale;

/**
 * End-to-end demo of the etrade library: Almgren-Chriss optimal trajectories
 * and efficient frontier, TWAP/POV/instantaneous baselines with an MC
 * cross-check, Avellaneda-Stoikov market making (A-S vs fixed-spread agent,
 * liquid and FX-like parameter sets, vol-stress scenario), and an
 * implementation-shortfall decomposition with a market-impact regression,
 * presets in bps, a pre-trade report, post-trade TCA, calibration with the
 * fitted ITCH presets, execution extensions, the market-making friction
 * ledger and limit-order placement.
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
        Analytics.ISDecomposition d = Analytics.decomposeShortfall("sell", 100.0, 99.5,
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

    /** Post-trade TCA (API_SPEC Section 10); deterministic. */
    private static void tcaSection() {
        hdr("Post-trade TCA: benchmarks, markouts, participation, A/B test");
        int[] fillBins = {0, 1, 1, 2};
        double[] fillQtys = {20_000.0, 15_000.0, 10_000.0, 25_000.0};
        double[] fillPrices = {99.96, 99.91, 99.89, 99.84};
        double[] binVolumes = {400_000.0, 300_000.0, 300_000.0, 500_000.0};
        double[] binVwaps = {99.95, 99.90, 99.85, 99.82};
        Tca.SlippageReport r = Tca.slippage("sell", 100.0, 99.80, fillBins, fillQtys, fillPrices,
                binVolumes, binVwaps);
        System.out.printf(Locale.ROOT, "sell 70,000 at avg %.4f: arrival %.4f bps, full-day VWAP "
                + "%.4f bps,%n", r.avgPrice(), r.arrivalBps(), r.vwapBps());
        System.out.printf(Locale.ROOT, "  interval VWAP %.4f bps, close %.4f bps (positive = "
                + "cost)%n", r.intervalVwapBps(), r.closeBps());
        Tca.ParticipationResult part = Tca.participation(fillBins, fillQtys, binVolumes);
        double[] mids = {100.00, 99.98, 99.95, 99.93, 99.90, 99.91, 99.88, 99.86, 99.85, 99.84};
        Tca.MarkoutResult mo = Tca.markouts("sell", new int[] {0, 2, 4, 6}, fillQtys, mids,
                new int[] {1, 3});
        System.out.printf(Locale.ROOT, "participation %.4f%%; markout per share after 1 step "
                + "%.6f $, after 3 steps %.6f $%n", 100.0 * part.overall(), mo.perShare()[0],
                mo.perShare()[1]);
        Tca.WelchResult w = Tca.welchTTest(new double[] {1.2, 3.4, 2.2, 5.1, 0.7, 2.9, 4.0, 3.3},
                new double[] {0.4, 1.1, -0.2, 0.9, 1.6, 0.3});
        System.out.printf(Locale.ROOT, "algo A vs B (8 vs 6 orders): diff %.4f bps, t = %.4f, "
                + "df = %.4f, p = %.6f%n", w.meanDiff(), w.tStat(), w.df(), w.pValue());
    }

    private static Market.ASMarket asMarket(double price, double vol, double touch,
                                            double halving, double skew, int qMax, double dt) {
        return new Market.ASMarket(price, 0.01, vol, touch, halving, skew, qMax, 1.0, dt,
                23_400.0, 1.0);
    }

    /** Calibration estimators and the fitted ITCH presets (API_SPEC Section 11). */
    private static void calibrationSection() {
        hdr("Calibration: fill intensity from 100 market orders; fitted ITCH presets");
        double[] dist = new double[100];
        java.util.Arrays.fill(dist, 0, 60, 0.005);
        java.util.Arrays.fill(dist, 60, 85, 0.015);
        java.util.Arrays.fill(dist, 85, 95, 0.025);
        java.util.Arrays.fill(dist, 95, 100, 0.035);
        Calibration.FillIntensity fi = Calibration.fillIntensity(dist, 0.01, 4, 1.0, 2.0);
        System.out.printf(Locale.ROOT, "toy fit: A = %.4f/day, k = %.4f per $, touch fills "
                + "%.4f/day, halving %.4f ticks%n", fi.aRate(), fi.k(), fi.touchFillsPerDay(),
                fi.halvingTicks());
        // mirrors data/presets.json, itch_*_20191230
        String[] names = {"itch_aapl_20191230", "itch_msft_20191230", "itch_qqq_20191230"};
        Market.ACMarket[] acs = {
            Market.ACMarket.of(289.72, 0.026401, 5_678_176.0, 0.01157, 283_900.0, 13,
                    1.0 / 13.0, 4.66e-8),
            Market.ACMarket.of(157.67, 0.012285, 3_625_411.0, 0.007432, 181_300.0, 13,
                    1.0 / 13.0, 2.78e-7),
            Market.ACMarket.of(212.16, 0.011602, 3_713_405.0, 0.005745, 185_700.0, 13,
                    1.0 / 13.0, 2.21e-7)};
        Market.ASMarket[] ass = {
            asMarket(289.72, 0.01157, 22_070.0, 0.46, 2.25, 50, 5.0),
            asMarket(157.67, 0.007432, 10_507.0, 0.251, 2.25, 50, 5.0),
            asMarket(212.16, 0.005745, 7_532.0, 0.214, 1.0, 200, 1.0)};
        System.out.println("fitted to one NASDAQ ITCH day, 2019-12-30 (one venue; caveats in "
                + "API_SPEC 11.6):");
        System.out.printf(Locale.ROOT, "%20s %7s %8s %7s %10s %8s %9s %12s%n", "preset", "price",
                "spread", "vol", "A /day", "k /$", "A-S open", "E[IS] 5% ADV");
        for (int i = 0; i < names.length; i++) {
            Market.ACMarket m = acs[i];
            ACParams p = Market.acParamsFromMarket(m);
            ASParams a = Market.asParamsFromMarket(ass[i]);
            double e = Market.bps(AlmgrenChriss.expectedCost(p), m.totalShares() * m.price());
            double o = Market.bps(AvellanedaStoikov.optimalSpread(a, 0.0), a.s0());
            System.out.printf(Locale.ROOT, "%20s %7.2f %6.3fbp %6.3f%% %10.1f %8.3f %7.4fbp "
                    + "%10.4fbp%n", names[i], m.price(), Market.bps(m.spread(), m.price()),
                    100.0 * m.dailyVol(), a.aRate(), a.k(), o, e);
        }
    }

    /** Transient impact, adaptive re-planning, internalisation (API_SPEC Section 12). */
    private static void executionExtSection() {
        hdr("Execution extensions: transient impact, adaptive vs static, internalisation");
        Execution.TransientParams tp = new Execution.TransientParams(1e6, 10, 0.1, 1e-6, 5.0,
                0.0);
        double[] x = Execution.transientOptimalSchedule(tp);
        double eOpt = Execution.transientExpectedCost(tp, x);
        double eTwap = Execution.transientExpectedCost(tp, AlmgrenChriss.twapHoldings(1e6, 10));
        System.out.printf(Locale.ROOT, "Obizhaeva-Wang, rho*tau = 0.5: first / middle / last "
                + "trade %.2f / %.2f / %.2f shares%n", x[0] - x[1], x[4] - x[5], x[9] - x[10]);
        System.out.printf(Locale.ROOT, "  E[IS] optimal %.2f $ vs TWAP %.2f $ (saves %.4f%%)%n",
                eOpt, eTwap, 100.0 * (1.0 - eOpt / eTwap));
        ACParams p = new ACParams(1_000_000.0, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6);
        System.out.println("adaptive re-planning, liquid set, 20k paths (each language samples "
                + "its own paths):");
        System.out.printf(Locale.ROOT, "%24s %13s %11s %9s %6s%n", "case", "static E[IS]",
                "adaptive", "saving", "se");
        String[] labels = {"no signal, full fills", "AR(1) signal 0.7, 0.5", "fill risk 0.5"};
        Execution.SignalParams[] sigs = {new Execution.SignalParams(0.0, 0.0, 0.0),
            new Execution.SignalParams(0.7, 0.5, 0.0), new Execution.SignalParams(0.0, 0.0, 0.5)};
        for (int i = 0; i < labels.length; i++) {
            Execution.AdaptiveSimResult s = Execution.simulateAdaptive(p, sigs[i], 20_000, 1L);
            System.out.printf(Locale.ROOT, "%24s %13.0f %11.0f %9.0f %6.0f%n", labels[i],
                    s.staticMean(), s.adaptiveMean(), s.savingMean(), s.savingSe());
        }
        Execution.InternalisationResult r = Execution.internalise(
                new double[] {200_000.0, 50_000.0}, new double[] {150_000.0}, 100.0, 0.01, 5e6,
                0.015, 13, 1.0 / 13.0, 0.5, 1.0, 0.10, Impact.usEquityVolumeCurve13());
        System.out.printf(Locale.ROOT, "internalise buys 200k + 50k against a 150k sell: cross "
                + "%.0f, send %.0f (%s)%n", r.crossedShares(), r.residualShares(),
                r.residualSide());
        System.out.printf(Locale.ROOT, "  impact %.2f $ -> %.2f $, saves %.4f bps of gross "
                + "notional%n", r.baselineCost(), r.internalCost(), r.savedBps());
    }

    /** Market making with frictions: the friction ledger (API_SPEC Section 9). */
    private static void mmFrictionsSection() {
        hdr("Market making with frictions: US large-cap A-S preset, one-share quotes");
        // mirrors data/presets.json, us_large_cap "as"
        ASParams a = Market.asParamsFromMarket(asMarket(100.0, 0.015, 1170.0, 0.462, 2.25, 50,
                5.0));
        System.out.printf(Locale.ROOT, "A = %.4f/day, k = %.4f per $, gamma = %.6f, %d steps of "
                + "5 s%n", a.aRate(), a.k(), a.gamma(), a.nSteps());
        for (double q : new double[] {0.0, 3.0}) {
            MarketMaking.HalfSpreads h = MarketMaking.glftHalfSpreads(a, q);
            System.out.printf(Locale.ROOT, "GLFT half spreads at q = %+.0f: bid %.8f $, ask "
                    + "%.8f $%n", q, h.deltaBid(), h.deltaAsk());
        }
        AvellanedaStoikov.Quotes rt = MarketMaking.roundToTick(99.9934, 100.0066, 0.01);
        System.out.printf(Locale.ROOT, "tick rounding away from fair: 99.9934 / 100.0066 -> "
                + "%.2f / %.2f; weighted mid %.3f%n", rt.bid(), rt.ask(),
                MarketMaking.weightedMid(99.99, 100.01, 900.0, 100.0));
        System.out.println("friction ledger, $ per day, 5k paths (each language samples its own "
                + "paths):");
        System.out.printf(Locale.ROOT, "%20s %7s %5s | %7s %6s %7s %8s | %6s %8s%n", "case",
                "PnL", "se", "spread", "fees", "invent", "markout", "fills", "informed");
        MarketMaking.Agent as = MarketMaking.Agent.AS;
        // Seeds 101-106 are the mm_sim golden cases of API_SPEC 9.7; the fee row is extra.
        String[] labels = {"frictionless", "tick + 0.2c rebate", "tick + 0.2c fee",
            "+ 15% informed flow", "+ 60% signal", "quotes 5 s stale", "GLFT, tick + rebate"};
        MarketMaking.Config[] cfgs = {
            new MarketMaking.Config(as, 0.0, 0.0, 0, 0.0, 0.5, null),
            new MarketMaking.Config(as, 0.01, 0.002, 0, 0.0, 0.5, null),
            new MarketMaking.Config(as, 0.01, -0.002, 0, 0.0, 0.5, null),
            new MarketMaking.Config(as, 0.01, 0.002, 0, 0.15, 0.5, null),
            new MarketMaking.Config(as, 0.01, 0.002, 0, 0.15, 0.6, null),
            new MarketMaking.Config(as, 0.01, 0.002, 1, 0.0, 0.5, null),
            new MarketMaking.Config(MarketMaking.Agent.GLFT, 0.01, 0.002, 0, 0.0, 0.5, null)};
        long[] seeds = {101, 102, 107, 103, 104, 105, 106};
        for (int i = 0; i < labels.length; i++) {
            MarketMaking.Result r = MarketMaking.simulate(a, 5_000, seeds[i], cfgs[i]);
            System.out.printf(Locale.ROOT, "%20s %7.2f %5.2f | %7.2f %6.2f %7.2f %8.2f | %6.1f "
                    + "%8.1f%n", labels[i], r.mean(MarketMaking.Stat.PNL),
                    r.se(MarketMaking.Stat.PNL), r.mean(MarketMaking.Stat.SPREAD),
                    r.mean(MarketMaking.Stat.FEES), r.mean(MarketMaking.Stat.INVENTORY),
                    r.mean(MarketMaking.Stat.MARKOUT), r.mean(MarketMaking.Stat.FILLS),
                    r.mean(MarketMaking.Stat.INFORMED_FILLS));
        }
        System.out.println("PnL = spread + fees + inventory - liquidation on every path; the "
                + "markout is part");
        System.out.println("of inventory (negative = adverse selection).");
    }

    /** Queue position, passive or aggressive, liquidation (API_SPEC Section 13). */
    private static void placementSection() {
        hdr("Limit-order placement: queue fill probability, post or cross, liquidation");
        System.out.printf(Locale.ROOT, "%8s %5s %6s %6s %8s %14s%n", "q_ahead", "size", "mu",
                "theta", "horizon", "P(full fill)");
        int[][] qs = {{0, 5}, {30, 5}, {50, 10}, {400, 50}};
        double[][] rates = {{10.0, 0.1, 0.5}, {10.0, 0.0, 2.0}, {20.0, 0.05, 3.0},
            {100.0, 0.2, 4.0}};
        for (int i = 0; i < qs.length; i++) {
            double pf = Placement.queueFillProbability(qs[i][0], qs[i][1], rates[i][0],
                    rates[i][1], rates[i][2]);
            System.out.printf(Locale.ROOT, "%8d %5d %6.1f %6.2f %8.1f %14.10f%n", qs[i][0],
                    qs[i][1], rates[i][0], rates[i][1], rates[i][2], pf);
        }
        String[] where = {"near the front", "deep in queue"};
        int[] ahead = {5, 60};
        double[][] dec = {{0.1, 1.0, 0.01}, {0.05, 3.0, 0.03}};
        for (int i = 0; i < where.length; i++) {
            double pf = Placement.queueFillProbability(ahead[i], 2, 10.0, dec[i][0], dec[i][1]);
            Placement.PlacementDecision d = Placement.placeOrCross(2000.0, 0.02, pf, dec[i][2],
                    1e-4, 0.5, 1.0 / 13.0);
            System.out.printf(Locale.ROOT, "sell 2,000 %s: P(fill) %.6f, post %.6f vs cross %.6f "
                    + "$/sh, breakeven %.4f: %s%n", where[i], pf, d.passiveCost(),
                    d.aggressiveCost(), d.breakevenFillProb(), d.passive() ? "post" : "cross");
        }
        System.out.println("liquidate 10 units over 50 periods (patient) or 20 with penalty "
                + "phi = 0.05 (urgent):");
        System.out.printf(Locale.ROOT, "%8s %9s %8s %4s %7s | %8s %8s %8s %17s%n", "case",
                "DP cost", "depth", "mkt", "all-mkt", "optimal", "all-mkt", "touch",
                "saving vs all-mkt");
        String[] cases = {"patient", "urgent"};
        Placement.LiquidationParams[] lps = {
            new Placement.LiquidationParams(10, 50, 0.02, 0.5, 0.02, 30.0, 0.3, 0.005, 0.0),
            new Placement.LiquidationParams(20, 50, 0.02, 0.5, 0.02, 50.0, 0.4, 0.005, 0.05)};
        long[] seeds = {201, 202};
        for (int i = 0; i < cases.length; i++) {
            Placement.LiquidationParams lp = lps[i];
            Placement.LiquidationPolicy pol = Placement.liquidationPolicy(lp);
            int nMkt = 0;
            for (boolean[] row : pol.market()) {
                for (int q = 1; q < row.length; q++) {
                    nMkt += row[q] ? 1 : 0;
                }
            }
            Placement.LiquidationSimResult s = Placement.simulateLiquidation(lp, 20_000,
                    seeds[i]);
            System.out.printf(Locale.ROOT, "%8s %9.5f %8.5f %4d %7.3f | %8.4f %8.4f %8.4f %9.4f "
                    + "+- %.4f%n", cases[i], pol.cost()[0][lp.totalUnits()],
                    pol.depth()[0][lp.totalUnits()], nMkt, Placement.aggressiveCost(lp),
                    s.optimalMean(), s.aggressiveMean(), s.passiveMean(),
                    s.savingVsAggressiveMean(), s.savingVsAggressiveSe());
        }
        System.out.println("costs in $ per position against the starting mid (negative = sold "
                + "above it); the");
        System.out.println("DP cost adds the urgency penalty, so it exceeds the simulated "
                + "shortfall when phi > 0;");
        System.out.println("depth = first posting depth in $; mkt = states where the policy "
                + "sends a market order;");
        System.out.println("simulations 20k paths, each language samples its own paths.");
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
        tcaSection();
        calibrationSection();
        executionExtSection();
        mmFrictionsSection();
        placementSection();
        System.out.println();
        System.out.println("done.");
    }
}
