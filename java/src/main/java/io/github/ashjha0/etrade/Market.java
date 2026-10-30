package io.github.ashjha0.etrade;

import static io.github.ashjha0.etrade.Validation.require;

/**
 * Market-unit layer: builds model parameters from quoted market quantities and reports
 * results in basis points (API_SPEC Section 7).
 *
 * <p>Almgren-Chriss impact follows the paper's own calibration: trading 1% of ADV per day
 * costs one spread of temporary impact, 10% of ADV moves the price permanently by one
 * spread. Avellaneda-Stoikov flow is described by the fill rate of a quote at the touch and
 * the distance over which that rate halves; risk aversion by the opening skew per unit of
 * inventory.
 */
public final class Market {

    /** Default fraction of ADV per day that costs one spread of temporary impact. */
    public static final double DEFAULT_TEMP_FRAC = 0.01;
    /** Default fraction of ADV that moves the price permanently by one spread. */
    public static final double DEFAULT_PERM_FRAC = 0.10;
    /** Default trading-session length in seconds (US equities, 6.5 hours). */
    public static final double DEFAULT_SESSION_SECONDS = 23_400.0;

    private Market() {
    }

    /**
     * Almgren-Chriss inputs in market units (API_SPEC 7.1).
     *
     * @param price price, &gt; 0
     * @param spread full quoted spread in currency, &gt; 0
     * @param adv average daily volume in shares per day, &gt; 0
     * @param dailyVol daily volatility as a fraction (0.015 = 1.5%), &gt; 0
     * @param totalShares shares to sell, &ge; 0
     * @param nIntervals number of trading intervals
     * @param tau interval length in trading days
     * @param lambdaRa risk aversion (1/currency), &ge; 0
     * @param tempFrac fraction of ADV per day costing one spread of temporary impact, in (0, 1]
     * @param permFrac fraction of ADV moving the price permanently by one spread, in (0, 1]
     */
    public record ACMarket(double price, double spread, double adv, double dailyVol,
                           double totalShares, int nIntervals, double tau, double lambdaRa,
                           double tempFrac, double permFrac) {

        /**
         * Inputs with the default impact fractions (0.01 and 0.10).
         *
         * @return the market inputs
         */
        public static ACMarket of(double price, double spread, double adv, double dailyVol,
                                  double totalShares, int nIntervals, double tau,
                                  double lambdaRa) {
            return new ACMarket(price, spread, adv, dailyVol, totalShares, nIntervals, tau,
                    lambdaRa, DEFAULT_TEMP_FRAC, DEFAULT_PERM_FRAC);
        }
    }

    /**
     * Avellaneda-Stoikov inputs in market units (API_SPEC 7.2).
     *
     * @param price price (initial mid), &gt; 0
     * @param tick tick size, &gt; 0
     * @param dailyVol daily volatility as a fraction, &gt; 0
     * @param touchFillsPerDay fills per side per day of a quote half a tick from the mid, &gt; 0
     * @param halvingTicks ticks over which the intensity halves, &gt; 0
     * @param skewTicks opening skew per unit of inventory in ticks, &gt; 0
     * @param qMax hard inventory cap, &ge; 0
     * @param liqTicks terminal liquidation penalty in ticks, &ge; 0
     * @param dtSeconds step length in seconds, &gt; 0
     * @param sessionSeconds session length in seconds, &gt; 0
     * @param horizonDays horizon in trading days, &gt; 0
     */
    public record ASMarket(double price, double tick, double dailyVol, double touchFillsPerDay,
                           double halvingTicks, double skewTicks, int qMax, double liqTicks,
                           double dtSeconds, double sessionSeconds, double horizonDays) {
    }

    private static void positive(String name, double v) {
        require(Double.isFinite(v) && v > 0.0, name + " must be finite and > 0");
    }

    /**
     * {@code sigma = price*dailyVol}, {@code eta = spread/(tempFrac*adv)},
     * {@code gammaP = spread/(permFrac*adv)}; the result is validated as any {@link ACParams}.
     *
     * @param m market inputs
     * @return model parameters
     * @throws IllegalArgumentException on invalid inputs
     */
    public static ACParams acParamsFromMarket(ACMarket m) {
        require(m != null, "market inputs must not be null");
        positive("price", m.price());
        positive("spread", m.spread());
        positive("adv", m.adv());
        positive("daily_vol", m.dailyVol());
        require(Double.isFinite(m.tempFrac()) && m.tempFrac() > 0.0 && m.tempFrac() <= 1.0,
                "temp_frac must be in (0, 1]");
        require(Double.isFinite(m.permFrac()) && m.permFrac() > 0.0 && m.permFrac() <= 1.0,
                "perm_frac must be in (0, 1]");
        return new ACParams(m.totalShares(), m.nIntervals(), m.tau(), m.price() * m.dailyVol(),
                m.spread() / (m.tempFrac() * m.adv()), m.spread() / (m.permFrac() * m.adv()),
                m.lambdaRa());
    }

    /**
     * {@code k = ln2/(halvingTicks*tick)}, {@code A = touchFillsPerDay*exp(k*tick/2)},
     * {@code gamma = skewTicks*tick/(sigma^2*horizonDays)}, {@code etaLiq = liqTicks*tick},
     * {@code dt = dtSeconds/sessionSeconds}; the result is validated as any {@link ASParams}.
     *
     * @param m market inputs
     * @return model parameters
     * @throws IllegalArgumentException on invalid inputs
     */
    public static ASParams asParamsFromMarket(ASMarket m) {
        require(m != null, "market inputs must not be null");
        positive("price", m.price());
        positive("tick", m.tick());
        positive("daily_vol", m.dailyVol());
        positive("touch_fills_per_day", m.touchFillsPerDay());
        positive("halving_ticks", m.halvingTicks());
        positive("skew_ticks", m.skewTicks());
        require(Double.isFinite(m.liqTicks()) && m.liqTicks() >= 0.0,
                "liq_ticks must be finite and >= 0");
        positive("dt_seconds", m.dtSeconds());
        positive("session_seconds", m.sessionSeconds());
        positive("horizon_days", m.horizonDays());
        double sigma = m.price() * m.dailyVol();
        double k = Math.log(2.0) / (m.halvingTicks() * m.tick());
        return new ASParams(m.price(), sigma, m.horizonDays(), m.dtSeconds() / m.sessionSeconds(),
                m.skewTicks() * m.tick() / (sigma * sigma * m.horizonDays()), k,
                m.touchFillsPerDay() * Math.exp(k * m.tick() / 2.0), m.qMax(),
                m.liqTicks() * m.tick());
    }

    /**
     * Basis points of a notional: {@code 1e4 * amount / notional}.
     *
     * @param amount finite amount
     * @param notional notional, &gt; 0
     * @return basis points
     * @throws IllegalArgumentException on invalid inputs
     */
    public static double bps(double amount, double notional) {
        require(Double.isFinite(amount), "amount must be finite");
        positive("notional", notional);
        return 1e4 * amount / notional;
    }

    /**
     * Cost of crossing half the quoted spread on every share: {@code 0.5*spread*X}.
     *
     * @param spread full quoted spread, &gt; 0
     * @param totalShares shares, &ge; 0
     * @return currency cost
     * @throws IllegalArgumentException on invalid inputs
     */
    public static double halfSpreadCost(double spread, double totalShares) {
        positive("spread", spread);
        require(Double.isFinite(totalShares) && totalShares >= 0.0,
                "total_shares must be finite and >= 0");
        return 0.5 * spread * totalShares;
    }

    /**
     * Power-law impact inputs in market units (API_SPEC 8.3). Build with
     * {@link #of(double, double, double, double, double, int, double)} and the
     * {@code with...} methods for the optional fields.
     *
     * @param price price, &gt; 0
     * @param spread full quoted spread, &gt; 0
     * @param adv average daily volume, &gt; 0
     * @param dailyVol daily volatility as a fraction, &gt; 0
     * @param totalShares shares to sell, &ge; 0
     * @param nIntervals number of intervals
     * @param tau interval length in trading days
     * @param lambdaRa risk aversion, &ge; 0
     * @param alpha impact exponent in (0, 1]
     * @param y square-root-law prefactor, &gt; 0
     * @param permFrac fraction of ADV moving the price permanently by one spread, in (0, 1]
     * @param driftBpsPerDay expected adverse drift in bps per day, &ge; 0
     * @param weights volume fractions per interval, or null for {@code tau} each
     */
    public record ImpactMarket(double price, double spread, double adv, double dailyVol,
                               double totalShares, int nIntervals, double tau, double lambdaRa,
                               double alpha, double y, double permFrac, double driftBpsPerDay,
                               double[] weights) {

        /** Defensive copy of the weights. */
        public ImpactMarket {
            weights = weights == null ? null : weights.clone();
        }

        /**
         * Core inputs with defaults: lambda 0, alpha 0.5, Y 1, permFrac 0.10, no drift,
         * uniform weights.
         *
         * @param price price
         * @param spread full quoted spread
         * @param adv average daily volume
         * @param dailyVol daily volatility as a fraction
         * @param totalShares shares to sell
         * @param nIntervals number of intervals
         * @param tau interval length in trading days
         * @return the market inputs
         */
        public static ImpactMarket of(double price, double spread, double adv, double dailyVol,
                                      double totalShares, int nIntervals, double tau) {
            return new ImpactMarket(price, spread, adv, dailyVol, totalShares, nIntervals, tau,
                    0.0, 0.5, 1.0, DEFAULT_PERM_FRAC, 0.0, null);
        }

        /** @return a copy of the weights, or null */
        @Override
        public double[] weights() {
            return weights == null ? null : weights.clone();
        }

        /**
         * @param v risk aversion
         * @return a copy with this value
         */
        public ImpactMarket withLambdaRa(double v) {
            return new ImpactMarket(price, spread, adv, dailyVol, totalShares, nIntervals, tau, v,
                    alpha, y, permFrac, driftBpsPerDay, weights);
        }

        /**
         * @param v impact exponent
         * @return a copy with this value
         */
        public ImpactMarket withAlpha(double v) {
            return new ImpactMarket(price, spread, adv, dailyVol, totalShares, nIntervals, tau,
                    lambdaRa, v, y, permFrac, driftBpsPerDay, weights);
        }

        /**
         * @param v prefactor Y
         * @return a copy with this value
         */
        public ImpactMarket withY(double v) {
            return new ImpactMarket(price, spread, adv, dailyVol, totalShares, nIntervals, tau,
                    lambdaRa, alpha, v, permFrac, driftBpsPerDay, weights);
        }

        /**
         * @param v permanent fraction
         * @return a copy with this value
         */
        public ImpactMarket withPermFrac(double v) {
            return new ImpactMarket(price, spread, adv, dailyVol, totalShares, nIntervals, tau,
                    lambdaRa, alpha, y, v, driftBpsPerDay, weights);
        }

        /**
         * @param v drift in bps per day
         * @return a copy with this value
         */
        public ImpactMarket withDriftBpsPerDay(double v) {
            return new ImpactMarket(price, spread, adv, dailyVol, totalShares, nIntervals, tau,
                    lambdaRa, alpha, y, permFrac, v, weights);
        }

        /**
         * @param v volume weights (null = uniform)
         * @return a copy with these weights
         */
        public ImpactMarket withWeights(double[] v) {
            return new ImpactMarket(price, spread, adv, dailyVol, totalShares, nIntervals, tau,
                    lambdaRa, alpha, y, permFrac, driftBpsPerDay, v);
        }

        /**
         * @param v shares to sell
         * @return a copy with this value
         */
        public ImpactMarket withTotalShares(double v) {
            return new ImpactMarket(price, spread, adv, dailyVol, v, nIntervals, tau, lambdaRa,
                    alpha, y, permFrac, driftBpsPerDay, weights);
        }
    }

    /**
     * {@code eta = y*sigma/adv^alpha}, {@code gammaP = spread/(permFrac*adv)},
     * {@code drift = price*driftBpsPerDay*1e-4} (API_SPEC 8.3).
     *
     * @param m market inputs
     * @return model parameters
     * @throws IllegalArgumentException on invalid inputs
     */
    public static Impact.Params impactParamsFromMarket(ImpactMarket m) {
        require(m != null, "market inputs must not be null");
        positive("price", m.price());
        positive("spread", m.spread());
        positive("adv", m.adv());
        positive("daily_vol", m.dailyVol());
        positive("y", m.y());
        require(Double.isFinite(m.permFrac()) && m.permFrac() > 0.0 && m.permFrac() <= 1.0,
                "perm_frac must be in (0, 1]");
        require(Double.isFinite(m.driftBpsPerDay()) && m.driftBpsPerDay() >= 0.0,
                "drift_bps_per_day must be finite and >= 0");
        require(Double.isFinite(m.alpha()) && m.alpha() > 0.0 && m.alpha() <= 1.0,
                "alpha must be in (0, 1]");
        double sigma = m.price() * m.dailyVol();
        return new Impact.Params(m.totalShares(), m.nIntervals(), m.tau(), sigma,
                m.y() * sigma / Math.pow(m.adv(), m.alpha()), m.alpha(),
                m.spread() / (m.permFrac() * m.adv()), m.lambdaRa(),
                m.price() * m.driftBpsPerDay() * 1e-4, m.weights());
    }

    /**
     * One strategy in a pre-trade report (bps of {@code X * price}).
     *
     * @param label "twap", "vwap" or "optimal"
     * @param lambdaRa risk aversion of the row
     * @param impactBps bps of E[IS]
     * @param sdBps bps of sqrt(Var)
     * @param totalBps impactBps + halfSpreadBps
     * @param horizon horizon used
     */
    public record PretradeRow(String label, double lambdaRa, double impactBps, double sdBps,
                              double totalBps, int horizon) {
    }

    /**
     * Pre-trade report (API_SPEC 8.4).
     *
     * @param sqrtLawBps closed-form square-root-law estimate
     * @param halfSpreadBps half-spread cost in bps
     * @param rows TWAP, VWAP, then optimal per lambda
     */
    public record PretradeReport(double sqrtLawBps, double halfSpreadBps,
                                 java.util.List<PretradeRow> rows) {
        /** Immutable copy of the rows. */
        public PretradeReport {
            rows = java.util.List.copyOf(rows);
        }
    }

    /**
     * TWAP, VWAP and the optimal schedule per lambda, costed on the power-law model;
     * {@code m.lambdaRa()} is ignored (API_SPEC 8.4).
     *
     * @param m market inputs
     * @param lambdas non-empty grid of risk aversions
     * @return the report
     */
    public static PretradeReport pretradeReport(ImpactMarket m, double[] lambdas) {
        require(lambdas != null && lambdas.length > 0, "lambdas must be non-empty");
        Impact.Params p0 = impactParamsFromMarket(m.withLambdaRa(0.0));
        double notional = m.totalShares() * m.price();
        positive("total_shares * price", notional);
        double hs = bps(halfSpreadCost(m.spread(), m.totalShares()), notional);
        java.util.List<PretradeRow> rows = new java.util.ArrayList<>();
        rows.add(row("twap", 0.0, AlmgrenChriss.twapHoldings(m.totalShares(), m.nIntervals()),
                m.nIntervals(), p0, notional, hs));
        rows.add(row("vwap", 0.0, Impact.vwapHoldings(m.totalShares(), p0.weights()),
                m.nIntervals(), p0, notional, hs));
        for (double lam : lambdas) {
            require(Double.isFinite(lam) && lam >= 0.0, "lambdas must be finite and >= 0");
            Impact.Params p = impactParamsFromMarket(m.withLambdaRa(lam));
            Impact.ScheduleResult s = Impact.optimalSchedule(p);
            rows.add(row("optimal", lam, s.holdings(), s.horizon(), p, notional, hs));
        }
        return new PretradeReport(
                Impact.sqrtLawCostBps(m.dailyVol(), m.totalShares(), m.adv(), m.y()), hs, rows);
    }

    private static PretradeRow row(String label, double lam, double[] x, int horizon,
                                   Impact.Params p, double notional, double hs) {
        double e = bps(Impact.expectedCost(p, x), notional);
        double sd = bps(Math.sqrt(Impact.varianceCost(p, x)), notional);
        return new PretradeRow(label, lam, e, sd, e + hs, horizon);
    }
}
