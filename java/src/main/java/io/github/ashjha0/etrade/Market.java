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
}
