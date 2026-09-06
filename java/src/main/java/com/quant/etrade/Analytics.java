package com.quant.etrade;

import static com.quant.etrade.Validation.require;
import static com.quant.etrade.Validation.requireFinite;

/**
 * Execution analytics: implementation-shortfall decomposition (Perold
 * convention), participation rate, and a tiny native OLS for market-impact
 * estimation.
 *
 * <p>Implementation shortfall of a parent order of {@code X} shares with
 * decision price {@code P_d}, arrival price {@code P_0}, close price
 * {@code P_T}, fills {@code (n_j, p_j)} (executed {@code Q = sum n_j},
 * remainder {@code R = X - Q}); all components are currency costs (positive =
 * loss).  For a sell ({@code sign = +1}; a buy mirrors with {@code sign = -1}):
 *
 * <pre>
 *   delay       = Q * sign*(P_d - P_0)
 *   trading     = sum_j n_j * sign*(P_0 - p_j)
 *   opportunity = R * sign*(P_d - P_T)
 *   total       = delay + trading + opportunity
 *               = sign*(X*P_d - sum_j n_j*p_j - R*P_T)     (exact identity)
 * </pre>
 */
public final class Analytics {

    private Analytics() {
    }

    /** Implementation-shortfall components (currency costs; positive = loss).
     *
     * @param total        total shortfall = delay + trading + opportunity
     * @param delay        drift cost before trading started
     * @param trading      execution slippage versus the arrival price
     * @param opportunity  cost of the unfilled remainder versus the decision price
     * @param executedQty  executed quantity Q
     * @param remainingQty unfilled quantity R = X - Q
     */
    public record ISDecomposition(double total, double delay, double trading,
                                  double opportunity, double executedQty,
                                  double remainingQty) {
    }

    /** Simple linear regression {@code y = intercept + slope*x}.
     *
     * @param slope     fitted slope
     * @param intercept fitted intercept
     * @param rSquared  coefficient of determination (1.0 when SST = 0)
     * @param nObs      number of observations
     */
    public record OLSResult(double slope, double intercept, double rSquared, int nObs) {
    }

    /**
     * Decomposes implementation shortfall into delay/trading/opportunity.
     * The identity {@code total == delay + trading + opportunity} holds exactly.
     *
     * @param side          {@code "buy"} or {@code "sell"}
     * @param decisionPrice price when the decision was made (finite)
     * @param arrivalPrice  price when trading started (finite)
     * @param closePrice    closing price (finite)
     * @param totalQty      parent order size {@code X >= 0}
     * @param fillQtys      fill quantities, each {@code >= 0}, same length as prices
     * @param fillPrices    fill prices, finite
     * @return the decomposition
     * @throws IllegalArgumentException on inconsistent inputs (mismatched fill
     *         arrays, negative quantities, over-execution, non-finite prices)
     */
    public static ISDecomposition decomposeShortfall(String side, double decisionPrice,
                                                     double arrivalPrice, double closePrice,
                                                     double totalQty, double[] fillQtys,
                                                     double[] fillPrices) {
        require("buy".equals(side) || "sell".equals(side), "side must be 'buy' or 'sell'");
        requireFinite(decisionPrice, "decisionPrice");
        requireFinite(arrivalPrice, "arrivalPrice");
        requireFinite(closePrice, "closePrice");
        require(Double.isFinite(totalQty) && totalQty >= 0.0, "totalQty must be >= 0");
        require(fillQtys != null && fillPrices != null
                        && fillQtys.length == fillPrices.length,
                "fillQtys and fillPrices must be arrays of equal length");
        double executed = 0.0;
        for (int j = 0; j < fillQtys.length; j++) {
            require(Double.isFinite(fillQtys[j]) && fillQtys[j] >= 0.0,
                    "fill quantities must be >= 0");
            requireFinite(fillPrices[j], "fill prices");
            executed += fillQtys[j];
        }
        require(executed <= totalQty * (1.0 + 1e-12) + 1e-12,
                "executed quantity exceeds totalQty");
        double remaining = totalQty - executed;
        double sign = "sell".equals(side) ? 1.0 : -1.0;
        double delay = executed * sign * (decisionPrice - arrivalPrice);
        double trading = 0.0;
        for (int j = 0; j < fillQtys.length; j++) {
            trading += fillQtys[j] * sign * (arrivalPrice - fillPrices[j]);
        }
        double opportunity = remaining * sign * (decisionPrice - closePrice);
        double total = delay + trading + opportunity;
        return new ISDecomposition(total, delay, trading, opportunity, executed, remaining);
    }

    /**
     * Executed volume as a fraction of market volume over the same window.
     * Market volume includes the trader's own fills, so a rate above 1 is an
     * input error.
     *
     * @param executedVolume executed volume, {@code >= 0} and {@code <= marketVolume}
     * @param marketVolume   market volume, {@code > 0}
     * @return the participation rate
     */
    public static double participationRate(double executedVolume, double marketVolume) {
        require(Double.isFinite(executedVolume) && executedVolume >= 0.0,
                "executedVolume must be >= 0");
        require(Double.isFinite(marketVolume) && marketVolume > 0.0,
                "marketVolume must be > 0");
        require(executedVolume <= marketVolume,
                "executed volume exceeds market volume - market volume must include own fills");
        return executedVolume / marketVolume;
    }

    /**
     * Tiny native OLS: {@code slope = Sxy/Sxx}, {@code intercept = mean(y) -
     * slope*mean(x)}, {@code R^2 = 1 - SSR/SST} ({@code 1.0} when SST = 0).
     *
     * @param x regressor values (length {@code >= 2}, finite, non-degenerate)
     * @param y response values (same length, finite)
     * @return the fitted regression
     */
    public static OLSResult ols(double[] x, double[] y) {
        require(x != null && y != null && x.length == y.length,
                "x and y must be arrays of equal length");
        require(x.length >= 2, "need at least 2 observations");
        double xm = 0.0;
        double ym = 0.0;
        for (int i = 0; i < x.length; i++) {
            requireFinite(x[i], "x values");
            requireFinite(y[i], "y values");
            xm += x[i];
            ym += y[i];
        }
        xm /= x.length;
        ym /= y.length;
        double sxx = 0.0;
        double sxy = 0.0;
        for (int i = 0; i < x.length; i++) {
            sxx += (x[i] - xm) * (x[i] - xm);
            sxy += (x[i] - xm) * (y[i] - ym);
        }
        require(sxx > 0.0, "x is degenerate (zero variance)");
        double slope = sxy / sxx;
        double intercept = ym - slope * xm;
        double ssr = 0.0;
        double sst = 0.0;
        for (int i = 0; i < x.length; i++) {
            double resid = y[i] - (intercept + slope * x[i]);
            ssr += resid * resid;
            sst += (y[i] - ym) * (y[i] - ym);
        }
        double r2 = sst == 0.0 ? 1.0 : 1.0 - ssr / sst;
        return new OLSResult(slope, intercept, r2, x.length);
    }

    /**
     * Deterministic synthetic (participation, per-share cost) fills — the
     * pinned, RNG-free construction identical across languages:
     * {@code part_i = 0.02 + 0.18*i/(n-1)},
     * {@code cost_i = intercept + slope*part_i + noiseAmp*sin(2.5*i + 0.3)}.
     *
     * @param nObs      number of observations, {@code >= 2}
     * @param intercept true intercept of the synthetic line
     * @param slope     true slope of the synthetic line
     * @param noiseAmp  amplitude of the deterministic sinusoidal "noise"
     * @return {@code {participations, costs}} (two arrays of length {@code nObs})
     */
    public static double[][] syntheticImpactFills(int nObs, double intercept,
                                                  double slope, double noiseAmp) {
        require(nObs >= 2, "nObs must be an integer >= 2");
        double[] part = new double[nObs];
        double[] cost = new double[nObs];
        for (int i = 0; i < nObs; i++) {
            part[i] = 0.02 + 0.18 * i / (nObs - 1);
            cost[i] = intercept + slope * part[i] + noiseAmp * Math.sin(2.5 * i + 0.3);
        }
        return new double[][] {part, cost};
    }

    /**
     * Market-impact regression: per-share cost on participation rate.
     *
     * @param participations participation rates
     * @param costs          per-share costs
     * @return the fitted regression
     */
    public static OLSResult estimateImpact(double[] participations, double[] costs) {
        return ols(participations, costs);
    }
}
