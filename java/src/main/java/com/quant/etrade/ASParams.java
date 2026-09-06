package com.quant.etrade;

import static com.quant.etrade.Validation.require;

/**
 * Avellaneda-Stoikov market-making parameters (2008 paper, finite horizon).
 *
 * <p>The mid price is an arithmetic Brownian motion (prices may go negative —
 * a documented model property).  {@code gamma} must be strictly positive:
 * the {@code gamma -> 0} limit blows up the utility derivation, so
 * {@code gamma <= 0} is an input error per spec.
 *
 * @param s0      initial mid price (any finite real; arithmetic model)
 * @param sigma   arithmetic mid volatility per sqrt(time unit), {@code >= 0}
 * @param horizon terminal time {@code T > 0}
 * @param dt      simulation step, {@code 0 < dt <= T}, with {@code T/dt} integral (1e-9 tol)
 *                and {@code T/dt <= AvellanedaStoikov.MAX_AS_STEPS}
 * @param gamma   risk aversion, strictly {@code > 0}
 * @param k       intensity decay, {@code > 0}
 * @param aRate   intensity scale {@code A > 0}
 * @param qMax    hard inventory cap (units), integer {@code >= 0}
 * @param etaLiq  terminal liquidation penalty per unit inventory, {@code >= 0}
 */
public record ASParams(double s0, double sigma, double horizon, double dt, double gamma,
                       double k, double aRate, int qMax, double etaLiq) {

    /** Validates all fields; see the class javadoc for the constraints. */
    public ASParams {
        require(Double.isFinite(s0), "s0 must be finite");
        require(Double.isFinite(sigma) && sigma >= 0.0, "sigma must be >= 0");
        require(Double.isFinite(horizon) && horizon > 0.0, "horizon must be > 0");
        require(Double.isFinite(dt) && dt > 0.0 && dt <= horizon,
                "dt must satisfy 0 < dt <= horizon");
        require(Double.isFinite(gamma) && gamma > 0.0,
                "gamma must be > 0 (gamma = 0 is outside the A-S model)");
        require(Double.isFinite(k) && k > 0.0, "k must be > 0");
        require(Double.isFinite(aRate) && aRate > 0.0, "aRate (A) must be > 0");
        require(qMax >= 0, "qMax must be an integer >= 0");
        require(Double.isFinite(etaLiq) && etaLiq >= 0.0, "etaLiq must be >= 0");
        double n = horizon / dt;
        require(Math.abs(n - Math.round(n)) < 1e-9,
                "horizon must be an integer multiple of dt");
        require(n <= AvellanedaStoikov.MAX_AS_STEPS,
                "horizon/dt must be <= 10000000 steps");
    }

    /** Number of simulation steps {@code T/dt}. */
    public int nSteps() {
        return (int) Math.round(horizon / dt);
    }
}
