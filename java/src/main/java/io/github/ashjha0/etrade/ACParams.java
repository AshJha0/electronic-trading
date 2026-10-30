package io.github.ashjha0.etrade;

import static io.github.ashjha0.etrade.Validation.require;

/**
 * Parameters of the discrete Almgren-Chriss optimal-execution problem
 * (sell {@code totalShares} shares over {@code nIntervals} intervals of
 * length {@code tau}; Almgren &amp; Chriss, 2000).
 *
 * <p>Validity (paper footnote): the adjusted temporary impact
 * {@code etaTilde = eta - gammaP*tau/2} must be strictly positive, i.e.
 * {@code gammaP*tau < 2*eta}; violation (including equality) throws
 * {@link IllegalArgumentException} at construction time.
 *
 * @param totalShares shares to sell, {@code X >= 0} ({@code 0} gives the empty trajectory)
 * @param nIntervals  number of trading intervals, {@code 1 <= N <= AlmgrenChriss.MAX_AC_INTERVALS}
 * @param tau         interval length, {@code > 0}
 * @param sigma       arithmetic volatility per sqrt(time unit), {@code >= 0}
 * @param eta         linear temporary impact coefficient, {@code > 0}
 * @param gammaP      linear permanent impact coefficient, {@code >= 0}
 * @param lambdaRa    risk aversion, {@code >= 0} ({@code 0} gives the TWAP limit)
 */
public record ACParams(double totalShares, int nIntervals, double tau, double sigma,
                       double eta, double gammaP, double lambdaRa) {

    /** Validates all fields; see the class javadoc for the constraints. */
    public ACParams {
        require(Double.isFinite(totalShares) && totalShares >= 0.0,
                "totalShares must be finite and >= 0");
        require(nIntervals >= 1 && nIntervals <= AlmgrenChriss.MAX_AC_INTERVALS,
                "nIntervals must be an integer in [1, 10000000]");
        require(Double.isFinite(tau) && tau > 0.0, "tau must be > 0");
        require(Double.isFinite(sigma) && sigma >= 0.0, "sigma must be >= 0");
        require(Double.isFinite(eta) && eta > 0.0, "eta must be > 0");
        require(Double.isFinite(gammaP) && gammaP >= 0.0, "gammaP must be >= 0");
        require(Double.isFinite(lambdaRa) && lambdaRa >= 0.0, "lambdaRa must be >= 0");
        // Paper validity condition: etaTilde = eta - gammaP*tau/2 must be > 0.
        require(gammaP * tau < 2.0 * eta,
                "model invalid: gammaP*tau >= 2*eta (etaTilde <= 0); "
                        + "see Almgren-Chriss (2000) footnote");
    }

    /** Total horizon {@code T = N * tau}. */
    public double horizon() {
        return nIntervals * tau;
    }

    /** Adjusted temporary impact {@code etaTilde = eta*(1 - gammaP*tau/(2*eta))}. */
    public double etaTilde() {
        return eta * (1.0 - gammaP * tau / (2.0 * eta));
    }

    /**
     * Copy of these parameters with a different risk aversion.
     *
     * @param newLambdaRa the new risk aversion, {@code >= 0}
     * @return a new validated parameter set
     */
    public ACParams withLambda(double newLambdaRa) {
        return new ACParams(totalShares, nIntervals, tau, sigma, eta, gammaP, newLambdaRa);
    }
}
