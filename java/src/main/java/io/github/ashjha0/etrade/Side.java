package io.github.ashjha0.etrade;

/** Direction of a parent order. */
public enum Side {
    /** Buy order: costs are paid when prices rise. */
    BUY,
    /** Sell order: costs are paid when prices fall. */
    SELL;

    /** +1 for {@link #SELL}, -1 for {@link #BUY} (the sign used in shortfall formulas). */
    double sellSign() {
        return this == SELL ? 1.0 : -1.0;
    }

    /**
     * Parses {@code "buy"} or {@code "sell"} (the spec's wire names).
     *
     * @param side the name
     * @return the side
     * @throws IllegalArgumentException for any other value, including null
     */
    public static Side parse(String side) {
        if ("buy".equals(side)) {
            return BUY;
        }
        if ("sell".equals(side)) {
            return SELL;
        }
        throw new IllegalArgumentException("side must be 'buy' or 'sell', got " + side);
    }
}
