package io.github.ashjha0.etrade;

/** Package-private input-validation helpers (throw {@link IllegalArgumentException}). */
final class Validation {

    private Validation() {
    }

    /** Throws {@link IllegalArgumentException} with {@code msg} when {@code cond} is false. */
    static void require(boolean cond, String msg) {
        if (!cond) {
            throw new IllegalArgumentException(msg);
        }
    }

    /** Requires {@code v} to be non-null; the message names the argument. */
    static void requireNonNull(Object v, String name) {
        require(v != null, name + " must not be null");
    }

    /** Requires {@code v} to be finite (not NaN/inf). */
    static void requireFinite(double v, String name) {
        require(Double.isFinite(v), name + " must be finite");
    }
}
