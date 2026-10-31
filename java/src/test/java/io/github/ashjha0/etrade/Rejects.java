package io.github.ashjha0.etrade;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.function.Executable;
import org.junit.jupiter.params.provider.Arguments;

/** Helpers for parameterized invalid-input tests. */
final class Rejects {

    private Rejects() {
    }

    /**
     * One invalid-input case.
     *
     * @param fragment text the exception message must contain (the offending argument)
     * @param call     the rejected call
     * @return the {@code (fragment, call)} arguments
     */
    static Arguments bad(String fragment, Executable call) {
        return Arguments.of(fragment, call);
    }

    /**
     * Asserts {@code call} throws {@link IllegalArgumentException} whose message
     * contains {@code fragment}.
     *
     * @param fragment expected message fragment
     * @param call     the call
     */
    static void assertRejects(String fragment, Executable call) {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, call);
        String msg = e.getMessage();
        assertTrue(msg != null && msg.contains(fragment),
                () -> "message '" + msg + "' should mention '" + fragment + "'");
    }
}
