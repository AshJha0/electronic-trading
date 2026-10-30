package io.github.ashjha0.etrade;

import java.util.SplittableRandom;

/**
 * Seeded uniform and standard-normal draws with a pinned algorithm.
 *
 * <p>Uniforms are {@link SplittableRandom#nextDouble()} (its algorithm is
 * specified in the JDK javadoc). Normals use the Marsaglia polar method on
 * those uniforms, written out here rather than delegating to
 * {@code nextGaussian()}, whose algorithm the JDK leaves unspecified:
 * draw {@code u, v = 2 nextDouble() - 1} until {@code 0 < s = u^2 + v^2 < 1},
 * then {@code f = StrictMath.sqrt(-2 StrictMath.log(s) / s)} gives the two
 * independent normals {@code u f} (returned) and {@code v f} (cached for the
 * next call). With {@link StrictMath} the sequence is bit-identical on every
 * JVM and platform for a given seed and call order.
 */
final class NormalRng {

    private final SplittableRandom rng;
    private double spare;
    private boolean hasSpare;

    NormalRng(long seed) {
        this.rng = new SplittableRandom(seed);
    }

    /** Uniform draw in {@code [0, 1)}. */
    double nextDouble() {
        return rng.nextDouble();
    }

    /** Standard normal draw (Marsaglia polar method, see the class javadoc). */
    double nextNormal() {
        if (hasSpare) {
            hasSpare = false;
            return spare;
        }
        double u;
        double v;
        double s;
        do {
            u = 2.0 * rng.nextDouble() - 1.0;
            v = 2.0 * rng.nextDouble() - 1.0;
            s = u * u + v * v;
        } while (s >= 1.0 || s == 0.0);
        double f = StrictMath.sqrt(-2.0 * StrictMath.log(s) / s);
        spare = v * f;
        hasSpare = true;
        return u * f;
    }
}
