package io.github.ashjha0.etrade.bench;

import io.github.ashjha0.etrade.ACParams;
import io.github.ashjha0.etrade.ASParams;
import io.github.ashjha0.etrade.ASSimResult;
import io.github.ashjha0.etrade.AlmgrenChriss;
import io.github.ashjha0.etrade.AvellanedaStoikov;
import io.github.ashjha0.etrade.Impact;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

/**
 * Hot-path benchmarks (JMH). Build and run with
 * {@code mvn -Pbench package && java -jar target/benchmarks.jar}.
 * Parameter sets match the Python, C++ and Rust benchmarks; measured numbers
 * live in docs/BENCHMARKS.md.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
public class HotPathBenchmarks {

    static final ASParams AS = new ASParams(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, 20, 0.5);

    /** Almgren-Chriss trajectory: liquid preset over a 5-day horizon. */
    @State(Scope.Benchmark)
    public static class AcState {
        @Param({"1000", "100000"})
        public int n;
        ACParams params;

        @Setup
        public void setup() {
            params = new ACParams(1e6, n, 5.0 / n, 0.95, 2.5e-6, 2.5e-7, 2e-6);
        }
    }

    /** Power-law schedule: one trading day in n intervals, alpha = 0.5. */
    @State(Scope.Benchmark)
    public static class ImpactState {
        @Param({"13", "390"})
        public int n;
        Impact.Params params;

        @Setup
        public void setup() {
            params = new Impact.Params(1e6, n, 1.0 / n, 0.95, 0.05, 0.5, 2.5e-7, 2e-6, 0.0, null);
        }
    }

    /** Market state for the quote function, read from fields so it is not constant-folded. */
    @State(Scope.Benchmark)
    public static class QuoteState {
        public double s = 100.0;
        public double q = 3.0;
        public double t = 0.5;
    }

    @Benchmark
    public double[] acTrajectory(AcState st) {
        return AlmgrenChriss.trajectory(st.params);
    }

    @Benchmark
    public Impact.ScheduleResult impactOptimalSchedule(ImpactState st) {
        return Impact.optimalSchedule(st.params);
    }

    @Benchmark
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    public ASSimResult asSimulate2000Paths() {
        return AvellanedaStoikov.simulate(AS, 2000, 42L);
    }

    @Benchmark
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public AvellanedaStoikov.Quotes asQuotes(QuoteState st) {
        return AvellanedaStoikov.quotes(AS, st.s, st.q, st.t);
    }
}
