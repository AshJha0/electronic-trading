# Benchmarks

Hot-path timings for the four ports. The benchmarks are separate targets;
neither CI nor the default builds run them.

## What is measured

The same inputs in every language:

| case | inputs |
|---|---|
| Almgren-Chriss trajectory (`ac_trajectory`, Section 1.4) | X = 1e6, N = 1,000 and 100,000, tau = 5/N, sigma = 0.95, eta = 2.5e-6, gamma_p = 2.5e-7, lambda = 2e-6 (the liquid preset over five days) |
| Power-law optimal schedule (`optimal_schedule`, Section 8.2) | X = 1e6, N = 13 and 390 over one day (tau = 1/N), sigma = 0.95, eta = 0.05, alpha = 0.5, gamma_p = 2.5e-7, lambda = 2e-6, uniform volume |
| Avellaneda-Stoikov simulation (`simulate`, Section 2.2) | 2,000 paths, seed 42, A-S agent; the liquid preset: s0 = 100, sigma = 2, T = 1, dt = 0.005 (200 steps), gamma = 0.1, k = 1.5, A = 140, q_max = 20, eta_liq = 0.5 |
| Avellaneda-Stoikov quotes (`quotes`, Section 2.1) | the same parameters, s = 100, q = 3, t = 0.5 |

## How to run

| port | command | harness |
|---|---|---|
| Python | `pip install -e "python[bench]"`, then in `python/`: `pytest benchmarks --benchmark-only` | pytest-benchmark 5.x |
| C++ | `cmake -S cpp -B build-bench -DETRADE_BUILD_BENCHMARKS=ON -DETRADE_BUILD_TESTS=OFF`, `cmake --build build-bench --config Release --target etrade_bench`, then run `etrade_bench --benchmark_repetitions=5` | Google Benchmark 1.9.1 (fetched, SHA-256 pinned) |
| Rust | in `rust/`: `cargo bench --bench hot_paths` | criterion 0.5 |
| Java | in `java/`: `mvn -Pbench package`, then `java -jar target/benchmarks.jar` | JMH 1.37, 1 fork, 3 x 1 s warm-up, 5 x 1 s measurement |

## Results

Measured on 2026-10-30 on **one Windows laptop**: Intel Core 5 120U (10
cores / 12 threads, 2.5 GHz base, 12 MB L3), 16 GB RAM, Windows 11 Home
(10.0.26200), on mains power with a normal desktop session running.
Toolchains: Python 3.12.10 with NumPy 2.4.6; MSVC 19.51 (Visual Studio 2026,
Release, `/O2`); rustc 1.99.1 (release profile); OpenJDK 17.0.20.1 (HotSpot,
default flags).

| case | Python | C++ | Rust | Java |
|---|---:|---:|---:|---:|
| AC trajectory, N = 1,000 | 15.0 µs | 8.7 µs | 33.2 µs | 153 µs |
| AC trajectory, N = 100,000 | 2.47 ms | 2.88 ms | 3.28 ms | 15.7 ms |
| power-law schedule, N = 13 | 48.4 µs | 2.67 µs | 11.0 µs | 6.06 µs |
| power-law schedule, N = 390 | 1.61 ms | 210 µs | 255 µs | 178 µs |
| A-S simulation, 2,000 paths | 27.6 ms | 68.1 ms | 22.9 ms | 77.3 ms |
| A-S quotes, one call | 0.60 µs | 34 ns | 36 ns | 55 ns |

Statistic per column: Python, the pytest-benchmark median; C++, the median
of 5 repetitions; Rust, criterion's point estimate (the middle of its
confidence interval); Java, the JMH average.

## Reading the numbers

- **One machine, one run each.** These figures come from a single Windows
  laptop with frequency scaling and background load, not an isolated
  benchmark host. Run-to-run spread was large in places: the C++ N = 1,000
  trajectory had a 47% coefficient of variation across its 5 repetitions
  (mean 11.1 µs, median 8.7 µs) and the N = 390 schedule 37%; the Java
  N = 1,000 trajectory had a JMH error of ±43 µs. Treat differences under
  about 2x as noise and compare ports only on the same machine.
- **The work differs where the spec allows it.** The Python reference
  vectorises with NumPy: the A-S simulation advances all 2,000 paths as
  arrays per step, so it is faster than the scalar C++ and Java loops even
  though each operation goes through the interpreter. The tiny quote call
  shows the interpreter cost directly (about 17x the compiled ports).
- **The Java trajectory is the outlier** at about 5x the other ports for
  N = 100,000. This was not profiled for this note; it is a candidate for
  a follow-up rather than a conclusion about the JVM.
- None of these paths is latency-critical in the library's intended use
  (research and pre-trade analysis); the numbers are a regression baseline
  and a rough check on complexity. Going from N = 1,000 to 100,000 (100x)
  multiplied the trajectory time by 99-331x across ports, and going from
  N = 13 to 390 (30x) multiplied the schedule time by 23-79x: close to
  linear in most ports, with the C++ ratios inflated by the noisy small-N
  and N = 390 measurements noted above.
