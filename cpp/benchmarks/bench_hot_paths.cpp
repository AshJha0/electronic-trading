// Hot-path benchmarks (Google Benchmark). Parameter sets match the Python,
// Rust and Java benchmarks; measured numbers live in docs/BENCHMARKS.md.
#include <benchmark/benchmark.h>

#include "etrade/almgren_chriss.hpp"
#include "etrade/avellaneda_stoikov.hpp"
#include "etrade/impact.hpp"

namespace {

// Liquid preset (data/params_liquid.json) spread over a 5-day horizon.
etrade::ACParams ac_params(int n) {
    return etrade::ACParams(1e6, n, 5.0 / n, 0.95, 2.5e-6, 2.5e-7, 2e-6);
}

// One trading day split into n intervals, square-root temporary impact.
etrade::ImpactParams impact_params(int n) {
    return etrade::ImpactParams(1e6, n, 1.0 / n, 0.95, 0.05, 0.5, 2.5e-7, 2e-6);
}

etrade::ASParams as_params() {
    return etrade::ASParams(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, 20, 0.5);
}

void BM_AcTrajectory(benchmark::State& state) {
    const auto p = ac_params(static_cast<int>(state.range(0)));
    for (auto _ : state) {
        auto x = etrade::ac_trajectory(p);
        benchmark::DoNotOptimize(x.data());
    }
}
BENCHMARK(BM_AcTrajectory)->Arg(1000)->Arg(100000)->Unit(benchmark::kMicrosecond);

void BM_ImpactOptimalSchedule(benchmark::State& state) {
    const auto p = impact_params(static_cast<int>(state.range(0)));
    for (auto _ : state) {
        auto r = etrade::optimal_schedule(p);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_ImpactOptimalSchedule)->Arg(13)->Arg(390)->Unit(benchmark::kMicrosecond);

void BM_AsSimulate2000Paths(benchmark::State& state) {
    const auto p = as_params();
    for (auto _ : state) {
        auto r = etrade::simulate(p, 2000, 42);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_AsSimulate2000Paths)->Unit(benchmark::kMillisecond);

void BM_AsQuotes(benchmark::State& state) {
    const auto p = as_params();
    double s = 100.0;
    double q = 3.0;
    double t = 0.5;
    for (auto _ : state) {
        benchmark::DoNotOptimize(s);
        benchmark::DoNotOptimize(q);
        benchmark::DoNotOptimize(t);
        auto bq = etrade::quotes(p, s, q, t);
        benchmark::DoNotOptimize(bq);
    }
}
BENCHMARK(BM_AsQuotes);

}  // namespace

BENCHMARK_MAIN();
