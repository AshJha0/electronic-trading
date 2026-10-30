//! Hot-path benchmarks (criterion). Run with `cargo bench --bench hot_paths`.
//!
//! Parameter sets match the Python, C++ and Java benchmarks; see
//! `docs/BENCHMARKS.md` for measured numbers.
#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use etrade::impact::{optimal_schedule, ImpactParams};
use etrade::{ac_trajectory, quotes, simulate, ACParams, ASParams, SimConfig};

fn ac_params(n: usize) -> ACParams {
    // Liquid preset (data/params_liquid.json) spread over a 5-day horizon.
    ACParams::new(1e6, n, 5.0 / n as f64, 0.95, 2.5e-6, 2.5e-7, 2e-6).unwrap()
}

fn impact_params(n: usize) -> ImpactParams {
    // One trading day split into n intervals, square-root temporary impact.
    ImpactParams::new(
        1e6,
        n,
        1.0 / n as f64,
        0.95,
        0.05,
        0.5,
        2.5e-7,
        2e-6,
        0.0,
        None,
    )
    .unwrap()
}

fn as_params() -> ASParams {
    ASParams::new(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, 20, 0.5).unwrap()
}

fn bench_ac(c: &mut Criterion) {
    let mut g = c.benchmark_group("ac_trajectory");
    for n in [1_000usize, 100_000] {
        let p = ac_params(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &p, |b, p| {
            b.iter(|| ac_trajectory(black_box(p)))
        });
    }
    g.finish();
}

fn bench_impact(c: &mut Criterion) {
    let mut g = c.benchmark_group("impact_optimal_schedule");
    for n in [13usize, 390] {
        let p = impact_params(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &p, |b, p| {
            b.iter(|| optimal_schedule(black_box(p)).unwrap())
        });
    }
    g.finish();
}

fn bench_as(c: &mut Criterion) {
    let p = as_params();
    let cfg = SimConfig::default();
    let mut g = c.benchmark_group("as");
    g.sample_size(10);
    g.bench_function("simulate_2000_paths", |b| {
        b.iter(|| simulate(black_box(&p), 2000, 42, &cfg).unwrap())
    });
    g.finish();
    c.bench_function("as/quotes", |b| {
        b.iter(|| {
            quotes(
                black_box(&p),
                black_box(100.0),
                black_box(3.0),
                black_box(0.5),
            )
            .unwrap()
        })
    });
}

criterion_group!(benches, bench_ac, bench_impact, bench_as);
criterion_main!(benches);
