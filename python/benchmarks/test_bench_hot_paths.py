"""Hot-path benchmarks (pytest-benchmark). Not collected by the default test run.

Run from ``python/``::

    pip install -e ".[bench]"
    pytest benchmarks --benchmark-only

The parameter sets match the C++, Rust and Java benchmarks so the numbers in
``docs/BENCHMARKS.md`` compare like with like.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pytest_benchmark", reason="benchmarks need pytest-benchmark")

import etrade as et
from etrade.impact import ImpactParams, optimal_schedule


def ac_params(n: int) -> et.ACParams:
    # Liquid preset (data/params_liquid.json) spread over a 5-day horizon.
    return et.ACParams(
        total_shares=1e6,
        n_intervals=n,
        tau=5.0 / n,
        sigma=0.95,
        eta=2.5e-6,
        gamma_p=2.5e-7,
        lambda_ra=2e-6,
    )


def impact_params(n: int) -> ImpactParams:
    # One trading day split into n intervals, square-root temporary impact.
    return ImpactParams(
        total_shares=1e6,
        n_intervals=n,
        tau=1.0 / n,
        sigma=0.95,
        eta=0.05,
        alpha=0.5,
        gamma_p=2.5e-7,
        lambda_ra=2e-6,
    )


AS = et.ASParams(
    s0=100.0,
    sigma=2.0,
    horizon=1.0,
    dt=0.005,
    gamma=0.1,
    k=1.5,
    a_rate=140.0,
    q_max=20,
    eta_liq=0.5,
)


@pytest.mark.parametrize("n", [1_000, 100_000])
def test_ac_trajectory(benchmark, n: int) -> None:
    p = ac_params(n)
    x = benchmark(et.ac_trajectory, p)
    assert len(x) == n + 1


@pytest.mark.parametrize("n", [13, 390])
def test_impact_optimal_schedule(benchmark, n: int) -> None:
    p = impact_params(n)
    r = benchmark(optimal_schedule, p)
    assert r.horizon >= 1


def test_as_simulate_2000_paths(benchmark) -> None:
    r = benchmark.pedantic(et.simulate, args=(AS, 2000, 42), rounds=10, iterations=1)
    assert r is not None


def test_as_quotes(benchmark) -> None:
    bid, ask = benchmark(et.quotes, AS, 100.0, 3.0, 0.5)
    assert bid < ask
