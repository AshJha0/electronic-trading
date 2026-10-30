---
title: Execution and market-making research
hide:
  - navigation
---

# electronic-trading

A research platform for executing large orders at low cost and making
markets profitably. It starts from the canonical models (Almgren-Chriss
optimal execution, Avellaneda-Stoikov market making, implementation-shortfall
analytics) and builds towards what decides money in practice: realistic
costs, informed flow, fees and ticks, fair-value signals and post-trade
measurement. Implemented in Python, C++, Rust and Java from one pinned
contract, cross-checked by shared golden values in CI on Linux, Windows and
macOS.

[Learn (guide)](learn.md){ .md-button .md-button--primary }
[Cookbook](cookbook.md){ .md-button }
[Architecture](architecture.md){ .md-button }
[API contract](api-spec.md){ .md-button }
[Diagrams](diagrams.md){ .md-button }
[GitHub](https://github.com/AshJha0/electronic-trading){ .md-button }

## Measured, not claimed

[![CI status](https://github.com/AshJha0/electronic-trading/actions/workflows/ci.yml/badge.svg)](https://github.com/AshJha0/electronic-trading/actions/workflows/ci.yml)
Every number below is re-checked by CI on Linux, Windows and macOS on every
push and pull request.

<div class="et-kv" markdown>

| | |
|---|---|
| Cross-language parity | Python 487, C++ 153, Rust 200, Java 142 tests, 982 in total; every suite loads the same 95 golden cases: 19 model, 15 market, 8 impact, 11 TCA, 9 calibration, 8 execution, 14 market-making and 11 placement |
| Golden suite | exact cases (AC `kappa` / `x_j` / `n_1` / E / Var on two parameter sets, TWAP/POV costs, A-S closed forms, both IS decompositions, the impact fit) at \|actual − expected\| ≤ tol·max(1, \|expected\|), tol between 1e-10 and 1e-12; market, TCA and calibration cases at 1e-12, power-law impact and execution cases at 1e-9 |
| Statistical cases | Python 20,000-path Monte Carlo references stored with their standard errors; each port runs M ≥ 2000 paths on its own RNG and must satisfy \|stat − ref\| ≤ 4·sqrt(se_ref² + se_own²) |
| Literature check | the paper's hyperbolic closed forms for E[IS] and Var[IS] equal the discrete sums to 1e-12 in every suite (measured rel. diff 2e-16 on the liquid set) |
| Numerics | `kappa` via `log1p(x + sqrt(x(2+x)))/tau` (cancellation-free at any `tau`; naive `acosh` loses 8 digits at millisecond `tau`); trajectory via an `exp/expm1` ratio, finite for any `kappa·T` (the naive `sinh` ratio is NaN above `kappa·T` ≈ 710); TWAP branch at `kappa·T < 1e-6` |
| Pinned bounds | `MAX_AC_INTERVALS` / `MAX_POV_INTERVALS` / `MAX_AS_STEPS` / `MAX_PATHS` = 10 million, checked on the double ratio before any cast or allocation; participation above 1 and non-liquidating schedules are input errors |
| Determinism | same seed gives identical results within a language (NumPy PCG64, C++ `mt19937_64`, Rust `ChaCha12Rng` pinned, Java `SplittableRandom`); cross-language agreement is statistical, never bitwise |
| Demo agreement | A-S mean fills 85.6 (Python) vs 85.2 / 85.5 / 85.3 (C++ / Rust / Java) on 2,000-path samples; MC E[IS] 1,075,336 vs closed form 1,078,215 with se 3,177 |

</div>

## Market presets in basis points

Printed by every demo from `data/presets.json` (sell orders over one day;
A-S spread at the open and at the close of the session).

| preset | order | E[IS] | sd | half-spread | A-S spread open / close |
|---|---|---|---|---|---|
| US large cap (\$100, 1 bp spread) | 5% ADV | 5.4 bps | 75.5 bps | 0.5 bps | 3.6 / 1.3 bps |
| US small cap (\$20, 25 bp spread) | 10% ADV | 265.2 bps | 178.9 bps | 12.5 bps | 53.8 / 28.8 bps |
| US ETF (\$500, 0.2 bp spread) | 1% ADV | 0.2 bps | 50.6 bps | 0.1 bps | 0.5 / 0.3 bps |
| EUR/USD (0.45 bp spread) | 5% of volume | 2.4 bps | 26.2 bps | 0.2 bps | 0.8 / 0.5 bps |
| Almgren-Chriss (2000) example | 20% ADV | 215.6 bps | 89.9 bps | 12.5 bps | |
| AAPL, fitted to ITCH 2019-12-30 (\$289.72, 0.91 bp spread) | 5% of NASDAQ volume | 4.9 bps | 58.2 bps | 0.5 bps | 1.2 / 0.5 bps |
| MSFT, fitted to ITCH 2019-12-30 (\$157.67, 0.78 bp spread) | 5% of NASDAQ volume | 4.2 bps | 37.4 bps | 0.4 bps | 1.9 / 0.5 bps |
| QQQ, fitted to ITCH 2019-12-30 (\$212.16, 0.55 bp spread) | 5% of NASDAQ volume | 2.9 bps | 29.0 bps | 0.3 bps | 0.8 / 0.3 bps |

The first five presets are illustrative. The last three take price,
spread, volatility, volume and fill intensity from one NASDAQ ITCH day,
with the caveats of [`API_SPEC` Section 11.6](api-spec.md) (one day, one
venue, the week after Christmas). This table uses the
Almgren-Chriss linear impact calibration (`API_SPEC` Section 7); the
pre-trade report below uses square-root impact.

## Pre-trade report (square-root impact)

US large cap, sell 5% of ADV in one day, U-shaped volume curve (`API_SPEC`
Section 8). The closed-form square-root-law estimate is 33.54 bps.

| strategy | lambda | impact | sd | total (with half spread) |
|---|---|---|---|---|
| TWAP | 0 | 34.96 bps | 81.59 bps | 35.46 bps |
| VWAP | 0 | 33.77 bps | 79.91 bps | 34.27 bps |
| optimal | 0 | 33.77 bps | 79.91 bps | 34.27 bps |
| optimal | 1e-7 | 33.83 bps | 76.51 bps | 34.33 bps |
| optimal | 1e-6 | 36.95 bps | 56.60 bps | 37.45 bps |
| optimal | 1e-5 | 58.26 bps | 21.56 bps | 58.76 bps |

With a 50 bps/day adverse drift behind the order, the drift-aware schedule
costs 52.13 bps against 57.31 bps for VWAP. The schedule comes from a pinned
Newton method with a KKT certificate; at `alpha = 1` it reproduces the
Almgren-Chriss trajectory (holdings within 1e-10 of `X`, costs within 1e-12
relative), and the Python suite cross-checks it against SciPy.

## Market making with frictions

US large-cap A-S preset, one-share quotes, 5-second steps, one day
(`API_SPEC` Section 9). Dollars per day, 20,000-path Python sample printed
by the demo; the other ports print their own 5,000-path samples. Every path
satisfies PnL = spread + fees + inventory − liquidation exactly.

| case | PnL (SE) | spread | fees | inventory | markout | fills |
|---|---|---|---|---|---|---|
| frictionless | 7.41 (0.01) | 7.44 | 0.00 | 0.01 | 0.01 | 1,076 |
| 1-cent tick, 0.2-cent rebate | 7.33 (0.01) | 6.20 | 1.16 | −0.01 | 0.00 | 580 |
| 1-cent tick, 0.2-cent fee | 5.00 (0.01) | 6.20 | −1.16 | −0.00 | −0.01 | 580 |
| tick, rebate, 15% informed flow | 4.01 (0.01) | 10.25 | 1.86 | −8.07 | −8.06 | 931 |
| as above, 60%-accurate signal | 5.00 (0.01) | 9.81 | 1.97 | −6.75 | −6.74 | 985 |
| tick, rebate, quotes 5 s stale | −23.73 (0.02) | −28.33 | 4.65 | 0.00 | 0.00 | 2,327 |
| GLFT quotes, tick, rebate | 11.00 (0.04) | 9.27 | 1.76 | −0.00 | −0.00 | 880 |

Informed flow raises spread capture and lowers PnL; the markout shows where
the money went ([Learn, Sections 6.7-6.12](learn.md)).

## What's inside

<div class="grid cards" markdown>

-   **Almgren-Chriss execution**

    ---

    Closed-form optimal sell trajectory (`kappa`, holdings `x_j`, trades
    `n_j`) for linear temporary and permanent impact under an arithmetic
    random walk, with exact discrete `E[IS]` / `Var[IS]` sums as the single
    code path for optimal, TWAP, POV and hand-built schedules. `x_N = 0` is
    forced exactly; the `lambda_ra = 0` TWAP limit is an explicit branch.

-   **Frontier and baselines**

    ---

    Efficient-frontier sweep over a risk-aversion grid with half-life
    (`1/kappa`, an e-folding time) reporting; TWAP, POV and
    instantaneous-liquidation baselines on the same cost model; a Monte Carlo
    IS simulator whose mean and variance must land within 4 SE of the closed
    forms. Sell programs only (`X ≥ 0`); the paper's fixed cost term is
    omitted and documented.

-   **Square-root impact and pre-trade**

    ---

    Power-law temporary impact (`alpha = 1/2` by default) with per-interval
    liquidity weights, an illustrative U-shaped volume curve, VWAP, an
    adverse drift for alpha-aware schedules, and a pre-trade report next to
    the closed-form square-root-law estimate. Parameters can be stated in
    market units (price, spread, ADV, daily volatility, tick) with results in
    basis points.

-   **Avellaneda-Stoikov market making**

    ---

    Reservation price, optimal spread `(2/γ)·ln(1 + γ/k)` and quotes;
    exponential arrival intensity `A·exp(−k·δ)` with thinned-Poisson fills; a
    simulation loop with pinned draw order, hard inventory cap, terminal
    liquidation penalty and an optional volatility stress after `T/2`. A
    fixed-spread comparison agent shares the same RNG mechanics.

-   **Execution analytics**

    ---

    Perold (1988) implementation-shortfall decomposition into delay, trading
    and opportunity components for buy and sell orders (with the identity
    and price-shift invariance tested), participation rate, and a native
    two-variable OLS used to fit cost per share against participation on
    pinned RNG-free synthetic fills.

-   **Post-trade TCA**

    ---

    Slippage against arrival, the day's VWAP, the interval VWAP of the bins
    the order traded in, and the close; markouts per fill and per share;
    per-bin participation; and a Welch t-test for comparing algorithms,
    with the incomplete beta and Student t CDF pinned so every port gives
    the same p-value.

-   **Calibration from recorded data**

    ---

    Volume curve, realised-volatility curve, spread statistics and a
    weighted fill-intensity fit on plain arrays; three presets and a volume
    curve fitted to the NASDAQ ITCH sample day 2019-12-30, with only
    aggregates committed.

-   **Execution extensions**

    ---

    Transient impact with exponential resilience (Obizhaeva-Wang), its
    optimum from one tridiagonal solve; adaptive re-planning against the
    static schedule under a drift signal or partial fills, with the zero
    saving of the time-consistent case asserted; internalisation of
    opposite client orders.

-   **Market making with frictions**

    ---

    The A-S quoting logic with a price grid, maker fees or rebates, stale
    quotes, informed traders and a fair-value signal of given accuracy,
    switched on one at a time; GLFT quotes beside A-S; every path's PnL split
    exactly into spread, fees, inventory and liquidation, with one-step
    markouts.

-   **Limit-order placement**

    ---

    Fill probability of a resting order from its place in the queue
    (uniformisation, no cancellation-prone closed form); the choice between
    posting and crossing for a child order; liquidation with limit and
    market orders by backward induction, checked against all-market and
    post-then-sweep on shared paths.

-   **Golden-value testing**

    ---

    Eight golden files with 95 cases (19 model, 15 market, 8 impact, 11 TCA,
    9 calibration, 8 execution, 14 market-making, 11 placement), loaded by
    all four suites: deterministic
    cases under a relative-or-absolute tolerance of 1e-9 to 1e-12, Monte
    Carlo cases as reference values plus standard errors. Sample values
    differ across languages by design; distributions must agree.

-   **Property, numeric and validation tests**

    ---

    Per language: trajectory monotonicity, inventory cap respected at every
    step, frontier monotonicity, continuity across the TWAP branch, overflow
    safety at `kappa·T > 2500`, `ac_kappa_small_tau` at millisecond `tau`,
    and every rule in `API_SPEC` Section 6: NaN/inf, empty grids,
    non-liquidating schedules, participation above 1, sizes beyond the
    pinned bounds.

-   **Error contract**

    ---

    Identical rules in all four languages: Python `ValueError`, C++
    `std::invalid_argument` (`std::domain_error` for the AC validity
    condition), Rust `Err(EtradeError)` with no panics, Java
    `IllegalArgumentException`; solver failures are reported separately
    (`std::runtime_error`, `SolverFailed`, `IllegalStateException`). No
    function returns NaN or infinity from valid finite inputs.

-   **Units and what is out of scope**

    ---

    `tau`, `dt`, `T` in an abstract time unit (bundled sets and the market
    layer use one trading day); `sigma` is arithmetic; prices may go
    negative; AC shares are fractional, A-S fills are one unit. Ticks,
    fees, latency and informed flow are stylised switches in the
    market-making simulator, not a matching engine. Not covered yet: venues,
    lots, an order book, hedging and risk limits, calendars, and any fit of
    the impact parameters `eta`, `alpha`, `gamma_p` and `Y` to real parent
    orders.

-   **Reference papers**

    ---

    Almgren and Chriss (2000), *Journal of Risk*; Avellaneda and Stoikov
    (2008), *Quantitative Finance*; Perold (1988) and Wagner and Edwards
    (1993) for the shortfall decomposition and its naming; Guéant, Lehalle
    and Fernandez-Tapia (2013) for the inventory-capped variant behind
    `q_max`; Almgren et al. (2005), Tóth et al. (2011) and Bouchaud et al.
    (2018) for the square-root law; Obizhaeva and Wang (2013) for transient
    impact; Almgren and Lorenz (2007) for adaptive execution; Glosten and
    Milgrom (1985) for adverse selection; Cont, Stoikov and Talreja (2010)
    and Cartea and Jaimungal (2015) for queues and liquidation; Andersen et
    al. (2001) for realised variance; Welch (1947) for the A/B test; MiFID II Art. 17/27, Reg NMS Rules
    605/606 and SEC Rule 15c3-5 in the guide's regulatory section.

</div>

## Part of a wider portfolio

<div class="grid cards" markdown>

-   **[quant-portfolio](https://ashjha0.github.io/quant-portfolio/)**

    ---

    The end-to-end quantitative finance portfolio this project sits
    alongside: 10 project areas built twice each (equity + FX) in Python
    with C++ and Rust performance twins, 31 sub-projects with golden-vector
    cross-language validation, a 481-question learning guide and a verified
    cookbook. [GitHub](https://github.com/AshJha0/quant-portfolio)

-   **[intraday-alpha-platform](https://ashjha0.github.io/intraday-alpha-platform/)**

    ---

    A polyglot (Python/C++/Rust/Java) research-to-production intraday
    trading platform: market data, order books, a 205-feature factory, 24
    alphas, validation, portfolio, fail-closed risk, execution and TCA, with
    an adaptive drift/lifecycle layer and 1,160 golden-parity tests.
    [GitHub](https://github.com/AshJha0/intraday-alpha-platform)

-   **[quantfinlib](https://github.com/AshJha0/Quant-Finance-Library)**

    ---

    A zero-dependency quantitative finance platform for the JVM: research,
    validation, execution and connectivity, from walk-forward backtesting to
    a measured sub-microsecond tick-to-order path, with 1,090+ tests and CI
    on every push.

</div>

## Quick start

```bash
git clone https://github.com/AshJha0/electronic-trading.git
cd electronic-trading

# Python (3.11+)
pip install -e "python[dev]"
python python/demo.py
pytest python                                   # 487 tests

# C++ (C++17, CMake >= 3.20; GCC, Clang or MSVC)
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build -C Release               # 153 tests

# Rust (stable, MSRV 1.78)
(cd rust && cargo test --release && cargo run --release --example demo)   # 200 tests

# Java (JDK 17+, Maven)
(cd java && mvn verify && mvn -q exec:java)     # 142 tests
```

electronic-trading is MIT-licensed and provided without warranty: a
validated reference implementation of published models, not investment
advice and not an order-management or risk system. See the
[README](https://github.com/AshJha0/electronic-trading#readme) and
[LICENSE](https://github.com/AshJha0/electronic-trading/blob/main/LICENSE).
