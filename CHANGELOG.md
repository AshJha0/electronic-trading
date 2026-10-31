# Changelog

All notable changes to this project. Dates are UTC.

## Unreleased

Nothing yet.

## 2.0.1 — 2026-10-31

Maintenance release. No API or numerical changes; every golden case and
seeded output is unchanged.

- Java test dependencies: Jackson `jackson-databind` 2.18.2 to 2.22.4 and
  JUnit 5.13.4 to 6.1.3 (the JUnit BOM). Test code only; the library jar has
  no runtime dependencies.
- Java tests: `TestData` reads JSON objects with `JsonNode.properties()`
  because `fields()` is deprecated in Jackson 2.22 and the build treats
  warnings as errors.

## 2.0.0 — 2026-10-31

Version 2 turns the reference library into an execution and market-making
research platform. Everything is implemented in Python, C++26, Rust and
Java 25 from one pinned contract (`API_SPEC.md`, Sections 0-14) and held
together by 120 shared golden cases checked in CI on Linux, Windows and
macOS (1,389 tests).

### Highlights

- **Market units** (Section 7): parameters from price, tick, spread, daily
  volatility and ADV; results in basis points; presets for a US large cap,
  small cap, ETF and EUR/USD, plus three presets fitted to a recorded NASDAQ
  ITCH day.
- **Execution a desk would recognise** (Sections 8, 12, 13): square-root and
  power-law impact with a certified Newton solver, intraday volume curves,
  alpha-aware (urgency) schedules, a pre-trade report, transient impact with
  adaptive re-planning, internalisation, and limit-order placement.
- **Market making that can lose money** (Sections 9, 14): informed flow, tick
  grid, fees and rebates, quote latency, a fair-value signal, GLFT quotes,
  exact per-path PnL attribution and markouts; hedging in a correlated
  instrument and risk controls (loss limit, position limit, quote throttle,
  kill switch).
- **Measurement** (Sections 10, 11): post-trade TCA (benchmarks, markouts,
  Welch A/B test) and calibration estimators for volume, volatility, spread
  and fill intensity.
- **Engineering**: CI on three operating systems, installable packages,
  coverage, CodeQL, Dependabot, benchmarks, property-based tests in all four
  languages, whole-trajectory goldens, and a MkDocs documentation site.

### Breaking changes

- Java: package renamed to `io.github.ashjha0.etrade`; Java 25 is the floor;
  Maven build; JUnit 5; `StrictMath` and a pinned normal generator change
  seeded Java simulation samples (statistical goldens are unaffected);
  `String`-typed `side` and `stat` overloads deprecated in favour of enums;
  `Demo` is no longer in the library jar.
- C++: C++26 and CMake 3.25 are required; validated parameter members are
  `const`; `se(std::string)` and `kDefaultFixedSpread` are deprecated in
  favour of `se(ASStat)` and `std::optional<double>`.
- Rust: result vectors are private behind accessors; `EtradeError` is
  `#[non_exhaustive]` with a new `SolverFailed` variant; `serde_json` is a
  dev-dependency only and the demo is an example.
- Python: `etrade` is an installable package (`pip install -e python`); no
  more `PYTHONPATH=src`.
- Behaviour: degenerate inputs that used to return NaN or a malformed path
  now return the defined limit or an input error (POV clip overflow,
  infinite urgency, overflowing horizon, huge orders); see API_SPEC 1.4,
  1.6 and 6.

### Dependency updates in this release

- GitHub Actions: checkout v7, setup-python v7, setup-java v6,
  upload-pages-artifact v5, deploy-pages v5. The `dtolnay/rust-toolchain`
  pin stays at the MSRV 1.78 (Dependabot misread it as an action version).
- Maven plugins: compiler 3.16.0, surefire 3.6.1, shade 3.6.2, exec 3.6.4,
  build-helper 3.6.2, JaCoCo 0.8.16. Rust: `serde_json` 1.0.154.
- Not taken: `rand`, `rand_chacha` and `rand_distr` major upgrades (they would
  change every seeded stream) and `criterion` 0.8 (needs a newer Rust than the
  MSRV); Dependabot now ignores these.

### Detailed changes

#### Market making with frictions (F3), placement (F5c), hedging and risk controls (F5b)

- Section 9: a market-making simulator with informed flow, tick grid, maker fee or rebate, quote latency and a fair-value signal; A-S, GLFT and fixed-spread agents; exact per-path attribution (spread + fees + inventory - liquidation) asserted on every path; one-step markouts; weighted-mid and order-flow-imbalance helpers. 14 golden cases.
- Section 13: queue-position fill probability, the passive-or-aggressive choice for a child order, and liquidation with limit and market orders. 11 golden cases.
- Section 14: hedging in a correlated instrument with hedge PnL and costs in the attribution, and risk controls (running loss limit, position limit with flattening, quote-rate throttle, kill switch) with trigger counts and tail quantiles. 18 golden cases.

#### Hardening, numerics and property tests

- Java: Java 25 floor (CI on 25 and 27), JUnit 5 with the Vintage engine, `StrictMath` and a pinned Marsaglia-polar normal generator, enum APIs, exposed per-path arrays, Jackson for goldens, `Demo` moved to `src/demo/java`.
- C++: C++26 (CMake 3.25, GCC 14 in CI), `const` members on validated parameters, default-initialised results, `[[nodiscard]]`/`noexcept`, one `detail/require.hpp`, enum `se()` and `std::optional` fixed spread, a locale-independent test JSON parser with a named member struct.
- Rust: one test per golden case with a typed schema and every key asserted, `#[non_exhaustive]` errors, private result fields, named-field spec structs, 16 doctests.
- Verification: numeric checks of every standard-error formula, draw-order guards (a wrong-order reference must diverge), whole-trajectory goldens (schema v2), total-work bounds, and property-based tests in all four languages (hypothesis, proptest, seeded loops), which found five extreme-value bugs fixed in this release.

#### Documentation

- Demos in all four languages print five more sections: post-trade TCA, calibration with the fitted ITCH presets, execution extensions (transient impact, adaptive vs static, internalisation), the market-making friction ledger (frictionless, tick with rebate or fee, informed flow, signal, latency, GLFT, with the spread / fees / inventory / markout attribution) and limit-order placement. Deterministic lines agree across the four languages; simulated lines are each language's own sample and say so.
- LEARN Sections 3.14-3.16 (queue fill probability, posting or crossing, liquidation with limit and market orders) and 6.7-6.12 (the friction ledger and how to read an attribution, informed flow and why the skew does not protect against it, tick and fee economics, latency and pick-offs, the value of a fair-value signal, GLFT and A-S); COOKBOOK recipes 24 and 25 in all four languages, every snippet run; ARCHITECTURE adds the `mm` and `placement` modules, their golden files (95 cases in eight files) and numerics notes 16-17.
- README and the landing pages: roadmap (F1-F4 done; F5: adaptive scheduling, internalisation and limit-order placement done, hedged market making and risk controls in progress), the friction ledger, feature-matrix rows, and test counts measured by running the suites (Python 487, C++ 153, Rust 200, Java 142; 982 in total). The site navigation adds the benchmark notes.
- COOKBOOK recipes 15-23: benchmark slippage, participation and markouts; Welch A/B comparison; calibration from arrays; loading a fitted preset; transient impact and its optimum; adaptive re-planning vs static; internalisation; an end-to-end workflow (fitted preset, pre-trade report, optimal schedule, simulated execution, TCA); and the common validation errors per language. Each recipe was run as a program in all four languages and every quoted number is the printed output.
- LEARN Sections 3.10-3.13 (calibration and what one day can tell you, transient impact, time consistency and when re-planning helps, internalisation economics) and 4.2-4.4 (benchmarks and interval VWAP, markouts, comparing algorithms); claims carry an evidence class; new knowledge-check questions Q11-Q17, a glossary (Section 15) and further reading.
- ARCHITECTURE: eight-module table, a module-dependency diagram, the data flow from the ITCH file to `data/presets.json`, an inventory of all six golden files, numerics notes for the KKT certificate, the transient-impact Thomas solve, the re-plan coefficients and the pinned t CDF, the CI matrix, and a guide to adding a module in all four languages.
- README and the landing pages: roadmap (F1, F2, F4 done; F3 in progress; F5 partly done), feature-matrix rows for TCA, calibration and execution, the three fitted presets, and test counts measured by running the suites (Python 371, C++ 127, Rust 151, Java 120; 769 in total). The site navigation adds the publishing notes.

#### Execution extensions (F5a)

- Transient impact in all four languages (API_SPEC Section 12): a discrete Obizhaeva-Wang propagator with exponential resilience `exp(-rho*t)`, expected cost of any schedule by the displacement recursion, and the exact static optimum from one tridiagonal solve (the Kac-Murdock-Szego inverse). The Obizhaeva-Wang shape (equal blocks at both ends, constant rate between, ratio `1 - exp(-rho*tau)`) and the reduction to Section 1 linear impact as `rho -> infinity` with `kappa = 2*eta/tau` are asserted.
- Adaptive re-planning: the Almgren-Chriss problem re-solved every interval from realised holdings and an observable AR(1) drift signal, with random partial fills; a seeded Monte Carlo compares static and adaptive cost (paired). Without a signal or fill risk the two coincide path by path (Almgren-Chriss is time consistent); with either, re-planning saves cost.
- Internalisation: opposite parent orders are crossed at the mid and only the residual is costed with the Section 8 model; per-order and total savings in bps. Crossing never increases total impact cost (asserted).
- Eight golden cases in `data/golden/execution_golden.json` (`data/generate_execution.py`), three of them statistical.
#### Calibration from recorded ITCH data (F4b)

- Calibration estimators in all four languages (API_SPEC Section 11): intraday volume curve, realised-variance volatility curve and daily volatility, quoted and effective spread statistics in ticks and bps, and the Avellaneda-Stoikov fill intensity `A exp(-k delta)`. The intensity is fitted by count-weighted least squares on the log rate at which market orders reach each distance, and reported in the Section 7.2 units (`touch_fills_per_day`, `halving_ticks`).
- Fitted presets `itch_aapl_20191230`, `itch_msft_20191230`, `itch_qqq_20191230` in `data/presets.json`, from the NASDAQ TotalView-ITCH sample day 2019-12-30. Every preset now carries a `source` field. `FITTED_VOLUME_CURVE_13` is added next to the illustrative `US_EQUITY_VOLUME_CURVE_13`.
- The ingestion tooling `data/calibration/fit_itch.py` (Python only) reads the sibling project's ingested books and writes only aggregates (`data/calibration/fitted_20191230.json`). Section 11.6 lists the caveats: one day, three names, one venue, a holiday week.
- Nine calibration golden cases in `data/golden/calibration_golden.json`. `data/golden/market_golden.json` was regenerated with six cases for the fitted presets (15 in total; the nine existing cases are unchanged).

#### Post-trade TCA (F4a)

- New `tca` module in all four languages (API_SPEC Section 10): slippage in bps against arrival, full-period market VWAP, interval VWAP (only the bins the order traded in) and close, signed as the Perold decomposition (positive = cost); per-fill markouts at chosen horizons (negative = adverse), averaged per fill and per share; participation per bin and overall.
- A/B comparison of two strategies' per-order costs: Welch t statistic, Welch-Satterthwaite degrees of freedom and a two-sided p-value. The Student-t tail uses one pinned algorithm everywhere (Lanczos log-gamma, regularized incomplete beta by the modified Lentz continued fraction); the Python suite checks it against SciPy.
- Eleven TCA golden cases in `data/golden/tca_golden.json`, generated by `data/generate_tca.py`.
#### Documentation accuracy pass and site

- Every number in README, LEARN, COOKBOOK, ARCHITECTURE and the landing page re-checked against the four demos and test suites; every COOKBOOK snippet run (Python) or compiled and run in a scratch harness (C++, Rust, Java). Corrections: two stale trajectory values in COOKBOOK recipe 1, deterministic values quoted to 15 significant digits (the last ulp differs by platform), the pre-trade table now lists every row the demos print, the scope and units notes no longer say square-root impact and volume curves are missing, ARCHITECTURE describes all five modules and the three golden files, and the roadmap marks F1 and F2 done.
- LEARN gains Section 3.8 (parameters in market units); the power-law section becomes 3.9, and its references (Almgren et al. 2005, Tóth et al. 2011, Bouchaud et al. 2018) are in the reading list.
- Documentation site built with MkDocs Material from the Markdown sources (`mkdocs.yml`), checked with `mkdocs build --strict` on every pull request and deployed to GitHub Pages from `main` by `.github/workflows/docs.yml`.

#### Execution a desk would recognise (F2)

- Power-law impact execution in all four languages (API_SPEC Section 8): temporary cost `eta_j * n^(1+alpha)` with per-interval liquidity weights, linear permanent impact and an optional adverse drift. `alpha = 1/2` is the square-root law; `alpha = 1` reproduces Almgren-Chriss exactly (asserted).
- Market calibration `eta = Y * sigma / ADV^alpha`: a one-day TWAP costs exactly the square-root-law figure.
- Pinned optimal-schedule solver: Newton with a tridiagonal Hessian and Armijo line search; when drift makes it optimal to finish early, horizon bisection plus a KKT certificate. Cross-checked against SciPy in the Python suite.
- Illustrative US-equity intraday volume curve, VWAP schedule and a pre-trade report (TWAP, VWAP, optimal per risk aversion, square-root-law estimate). Every demo prints it.
- Eight impact golden cases in `data/golden/impact_golden.json`.
- Test counts: Python 174, C++ 91, Rust 78, Java 85 (428 in total).

#### Market units and repositioning (F1)

- New market-unit layer in all four languages (API_SPEC Section 7): Almgren-Chriss parameters from price, quoted spread, ADV and daily volatility using the paper's own impact calibration; Avellaneda-Stoikov parameters from tick, touch fill rate, halving distance and opening skew; `bps` and `half_spread_cost` reporting helpers.
- Five presets in `data/presets.json` (US large cap, US small cap, US ETF, EUR/USD, the Almgren-Chriss example) and nine market golden cases in `data/golden/market_golden.json`. The paper preset reproduces `params_liquid.json` exactly.
- Every demo prints the presets in basis points.
- Repositioned as an execution and market-making research platform (README, landing page, LEARN introduction), with the roadmap and current limitations stated.
- Test counts: Python 139, C++ 83, Rust 62, Java 77 (361 in total).

#### Build and CI (Phase 0+1)

- Continuous integration on Linux, Windows and macOS for all four languages.
- Python is an installable package (`pip install -e python`), linted with ruff and type-checked with mypy.
- C++ builds on MSVC as well as GCC and Clang; GoogleTest is fetched when not installed; tests are optional.
- Java builds with Maven (`mvn -f java/pom.xml verify`) on any OS; package renamed to `io.github.ashjha0.etrade`; Java 17 is the minimum.
- Rust crate metadata completed; `serde_json` is now a dev-dependency only and the demo is an example (`cargo run --example demo`).
- Documentation corrections: Java determinism scoped to one JVM build and platform; the `sinh` overflow motivation fixed (`kappa*T` does not depend on the grid); "1 ulp" claims replaced with "cancellation-free"; the LEARN knowledge-check section renamed "Knowledge check" (now Section 13) and sections renumbered.

## 1.1.0 — 2026-09-06

The review found that mean_fills in the Avellaneda-Stoikov simulation was a logical OR, not a count: the Python reference summed numpy bool arrays, so a step where both bid and ask filled counted once, and the three ports had copied the OR to match the golden. That undercounted fills by about 10% on the liquid set (77.2 vs 85.5). The fix is pinned in API_SPEC §2.2, the golden as_sim_liquid.mean_fills was regenerated (77.18 → 85.53), and a saturated-book test (P(fill) = 1 both sides ⇒ fills = 2·n_steps exactly) plus an independent scalar re-implementation of the loop keep the OR from returning. The same pass replaced the overflow-prone sinh ratio and the cancellation-prone acosh kappa, added the 10-million size bounds (previously a MemoryError / float-to-int UB / capacity abort / NegativeArraySizeException across the four ports) and made simulate_is enforce the liquidation check. Still by design: A-S fill probability is the exact thinned 1 − exp(−λdt), so mean PnL sits ~12% below the paper's Table 1 while the PnL standard deviations agree.
