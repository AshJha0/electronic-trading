# Electronic Trading (`etrade`): execution and market-making research

[![CI](https://github.com/AshJha0/electronic-trading/actions/workflows/ci.yml/badge.svg)](https://github.com/AshJha0/electronic-trading/actions/workflows/ci.yml)

A research platform for the two businesses at the centre of electronic
trading: **executing large orders at low cost** and **making markets
profitably**. It starts from the canonical models, Almgren-Chriss (2000)
optimal execution and Avellaneda-Stoikov (2008) market making, plus
implementation-shortfall analytics, and builds outward towards what decides
money in practice: realistic costs, informed flow, fees and ticks, fair-value
signals and measurement after the fact.

Everything is implemented four times, in Python (the reference), C++, Rust
and Java, from one pinned contract (`API_SPEC.md`), and held together by
shared golden values checked in CI on Linux, Windows and macOS.

**Where it stands.** The models are exact and cross-validated; parameters
can be stated in market units (price, tick, spread, daily volatility, ADV)
with results in basis points. Execution now has a power-law impact model
(square-root by default) with intraday volume curves, alpha-aware schedules
and a pre-trade cost report; at `alpha = 1` it reproduces the Almgren-Chriss
trajectory (holdings within 1e-10 of `X`, costs within 1e-12 relative,
asserted in every suite). Post-trade TCA measures slippage against four
benchmarks, markouts and participation, and compares algorithms with a
Welch t-test. Calibration estimators turn recorded data into model inputs,
and three presets are fitted to one NASDAQ ITCH day (AAPL, MSFT, QQQ on
2019-12-30, with the caveats of `API_SPEC.md` Section 11.6). Execution
extensions add transient (decaying) impact, adaptive re-planning and
internalisation. Market making runs with a price grid, maker fees or
rebates, stale quotes, informed flow and a fair-value signal, with an exact
PnL attribution and markouts, and GLFT quotes beside A-S. Limit-order
placement covers queue fill probability, the choice between posting and
crossing, and liquidation with limit and market orders. The other five
presets and the pinned volume curve are illustrative, not fitted to data.

Documentation site: https://ashjha0.github.io/electronic-trading/ (the guide,
cookbook, contract, architecture notes and changelog, built from the Markdown
files in this repository).

### Pre-trade report

Every demo prints this for a US large cap selling 5% of ADV in one day with
square-root impact and a U-shaped volume curve (`API_SPEC.md` Section 8):

| strategy | lambda | impact | sd | total (with half spread) |
|---|---|---|---|---|
| TWAP | 0 | 34.96 bps | 81.59 bps | 35.46 bps |
| VWAP | 0 | 33.77 bps | 79.91 bps | 34.27 bps |
| optimal | 0 | 33.77 bps | 79.91 bps | 34.27 bps |
| optimal | 1e-7 | 33.83 bps | 76.51 bps | 34.33 bps |
| optimal | 1e-6 | 36.95 bps | 56.60 bps | 37.45 bps |
| optimal | 1e-5 | 58.26 bps | 21.56 bps | 58.76 bps |

At `lambda = 0` the optimum matches VWAP to the printed precision: with no
risk penalty and no drift, the temporary cost is minimised by trading in
proportion to liquidity (the small permanent-impact term moves it slightly). The
closed-form square-root-law estimate is 33.54 bps. With a 50 bps/day
adverse drift behind the order, the drift-aware schedule costs 52.13 bps
against 57.31 bps for VWAP, a saving of 5.18 bps.

### Market presets at a glance

Produced by every demo from `data/presets.json` (sell orders over one day;
A-S spread at the open and at the close of the session):

| preset | order | E[IS] | sd | half-spread | A-S spread open / close |
|---|---|---|---|---|---|
| US large cap ($100, 1 bp spread) | 5% ADV | 5.4 bps | 75.5 bps | 0.5 bps | 3.6 / 1.3 bps |
| US small cap ($20, 25 bp spread) | 10% ADV | 265.2 bps | 178.9 bps | 12.5 bps | 53.8 / 28.8 bps |
| US ETF ($500, 0.2 bp spread) | 1% ADV | 0.2 bps | 50.6 bps | 0.1 bps | 0.5 / 0.3 bps |
| EUR/USD (0.45 bp spread) | 5% of volume | 2.4 bps | 26.2 bps | 0.2 bps | 0.8 / 0.5 bps |
| Almgren-Chriss (2000) example | 20% ADV | 215.6 bps | 89.9 bps | 12.5 bps | |
| AAPL, fitted to ITCH 2019-12-30 ($289.72, 0.91 bp spread) | 5% of NASDAQ volume | 4.9 bps | 58.2 bps | 0.5 bps | 1.2 / 0.5 bps |
| MSFT, fitted to ITCH 2019-12-30 ($157.67, 0.78 bp spread) | 5% of NASDAQ volume | 4.2 bps | 37.4 bps | 0.4 bps | 1.9 / 0.5 bps |
| QQQ, fitted to ITCH 2019-12-30 ($212.16, 0.55 bp spread) | 5% of NASDAQ volume | 2.9 bps | 29.0 bps | 0.3 bps | 0.8 / 0.3 bps |

The fitted rows take price, spread, volatility, volume and the fill
intensity from one day of NASDAQ data; order size, risk aversion and the
market-making settings follow the US large-cap and ETF templates. One day
on one venue in the week after Christmas: read them as realistic magnitudes,
not as estimates with known error (`LEARN.md` Section 3.10).

### Market making with frictions

Every demo prints this friction ledger for the US large-cap A-S preset
(one-share quotes, 5-second steps, one day; `API_SPEC.md` Section 9).
Dollars per day, 20,000-path Python sample; each row adds or changes one
friction, and every path satisfies
`PnL = spread + fees + inventory - liquidation` exactly:

| case | PnL (SE) | spread | fees | inventory | markout | fills |
|---|---|---|---|---|---|---|
| frictionless | 7.41 (0.01) | 7.44 | 0.00 | 0.01 | 0.01 | 1,076 |
| 1-cent tick, 0.2-cent rebate | 7.33 (0.01) | 6.20 | 1.16 | -0.01 | 0.00 | 580 |
| 1-cent tick, 0.2-cent fee | 5.00 (0.01) | 6.20 | -1.16 | -0.00 | -0.01 | 580 |
| tick, rebate, 15% informed flow | 4.01 (0.01) | 10.25 | 1.86 | -8.07 | -8.06 | 931 |
| as above, 60%-accurate signal | 5.00 (0.01) | 9.81 | 1.97 | -6.75 | -6.74 | 985 |
| tick, rebate, quotes 5 s stale | -23.73 (0.02) | -28.33 | 4.65 | 0.00 | 0.00 | 2,327 |
| GLFT quotes, tick, rebate | 11.00 (0.04) | 9.27 | 1.76 | -0.00 | -0.00 | 880 |

Informed flow raises spread capture and lowers PnL: the markout, the loss in
the step after each fill, takes it back, and the inventory skew does not
prevent it. Quotes 5 seconds stale are picked off and lose money on every
day. The C++, Rust and Java demos print the same table from their own
5,000-path samples (`LEARN.md` Sections 6.7-6.12 read it row by row).

### Roadmap

1. **F1. Market units and realistic presets** (done; `API_SPEC.md`
   Section 7).
2. **F2. Execution a desk would recognise** (done; Section 8): square-root
   and power-law impact, alpha-aware (urgency) schedules, intraday volume
   curve, pre-trade cost report.
3. **F3. Market making that can lose money** (done; Section 9): informed
   flow, fees and rebates, tick grid, latency, PnL attribution, markouts,
   fair-value helpers, GLFT quotes.
4. **F4. Calibration on recorded order-book data and post-trade TCA**
   (done; Sections 10 and 11): benchmark slippage, markouts, participation,
   Welch A/B comparison; volume, volatility, spread and fill-intensity
   estimators; three presets fitted to one NASDAQ ITCH day.
5. **F5. Strategy extensions** (partly done; Sections 12 and 13): adaptive
   scheduling, transient impact, internalisation and limit-order placement
   are done; hedged market making and risk controls are in progress.

Not investment advice and not a trading system; see the usage notes below.

## Feature matrix

| Feature | Python | C++ | Rust | Java |
|---|---|---|---|---|
| AC closed-form optimal trajectory (`kappa`, `x_j`, `n_j`) | yes | yes | yes | yes |
| AC exact discrete `E[IS]` / `Var[IS]` sums | yes | yes | yes | yes |
| `lambda_ra = 0` exact TWAP limit branch (`kappa*T < 1e-6`) | yes | yes | yes | yes |
| Efficient frontier sweep + half-life reporting | yes | yes | yes | yes |
| TWAP / POV / instantaneous-liquidation baselines | yes | yes | yes | yes |
| AC Monte Carlo IS check (closed forms within 4 SE) | yes | yes | yes | yes |
| A-S reservation price, optimal spread, quotes | yes | yes | yes | yes |
| A-S intensity `A*exp(-k*delta)`, thinned-Poisson fills | yes | yes | yes | yes |
| A-S simulation (pinned loop order, inventory cap, stress vol) | yes | yes | yes | yes |
| Fixed-spread comparison agent (same RNG stream mechanics) | yes | yes | yes | yes |
| IS decomposition (delay / trading / opportunity, buy & sell) | yes | yes | yes | yes |
| Participation rate + market-impact OLS | yes | yes | yes | yes |
| Golden-value suite (`data/golden/golden.json`, 19 cases) | yes | yes | yes | yes |
| Determinism-by-seed tests (own RNG per language) | yes | yes | yes | yes |
| Overflow-free trajectory (`exp/expm1` form, any `kappa*T`) | yes | yes | yes | yes |
| Cancellation-free `kappa` (`log1p` form, any `tau`) | yes | yes | yes | yes |
| Pinned size bounds (intervals, POV clips, steps, paths) — input errors, never UB/OOM | yes | yes | yes | yes |
| Literature check: paper closed forms = discrete sums (1e-12) | yes | yes | yes | yes |
| Market units: AC from price/spread/ADV/vol, A-S from tick/touch rate/halving/skew | yes | yes | yes | yes |
| Reporting in bps; half-spread cost; 8 presets (3 fitted) | yes | yes | yes | yes |
| Power-law (square-root) impact, volume curve, drift-aware schedules; pinned Newton solver with KKT check | yes | yes | yes | yes |
| Pre-trade report (TWAP, VWAP, optimal per lambda, square-root-law estimate); 8 impact golden cases | yes | yes | yes | yes |
| TCA: slippage vs arrival / market VWAP / interval VWAP / close, markouts, per-bin participation | yes | yes | yes | yes |
| TCA: Welch t-test with pinned `ln_gamma`, incomplete beta and Student t CDF; 11 TCA golden cases | yes | yes | yes | yes |
| Calibration: volume curve, realised-volatility curve, spread statistics, weighted fill-intensity fit; 9 calibration golden cases | yes | yes | yes | yes |
| Fitted presets (AAPL, MSFT, QQQ, one ITCH day) and fitted volume curve; 6 more market golden cases (15 in all) | yes | yes | yes | yes |
| ITCH ingestion and fitting (`data/calibration/fit_itch.py`, tooling) | yes | | | |
| Execution: transient impact (Obizhaeva-Wang) with exact Thomas-solved optimum | yes | yes | yes | yes |
| Execution: adaptive re-planning vs static (drift signal, partial fills), internalisation; 8 execution golden cases | yes | yes | yes | yes |
| Market making with frictions: tick grid, maker fee or rebate, quote latency, informed flow, fair-value signal | yes | yes | yes | yes |
| Exact per-path PnL attribution (spread, fees, inventory, liquidation) and one-step markouts | yes | yes | yes | yes |
| GLFT closed-form quotes; weighted mid and order-flow imbalance helpers; 14 market-making golden cases | yes | yes | yes | yes |
| Queue fill probability by uniformisation; passive-or-aggressive decision for a child order | yes | yes | yes | yes |
| Liquidation with limit and market orders (backward induction, paired simulation); 11 placement golden cases | yes | yes | yes | yes |

## Directory layout

```
electronic-trading/
  README.md                # this file
  LEARN.md                 # theory: microstructure, AC, A-S, impact, calibration, TCA, frictions, placement, regulation
  COOKBOOK.md              # task-oriented recipes in all four languages
  API_SPEC.md              # the pinned cross-language contract (normative)
  CHANGELOG.md
  mkdocs.yml               # documentation site (MkDocs Material)
  docs/
    ARCHITECTURE.md        # design, data flow, numerics, testing strategy
    GITHUB_PAGES.md        # how the docs site is built and published
    index.html             # stand-alone landing page (served until Pages switches to the Actions build)
    site/                  # MkDocs landing page and stylesheet
    diagrams/*.mmd         # raw Mermaid sources
  data/
    params_liquid.json     # liquid large-cap parameter set
    params_illiquid.json   # small-cap illiquid parameter set
    params_fx.json         # FX-like parameter set (tight spread, high k)
    parent_order.json      # parent order, lambda grid, IS decomposition inputs
    presets.json           # market-unit presets (5 illustrative, 3 fitted to one ITCH day)
    calibration/fit_itch.py           # ITCH dataset -> fitted aggregates (tooling, Python only)
    calibration/fitted_20191230.json  # the fitted aggregates (no raw data)
    generate_data.py       # regenerates data + golden values (seeded)
    generate_market.py     # regenerates presets.json + market golden cases
    generate_impact.py     # regenerates the power-law impact golden cases
    generate_tca.py  generate_calibration.py  generate_execution.py
    generate_mm.py  generate_placement.py
    golden/golden.json     # 19 cross-language golden cases
    golden/market_golden.json       # 15 market-unit golden cases
    golden/impact_golden.json       # 8 power-law impact golden cases
    golden/tca_golden.json          # 11 TCA golden cases
    golden/calibration_golden.json  # 9 calibration golden cases
    golden/execution_golden.json    # 8 execution-extension golden cases
    golden/mm_golden.json           # 14 market-making golden cases
    golden/placement_golden.json    # 11 limit-order placement golden cases
  python/
    src/etrade/            # reference package (almgren_chriss, avellaneda_stoikov, analytics, impact, market, tca, calibration, execution, mm, placement)
    tests/                 # pytest suites incl. golden-value tests
    demo.py                # end-to-end demo report
    pyproject.toml  requirements.txt
  cpp/
    include/etrade/*.hpp   # C++17 library headers
    src/*.cpp  tests/*.cpp
    CMakeLists.txt  build.sh
  rust/
    Cargo.toml  src/lib.rs src/*.rs
    examples/demo.rs  tests/*.rs
  java/
    pom.xml
    src/main/java/io/github/ashjha0/etrade/*.java
    src/test/java/io/github/ashjha0/etrade/*Test.java
```

## Build, run, test

Every command below runs in CI on Linux, Windows and macOS on every push
and pull request (`.github/workflows/ci.yml`).

### Python (3.11+)

```bash
pip install -e "python[dev]"         # installs etrade + pytest, ruff, mypy
python python/demo.py                # run the demo
pytest python                        # run the tests (487 tests; 6 benchmarks more with pytest-benchmark)
ruff check . && (cd python && mypy)  # lint and type-check
```

### C++ (C++17, CMake >= 3.20; GCC, Clang or MSVC)

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release   # GoogleTest is fetched if not installed
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure   # run the tests (153 tests)
```

The demo is `build/etrade_demo` (`build/Release/etrade_demo.exe` with MSVC).
Options: `-DETRADE_BUILD_TESTS=OFF`, `-DETRADE_WERROR=ON`,
`-DETRADE_SANITIZE=ON` (ASan + UBSan, GCC/Clang).

### Rust (stable, MSRV 1.78)

```bash
cd rust
cargo test --release                 # run the tests (200 tests, 4 of them doc-tests)
cargo run --release --example demo   # run the demo
```

### Java (JDK 17+, Maven)

```bash
cd java
mvn verify                           # compile with -Xlint:all -Werror and run the tests (142 tests)
mvn -q exec:java                     # run the demo
```

### Documentation site (MkDocs Material)

```bash
pip install -r docs/site/requirements.txt
mkdocs serve                         # preview at http://127.0.0.1:8000/electronic-trading/
mkdocs build --strict                # what the Docs workflow runs on every PR
```

Test counts above are the suite sizes measured by running each suite on
2026-10-30 (Python 487, C++ 153, Rust 200, Java 142; 982 in total; the six
Python benchmarks run only when pytest-benchmark is installed); every
suite loads the same eight golden files, 95 cases: 19 model, 15 market, 8
impact, 11 TCA, 9 calibration, 8 execution, 14 market-making and 11
placement.

## Sample demo output (Python reference, abridged)

```
etrade demo — Almgren-Chriss execution & Avellaneda-Stoikov market making

=== Almgren-Chriss optimal trajectories (sell 1.0e6 shares, N=5) ======
 lambda_ra     kappa  half-life | holdings x_j (thousands)
   0.0e+00    0.0000        inf |   1000.0    800.0    600.0    400.0    200.0      0.0
   5.0e-07    0.4325      2.312 |   1000.0    636.9    394.8    227.8    104.0      0.0
   2.0e-06    0.8463      1.182 |   1000.0    428.6    182.9     76.3     27.6      0.0
   1.0e-05    1.7267      0.579 |   1000.0    177.9     31.6      5.6      1.0      0.0

=== Efficient frontier of execution (E[cost] vs sqrt(Var[cost])) ======
 lambda_ra      E[cost] $    sqrt(Var) $     kappa
   0.0e+00        600,000      1,040,673    0.0000
   1.0e-08        600,089      1,032,060    0.0616
   1.0e-07        607,557        961,623    0.1946
   5.0e-07        705,633        750,597    0.4325
   1.0e-06        848,727        603,431    0.6071
   2.0e-06      1,078,215        449,368    0.8463
   5.0e-06      1,472,888        272,687    1.2876
   1.0e-05      1,782,711        171,712    1.7267

=== Baselines: optimal vs TWAP vs POV vs instantaneous ================
    strategy      E[cost] $    sqrt(Var) $
  AC optimal      1,078,215        449,368
        TWAP        600,000      1,040,673
     POV 15%        941,406        639,488
     Instant      2,500,000              0
MC check (20k paths): E=1,075,336 (closed 1,078,215, se 3,177); Var/1e9=201.9 (closed 201.9)

=== Avellaneda-Stoikov market making (liquid large-cap) ===============
params: s0=100.0 sigma=2.0 gamma=0.1 k=1.5 A=140.0 T=1.0 dt=0.005 q_max=20
example quote at t=0, q=+3: bid=97.9546 ask=99.6454 (mid 100.0, reservation 98.8000)
         agent   mean PnL   std PnL  Sharpe  mean|q| mean max|q|   rms q   fills
           A-S    55.9191    6.3677    8.78    1.063       4.096   1.452    85.6
  fixed-spread    57.8418   12.8145    4.51    4.226       9.499   4.964    81.8
inventory control: A-S rms q = 1.452 vs fixed 4.964 (similar PnL, much tighter inventory)
stress (vol x2 after T/2): std PnL 6.3677 -> 7.9045

[FX-like market-making section omitted]

=== Implementation-shortfall decomposition (sell parent order) ========
decision 100.00, arrival 99.50, close 98.80; executed 900/1000
         delay:     450.00 $
       trading:     230.00 $
   opportunity:     120.00 $
         TOTAL:     800.00 $
participation rate example: 120k / 1.5m = 8.00%

=== Market-impact regression (cost/share vs participation, native OLS)
fitted: cost = 0.5035 + 7.9707 * participation (true 0.50 + 8.00), R^2 = 0.9934, n = 25

[presets, pre-trade and friction-ledger sections omitted: they are the tables at the top;
 TCA, calibration, execution-extension and placement sections omitted]

done.
```

The demo prints every section in full; the excerpt leaves out the FX-like
market-making run, the tables already shown at the top of this file, and
the sections for Sections 10-13 of the contract, whose numbers `LEARN.md`
and `COOKBOOK.md` quote.
The simulation rows are 2,000-path samples from the Python demo; the C++,
Rust and Java demos print their own samples (e.g. A-S fills 85.2 / 85.5 /
85.3) that agree within the statistical bands. `fills` is the total number of
fills per path, both sides counted.

## Golden-value cross-language testing

`data/golden/golden.json` holds 19 named cases with inputs, expected outputs
and a tolerance. Every language's test suite loads the same file and asserts
agreement, so all four implementations are pinned to identical numbers:

- **Exact cases** (closed forms, deterministic constructions): each key must
  satisfy `|actual - expected| <= tol * max(1, |expected|)` with `tol`
  between `1e-10` and `1e-12`. Examples: AC `kappa`/`x_j`/`n_1`/`E`/`Var` for
  two parameter sets, `kappa` at millisecond-scale `tau` (where the naive
  `acosh` form loses 8 digits), TWAP/POV costs, the `lambda_ra = 0` = TWAP
  identity, A-S reservation price / spread / fill probability, both IS
  decompositions, the impact-regression fit on pinned RNG-free synthetic
  fills.
- **Statistical cases** (Monte Carlo): the file stores the Python 20,000-path
  reference value *plus its standard error* per statistic. Ports use their own
  native RNG (C++ `std::mt19937_64`, Rust `ChaCha12Rng`, Java
  `SplittableRandom`) with the same pinned logical draw order, run `M >= 2000`
  paths, and must satisfy `|stat - ref| <= 4*sqrt(se_ref^2 + se_own^2)`.
  Sample values differ across languages by design; distributions must agree.

Each suite additionally verifies determinism (same seed twice gives identical
results), property-style invariants (trajectory monotonicity, inventory cap
respected at every step, IS decomposition identity and price-shift
invariance, frontier monotonicity), the literature identity between the
Almgren-Chriss closed forms and the discrete sums, overflow/underflow safety
at `kappa*T > 2500`, and every validation rule of `API_SPEC.md` Section 6
(non-finite inputs, size bounds, non-liquidating schedules, participation
above 1, ...).

Seven further golden files follow the same schema and comparison rule:
`market_golden.json` (15 market-unit cases, 1e-12, `API_SPEC.md` Sections
7.4 and 11.7), `impact_golden.json` (8 power-law impact cases, 1e-9,
Section 8.5), `tca_golden.json` (11 cases, 1e-12, Section 10.6),
`calibration_golden.json` (9 cases, 1e-12, Section 11.7),
`execution_golden.json` (8 cases, exact at 1e-9 and statistical at 4 SE,
Section 12.4), `mm_golden.json` (14 cases, exact at 1e-12 and statistical
at 4 SE, Section 9.7) and `placement_golden.json` (11 cases, exact at 1e-10
and statistical at 4 SE, Section 13.4). Each has a generator, `data/generate_<name>.py`
(`generate_data.py` for `golden.json`); `docs/ARCHITECTURE.md` lists them
all. Regenerate on Linux or macOS: the committed values come from there,
and the last bit of some libm results differs on Windows.

## Real-world usage notes

What a practitioner gets, and what is deliberately left out:

- **Scope.** Almgren-Chriss (2000) *static* mean-variance execution with
  linear temporary and permanent impact and an arithmetic random walk;
  Avellaneda-Stoikov (2008) finite-horizon market making with exponential
  arrival intensities, plus a hard inventory cap and a terminal liquidation
  penalty; Perold (1988) implementation-shortfall decomposition; participation
  rate; a two-variable OLS; a power-law (square-root by default)
  generalisation of the execution problem with intraday liquidity weights and
  an adverse drift, solved numerically (`API_SPEC.md` Section 8); conversion
  from market units with reporting in basis points (Section 7); post-trade
  TCA (Section 10); calibration estimators on plain arrays (Section 11);
  transient impact with exponential resilience, adaptive re-planning under a
  drift signal and partial fills, and internalisation (Section 12); market
  making with a tick grid, fees, stale quotes, informed flow and a signal
  of given accuracy (Section 9); queue fill probability and liquidation
  with limit and market orders (Section 13). Nothing else: no venues, lots
  or order book; the frictions are stylised switches, not a matching
  engine.
- **Units and conventions** (`API_SPEC.md` Section 0): `tau`, `dt`, `T` are in
  an abstract time unit (the bundled sets use 1 = one trading day) — nothing
  ties them to a clock or a calendar (no day-count conventions or holidays;
  the intraday volume curve of Section 8 is a vector of per-interval weights,
  not a clock); `sigma` is *arithmetic* (currency per share per
  sqrt(time unit)); prices may go negative; AC shares are fractional (no lot
  rounding); A-S fills are one unit each; all costs are in currency (positive
  = loss); `half_life` is the paper's `1/kappa` (an e-folding time), not
  `ln 2 / kappa`.
- **Sell programs only** in AC (`X >= 0`); mirror prices by hand for a buy
  program. The paper's fixed cost `epsilon*sum|n_j|` (half-spread, fees) is
  omitted — it equals `epsilon*X` for any one-sided schedule and does not
  change the trajectory; add it to `E[IS]` if you report total cost.
- **A-S fill probability** is the exact thinned `1 - exp(-lambda*dt)`, not
  the paper's first-order `lambda*dt` — mean PnL is ~12% below the paper's
  Table 1 for that reason; the PnL standard deviations agree.
- **Validated at every public entry point** (same rules in all four
  languages): NaN/inf anywhere, negative-where-invalid, empty grids,
  non-liquidating schedules, participation above 1, and every size beyond the
  pinned bounds (10 million intervals / POV clips / steps / paths) are input
  errors — Python `ValueError`, C++ `std::invalid_argument`
  (`std::domain_error` for the AC validity condition), Rust
  `Err(EtradeError)` (no panics), Java `IllegalArgumentException`. No
  function returns NaN or infinity from valid finite inputs; the AC
  trajectory is finite for any `kappa*T` (the `sinh` ratio is evaluated in
  `exp/expm1` form) and `kappa` is free of cancellation for any `tau` (`log1p`
  form).
- **Determinism.** Same seed, same results within a language. Python is
  stable across NumPy releases (PCG64 is versioned) and Rust across `rand`
  0.8.x; Java is guaranteed within one JVM build and platform (`nextGaussian`
  and `Math.*` are not bit-specified); C++ is deterministic per standard
  library because `std::normal_distribution` is implementation-defined.
  Cross-language agreement is statistical (4-SE bands), never bitwise.
- **What is verified.** 982 tests across the four suites: 95 shared golden
  cases (19 model at 1e-10..1e-12, 15 market-unit at 1e-12, 8 power-law
  impact at 1e-9, 11 TCA and 9 calibration at 1e-12, 8 execution at 1e-9 or
  4 SE, 14 market-making at 1e-12 or 4 SE, 11 placement at 1e-10 or 4 SE), the paper's closed forms against the discrete sums at 1e-12, Monte
  Carlo against closed forms at 4 SE, and every validation rule; in the
  Python suite, the power-law solver against SciPy's constrained optimiser
  and the t-test against SciPy. What is *not* verified: the impact
  parameters `eta`, `gamma_p`, `alpha` and `Y` against real parent orders
  (the fitted presets take them from the literature), the sampling error of
  the one-day fits, and behaviour outside the documented model assumptions
  (`LEARN.md` Section 7).

## References

Papers and standards the code implements or the documentation relies on:

- R. Almgren and N. Chriss (2000), "Optimal execution of portfolio
  transactions", *Journal of Risk* 3(2), 5–39.
  https://doi.org/10.21314/JOR.2001.041 (author copy:
  https://www.math.nyu.edu/~almgren/papers/optliq.pdf)
- M. Avellaneda and S. Stoikov (2008), "High-frequency trading in a limit
  order book", *Quantitative Finance* 8(3), 217–224.
  https://doi.org/10.1080/14697680701381228
- A. F. Perold (1988), "The implementation shortfall: paper versus reality",
  *Journal of Portfolio Management* 14(3), 4–9.
  https://doi.org/10.3905/jpm.1988.409150
- W. H. Wagner and M. Edwards (1993), "Best execution", *Financial Analysts
  Journal* 49(1), 65–71. https://doi.org/10.2469/faj.v49.n1.65 (the
  delay / trading / opportunity naming of the shortfall components)
- O. Guéant, C.-A. Lehalle and J. Fernandez-Tapia (2013), "Dealing with the
  inventory risk: a solution to the market making problem", *Mathematics
  and Financial Economics* 7(4), 477–507.
  https://doi.org/10.1007/s11579-012-0087-0 (the inventory-capped variant
  motivating `q_max`, and the GLFT quotes of Section 9.3)
- A. A. Obizhaeva and J. Wang (2013), "Optimal trading strategy and
  supply/demand dynamics", *Journal of Financial Markets* 16(1), 1–32
  (transient impact, `API_SPEC.md` Section 12.1).
- R. Almgren and J. Lorenz (2007), "Adaptive arrival price", *Trading*
  2007(1), 59–66 (adaptive execution, Section 12.2).
- T. G. Andersen, T. Bollerslev, F. X. Diebold and P. Labys (2001), "The
  distribution of realized exchange rate volatility", *Journal of the
  American Statistical Association* 96(453), 42–55 (realised variance,
  Section 11.3).
- B. L. Welch (1947), "The generalization of 'Student's' problem when
  several different population variances are involved", *Biometrika*
  34(1–2), 28–35 (the A/B test, Section 10.4).
- L. R. Glosten and P. R. Milgrom (1985), "Bid, ask and transaction prices
  in a specialist market with heterogeneously informed traders", *Journal of
  Financial Economics* 14(1), 71–100 (adverse selection, Section 9).
- R. Cont, S. Stoikov and R. Talreja (2010), "A stochastic model for order
  book dynamics", *Operations Research* 58(3), 549–563 (the queue model,
  Section 13.1).
- A. Cartea and S. Jaimungal (2015), "Optimal execution with limit and
  market orders", *Quantitative Finance* 15(8), 1279–1291 (liquidation with
  both order types, Section 13.3).
- NASDAQ TotalView-ITCH 5.0 sample file `12302019.NASDAQ_ITCH50`
  (emi.nasdaq.com/ITCH/), the source of the fitted presets (Section 11.6).
- Regulatory texts cited in `LEARN.md` Section 12: Directive 2014/65/EU
  (MiFID II), Art. 17 and Art. 27; U.S. SEC Regulation NMS (17 CFR 242.600
  ff.), Rules 605/606; SEC Rule 15c3-5 (Market Access).

## License

MIT License — see the `LICENSE` file (Copyright (c) 2026 Ashish Jha).
Provided without warranty of any kind; this is a validated reference
implementation of published models, not investment advice and not an
order-management or risk system.
