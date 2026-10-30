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
and a pre-trade cost report; at `alpha = 1` it reproduces Almgren-Chriss
exactly. Market-making simulation does not yet include informed flow, fees or
latency; that is the next release. Presets and the volume curve are
illustrative, not fitted to data.

### Pre-trade report

Every demo prints this for a US large cap selling 5% of ADV in one day with
square-root impact and a U-shaped volume curve (`API_SPEC.md` Section 8):

| strategy | lambda | impact | sd | total (with half spread) |
|---|---|---|---|---|
| TWAP | 0 | 34.96 bps | 81.59 bps | 35.46 bps |
| VWAP | 0 | 33.77 bps | 79.91 bps | 34.27 bps |
| optimal | 1e-6 | 36.95 bps | 56.60 bps | 37.45 bps |
| optimal | 1e-5 | 58.26 bps | 21.56 bps | 58.76 bps |

The closed-form square-root-law estimate is 33.54 bps. With a 50 bps/day
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

### Roadmap

1. **Market units and realistic presets** (done).
2. **Execution a desk would recognise** (this release): square-root and
   power-law impact, alpha-aware (urgency) schedules, intraday volume curve,
   pre-trade cost report.
3. **Market making that can lose money:** informed flow, fees and rebates,
   tick grid, latency, PnL attribution, markouts, fair-value models.
4. **Calibration on recorded order-book data** and post-trade TCA.
5. **Strategy extensions:** limit-order placement, adaptive scheduling,
   internalisation, hedged market making, risk controls.

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
| Reporting in bps; half-spread cost; 5 presets; 9 market golden cases | yes | yes | yes | yes |
| Power-law (square-root) impact, volume curve, drift-aware schedules; pinned Newton solver with KKT check | yes | yes | yes | yes |
| Pre-trade report (TWAP, VWAP, optimal per lambda, square-root-law estimate); 8 impact golden cases | yes | yes | yes | yes |

## Directory layout

```
electronic-trading/
  README.md                # this file
  LEARN.md                 # theory: microstructure, AC, A-S, IS, regulation
  COOKBOOK.md              # task-oriented recipes in all four languages
  API_SPEC.md              # the pinned cross-language contract (normative)
  docs/
    ARCHITECTURE.md        # design, data flow, numerics, testing strategy
    GITHUB_PAGES.md        # the docs site (live at https://ashjha0.github.io/electronic-trading/) and how to publish it
    diagrams/*.mmd         # raw Mermaid sources
  data/
    params_liquid.json     # liquid large-cap parameter set
    params_illiquid.json   # small-cap illiquid parameter set
    params_fx.json         # FX-like parameter set (tight spread, high k)
    parent_order.json      # parent order, lambda grid, IS decomposition inputs
    presets.json           # market-unit presets (large cap, small cap, ETF, EUR/USD, paper)
    generate_data.py       # regenerates data + golden values (seeded)
    generate_market.py     # regenerates presets.json + market golden cases
    golden/golden.json     # 19 cross-language golden cases
    generate_impact.py     # regenerates the power-law impact golden cases
    golden/market_golden.json  # 9 market-unit golden cases
    golden/impact_golden.json  # 8 power-law impact golden cases
  python/
    src/etrade/            # reference package (almgren_chriss, avellaneda_stoikov, analytics)
    tests/                 # pytest suites incl. golden-value tests
    demo.py                # end-to-end demo report
    requirements.txt
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
pytest python                        # run the tests (174 tests)
ruff check . && (cd python && mypy)  # lint and type-check
```

### C++ (C++17, CMake >= 3.20; GCC, Clang or MSVC)

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release   # GoogleTest is fetched if not installed
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure   # run the tests (91 tests)
```

The demo is `build/etrade_demo` (`build/Release/etrade_demo.exe` with MSVC).
Options: `-DETRADE_BUILD_TESTS=OFF`, `-DETRADE_WERROR=ON`,
`-DETRADE_SANITIZE=ON` (ASan + UBSan, GCC/Clang).

### Rust (stable, MSRV 1.78)

```bash
cd rust
cargo test --release                 # run the tests (78 tests)
cargo run --release --example demo   # run the demo
```

### Java (JDK 17+, Maven)

```bash
cd java
mvn verify                           # compile with -Xlint:all -Werror and run the tests (85 tests)
mvn -q exec:java                     # run the demo
```

Test counts above are the current suite sizes (Python 174, C++ 91, Rust 78,
Java 85, 428 in total); every suite loads the same 19 model golden cases, the
9 market golden cases and the 8 impact golden cases.

## Sample demo output (Python reference, truncated)

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
   5.0e-07        705,633        750,597    0.4325
   2.0e-06      1,078,215        449,368    0.8463
   1.0e-05      1,782,711        171,712    1.7267

=== Baselines: optimal vs TWAP vs POV vs instantaneous ================
    strategy      E[cost] $    sqrt(Var) $
  AC optimal      1,078,215        449,368
        TWAP        600,000      1,040,673
     POV 15%        941,406        639,488
     Instant      2,500,000              0
MC check (20k paths): E=1,075,336 (closed 1,078,215, se 3,177); Var/1e9=201.9 (closed 201.9)

=== Avellaneda-Stoikov market making (liquid large-cap) ===============
example quote at t=0, q=+3: bid=97.9546 ask=99.6454 (mid 100.0, reservation 98.8000)
         agent   mean PnL   std PnL  Sharpe  mean|q| mean max|q|   rms q   fills
           A-S    55.9191    6.3677    8.78    1.063       4.096   1.452    85.6
  fixed-spread    57.8418   12.8145    4.51    4.226       9.499   4.964    81.8
inventory control: A-S rms q = 1.452 vs fixed 4.964 (similar PnL, much tighter inventory)
stress (vol x2 after T/2): std PnL 6.3677 -> 7.9045

=== Implementation-shortfall decomposition (sell parent order) ========
decision 100.00, arrival 99.50, close 98.80; executed 900/1000
         delay:     450.00 $
       trading:     230.00 $
   opportunity:     120.00 $
         TOTAL:     800.00 $

=== Market-impact regression (cost/share vs participation, native OLS)
fitted: cost = 0.5035 + 7.9707 * participation (true 0.50 + 8.00), R^2 = 0.9934, n = 25

done.
```

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
above 1, ...). Regenerate data and golden values with
`python3 data/generate_data.py`.

## Real-world usage notes

What a practitioner gets, and what is deliberately left out:

- **Scope.** Almgren-Chriss (2000) *static* mean-variance execution with
  linear temporary and permanent impact and an arithmetic random walk;
  Avellaneda-Stoikov (2008) finite-horizon market making with exponential
  arrival intensities, plus a hard inventory cap and a terminal liquidation
  penalty; Perold (1988) implementation-shortfall decomposition; participation
  rate; a two-variable OLS. Nothing else — no square-root impact, no
  adaptive re-optimisation, no adverse selection, no order types, venues,
  ticks, lots or latency.
- **Units and conventions** (`API_SPEC.md` Section 0): `tau`, `dt`, `T` are in
  an abstract time unit (the bundled sets use 1 = one trading day) — nothing
  ties them to a clock or a calendar (no day-count conventions, holidays or
  intraday volume profiles); `sigma` is *arithmetic* (currency per share per
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
- **What is verified.** 428 tests across the four suites: 19 shared golden
  cases at 1e-10..1e-12, the paper's closed forms against the discrete sums
  at 1e-12, Monte Carlo against closed forms at 4 SE, and every validation
  rule. What is *not* verified: calibration of `eta`, `gamma_p`, `A`, `k` to
  any real market, and behaviour outside the documented model assumptions
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
  motivating `q_max`; cited in `LEARN.md`)
- Regulatory texts cited in `LEARN.md` Section 12: Directive 2014/65/EU
  (MiFID II), Art. 17 and Art. 27; U.S. SEC Regulation NMS (17 CFR 242.600
  ff.), Rules 605/606; SEC Rule 15c3-5 (Market Access).

## License

MIT License — see the `LICENSE` file (Copyright (c) 2026 Ashish Jha).
Provided without warranty of any kind; this is a validated reference
implementation of published models, not investment advice and not an
order-management or risk system.
