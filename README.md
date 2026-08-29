# Electronic Trading (`etrade`)

Optimal execution and market making, implemented four times over from the same
pinned contract: **Almgren-Chriss (2000)** discrete optimal execution,
**Avellaneda-Stoikov (2008)** finite-horizon market making, and execution
analytics (Perold implementation-shortfall decomposition, participation rate,
a tiny native OLS for market-impact estimation).

The Python package `etrade` is the reference implementation; the C++
(`etrade::` namespace), Rust (`etrade` crate) and Java (`com.quant.etrade`)
ports mirror `API_SPEC.md` exactly and are validated against shared golden
values.

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
| Golden-value suite (`data/golden/golden.json`, 18 cases) | yes | yes | yes | yes |
| Determinism-by-seed tests (own RNG per language) | yes | yes | yes | yes |

## Directory layout

```
12-electronic-trading/
  README.md                # this file
  LEARN.md                 # theory: microstructure, AC, A-S, IS, regulation
  COOKBOOK.md              # task-oriented recipes in all four languages
  API_SPEC.md              # the pinned cross-language contract (normative)
  docs/
    ARCHITECTURE.md        # design, data flow, numerics, testing strategy
    diagrams/*.mmd         # raw Mermaid sources
  data/
    params_liquid.json     # liquid large-cap parameter set
    params_illiquid.json   # small-cap illiquid parameter set
    params_fx.json         # FX-like parameter set (tight spread, high k)
    parent_order.json      # parent order, lambda grid, IS decomposition inputs
    generate_data.py       # regenerates data + golden values (seeded)
    golden/golden.json     # 18 cross-language golden cases
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
    src/bin/demo.rs  tests/*.rs
  java/
    src/main/java/com/quant/etrade/*.java
    src/test/java/com/quant/etrade/*Test.java
    build.sh  test.sh  demo.sh
```

## Build, run, test

### Python (3.11, numpy)

```bash
cd python
PYTHONPATH=src python3 demo.py       # run the demo
PYTHONPATH=src pytest -q             # run the tests
```

### C++ (C++17, CMake, GoogleTest)

```bash
cd cpp
./build.sh                           # cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j2
./build/demo                         # run the demo
ctest --test-dir build --output-on-failure   # run the tests
```

### Rust (edition 2021)

```bash
cd rust
cargo build --release
cargo run --release --bin demo       # run the demo
cargo test                           # run the tests
```

### Java (21, JUnit 4 at /usr/share/java/junit4.jar)

```bash
cd java
./build.sh                           # javac everything into out/
./demo.sh                            # run the demo main
./test.sh                            # JUnitCore on all *Test classes
```

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
           A-S    55.9191    6.3677    8.78    1.063       4.096   1.452    77.2
  fixed-spread    57.8418   12.8145    4.51    4.226       9.499   4.964    73.4
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

## Golden-value cross-language testing

`data/golden/golden.json` holds 18 named cases with inputs, expected outputs
and a tolerance. Every language's test suite loads the same file and asserts
agreement, so all four implementations are pinned to identical numbers:

- **Exact cases** (closed forms, deterministic constructions): each key must
  satisfy `|actual - expected| <= tol * max(1, |expected|)` with `tol`
  between `1e-10` and `1e-12`. Examples: AC `kappa`/`x_j`/`n_1`/`E`/`Var` for
  two parameter sets, TWAP/POV costs, the `lambda_ra = 0` = TWAP identity,
  A-S reservation price / spread / fill probability, both IS decompositions,
  the impact-regression fit on pinned RNG-free synthetic fills.
- **Statistical cases** (Monte Carlo): the file stores the Python 20,000-path
  reference value *plus its standard error* per statistic. Ports use their own
  native RNG (C++ `std::mt19937_64`, Rust `rand`, Java `Random`) with the same
  pinned logical draw order, run `M >= 2000` paths, and must satisfy
  `|stat - ref| <= 4*sqrt(se_ref^2 + se_own^2)`. Sample values differ across
  languages by design; distributions must agree.

Each suite additionally verifies determinism (same seed twice gives identical
results) and property-style invariants (trajectory monotonicity, inventory cap
respected at every step, IS decomposition identity, frontier monotonicity).
Regenerate data and golden values with `python3 data/generate_data.py`.

## License

Educational use. No license restrictions; provided as teaching material with
no warranty — not investment advice and not production trading software.
