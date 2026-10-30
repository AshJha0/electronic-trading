# Changelog

All notable changes to this project. Dates are UTC.

## Unreleased

### Documentation accuracy pass and site

- Every number in README, LEARN, COOKBOOK, ARCHITECTURE and the landing page re-checked against the four demos and test suites; every COOKBOOK snippet run (Python) or compiled and run in a scratch harness (C++, Rust, Java). Corrections: two stale trajectory values in COOKBOOK recipe 1, deterministic values quoted to 15 significant digits (the last ulp differs by platform), the pre-trade table now lists every row the demos print, the scope and units notes no longer say square-root impact and volume curves are missing, ARCHITECTURE describes all five modules and the three golden files, and the roadmap marks F1 and F2 done.
- LEARN gains Section 3.8 (parameters in market units); the power-law section becomes 3.9, and its references (Almgren et al. 2005, Tóth et al. 2011, Bouchaud et al. 2018) are in the reading list.
- Documentation site built with MkDocs Material from the Markdown sources (`mkdocs.yml`), checked with `mkdocs build --strict` on every pull request and deployed to GitHub Pages from `main` by `.github/workflows/docs.yml`.

### Execution a desk would recognise (F2)

- Power-law impact execution in all four languages (API_SPEC Section 8): temporary cost `eta_j * n^(1+alpha)` with per-interval liquidity weights, linear permanent impact and an optional adverse drift. `alpha = 1/2` is the square-root law; `alpha = 1` reproduces Almgren-Chriss exactly (asserted).
- Market calibration `eta = Y * sigma / ADV^alpha`: a one-day TWAP costs exactly the square-root-law figure.
- Pinned optimal-schedule solver: Newton with a tridiagonal Hessian and Armijo line search; when drift makes it optimal to finish early, horizon bisection plus a KKT certificate. Cross-checked against SciPy in the Python suite.
- Illustrative US-equity intraday volume curve, VWAP schedule and a pre-trade report (TWAP, VWAP, optimal per risk aversion, square-root-law estimate). Every demo prints it.
- Eight impact golden cases in `data/golden/impact_golden.json`.
- Test counts: Python 174, C++ 91, Rust 78, Java 85 (428 in total).

### Market units and repositioning (F1)

- New market-unit layer in all four languages (API_SPEC Section 7): Almgren-Chriss parameters from price, quoted spread, ADV and daily volatility using the paper's own impact calibration; Avellaneda-Stoikov parameters from tick, touch fill rate, halving distance and opening skew; `bps` and `half_spread_cost` reporting helpers.
- Five presets in `data/presets.json` (US large cap, US small cap, US ETF, EUR/USD, the Almgren-Chriss example) and nine market golden cases in `data/golden/market_golden.json`. The paper preset reproduces `params_liquid.json` exactly.
- Every demo prints the presets in basis points.
- Repositioned as an execution and market-making research platform (README, landing page, LEARN introduction), with the roadmap and current limitations stated.
- Test counts: Python 139, C++ 83, Rust 62, Java 77 (361 in total).

### Build and CI (Phase 0+1)

- Continuous integration on Linux, Windows and macOS for all four languages.
- Python is an installable package (`pip install -e python`), linted with ruff and type-checked with mypy.
- C++ builds on MSVC as well as GCC and Clang; GoogleTest is fetched when not installed; tests are optional.
- Java builds with Maven (`mvn -f java/pom.xml verify`) on any OS; package renamed to `io.github.ashjha0.etrade`; Java 17 is the minimum.
- Rust crate metadata completed; `serde_json` is now a dev-dependency only and the demo is an example (`cargo run --example demo`).
- Documentation corrections: Java determinism scoped to one JVM build and platform; the `sinh` overflow motivation fixed (`kappa*T` does not depend on the grid); "1 ulp" claims replaced with "cancellation-free"; the LEARN knowledge-check section renamed "Knowledge check" (now Section 13) and sections renumbered.

## 1.1.0 — 2026-09-06

The review found that mean_fills in the Avellaneda-Stoikov simulation was a logical OR, not a count: the Python reference summed numpy bool arrays, so a step where both bid and ask filled counted once, and the three ports had copied the OR to match the golden. That undercounted fills by about 10% on the liquid set (77.2 vs 85.5). The fix is pinned in API_SPEC §2.2, the golden as_sim_liquid.mean_fills was regenerated (77.18 → 85.53), and a saturated-book test (P(fill) = 1 both sides ⇒ fills = 2·n_steps exactly) plus an independent scalar re-implementation of the loop keep the OR from returning. The same pass replaced the overflow-prone sinh ratio and the cancellation-prone acosh kappa, added the 10-million size bounds (previously a MemoryError / float-to-int UB / capacity abort / NegativeArraySizeException across the four ports) and made simulate_is enforce the liquidation check. Still by design: A-S fill probability is the exact thinned 1 − exp(−λdt), so mean PnL sits ~12% below the paper's Table 1 while the PnL standard deviations agree.
