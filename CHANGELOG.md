# Changelog

All notable changes to this project. Dates are UTC.

## Unreleased

- Continuous integration on Linux, Windows and macOS for all four languages.
- Python is an installable package (`pip install -e python`), linted with ruff and type-checked with mypy.
- C++ builds on MSVC as well as GCC and Clang; GoogleTest is fetched when not installed; tests are optional.
- Java builds with Maven (`mvn -f java/pom.xml verify`) on any OS; package renamed to `io.github.ashjha0.etrade`; Java 17 is the minimum.
- Rust crate metadata completed; `serde_json` is now a dev-dependency only and the demo is an example (`cargo run --example demo`).
- Documentation corrections: Java determinism scoped to one JVM build and platform; the `sinh` overflow motivation fixed (`kappa*T` does not depend on the grid); "1 ulp" claims replaced with "cancellation-free"; LEARN Section 12 renamed "Knowledge check" and sections renumbered.

## 1.1.0 — 2026-09-06

The review found that mean_fills in the Avellaneda-Stoikov simulation was a logical OR, not a count: the Python reference summed numpy bool arrays, so a step where both bid and ask filled counted once, and the three ports had copied the OR to match the golden. That undercounted fills by about 10% on the liquid set (77.2 vs 85.5). The fix is pinned in API_SPEC §2.2, the golden as_sim_liquid.mean_fills was regenerated (77.18 → 85.53), and a saturated-book test (P(fill) = 1 both sides ⇒ fills = 2·n_steps exactly) plus an independent scalar re-implementation of the loop keep the OR from returning. The same pass replaced the overflow-prone sinh ratio and the cancellation-prone acosh kappa, added the 10-million size bounds (previously a MemoryError / float-to-int UB / capacity abort / NegativeArraySizeException across the four ports) and made simulate_is enforce the liquidation check. Still by design: A-S fill probability is the exact thinned 1 − exp(−λdt), so mean PnL sits ~12% below the paper's Table 1 while the PnL standard deviations agree.
