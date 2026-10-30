# Contributing

`etrade` is one library written four times: a Python reference and C++26,
Rust and Java 25 ports. They agree because each implements the same pinned
contract in [`API_SPEC.md`](API_SPEC.md) and each is checked against the same
golden files in `data/golden/`. A change is finished when all four ports pass
those goldens on Linux, Windows and macOS in CI.

## Local toolchains

| language | version | build and test (from the repo root) |
|---|---|---|
| Python | 3.11+ (CI: 3.11 and 3.13) | `pip install -e "python[dev]"`, then `ruff check . && ruff format --check .`, `mypy` (in `python/`), `pytest -q python` |
| C++ | C++26, CMake 3.25+; GCC 14+, recent Clang, MSVC (`/std:c++latest`), LLVM Clang on macOS if AppleClang lacks C++26 mode; only C++26 features all four compilers implement | `cmake -S cpp -B build -DETRADE_WERROR=ON`, `cmake --build build --config Release`, `ctest --test-dir build -C Release` |
| Rust | MSRV 1.78, stable for development | in `rust/`: `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test --release` |
| Java | JDK 25 (latest LTS; CI on Temurin 25), Maven 3.9 | in `java/`: `mvn -B verify` (compiles with `-Xlint:all -Werror`) |

GoogleTest is taken from the system when CMake finds it, otherwise fetched at
a pinned hash. The Rust MSRV job resolves dependencies with
`CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=fallback cargo update` and then
builds on 1.78; do the same before adding a Rust dependency.

On Windows with an antivirus scanner, `cargo` can fail with `LNK1104` while a
test binary is locked; building with `-j1` or deleting
`rust/target/release/deps/*.exe` clears it.

Optional tooling, none of which the default builds need:

- `pre-commit install` runs the hooks in `.pre-commit-config.yaml`
  (ruff, ruff-format, cargo fmt, whitespace and YAML checks).
- Benchmarks: see [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).
- Coverage: `pip install -e "python[coverage]"` then `pytest --cov=etrade`
  in `python/`; `cargo llvm-cov` in `rust/`; `-DETRADE_COVERAGE=ON` with
  GCC or Clang plus `lcov`; `mvn -Pcoverage verify` in `java/`. The
  `Coverage` workflow runs all four and uploads the reports as artifacts.

## Adding a module in all four languages

Work in this order; each step is the input of the next.

1. **Spec section.** Add a numbered section to `API_SPEC.md`: model,
   pinned algorithm (loop order, tolerances, iteration caps), size bounds
   (Section 0), units, error messages (Section 6 style, each naming the
   argument), and the per-language names. Anything a port could reasonably
   do differently must be pinned here, or the goldens will not agree.
2. **Python reference.** Implement it in `python/src/etrade/<module>.py`
   and export the public names from `python/src/etrade/__init__.py`.
3. **Generator.** Add `data/generate_<module>.py`, which imports the Python
   reference and writes `data/golden/<module>_golden.json` as a list of
   `{"name", "kind", "inputs", "expect", "tol"}` cases (see
   `data/generate_impact.py`). Cover the paper's worked example, the limits
   that reduce to an existing module, and the edge cases of the validity
   checks. Re-running the generator must reproduce the file byte for byte.
4. **Goldens.** Commit the generated JSON. Never edit it by hand; change
   the generator and regenerate.
5. **Tests, per language,** each reading the same golden file and asserting
   every case within its `tol`, plus validation tests for every error
   message, plus a guard that fails when the file gains a case no test
   handles:
   - Python: `python/tests/test_<module>.py`
   - C++: `cpp/tests/test_<module>.cpp` (golden path from `ETRADE_DATA_DIR`)
   - Rust: `rust/tests/<module>.rs` (path from `CARGO_MANIFEST_DIR`)
   - Java: `java/src/test/java/io/github/ashjha0/etrade/<Module>Test.java`
6. **Registration lines,** which are easy to forget:
   - Python: the names in `__init__.py` (and `__all__`).
   - C++: `include/etrade/<module>.hpp`, `src/<module>.cpp` in
     `add_library(etrade ...)` and `tests/test_<module>.cpp` in
     `add_executable(etrade_tests ...)` in `cpp/CMakeLists.txt`.
   - Rust: `pub mod <module>;` (and any `pub use`) in `rust/src/lib.rs`.
   - Java: a public class in `io.github.ashjha0.etrade`; Maven and Surefire
     pick up new sources and `*Test` classes on their own.
7. **Docs and changelog:** a page or section in the docs site, a
   `CHANGELOG.md` entry, and demo lines where the module has a natural use.

Run all four test suites locally before pushing; CI runs the same commands.

## Branches and pull requests

Work is organised in rounds. Parallel pieces of a round each go on their own
branch from `main`, and the round lands as **one** integration pull request
that merges those branches, so `main` moves once per round with all four
ports in agreement. Do not open a pull request per piece unless the round's
coordinator asks for one.

Commits are small and say what changed and why. The pull request template
lists the checks a reviewer expects to have been run.

## Writing style

Docs are written for a careful engineer: measured numbers with their source
and conditions (machine, toolchain, parameters), stated assumptions, and
limitations said plainly. Avoid claims a test or a reference does not back.
