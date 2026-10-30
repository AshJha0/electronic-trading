# Security policy

## Scope

`etrade` is a research library: it computes execution schedules, market-making
quotes and cost analytics from parameters you supply. It does not connect to
exchanges or brokers, place orders, handle credentials or open network
sockets. The issues most likely to matter are therefore:

- input handling: an input that causes undefined behaviour, a crash, an
  integer overflow, unbounded memory use or a hang instead of the input error
  the spec (`API_SPEC.md`, Sections 0 and 6) requires;
- the build and supply chain: a dependency, fetched archive or CI workflow
  that could be tampered with.

Numerical disagreement between ports, or with a paper, is a bug rather than a
security issue; please open a normal issue for it.

## Supported versions

Only the latest release on `main` receives fixes.

## Reporting a vulnerability

Please report privately through GitHub's
[private vulnerability reporting](https://github.com/AshJha0/electronic-trading/security/advisories/new)
rather than a public issue. Include the language port, version or commit, a
minimal input that reproduces the problem and what you observed.

You should get an acknowledgement within a week. Once a fix is ready it is
released with a note in `CHANGELOG.md`, crediting the reporter unless they
prefer otherwise.

## What the repository does to reduce risk

- Size bounds are checked before any allocation or cast (`API_SPEC.md`
  Section 0); the C++ port is tested under AddressSanitizer and UBSan, and
  the Rust crate forbids `unsafe` code.
- Fetched C++ dependencies are pinned by SHA-256; `rust/Cargo.lock` is
  committed; GitHub Actions in the coverage and CodeQL workflows are pinned
  by commit SHA.
- Dependabot proposes weekly updates for pip, cargo, Maven and Actions, and
  CodeQL scans Python, C++, Java and Rust.
