# ARCHITECTURE — `etrade`

Design notes for the four parallel implementations (Python reference, C++,
Rust, Java) of the Almgren-Chriss / Avellaneda-Stoikov / analytics stack.
The normative contract is `../API_SPEC.md`; this document explains *why* it
is shaped the way it is.

## 1. Component responsibilities

Three independent modules per language, mirrored one-to-one across ports:

| Module | Owns | Depends on |
|---|---|---|
| `almgren_chriss` | `ACParams` validation (incl. size bounds), `kappa` (`log1p` form), `exp/expm1`-form sinh trajectory with exact TWAP branch, trade lists, exact discrete E/Var sums, TWAP/POV/instantaneous baselines, efficient frontier, Monte Carlo IS check | RNG (MC only) |
| `avellaneda_stoikov` | `ASParams` validation, reservation price / optimal spread / quotes closed forms, intensity and fill probability, the pinned path simulation with inventory cap, stress vol and terminal liquidation, per-path and aggregate statistics | RNG (simulation only) |
| `analytics` | Perold IS decomposition (buy/sell), participation rate, native closed-form OLS, pinned RNG-free synthetic impact fills | nothing stochastic |
| `market` | conversion from market units (price, spread, tick, ADV, daily vol, touch fill rate, halving distance, opening skew) to `ACParams`/`ASParams`; bps and half-spread reporting helpers (API_SPEC Section 7) | `almgren_chriss`, `avellaneda_stoikov` (parameter types only) |

The three model modules do not import each other; `market` only builds their parameter objects. Parameter objects are immutable value
types (frozen dataclasses in Python, plain structs/records elsewhere) that
validate **at construction**, so every downstream function can assume valid
inputs and stay branch-free. Functions that take user data (holdings paths,
grids, quote state, volumes) validate it at entry — finiteness, lengths,
liquidation, and the pinned size bounds of `API_SPEC.md` Section 0, checked
*before* any cast or allocation so that an absurd request (a POV clip of
1e-9 shares, `usize::MAX` paths) is an input error rather than undefined
behaviour, an integer overflow, an out-of-memory abort or a panic. The only
shared state anywhere is the RNG inside a single simulation call.

### Data flow

```mermaid
flowchart TB
    subgraph DATA["data/ (shared, committed)"]
        PJ["params_liquid.json<br/>params_illiquid.json<br/>params_fx.json"]
        PO["parent_order.json<br/>(X, lambda grid, IS fills)"]
        GEN["generate_data.py<br/>(seeded, asserts AC closed forms vs MC)"]
        GOLD["golden/golden.json<br/>19 cases: exact + statistical"]
        GEN --> PJ
        GEN --> PO
        GEN --> GOLD
    end

    subgraph CORE["etrade core (identical in Python / C++ / Rust / Java)"]
        ACP["ACParams<br/>validate: eta_tilde > 0, X >= 0,<br/>N <= MAX_AC_INTERVALS, ..."]
        AC["almgren_chriss<br/>kappa, trajectory, trade list,<br/>E/Var sums, frontier,<br/>TWAP / POV / instant, MC check"]
        ASP["ASParams<br/>validate: gamma > 0, T/dt integral,<br/>T/dt <= MAX_AS_STEPS, ..."]
        AS["avellaneda_stoikov<br/>reservation price, spread, quotes,<br/>intensity, fill prob,<br/>pinned simulation loop"]
        AN["analytics<br/>IS decomposition, participation,<br/>native OLS impact regression"]
        ACP --> AC
        ASP --> AS
    end

    PJ --> ACP
    PJ --> ASP
    PO --> AC
    PO --> AN

    subgraph OUT["outputs"]
        DEMO["demo report<br/>(trajectories, frontier, baselines,<br/>A-S vs fixed, IS, impact fit)"]
        TESTS["4 language test suites<br/>exact cases: 1e-10..1e-12<br/>statistical cases: 4-SE band"]
    end

    AC --> DEMO
    AS --> DEMO
    AN --> DEMO
    AC --> TESTS
    AS --> TESTS
    AN --> TESTS
    GOLD --> TESTS
```

`generate_data.py` is the single source of truth for the bundled JSON: it
writes the parameter sets, the parent order, and the golden values (after
asserting the AC closed forms against its own Monte Carlo), all seeded.
Ports never regenerate data; they only consume it.

### Module surface

```mermaid
classDiagram
    class ACParams {
        +float total_shares
        +int n_intervals
        +float tau
        +float sigma
        +float eta
        +float gamma_p
        +float lambda_ra
        +horizon() float
        +eta_tilde() float
        +with_lambda(lambda_ra) ACParams
    }
    class almgren_chriss {
        +ac_kappa(p) float
        +half_life(p) float
        +characteristic_time(p) float
        +ac_trajectory(p) array
        +ac_trade_list(p) array
        +twap_holdings(X, N) array
        +pov_holdings(X, tau, V, p) array
        +expected_cost(p, holdings) float
        +variance_cost(p, holdings) float
        +instantaneous_cost(p) tuple
        +efficient_frontier(p, lambdas) FrontierPoint[]
        +simulate_is(p, holdings, n_paths, seed) ISSimResult
    }
    class FrontierPoint {
        +float lambda_ra
        +float kappa
        +float e_cost
        +float var_cost
        +float std_cost
    }
    class ISSimResult {
        +float mean
        +float var
        +float se_mean
        +float se_var
        +int n_paths
    }
    class ASParams {
        +float s0
        +float sigma
        +float horizon
        +float dt
        +float gamma
        +float k
        +float a_rate
        +int q_max
        +float eta_liq
        +n_steps() int
    }
    class avellaneda_stoikov {
        +reservation_price(p, s, q, t) float
        +optimal_spread(p, t) float
        +quotes(p, s, q, t) tuple
        +intensity(p, delta) float
        +fill_probability(p, delta) float
        +average_as_spread(p) float
        +simulate(p, n_paths, seed, agent, ...) ASSimResult
    }
    class ASSimResult {
        +array pnl
        +array rms_q_path
        +float mean_pnl
        +float std_pnl
        +float sharpe
        +float mean_abs_q
        +float mean_max_abs_q
        +float rms_q
        +float mean_fills
        +bool cap_violated
        +se(stat) float
    }
    class analytics {
        +decompose_shortfall(side, Pd, P0, PT, X, qtys, prices) ISDecomposition
        +participation_rate(exec, mkt) float
        +ols(x, y) OLSResult
        +synthetic_impact_fills(n, a, b, amp) tuple
        +estimate_impact(parts, costs) OLSResult
    }
    class ISDecomposition {
        +float total
        +float delay
        +float trading
        +float opportunity
        +float executed_qty
        +float remaining_qty
    }
    class OLSResult {
        +float slope
        +float intercept
        +float r_squared
        +int n_obs
    }

    almgren_chriss ..> ACParams : validates on construction
    almgren_chriss --> FrontierPoint
    almgren_chriss --> ISSimResult
    avellaneda_stoikov ..> ASParams : validates on construction
    avellaneda_stoikov --> ASSimResult
    analytics --> ISDecomposition
    analytics --> OLSResult
```

## 2. The pinned A-S simulation loop

Cross-language reproducibility of a *stochastic* simulation is the hardest
part of this project. The loop below is pinned to the operation: any
reordering (quoting after the mid move, drawing `u_bid` before `u_ask`,
skipping a suppressed side's draw) produces statistically different — or
subtly stream-forked — results.

```mermaid
sequenceDiagram
    participant Sim as Simulator (per step i, t = i*dt)
    participant Agent as Quoting agent ("as" / "fixed")
    participant RNG as RNG (pinned draw order)
    participant Mkt as Market state (s, q, cash)

    Note over Sim,Mkt: one step of the pinned loop (Section 2.2 of API_SPEC)
    Sim->>Agent: 1. quote(s, q, T - t)
    Agent-->>Sim: r, spread  (as: r = s - q*gamma*sigma^2*(T-t),<br/>fixed: r = s, fixed spread)
    Sim->>Sim: ask = r + spread/2, bid = r - spread/2
    Sim->>Sim: cap rule: quote bid iff q < q_max, ask iff q > -q_max
    Sim->>RNG: 2. draw u_ask, then u_bid (ALWAYS both, even if a side suppressed)
    RNG-->>Sim: u_ask, u_bid ~ U(0,1)
    Sim->>Sim: side fills iff quoted and u < 1 - exp(-A*exp(-k*delta)*dt)
    Sim->>Mkt: 3. ask fill: q -= 1, cash += ask, fills += 1 --- bid fill: q += 1, cash -= bid, fills += 1
    Sim->>RNG: 4. draw z ~ N(0,1)
    RNG-->>Sim: z
    Sim->>Mkt: s += sigma_t*sqrt(dt)*z  (sigma_t stressed for t >= stress_time_frac*T)
    Note over Sim,Mkt: after last step: PnL = cash + q*S_T - eta_liq*|q|
```

Design points:

- **Always-draw rule.** Both uniforms are consumed every step regardless of
  the cap state, so the random stream is a pure function of
  `(seed, n_paths, n_steps)` and does not fork the first time a path hits
  `q_max`. This makes the A-S vs fixed-spread comparison a controlled
  experiment on the *same* randomness per (seed, agent).
- **Cap by construction.** The bid is suppressed at `q >= q_max`, the ask at
  `q <= -q_max`, so `|q| <= q_max` is an invariant, and the simulation still
  *asserts* it every step (`cap_violated` flag) — belt and braces, and a
  cheap tripwire for port bugs.
- **Stress as misspecification.** `stress_factor` scales only the realised
  `sigma_t` in step 4; the agent's formulas keep the base sigma. This
  isolates "the world changed and the model didn't notice" from "the model
  was re-calibrated".
- **Vectorisation across paths.** Python draws per-step *vectors* across
  paths (`u_ask` vector, `u_bid` vector, `z` vector) — ~200 numpy ops per
  simulated step instead of a million Python-level iterations. Native ports
  loop per path with the same *logical* per-step draw order; that is exactly
  why cross-language agreement is defined statistically, not bitwise.

## 3. Numerical design decisions and trade-offs

1. **Exact discrete sums as the single code path.** `E[IS]` and `Var[IS]`
   are always evaluated through
   `(1/2)gamma_p X^2 + (eta_tilde/tau) sum n_j^2` and
   `sigma^2 tau sum x_j^2`. For the optimal trajectory these coincide with
   the paper's hyperbolic closed forms (evaluated with the discrete kappa)
   to machine precision — rel. diff 2e-16 on the liquid set, asserted at
   1e-12 by every suite as a literature check. The sums are pinned because
   one code path covers optimal, TWAP, POV and hand-built schedules, so
   there is one set of invariants to test. Only the *continuous* kappa
   substituted into the closed forms is an approximation.
2. **`log1p` form for kappa.** `kappa = arccosh(1 + x)/tau` with
   `x = kappa_tilde^2 tau^2/2` is the exact root of the discrete
   characteristic equation — no iterative solve, no tolerance to pin. It is
   evaluated as `log1p(x + sqrt(x(2 + x)))/tau`, which is the same function
   but free of cancellation for small `x` (a few ulp); the literal `acosh(1 + x)` loses
   `~eps/x` relative digits by cancellation, i.e. it degrades *worst*
   exactly as `tau -> 0` (measured: `x = 2e-9` gives 1.2e-8 relative error,
   above the 1e-10 golden tolerance; `x = 5e-16` gives 6%). Golden case
   `ac_kappa_small_tau` pins this.
3. **`exp/expm1` form for the sinh ratio.** `sinh(kappa(T-t_j))/sinh(kappa T)`
   overflows to `inf/inf = NaN` at `kappa T > 710` — a very urgent schedule or a
   very long horizon (`kappa T` is essentially grid-independent at fixed `T`). It is evaluated as
   `exp(-kappa t_j) expm1(-2 kappa (T-t_j)) / expm1(-2 kappa T)`, identical
   for every `kappa > 0`, finite for every valid input, with the ratio formed
   first so `x_0 == X` exactly. Far out on a very urgent schedule `exp`
   underflows and `x_j` is exactly 0 — correct, and documented.
4. **Explicit TWAP limit branch.** The ratio is 0/0 at `kappa = 0`, so
   `kappa*T < KAPPA_T_LIMIT = 1e-6` switches to the exact linear schedule.
   The threshold is pinned in the spec (a port choosing 1e-8 would still
   pass goldens today, but the branch point is observable in principle, so
   it is contract, not implementation detail). Continuity across the branch
   is a dedicated test.
5. **`x_N = 0` exactly.** The last holding is forced to zero after
   evaluating either branch, so "fully liquidates" is exact rather than
   epsilon-true, and the liquidation check `|x_N| <= 1e-9 max(1, X)` on
   user-supplied schedules (in `expected_cost` *and* `simulate_is`) never
   trips on our own output.
6. **`log1p` / `expm1` everywhere they matter.** The A-S markup
   `(2/gamma) ln(1 + gamma/k)` and fill probability `1 - exp(-lambda dt)`
   are computed as `log1p` and `-expm1`; with FX-scale inputs
   (`gamma/k = 0.05`, `lambda*dt ~ 1e-3`) naive forms lose enough digits to
   break 1e-12 goldens. The exact thinned probability is also a documented
   divergence from the paper's first-order `lambda dt` (see LEARN 6.4).
7. **Fills are counted per side.** `mean_fills` is the total number of
   fills (a both-sides step counts two). The original Python reference
   summed numpy *bool* arrays — a logical OR — and the ports had copied the
   OR to match the golden; that undercounted by ~10% on the liquid set
   (77.2 vs 85.5). The fix is pinned in `API_SPEC.md` 2.2, the golden was
   regenerated, and a saturated-book test (P(fill) = 1 on both sides →
   fills = 2·n_steps exactly) plus an independent scalar re-implementation
   of the loop in the Python suite keep the OR from coming back.
8. **Pinned size bounds.** `MAX_AC_INTERVALS`, `MAX_POV_INTERVALS`,
   `MAX_AS_STEPS`, `MAX_PATHS` (all 10 million) are checked on the *double*
   ratio before any `ceil`/cast/`new`/`Vec::with_capacity`. Without them
   `pov_holdings(1e6, 1, 1, 1e-9)` was a `MemoryError` in Python,
   float→int UB in C++, a capacity-overflow abort in Rust and a
   `NegativeArraySizeException` in Java.
9. **ddof = 1 sample moments, pinned SE formulas.** SE of a per-path-mean
   statistic is `sample std / sqrt(M)`; `SE(std_pnl)` and `SE(var)` use the
   normal approximation (`std/sqrt(2(M-1))`, `var*sqrt(2/(M-1))`). Pinning
   the SE formulas is what makes the 4-SE acceptance band well defined
   across languages.
10. **Money in whole currency units, shares in floats.** No basis-point or
    per-share normalisation inside the core: E and Var come out in currency,
    letting the golden values assert the raw sums. Reporting layers (demo)
    do the pretty-printing.
11. **RNG policy.** Determinism is per-language, agreement is statistical.
    Python: `numpy.random.default_rng(seed)` (PCG64, stream versioned by
    NumPy); C++: `std::mt19937_64` with `std::uniform_real_distribution` /
    `std::normal_distribution` — deterministic per standard library only,
    since the distribution algorithms are implementation-defined (libstdc++,
    libc++ and MSVC differ for the same seed); Rust:
    `rand_chacha::ChaCha12Rng::seed_from_u64` pinned explicitly (the `rand`
    crate documents `StdRng` as free to change between releases); Java:
    `SplittableRandom` (`nextDouble` is a specified algorithm; `nextGaussian`
    comes from the `RandomGenerator` default since Java 17 and is not
    specified, and `Math.*` may differ by 1 ulp across platforms, so Java
    determinism is guaranteed within one JVM build and platform only).
    Each language tests "same seed twice = identical results"; the golden
    file stores Python's 20k-path references *with standard errors* so any
    port with `M >= 2000` paths can verify
    `|stat - ref| <= 4 sqrt(se_ref^2 + se_own^2)`.

## 4. Error-handling strategy per language

All input violations are detected at parameter construction or function
entry, reported with a specific message, and never clamped or deferred:

| Language | Mechanism | Notes |
|---|---|---|
| Python | `raise ValueError(msg)` | `_require(cond, msg)` helper; dataclass `__post_init__` validation |
| C++ | `throw std::invalid_argument` | `std::domain_error` specifically for the `gamma_p*tau >= 2*eta` model-validity failure |
| Rust | `Result<_, EtradeError>` | manual thiserror-style enum; no panics on bad input; `?` propagation in demo/tests |
| Java | `throw IllegalArgumentException(msg)` | checked eagerly in constructors and static entry points |

The checked conditions are enumerated in `API_SPEC.md` Section 6 and are
identical in the four ports: NaN/inf anywhere (parameters, holdings paths,
grids, quote state `s`/`q`, volumes, `fixed_spread`, stress settings),
sign/range violations, empty grids, non-liquidating schedules (for `E` and
the MC check alike), participation above 1, integer-typed arguments that
are not integers (Python rejects `bool` and accepts NumPy integers), and
the size bounds of Section 0. Just as important is the pinned list of
**non-errors**: negative prices (arithmetic walk), `kappa = 0` (half-life
reported as infinity), signed quote distances (`delta < 0` when skew pushes
a quote through the mid), `q_max = 0` (quotes nothing, PnL identically
zero), `x_j` underflowing to exactly 0 far out on a very urgent schedule,
and a participation rate of exactly 1. Turning any of these into an
exception is a port bug that the tests catch.

## 5. Testing strategy

Five layers, replicated per language (Python 139 tests, C++ 83, Rust 62,
Java 77, 361 in total):

1. **Golden values** (`data/golden/golden.json`, 19 cases; every suite
   asserts the count exactly and that every case is handled).
   *Exact* cases (tol 1e-10..1e-12, relative for large magnitudes via
   `tol*max(1,|expected|)`) pin the closed forms: AC solutions on two
   parameter sets, `kappa` at millisecond `tau` (`ac_kappa_small_tau`),
   TWAP/POV/instant costs, a frontier point, the `lambda_ra = 0` = TWAP
   identity, A-S reservation/spread/fill probability, both IS
   decompositions, participation, and the impact regression on the pinned
   RNG-free synthetic fills.
   *Statistical* cases carry the Python 20,000-path reference plus SE per
   statistic; ports compare within the combined 4-SE band using their own
   RNG.
2. **Property-style invariants** (grid loops): optimal sell trajectory
   monotone decreasing and summing to X (1e-12); higher `lambda_ra` gives
   front-loaded schedules, higher E, lower Var (frontier monotonicity);
   trade list matches the cosh identity; sinh/TWAP branch continuity near
   the threshold; A-S spread positive and decreasing to the terminal floor;
   reservation price below mid when long; inventory cap invariant on every
   step; A-S rms inventory below fixed-spread rms inventory with margin;
   IS components summing to the total exactly.
3. **Literature and numerics**: the paper's closed-form E and V equal the
   discrete sums at 1e-12 on all three parameter sets; the trajectory is
   finite, monotone and sums to X at `kappa T ~ 846` and `~2683` (where
   `sinh` overflows), with `x_1/X = exp(-kappa tau)` to 1e-9; `kappa`
   matches the series `kappa_tilde(1 - kappa_tilde^2 tau^2/24)` to 1e-9 at
   `tau = 1e-3`; the `expm1` form equals the textbook sinh ratio to 1e-14
   at moderate `kappa T`; POV interval counts at the `ceil` boundary
   (`X/clip = 2` exactly → 2, one share-thousandth more → 3).
4. **Monte Carlo vs closed forms**: AC sample mean/variance within 4 SE of
   the exact sums for the optimal *and* the TWAP schedule in every
   language; the A-S comparison run on a reduced path count in tests.
5. **Error paths and determinism**: every validation rule of Section 6 has
   a negative test (non-finite inputs, size bounds, non-liquidating
   schedules, participation above 1, non-finite quote state, NaN/inf
   `fixed_spread` and stress settings, the `T/dt` integrality edge cases
   `dt = T/3` and `dt = T`); stress with `stress_time_frac = 1` or
   `stress_factor = 1` is bitwise identical to the unstressed run; every
   simulation is run twice with the same seed and asserted
   bitwise-identical within the language; the A-S fill count is checked
   against a saturated book (exactly `2·n_steps`) and, in Python, against an
   independent scalar re-implementation of the pinned loop.

## 6. Performance notes

- Closed-form AC quantities are O(N) with trivial constants; the frontier
  sweep is O(N * grid) — microseconds. Nothing to optimise.
- The A-S simulation dominates runtime: O(n_paths * n_steps) with two
  uniforms, one normal, and a handful of exp calls per path-step. Python
  vectorises across paths; native ports loop per path; none parallelise —
  determinism and portability outrank speed here. No timing is asserted or
  enforced anywhere; as a rough indication only, on a shared 2-core box the
  Python demo (three 2,000-path A-S runs plus a 20,000-path AC check) runs
  in a few seconds and each language's test suite in well under a minute.
- `exp` is the hot function in the A-S loop. The intensity for a suppressed
  side is still computed (branch-free style) in the reference; ports may
  skip it *only* where it cannot change results or draw order.
- Memory: per-path aggregates are accumulated on the fly (sums of |q|,
  q^2, running max), so the simulation stores O(n_paths) state, not
  O(n_paths * n_steps) paths. The MC IS check allocates its
  `(n_paths, N)` normal block up front in Python purely for draw-order
  pinning; native ports draw per step and stay O(n_paths).
