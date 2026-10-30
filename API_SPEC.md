# `etrade` — API contract (pinned, cross-language)

This document is the normative contract for all ports (Python `etrade`,
C++ `etrade::`, Rust crate `etrade`, Java `io.github.ashjha0.etrade`).  Every formula,
evaluation form, branch threshold, size bound, loop order and golden tolerance
below is pinned; ports must reproduce them exactly (up to floating point)
except where the RNG note says otherwise.  The Python implementation under
`python/src/etrade/` is the reference.

Notation: all prices are arithmetic (they may go negative — a documented model
property of both models, never an error).

---

## 0. Pinned size bounds and units

Every public entry point bounds its sizes **before** any cast or allocation,
so an absurd input is an input error (Section 6) and never undefined
behaviour, an integer overflow, an out-of-memory abort or a panic:

| constant | value | applies to |
|---|---|---|
| `MAX_AC_INTERVALS` | 10,000,000 | `ACParams.n_intervals`, `twap_holdings(n_intervals)` |
| `MAX_POV_INTERVALS` | 10,000,000 | `pov_holdings`: `X/clip` (before the `ceil`) |
| `MAX_AS_STEPS` | 10,000,000 | `ASParams`: `horizon/dt` |
| `MAX_PATHS` | 10,000,000 | `n_paths` of `simulate_is` and `simulate` |
| `MAX_IMPACT_INTERVALS` | 10,000 | `ImpactParams.n_intervals` (Section 8.1) |

Names per language: Python `et.MAX_AC_INTERVALS` etc.; C++
`etrade::kMaxAcIntervals`, `kMaxPovIntervals`, `kMaxPaths`, `kMaxAsSteps`,
`kMaxAsPaths`; Rust `etrade::MAX_AC_INTERVALS`, `MAX_POV_INTERVALS`,
`MAX_PATHS`, `MAX_AS_STEPS`, `MAX_AS_PATHS`; Java
`AlmgrenChriss.MAX_AC_INTERVALS`, `MAX_POV_INTERVALS`, `MAX_PATHS`,
`AvellanedaStoikov.MAX_AS_STEPS`, `MAX_AS_PATHS`. Python has no separate
`MAX_AS_PATHS`: `simulate` uses `MAX_PATHS`, which has the same value.
`MAX_IMPACT_INTERVALS` is `etrade.impact.MAX_IMPACT_INTERVALS` (Python),
`etrade::kMaxImpactIntervals` (C++), `etrade::impact::MAX_IMPACT_INTERVALS`
(Rust) and
`Impact.MAX_IMPACT_INTERVALS` (Java).

**Product bounds (pinned).** The per-axis bounds above still admit
`MAX_PATHS * MAX_AC_INTERVALS = 1e14` units of work, so the two Monte Carlo
entry points also bound the product, checked after the per-axis bounds and
before any allocation or draw:

| constant | value | applies to |
|---|---|---|
| `MAX_MC_WORK` | 1,000,000,000 | `simulate_is`: `n_paths * N` (`N` = intervals of the schedule simulated) |
| `MAX_AS_WORK` | 10,000,000,000 | `simulate` (Section 2): `n_paths * n_steps` |

Violation is the input error `"n_paths * n_intervals must be <= 1000000000"`
or `"n_paths * n_steps must be <= 10000000000"`. The product is formed in
64-bit integer (or exact double) arithmetic, so it cannot overflow. Names:
Python `et.MAX_MC_WORK`, `et.MAX_AS_WORK`; C++ `etrade::kMaxMcWork`,
`kMaxAsWork`; Rust `etrade::MAX_MC_WORK`, `MAX_AS_WORK`; Java
`AlmgrenChriss.MAX_MC_WORK`, `AvellanedaStoikov.MAX_AS_WORK`.

Units are abstract but must be consistent within a parameter set; nothing
ties `tau`/`dt` to a clock or calendar (no day-count, no holidays). The
intraday volume curve of Section 8.3 is a vector of per-interval weights,
not a clock. Sections 7 and 8 fix the time unit to one trading day:

| parameter | unit |
|---|---|
| `tau`, `horizon`, `dt`, `t` | time unit (the bundled sets use 1 = one trading day) |
| `sigma` (AC) | currency per share per sqrt(time unit) |
| `eta` | currency × time unit / share² (temporary impact `eta*v`, `v` in shares per time unit) |
| `gamma_p` | currency / share² (permanent impact `gamma_p*n`) |
| `lambda_ra` | 1 / currency |
| `sigma` (A-S) | price per sqrt(time unit) |
| `gamma`, `k` | 1 / price |
| `a_rate` (A) | fills per time unit |
| `eta_liq` | price per unit of inventory |
| shares / inventory | fractional shares in AC; whole units (one per fill) in A-S |

---

## 1. Almgren-Chriss discrete optimal execution (2000 paper)

### 1.1 Setup

Sell `X >= 0` shares over `N >= 1` intervals of length `tau > 0`; horizon
`T = N*tau`; times `t_j = j*tau`.  Holdings `x_0 = X, ..., x_N = 0`; trade
list `n_j = x_{j-1} - x_j` for `j = 1..N`.

Price dynamics (arithmetic random walk, linear impacts):

```
S_j    = S_{j-1} + sigma*sqrt(tau)*xi_j - gamma_p*n_j        (permanent, g(v)=gamma_p*v)
S~_j   = S_{j-1} - eta*n_j/tau                               (temporary, h(v)=eta*v)
IS     = X*S_0 - sum_{j=1}^{N} n_j * S~_j                    (implementation shortfall)
```

`IS` is independent of `S_0`; the reference simulation uses `S_0 = 0`.

The paper's fixed cost term `epsilon * sum_j |n_j|` (half-spread and fees;
`epsilon = 0.0625` in the paper's own example) is **deliberately omitted**:
for a one-sided program it equals `epsilon*X` for every schedule, so it does
not change the optimal trajectory.  Add it to `E[IS]` yourself if you report
total cost.

### 1.2 Adjusted temporary impact and validity

```
eta_tilde = eta * (1 - gamma_p*tau/(2*eta)) = eta - gamma_p*tau/2
```

**Validity check (pinned):** the model requires `gamma_p*tau < 2*eta`
(equivalently `eta_tilde > 0`).  Violation — including equality — is an input
error (Section 6), raised at parameter-construction time.

### 1.3 Exact discrete moments (pinned sums)

For ANY fully liquidating schedule (`x_N = 0`, enforced with tolerance
`|x_N| <= 1e-9 * max(1, X)`):

```
E[IS]   = (1/2)*gamma_p*X^2 + (eta_tilde/tau) * sum_{j=1}^{N} n_j^2
Var[IS] = sigma^2 * tau * sum_{j=1}^{N} x_j^2            (x_N = 0 contributes 0)
```

All E/Var values in golden.json are computed through these sums.  For the
optimal trajectory the sums coincide with the paper's hyperbolic closed
forms (evaluated with the discrete `kappa` of 1.4) to machine precision
(rel. diff ~2e-16, asserted by every suite); the sums are pinned because they
also cover TWAP/POV/arbitrary schedules.

**`X` is taken from the schedule (pinned):** when a holdings path is
supplied, the permanent-impact term uses `x_0` of that path, not
`params.total_shares` — the functions cost *that schedule*.  A supplied path
must be finite, of length `>= 2`, and (for `E` and the MC check) fully
liquidating.

### 1.4 Optimal trajectory (pinned closed form)

Objective `U(x) = E[IS] + lambda_ra * Var[IS]`, `lambda_ra >= 0`.

```
kappa_tilde^2 = lambda_ra * sigma^2 / eta_tilde
kappa         = arccosh(1 + kappa_tilde^2 * tau^2 / 2) / tau
                (i.e. the root of (2/tau^2)*(cosh(kappa*tau)-1) = kappa_tilde^2)
x_j           = X * sinh(kappa*(T - t_j)) / sinh(kappa*T),   j = 0..N
n_j           = x_{j-1} - x_j
              = (2*X*sinh(kappa*tau/2)/sinh(kappa*T)) * cosh(kappa*(T - t_{j-1/2})),
                t_{j-1/2} = (j - 1/2)*tau
```

**Evaluation forms (pinned — ports must use exactly these):**

```
x     = kappa_tilde^2 * tau^2 / 2
kappa = log1p(x + sqrt(x*(2 + x))) / tau          (== arccosh(1 + x)/tau; free of
                                                   cancellation for small x, where
                                                   acosh(1 + x) loses ~eps/x digits)
ratio_j = expm1(-2*kappa*(T - t_j)) / expm1(-2*kappa*T)
x_j     = X * exp(-kappa*t_j) * ratio_j            (== sinh(kappa(T-t_j))/sinh(kappa T);
                                                   no overflow for any kappa*T, whereas
                                                   sinh overflows at kappa*T > ~710)
```

The ratio is formed first so that `x_0 == X` exactly (`1 * 1 * X`).  Every
`x_j` is finite for every valid parameter set; for `kappa*t_j > ~745` the
`exp` underflows and `x_j` is (correctly) `0`.  Implementations compute
`n_j` as the difference of the `x_j`; the cosh form is the identity it must
satisfy.  Set `x_N = 0` exactly (no residual epsilon).

**Small-kappa limit branch (pinned):** if `kappa_tilde^2 == 0` (which includes
`lambda_ra == 0` and `sigma == 0`) then `kappa = 0`; and whenever
`kappa * T < 1e-6` (constant `KAPPA_T_LIMIT = 1e-6`) the trajectory is the
exact linear TWAP limit

```
x_j = X * (N - j) / N
```

with no evaluation of the sinh ratio (no 0/0).  The sinh branch is continuous
at the threshold to well below golden tolerance.

**Infinite-urgency limit (pinned):** if `kappa` is not finite (the
`kappa_tilde^2` or `log1p` evaluation overflows for absurd but finite
inputs), the trajectory is the instantaneous schedule `x_0 = X`,
`x_j = 0` for `j >= 1`, never NaN; `ac_kappa` then returns `+inf` (and
`half_life` 0). `Var[IS]` of a schedule whose holdings after `x_0` are all
zero is exactly 0 even when `sigma^2` overflows.

Special cases: `X = 0` → all-zero trajectory, `E = Var = 0`.  `N = 1` →
`x = [X, 0]`, single clip.

### 1.5 Derived quantities (pinned)

- **Half-life / characteristic time:** `theta = 1/kappa` — the paper's
  e-folding time (the trajectory decays like `exp(-t/theta)`), **not**
  `ln(2)/kappa`; `+inf` when `kappa = 0` (report as infinity, not an error).
  Exposed as `half_life` and, under the paper's name, `characteristic_time`
  (C++ `characteristic_time`, Rust `characteristic_time`, Java
  `characteristicTime`); both return the same value.
- **Efficient frontier:** sweep a grid of `lambda_ra` values; each point
  carries `(lambda_ra, kappa, E, Var, sqrt(Var))` computed per 1.3/1.4.
  Empty grid, or any entry non-finite or negative, is an input error.  Input
  order is preserved; duplicate and unsorted grids are reproduced as given.
- **Instantaneous liquidation:** sell all `X` in the first interval:
  `E = (1/2)*gamma_p*X^2 + eta_tilde*X^2/tau`, `Var = 0`.

### 1.6 TWAP and POV baselines (pinned)

- **TWAP:** `x_j = X*(N-j)/N`; E/Var from the Section 1.3 sums.
- **POV:** inputs market volume rate `V > 0` (shares per unit time) and
  participation `p in (0,1)` (both open endpoints; violation is an input
  error; all inputs finite).  Clip size `c = p*V*tau`.  The schedule sells
  `c` per interval until done, last clip partial: number of intervals
  `M = max(1, ceil(X/c - 1e-12))` for `X > 0` (a remainder below the
  `1e-12` slack still takes one interval), holdings `x_i = max(X - i*c, 0)`
  for `i = 0..M`, `x_M = 0` exactly.  A clip that is not finite (the product
  `p*V*tau` overflowing) is the input error "POV clip
  participation*market_volume_rate*tau must be finite"; a clip that
  underflows to 0 fails the bound below.  **Bound (pinned):** `X/c <= MAX_POV_INTERVALS` is
  checked on the double ratio *before* the `ceil`/cast/allocation; violation
  is the input error "POV clip too small: would need > 10000000 intervals".
  E/Var from the Section 1.3 sums (the horizon is `M*tau`, not `N*tau`).

### 1.7 Monte Carlo IS check

Simulate 1.1 path by path: per path, loop `j = 1..N`: execute
`n_j` at `S_{j-1} - eta*n_j/tau`, then update
`S_j = S_{j-1} + sigma*sqrt(tau)*xi_j - gamma_p*n_j`.  The schedule must
pass the same liquidation check as 1.3 (`|x_N| <= 1e-9*max(1, X)`) — a
partial schedule is an input error, since `-proceeds` would silently ignore
the mark-to-market of the unsold shares.  `2 <= n_paths <= MAX_PATHS`.
Report sample mean, sample variance (ddof = 1), `SE(mean) = sqrt(var/M)`,
`SE(var) = var*sqrt(2/(M-1))`.  Closed forms must match MC within 4 SE (for
the optimal schedule and for TWAP, in every language).

---

## 2. Avellaneda-Stoikov market making (2008 paper, finite horizon)

### 2.1 Closed forms (pinned)

Parameters: `s0`, `sigma >= 0`, horizon `T > 0`, step `dt` with
`0 < dt <= T`, `T/dt` integral (tolerance 1e-9) and `T/dt <= MAX_AS_STEPS`,
risk aversion
`gamma > 0` **strictly** (the `gamma -> 0` limit blows up the utility
derivation; `gamma <= 0` is an input error per spec), intensity parameters
`A > 0`, `k > 0`, hard inventory cap `q_max` (integer `>= 0`), terminal
liquidation penalty `eta_liq >= 0`.

```
reservation price   r(s,q,t) = s - q*gamma*sigma^2*(T - t)
total spread        spread(t) = gamma*sigma^2*(T - t) + (2/gamma)*ln(1 + gamma/k)
quotes              ask = r + spread/2 ,  bid = r - spread/2
distances           delta_a = ask - s ,  delta_b = s - bid      (signed; may be < 0)
arrival intensity   lambda(delta) = A * exp(-k*delta)
per-step fill prob  P(fill) = 1 - exp(-lambda(delta) * dt)      (in [0,1))
```

`t` must lie in `[0, T]` (outside is an input error); `s` and `q` must be
finite.  At `t = T` the spread equals its terminal floor
`(2/gamma)*ln(1 + gamma/k)` and `r = s`.  Fill size is 1 unit per fill per
side per step.

**Fill probability (pinned choice, documented divergence):** the exact
thinned-Poisson `1 - exp(-lambda*dt)` is used, not the first-order
`lambda*dt` of the paper's own simulation (as commonly reproduced).  With
`lambda*dt ~ 0.28` at the paper's parameters the first-order form over-fills
by ~12%, which is why the mean PnL here (~56 on the liquid set) is below the
paper's Table 1 value (~65 at `gamma = 0.1`) while the PnL standard
deviations (6.4 vs 6.4; symmetric strategy 12.8 vs 12.7) agree.  The hard
cap `q_max` and the terminal penalty `eta_liq` are additions to the paper.

Time-averaged A-S spread (the fixed agent's default):

```
avg_spread = gamma*sigma^2*T/2 + (2/gamma)*ln(1 + gamma/k)
```

### 2.2 Simulation loop (ORDER PINNED)

State per path: mid `s` (init `s0`), inventory `q` (init 0), cash (init 0).
`n_steps = round(T/dt)` steps, step `i = 0..n_steps-1`, `t = i*dt`.
Per step, in this exact order:

1. **quote** — from the *current* `s`, `q`, remaining time `T - t`:
   - agent `"as"`: `r` and `spread(t)` as in 2.1;
   - agent `"fixed"`: `r = s`, spread = the fixed spread (default
     `avg_spread`), no inventory skew.
   - **inventory cap rule:** the bid is quoted only if `q < q_max`; the ask
     only if `q > -q_max` (one-sided quoting at the cap; `q_max = 0` quotes
     nothing).  So `|q| <= q_max` at every step by construction, and the
     simulation asserts it.
2. **fill draws** — draw `u_ask ~ U(0,1)` first, then `u_bid ~ U(0,1)`
   (both drawn *always*, even for a suppressed side, so the stream does not
   depend on the cap state).  A side fills iff it is quoted and
   `u < P(fill)` with `delta` measured from the current mid.
3. **inventory/cash update** — ask fill: `q -= 1`, `cash += ask`;
   bid fill: `q += 1`, `cash -= bid` (both may fill in one step).
4. **mid move** — draw `z ~ N(0,1)`; `s += sigma_t*sqrt(dt)*z` where
   `sigma_t = sigma*stress_factor` for `t >= stress_time_frac*T` (default
   stress off: factor 1).  The *agent* always quotes with the base `sigma`
   (stress = model misspecification scenario).

**Terminal liquidation (pinned):** after the last step,
`PnL = cash + q*S_T - eta_liq*|q|` (liquidation cost is `eta_liq*|q|`).

Statistics over `M` paths (aggregates are over per-path values):

- `mean_pnl`, `std_pnl` (ddof = 1), Sharpe-like `mean_pnl/std_pnl`;
- `mean_abs_q` = mean over paths of the per-path *time average* of `|q|`
  (over the `n_steps` post-update values);
- `mean_max_abs_q` = mean over paths of the per-path maximum `|q|`;
- `rms_q` = mean over paths of the per-path `sqrt(time average of q^2)`
  (the "inventory std" measure used in the fixed-spread comparison);
- `mean_fills` = mean over paths of the **total number of fills, both
  sides counted**: a step where the bid and the ask both fill contributes
  two.  (Ports must count per side — `if fill_ask: fills += 1; if fill_bid:
  fills += 1` — never a per-step "any side filled" indicator.)

Standard errors: SE of a per-path-mean statistic = sample std of the per-path
values / sqrt(M); `SE(std_pnl) = std_pnl / sqrt(2*(M-1))` (normal approx).

---

## 3. Execution analytics

### 3.1 Implementation-shortfall decomposition (pinned, Perold convention)

Inputs: `side` (`"buy"`/`"sell"`), decision price `P_d`, arrival price `P_0`,
close price `P_T`, parent size `X >= 0`, fills `(n_j, p_j)` with `n_j >= 0`,
`Q = sum n_j <= X`, `R = X - Q`.  All components are currency costs
(positive = loss).  For a sell (`sign = +1`; buy mirrors with `sign = -1`):

```
delay       = Q * sign*(P_d - P_0)
trading     = sum_j n_j * sign*(P_0 - p_j)
opportunity = R * sign*(P_d - P_T)
total       = delay + trading + opportunity
            = sign*(X*P_d - sum_j n_j*p_j - R*P_T)     (identity, exact to 1e-10+)
```

### 3.2 Participation rate

`participation = executed_volume / market_volume`; requires finite
`executed_volume >= 0`, `market_volume > 0`, and **`executed_volume <=
market_volume`** (market volume includes the trader's own fills, so a rate
above 1 is an input error, message "executed volume exceeds market volume").

### 3.3 Market-impact regression (tiny native OLS)

OLS of per-share cost on participation rate, closed form (no library fit):
`slope = Sxy/Sxx`, `intercept = mean(y) - slope*mean(x)`,
`R^2 = 1 - SSR/SST` (`1.0` if `SST = 0`).  Requires `>= 2` points, finite
inputs, non-degenerate x.

**Pinned synthetic fills** (deterministic, RNG-free, identical in every
language), used by the `impact_regression` golden case:

```
part_i = 0.02 + 0.18*i/(n-1),                      i = 0..n-1
cost_i = intercept + slope*part_i + noise_amp*sin(2.5*i + 0.3)
```

with `n = 25, intercept = 0.5, slope = 8.0, noise_amp = 0.05`.

---

## 4. RNG note for ports

Only *deterministic* quantities are pinned bit-for-bit.  The Python reference
uses `numpy.random.default_rng(seed)` (PCG64) with the draw order pinned in
Sections 1.7 and 2.2, drawing per step **vectors across paths** (AS: `u_ask`
vector, `u_bid` vector, `z` vector per step; AC MC: a `(n_paths, N)` normal
block up front).  Other languages use their own RNG with the same per-step
*logical* draw order per path; they will NOT reproduce Python's sample
values.  The generators are pinned per language:

| language | generator | reproducibility of "same seed" |
|---|---|---|
| Python | `numpy.random.default_rng(seed)` (PCG64) | bit-stable across NumPy releases (PCG64 stream is versioned) |
| C++ | `std::mt19937_64(seed)` + `std::uniform_real_distribution` / `std::normal_distribution` | bit-stable within one standard library; the distribution algorithms are implementation-defined, so libstdc++ / libc++ / MSVC differ for the same seed |
| Rust | `rand_chacha::ChaCha12Rng::seed_from_u64(seed)` + `rand_distr::StandardNormal` | pinned explicitly (not `StdRng`, which `rand` documents as free to change between releases); bit-stable across `rand` 0.8.x |
| Java | `java.util.SplittableRandom(seed).nextDouble()` uniforms; normals by the Marsaglia polar method on those uniforms (`u, v = 2U - 1` until `0 < s = u^2 + v^2 < 1`, return `u*sqrt(-2 ln s / s)` then the cached `v*sqrt(-2 ln s / s)`), not `nextGaussian` | bit-identical across JVMs, JDK versions (25+) and platforms: `SplittableRandom.nextDouble` is specified, the normal generator is pinned in the library (`NormalRng`), and every transcendental call (`exp`, `expm1`, `log`, `log1p`, `pow`, `sqrt`, `sin`) uses `StrictMath` (fdlibm). Not guaranteed: equality with the Python/C++/Rust streams (statistical agreement only, checked by the 4-SE goldens) |

Therefore:

- deterministic golden cases must match within their stated tolerance;
- simulation golden cases store the Python 20k-path reference plus a standard
  error per statistic; a port running `M >= 2000` paths with any seeded RNG
  must satisfy `|stat - ref| <= 4*sqrt(se_ref^2 + se_own^2)`;
- each language's own suite must additionally verify determinism: same seed
  twice gives identical results.

## 5. Golden values (`data/golden/golden.json`)

Schema: `{"cases": [{"name", "kind", "inputs", "expect", "tol"}]}` — flat
scalar keys only.  `kind` is `"exact"` or `"statistical"`.

- **exact:** for every key, `|actual - expected| <= tol * max(1, |expected|)`
  (relative for large magnitudes, absolute near zero).
- **statistical:** `tol = 4.0` is the SE-band multiplier; `expect` holds each
  statistic plus a `<key>_se` companion; comparison rule per Section 4.

Case list (19):

| name | kind | expect keys | tol |
|---|---|---|---|
| ac_liquid_solution | exact | kappa, half_life, x_j (j=2), n_1, e_cost, var_cost | 1e-10 |
| ac_illiquid_solution | exact | kappa, half_life, x_j (j=4), n_1, e_cost, var_cost | 1e-10 |
| ac_twap_liquid | exact | e_cost, var_cost | 1e-10 |
| ac_pov_liquid | exact | e_cost, var_cost, n_intervals_used | 1e-10 |
| ac_frontier_point_liquid | exact | e_cost, std_cost, kappa (lambda_ra=5e-7) | 1e-10 |
| ac_kappa_small_tau | exact | kappa, half_life, x_1 (liquid set with tau=1e-3, N=5000, lambda_ra=1e-8: `kappa_tilde^2 tau^2/2 ~ 2e-9`; the `acosh(1+x)` form misses by ~1e-8) | 1e-10 |
| ac_lambda_zero_limit_liquid | exact | max_abs_diff_vs_twap (= 0) | 1e-12 |
| ac_instant_liquidation_liquid | exact | e_cost, var_cost (= 0) | 1e-10 |
| ac_mc_liquid | statistical | mean_cost, var_cost (+ _se) | 4 SE |
| as_reservation_liquid | exact | reservation_price (s=100, q=3, t=0.3) | 1e-12 |
| as_spread_liquid | exact | spread (t=0.3), spread_terminal | 1e-12 |
| as_fill_prob_liquid | exact | intensity, fill_probability (delta=0.6) | 1e-12 |
| as_sim_liquid | statistical | mean_pnl, std_pnl, mean_abs_q, mean_max_abs_q, mean_fills (+ _se; mean_fills counts both sides, ref 85.53) | 4 SE |
| as_sim_fx | statistical | mean_pnl, std_pnl, mean_abs_q (+ _se) | 4 SE |
| as_fixed_comparison_liquid | statistical | as_rms_q, fixed_rms_q (+ _se) | 4 SE |
| is_decomposition_sell | exact | total, delay, trading, opportunity | 1e-10 |
| is_decomposition_buy | exact | total, delay, trading, opportunity | 1e-10 |
| participation_rate | exact | participation | 1e-12 |
| impact_regression | exact | slope, intercept, r_squared | 1e-10 |

Inputs use prefixed flat keys (`ac_*`, `as_*`) matching
`data/params_{liquid,illiquid,fx}.json`; statistical cases carry `seed` and
`n_paths_ref` (20000).  Regenerate with `python3 data/generate_data.py`
(which first asserts the AC closed forms against Monte Carlo).

### 5.1 Vector cases

Schema v2 (`data/golden/trajectory_golden.json`, generated by
`data/generate_trajectories.py`): `{"schema": 2, "cases": [...]}` with the
case fields of Section 5. Vectors are stored as indexed flat keys
(`x_0, x_1, ..., x_{len-1}`), plus `len`, the number of points; a reader
asserts `len` and every indexed key. Comparison rule and tolerances as
Section 5 (`1e-10` for Section 1 paths, `1e-9` for Section 8).

| name | kind | expect keys |
|---|---|---|
| ac_trajectory_{liquid,illiquid,fx} | ac_trajectory | `x_j`, the Section 1.4 optimum on the bundled set |
| twap_liquid_10 | twap | `x_j` of `twap_holdings(X, 10)` |
| pov_liquid | pov | `x_j` of `pov_holdings` (Section 1.6) |
| frontier_liquid_8 | frontier | `kappa_i`, `e_cost_i`, `std_cost_i` over the 8-point grid `lambda_0..lambda_7` (inputs carry `n_lambdas`) |
| impact_alpha06_drift_curve_path | impact_schedule | `xfrac_j = x_j / X`, `j = 0..N`, of the Section 8.2 schedule (inputs as Section 8.5) |

Every suite asserts every key of every case and that the count is 7.

## 6. Error behavior

All input violations are reported, never silently clamped, and never crash
the process with UB: Python raises `ValueError` with a message; C++ throws
`std::invalid_argument` (or `std::domain_error` for the `gamma_p*tau >=
2*eta` model-validity failure); Rust returns `Result<_, EtradeError>`; Java
throws `IllegalArgumentException`.  Checked conditions include:

- AC: `X >= 0` finite; `N` integer in `[1, MAX_AC_INTERVALS]` (Python:
  any `numbers.Integral` except `bool`); `tau > 0`; `sigma >= 0`; `eta > 0`;
  `gamma_p >= 0`; `lambda_ra >= 0` (all finite); **`gamma_p*tau < 2*eta`**;
  TWAP/POV inputs finite, POV `p in (0,1)`, `V > 0`, `X/clip <=
  MAX_POV_INTERVALS`; frontier grid non-empty with finite entries `>= 0`;
  supplied holdings finite with length `>= 2`; schedules for `E` and the MC
  check must liquidate; `n_paths` integer in `[2, MAX_PATHS]`.
- AS: `s0` finite; `gamma > 0`; `k > 0`; `A > 0`; `sigma >= 0`; `T > 0`;
  `0 < dt <= T` with `T/dt` integral and `<= MAX_AS_STEPS`; `q_max` integer
  `>= 0`; `eta_liq >= 0`; `s`, `q` finite; `t in [0, T]`; `delta` finite;
  `n_paths` integer in `[2, MAX_PATHS]`; agent name valid; `fixed_spread`
  finite and `> 0` (C++ NaN sentinel `kDefaultFixedSpread` and Rust `None`
  select the average A-S spread; Java has no sentinel — pass
  `averageAsSpread(p)`); `stress_factor` finite and `> 0`;
  `stress_time_frac` finite in `[0, 1]`.
- Analytics: side valid; quantities `>= 0`; fills consistent (equal lengths,
  `Q <= X`); finite prices; OLS `n >= 2` and non-degenerate x;
  `market_volume > 0` and `executed_volume <= market_volume`.

No public function returns NaN from finite valid inputs, and no loop or
allocation is unbounded (Section 0). A result whose true value exceeds the
double range (for example a variance with `sigma` near `1e308`) is
returned as infinity; the infinite-urgency limits of Section 1.4 are
defined values, not errors. The property-based tests in every language
draw extreme finite inputs to keep this true.

Non-errors (documented behavior): negative prices (arithmetic walks);
`kappa = 0` (TWAP branch, half-life = infinity); a quote through the mid
(`delta < 0`); `q_max = 0` (no quoting, zero PnL); `x_j` underflowing to
exactly `0` far out on a very urgent schedule; a participation rate of
exactly 1.

## 7. Market units (conversion layer)

The core modules take parameters in model units. Section 7 pins how they are
built from the quantities a desk quotes, and how results are reported in
basis points. Every port exposes these functions with the same names in its
own style (C++ `etrade::ac_params_from_market`, Rust
`etrade::market::ac_params_from_market`, Java `Market.acParamsFromMarket`);
Python `etrade.market.*` is the reference. Golden cases live in a separate
file, `data/golden/market_golden.json`, so the 19-case contract of Section 5
is unchanged. Presets in market units are in `data/presets.json`.

### 7.1 Almgren-Chriss from market quantities (pinned)

Inputs: `price > 0`, full quoted `spread > 0` (currency), `adv > 0` (shares
per trading day), `daily_vol > 0` (a fraction: 0.015 means 1.5%),
`total_shares`, `n_intervals`, `tau` (trading days), `lambda_ra`, and
`temp_frac`, `perm_frac` in `(0, 1]` (defaults 0.01 and 0.10). All finite.

```
sigma   = price * daily_vol
eta     = spread / (temp_frac * adv)
gamma_p = spread / (perm_frac * adv)
```

This is the calibration of Almgren and Chriss (2000, Section 4): trading
1% of ADV per day costs one spread of temporary impact, 10% of ADV moves the
price permanently by one spread. With the paper's price 50, spread 0.125,
ADV 5e6 and daily vol 0.019 it reproduces `params_liquid.json` exactly
(`eta = 2.5e-6`, `gamma_p = 2.5e-7`, `sigma = 0.95`). The result is an
ordinary `ACParams` and is validated as in Section 1.2. The calibration is
linear, so it inherits the linear model's known under-pricing of small
orders relative to the square-root law.

### 7.2 Avellaneda-Stoikov from market quantities (pinned)

Inputs: `price > 0`, `tick > 0`, `daily_vol > 0`, `touch_fills_per_day > 0`
(fill rate per side of a quote half a tick from the mid), `halving_ticks > 0`
(distance in ticks over which the intensity halves), `skew_ticks > 0`
(reservation-price shift per unit of inventory at `t = 0`, in ticks),
integer `q_max >= 0`, `liq_ticks >= 0`, `dt_seconds > 0`,
`session_seconds > 0` (default 23400) and `horizon_days > 0` (default 1).
All finite.

```
sigma   = price * daily_vol
k       = ln(2) / (halving_ticks * tick)
A       = touch_fills_per_day * exp(k * tick / 2)
gamma   = skew_ticks * tick / (sigma^2 * horizon_days)
eta_liq = liq_ticks * tick
horizon = horizon_days
dt      = dt_seconds / session_seconds
```

Defining properties, asserted in every suite: `lambda(tick/2)` equals
`touch_fills_per_day`; `lambda` halves every `halving_ticks` ticks; the
opening skew `s - r(s, 1, 0)` equals `skew_ticks * tick`. The result is an
ordinary `ASParams` and is validated as in Section 2.1, so a `dt_seconds`
that does not divide the session is an input error.

### 7.3 Reporting helpers (pinned)

```
bps(amount, notional)          = 1e4 * amount / notional      (notional > 0, amount finite)
half_spread_cost(spread, X)    = 0.5 * spread * X              (spread > 0, X >= 0)
```

`half_spread_cost` is the paper's fixed cost `epsilon * X` with
`epsilon = spread / 2`, omitted from `E[IS]` by Section 1.1; add it when
reporting total expected cost.

### 7.4 Market golden cases

Schema as in Section 5. `kind` is `market_ac` or `market_as`; inputs are the
Section 7.1 or 7.2 argument names (integers stored as floats); tolerance
`1e-12` relative. Expected keys:

- `market_ac`: `sigma`, `eta`, `gamma_p`, `kappa`, `e_cost_bps`,
  `sd_cost_bps` (bps of `total_shares * price`), `spread_cost_bps`.
- `market_as`: `sigma`, `k`, `a_rate`, `gamma`, `eta_liq`, `dt`, `n_steps`,
  `spread_open_bps`, `spread_close_bps` (A-S spread at `t = 0` and `t = T`,
  in bps of `price`).

Nine cases: one `market_ac` for each of the five presets and one
`market_as` for each of the four presets that carry an A-S block. Every
suite asserts every expected key of every case and that the count is nine.

Validation messages name the offending argument (for example
`"spread must be finite and > 0"`).

## 8. Power-law impact execution (pinned)

Generalises Section 1 to the empirical square-root law, intraday liquidity
and an adverse drift. Python `etrade.impact` and the Section 8.3/8.4
functions in `etrade.market` are the reference; ports expose the same
names in their own style (C++ `etrade::ImpactParams`, Rust
`etrade::impact`, Java `Impact`). Golden cases: `data/golden/impact_golden.json`
(8 cases, generated by `data/generate_impact.py`).

### 8.1 Model

Parameters: `total_shares X >= 0`, integer `n_intervals N` in
`[1, MAX_IMPACT_INTERVALS = 10000]`, `tau > 0` (trading days), `sigma >= 0`,
`eta > 0`, `alpha` in `(0, 1]`, `gamma_p >= 0`, `lambda_ra >= 0`,
`drift >= 0` (expected adverse move per share per time unit), and liquidity
weights `w_j > 0`, `j = 1..N` (fraction of daily volume in interval `j`;
default `w_j = tau`). All finite.

```
eta_j   = eta * w_j^(-alpha)
c_j(n)  = eta_j * n^(1+alpha) - gamma_p * n^2 / 2          (n >= 0)
E[IS]   = gamma_p * X^2 / 2 + sum_{j=1}^{N} c_j(n_j) + drift * tau * sum_{k=1}^{N-1} x_k
Var[IS] = sigma^2 * tau * sum_{j=1}^{N} x_j^2
U       = E[IS] + lambda_ra * Var[IS]
```

`alpha = 1` with default weights is exactly Section 1 (`eta_j = eta/tau`,
`c_j = (eta_tilde/tau) n^2`). The cost functions take a holdings path of
length `N + 1` that is finite, fully liquidating (`|x_N| <= 1e-9 max(1, |x_0|)`)
and non-increasing (`x_j - x_{j-1} <= 1e-12 max(1, |x_0|)`); trades are
`max(x_{j-1} - x_j, 0)`; `X` in the permanent term is `x_0`.

**Validity (pinned):** for `X > 0`, every interval must satisfy
`alpha * (1 + alpha) * eta_j * X^(alpha - 1) > gamma_p` (each `c_j` convex
on `(0, X]`; for `alpha = 1` this is Section 1.2's `gamma_p * tau < 2 eta`).
Violation is an input error.

### 8.2 Optimal schedule (pinned algorithm)

`optimal_schedule(p)` returns holdings (length `N + 1`), the horizon `m`
used (`x_k = 0` for `k >= m`) and the Newton iteration count. `X = 0`
returns zeros, horizon `N`, 0 iterations.

Solve on horizon `m` (`x_m = 0`; `m = 1` is the single clip `[X, 0]`):

1. Start from the linear path `x_k = X (m - k)/m`.
2. Each iteration: with `n_j = x_{j-1} - x_j`,
   `c_j'(n) = (1+alpha) eta_j n^alpha - gamma_p n`,
   `c_j''(n) = alpha (1+alpha) eta_j n^(alpha-1) - gamma_p`, form for
   `k = 1..m-1` the gradient
   `g_k = c_{k+1}'(n_{k+1}) - c_k'(n_k) + 2 lambda_ra sigma^2 tau x_k + drift tau`
   and the tridiagonal Hessian
   `H_kk = c_k''(n_k) + c_{k+1}''(n_{k+1}) + 2 lambda_ra sigma^2 tau`,
   `H_{k,k+1} = -c_{k+1}''(n_{k+1})`; solve `H d = -g` by the Thomas algorithm.
3. Stop (converged) when `max_k |d_k| <= 1e-12 X` or
   `-g.d <= 1e-14 max(1, |U(x)|)`.
4. Otherwise backtrack `t = 1, 1/2, 1/4, ...` down to `2^-40`; accept the first
   `t` for which every trade of `x + t d` is strictly positive and
   `U(x + t d) <= U(x) + 1e-4 t g.d`. No acceptable `t` means the horizon is
   infeasible. At most 100 iterations; reaching the cap means infeasible.

Then: solve horizon `N`; if it converges, return it. Otherwise (only
possible with `drift > 0`; with `drift = 0` it is an error) find the
largest feasible `m` by bisection on `[1, N]` (`m = 1` is always feasible)
and verify the KKT condition
`c_m'(n_m) <= drift * tau * (1 + 1e-9) + 1e-12`; failure is an error. A point
that passes is the global optimum of the convex problem.

### 8.3 Market units

`impact_params_from_market(price, spread, adv, daily_vol, total_shares,
n_intervals, tau, lambda_ra, alpha = 0.5, y = 1.0, perm_frac = 0.10,
drift_bps_per_day = 0, weights = default)`, all of `price, spread, adv,
daily_vol, y > 0`, `perm_frac` in `(0, 1]`, `drift_bps_per_day >= 0`:

```
sigma   = price * daily_vol
eta     = y * sigma / adv^alpha        (participation p costs y*sigma*p^alpha per share)
gamma_p = spread / (perm_frac * adv)   (as Section 7.1)
drift   = price * drift_bps_per_day * 1e-4
```

With `alpha = 1/2` a one-day TWAP costs exactly the square-root law
`1e4 * y * daily_vol * sqrt(X/adv)` bps of temporary impact (asserted).
`sqrt_law_cost_bps(daily_vol, order_shares, adv, y = 1)` returns that
closed form (`daily_vol, adv, y > 0`, `order_shares >= 0`).

`US_EQUITY_VOLUME_CURVE_13` (pinned, illustrative, sums to 1): thirteen
30-minute fractions
`0.12, 0.09, 0.075, 0.065, 0.06, 0.055, 0.05, 0.055, 0.06, 0.065, 0.075, 0.09, 0.14`.
`vwap_holdings(X, w)`: `x_j = X (1 - cumsum(w)_j / sum(w))`, `x_N = 0`.

### 8.4 Pre-trade report

`pretrade_report(<market inputs of 8.3 without lambda_ra>, lambdas)` with a
non-empty grid of finite `lambda_ra >= 0` returns `sqrt_law_bps`,
`half_spread_bps` (Section 7.3) and rows in this order: `twap`, `vwap`
(weights as given), then `optimal` for each `lambda_ra` in grid order. Each
row carries `lambda_ra`, `impact_bps = bps(E[IS])`, `sd_bps = bps(sqrt(Var))`,
`total_bps = impact_bps + half_spread_bps` and the horizon, all in bps of
`X * price`; TWAP and VWAP rows are costed with `lambda_ra = 0` parameters
(the cost functions do not depend on `lambda_ra`).

### 8.5 Golden cases

Kinds: `impact_schedule` (expect `xfrac_01..xfrac_{N-1}` = `x_j / X`,
`e_cost`, `var_cost`, `horizon`), `impact_vwap` (`e_cost`, `var_cost`),
`pretrade` (`sqrt_law_bps`, `half_spread_bps`, `row{i}_impact_bps`,
`row{i}_sd_bps`, `row{i}_horizon`; inputs carry `n_lambdas` and
`lambda_0..`), `sqrt_law` (`bps`). Inputs use 8.3 argument names;
`volume_curve = 1` selects `US_EQUITY_VOLUME_CURVE_13`, `0` the default
weights. Tolerance `1e-9` with the rule of Section 5. Every suite asserts
every key of every case and that the count is 8.

## 9. Market making with frictions (pinned)

Section 2's simulator is the paper's world: fills are independent of the next
price move, quotes sit at any real price, trading is free and quotes never go
stale. Section 9 keeps the A-S quoting logic and adds those frictions one
switch at a time, plus the accounting a desk uses to see where the money
went. Python `etrade.mm` is the reference; ports expose the same names in
their own style (C++ `etrade::simulate_mm`, `etrade::MMConfig`; Rust
`etrade::mm`; Java `MarketMaking`). The Section 2 loop is unchanged. Golden
cases: `data/golden/mm_golden.json` (14 cases: 8 exact, 6 statistical),
generated by `data/generate_mm.py`.

### 9.1 Configuration

`MMConfig`: `agent` in {`as`, `glft`, `fixed`} (default `as`); `tick >= 0`
(0 = continuous prices); `maker_fee` finite (per share per fill, positive =
rebate received, negative = fee paid); integer `latency_steps >= 0` and
`<= n_steps`; `informed_prob` in `[0, 1]`; `signal_accuracy` in `[0.5, 1]`
(0.5 = no signal); optional `fixed_spread > 0` for the `fixed` agent (default
the average A-S spread of Section 2.1). All finite. `n_paths` integer in
`[2, MAX_PATHS]`.

### 9.2 Tick rounding (pinned)

`round_to_tick(bid, ask, tick)`: with `tick > 0`,
`bid -> floor(bid/tick + 1e-9) * tick`, `ask -> ceil(ask/tick - 1e-9) * tick`
(away from fair value); `tick = 0` returns the inputs. Bid and ask finite,
`tick >= 0`.

### 9.3 GLFT quotes (pinned)

The asymptotic closed-form approximation of Guéant, Lehalle and
Fernandez-Tapia (2013) for the inventory-constrained market maker,
time-independent:

```
c1        = log1p(gamma/k) / gamma
c2        = sqrt( sigma^2 * gamma / (2 k A) * (1 + gamma/k)^(1 + k/gamma) )
delta_bid = c1 + (2q + 1)/2 * c2
delta_ask = c1 - (2q - 1)/2 * c2
bid = fair - delta_bid,   ask = fair + delta_ask
```

`glft_half_spreads(params, q)` returns `(delta_bid, delta_ask)`; `q` finite.

### 9.4 Simulation loop (ORDER PINNED)

State per path: mid `s` (init `s0`), inventory `q` (init 0), cash (init 0),
and a history of the last `latency_steps + 1` mids (all `s0` initially;
`hist[0]` is the mid `latency_steps` steps ago). Per step `i`, `t = i*dt`:

1. **Draws, always all five, in this order:** `z ~ N(0,1)`, `u_sig`, `u_ask`,
   `u_bid`, `u_inf ~ U(0,1)`.
2. **Fair value:** `pred = sign(z)` if `u_sig < signal_accuracy` else
   `-sign(z)` (`sign(0) = 0`);
   `fair = hist[0] + (2*signal_accuracy - 1) * pred * sigma*sqrt(dt)*sqrt(2/pi)`.
3. **Quotes from `fair`:** `as`: `r = fair - q gamma sigma^2 (T - t)`,
   `half = (gamma sigma^2 (T - t) + (2/gamma) log1p(gamma/k)) / 2`, quotes
   `r -/+ half`; `glft`: Section 9.3 at the current `q`; `fixed`:
   `fair -/+ fixed_spread/2`. Then Section 9.2 rounding when `tick > 0`.
4. **Uninformed fills:** `fill_ask = (q > -q_max) and u_ask < P(ask - s)`,
   `fill_bid = (q < q_max) and u_bid < P(s - bid)` with
   `P(delta) = -expm1(-A exp(-k delta) dt)`, distances from the **true
   current** mid `s`.
5. **Informed fills:** `move = sigma*sqrt(dt)*z`; `informed = u_inf < informed_prob`;
   `inf_ask = informed and z > 0 and q > -q_max and not fill_ask and ask < s + move`;
   `inf_bid = informed and z < 0 and q < q_max and not fill_bid and bid > s + move`.
6. **Updates:** `fa = fill_ask or inf_ask`, `fb = fill_bid or inf_bid`;
   `q += fb - fa`; `cash += (fa ? ask : 0) - (fb ? bid : 0) + maker_fee*(fa + fb)`;
   accumulate `spread += (fa ? ask - s : 0) + (fb ? s - bid : 0)`,
   `fees += maker_fee*(fa + fb)`, `fills += fa + fb`,
   `informed_fills += inf_ask + inf_bid`, `inventory += q * move` (`q` after
   this step's fills), `markout += (fb - fa) * move`, `sum_q2 += q^2`.
7. **Mid move:** `s += move`; push `s` into the history (drop the oldest).

Terminal: `liquidation = eta_liq |q_T|`, `pnl = cash + q_T s_T - liquidation`,
`rms_q = sqrt(sum_q2 / n_steps)`.

### 9.5 Attribution and statistics (pinned)

Per path, exactly (asserted every simulation, relative to `max(1, |pnl|)`,
tolerance `1e-9`):

```
pnl = spread + fees + inventory - liquidation
```

`markout` is the one-step markout of the fills, a component of `inventory`;
negative means adverse selection. Statistics, each a per-path array:
`pnl, spread, fees, inventory, liquidation, markout, fills, informed_fills,
rms_q`. `mean(stat)` is the mean over paths; `se(stat)` is the sample
standard deviation (ddof 1) over `sqrt(n_paths)`. An unknown statistic name
is an input error. The result also reports the largest identity residual.

### 9.6 Fair-value helpers (pinned, exact)

- `weighted_mid(bid, ask, bid_size, ask_size) =
  (bid*ask_size + ask*bid_size) / (bid_size + ask_size)`: a size-weighted
  proxy for the microprice of Stoikov (2018). Prices finite with
  `ask >= bid`; sizes finite `>= 0` with a positive sum.
- `order_flow_imbalance(bid_px, bid_sz, ask_px, ask_sz)`: per-update OFI of
  Cont, Kukanov and Stoikov (2014) for consecutive top-of-book states,
  returning `len - 1` values
  `e_n = 1[Pb_n >= Pb_{n-1}] Qb_n - 1[Pb_n <= Pb_{n-1}] Qb_{n-1}
       - 1[Pa_n <= Pa_{n-1}] Qa_n + 1[Pa_n >= Pa_{n-1}] Qa_{n-1}`.
  At least 2 states, equal lengths, finite, sizes `>= 0`.

These are computed from book data; inside the simulator the signal is the
abstract accuracy model of 9.4.

### 9.7 Golden cases

Exact kinds (`tol 1e-12`): `mm_tick` (`bid`, `ask`), `mm_glft` (inputs: the
`us_large_cap` A-S preset in market units plus `q`; expect `delta_bid`,
`delta_ask`), `mm_wmid` (`wmid`), `mm_ofi` (inputs `bid_px_i, bid_sz_i,
ask_px_i, ask_sz_i` for `i = 0..4`; expect `e_1..e_4`, `ofi` = sum).
Statistical kind `mm_sim` (`tol = 4.0`, Section 4 rule): inputs are the
`us_large_cap` A-S preset in market units (build `ASParams` with Section 7.2)
plus `agent_code` (0 `as`, 1 `glft`, 2 `fixed`), `quote_tick` (the
config `tick`; the preset's own `tick` builds `ASParams`), `maker_fee`,
`latency_steps`, `informed_prob`, `signal_accuracy`, `seed`, `n_paths_ref`
(20000); expect `pnl, spread, fees, inventory, markout, fills,
informed_fills`, each with a `_se`. Ports run their own RNG (Section 4) with
at least 2000 paths. Every suite asserts every key of every case, that the
count is 14, and the attribution identity on every simulated path.

## 10. Post-trade transaction-cost analysis (pinned)

Measures an order after it has traded. Python `etrade.tca` is the reference;
ports expose the same names in their own style (C++ `etrade::slippage` in
`etrade/tca.hpp`, Rust `etrade::tca`, Java `Tca`). Golden cases:
`data/golden/tca_golden.json` (11 cases, generated by `data/generate_tca.py`).
All loops run in input order (fills `j = 0..J-1`, bins `b = 0..B-1`) and every
sum is a plain left-to-right accumulation, so the ports agree to a few ulps.

Inputs shared by 10.1-10.3: fills `(b_j, n_j, p_j)` — integer bin (or step)
`b_j` in `[0, B)`, quantity `n_j >= 0`, price `p_j > 0` — and per-bin market
volume `V_b >= 0` with `sum V_b > 0` and market VWAP `P_b > 0`. Market volume
includes the order's own fills. `Q = sum n_j`; zero-quantity fills are legal
but must not make `Q = 0`. All finite.

### 10.1 Benchmark slippage

`slippage(side, arrival_price, close_price, fill_bins, fill_qtys, fill_prices,
bin_volumes, bin_vwaps)`, `sign = +1` for a sell and `-1` for a buy (the
Section 3.1 convention, positive = cost):

```
pbar          = sum_j n_j p_j / Q
market_vwap   = sum_b V_b P_b / sum_b V_b
S             = { b : some fill j has b_j = b and n_j > 0 }
interval_vwap = sum_{b in S} V_b P_b / sum_{b in S} V_b         (ascending b)
cost_bps(B)   = 1e4 * sign * (B - pbar) / B
```

reported for `B` = arrival, `market_vwap` (`vwap_bps`), `interval_vwap` and
close, with `executed_qty = Q`, `avg_price`, `market_vwap`, `interval_vwap`.
`sum_{b in S} V_b > 0` is required. Properties asserted: buy and sell results
are exact negatives; scaling every price by `c > 0` leaves the bps unchanged;
adding a constant to every price leaves the per-share cost
`cost_bps * B / 1e4` unchanged; `arrival_bps * P_0 * Q / 1e4` equals the
Section 3.1 trading component.

### 10.2 Markouts

`markouts(side, fill_steps, fill_qtys, mids, horizons)` with a mid series
`m_0..m_{M-1}`, fill steps `t_j` in `[0, M)` and integer horizons `h >= 0`
with `t_j + h < M` for every pair; `d = +1` for a buy, `-1` for a sell:

```
mk_j(h)      = d * (m_{t_j + h} - m_{t_j})       (currency per share, negative = adverse)
per_fill(h)  = sum_j mk_j(h) / J
per_share(h) = sum_j n_j mk_j(h) / Q
```

Results are in horizon order. Fill prices are not inputs: the markout
measures the mid's move after the fill, not the spread paid.

### 10.3 Participation

`participation(fill_bins, fill_qtys, bin_volumes)`: `q_b = sum_{j: b_j = b} n_j`,
`per_bin_b = q_b / V_b` (`0` when `V_b = 0`), `overall = Q / sum_b V_b`. Each
bin requires `q_b <= V_b` exactly (Section 3.2's rule per bin). Fills may
have `Q = 0` here.

### 10.4 A/B comparison (Welch t-test) and the pinned t CDF

`welch_ttest(a, b)` on two samples of per-order costs (bps), each of length
`>= 2`, finite:

```
mean_x = sum x_i / n_x
v_x    = sum (x_i - mean_x)^2 / (n_x - 1)        (two-pass)
e_x    = v_x / n_x,  se2 = e_a + e_b              (se2 > 0 required)
mean_diff = mean_a - mean_b
t_stat    = mean_diff / sqrt(se2)
df        = se2^2 / (e_a^2/(n_a - 1) + e_b^2/(n_b - 1))      (Welch-Satterthwaite)
p_value   = I_{df/(df + t^2)}(df/2, 1/2)          (two-sided)
```

`student_t_cdf(t, df)` (`t` finite, `df > 0` finite, non-integer allowed):
`tail = 0.5 * I_{df/(df+t^2)}(df/2, 1/2)`; result `1 - tail` for `t >= 0`,
`tail` otherwise.

`betainc_reg(a, b, x)` (`a, b > 0` finite, `x` in `[0, 1]`), following
*Numerical Recipes* (Press et al., 3rd ed., Sections 6.1 and 6.4): `x = 0`
gives `0`, `x = 1` gives `1`; otherwise
`bt = exp(ln_gamma(a+b) - ln_gamma(a) - ln_gamma(b) + a ln x + b log1p(-x))`
and the result is `bt * cf(a, b, x) / a` when `x < (a+1)/(a+b+2)`, else
`1 - bt * cf(b, a, 1-x) / b`. `cf` is the modified Lentz evaluation:

```
qab = a+b; qap = a+1; qam = a-1
c = 1; d = 1 - qab x / qap; d = 1/fix(d); h = d
for m = 1 .. MAX_BETACF_ITER (= 10000):
    aa = m (b - m) x / ((qam + 2m)(a + 2m))
    d = 1/fix(1 + aa d); c = fix(1 + aa/c); h = h * d * c
    aa = -(a + m)(qab + m) x / ((a + 2m)(qap + 2m))
    d = 1/fix(1 + aa d); c = fix(1 + aa/c); del = d c; h = h * del
    if |del - 1| <= 1e-15: return h
error "incomplete beta continued fraction did not converge"
```

with `fix(v) = 1e-300` when `|v| < 1e-300`, else `v`.

`ln_gamma(x)` (`x > 0` finite): Lanczos with `g = 7` and nine coefficients

```
0.99999999999980993, 676.5203681218851, -1259.1392167224028,
771.32342877765313, -176.61502916214059, 12.507343278686905,
-0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
```

For `x < 0.5`: `ln(pi / |sin(pi x)|) - ln_gamma(1 - x)`. Otherwise `z = x - 1`,
`s = c_0 + sum_{i=1}^{8} c_i / (z + i)` (in order), `t = z + 7.5`, result
`0.5 ln(2 pi) + (z + 0.5) ln t - t + ln s`. Relative accuracy is about
`1e-15`; the Python suite checks `ln_gamma` against `math.lgamma` and
`betainc_reg`, `student_t_cdf` and `welch_ttest` against SciPy
(`rel 1e-12`, `1e-11`, `1e-10`). Not intended for `df` above about `1e5`,
where `ln_gamma` differences cancel.

Worked example (golden `welch_unequal_variance`): A = 8 orders with mean
2.85 bps, B = 6 orders with mean 0.683 bps give `t = 3.7675`,
`df = 10.184`, `p = 0.00356` (SciPy `ttest_ind(equal_var=False)` agrees).

### 10.5 Validation (messages pinned, each names the argument)

`"side must be 'buy' or 'sell'"`; `"arrival_price must be finite and > 0"`,
`"close_price must be finite and > 0"`; `"fills must be non-empty"`;
`"fill arrays must have equal length"`; `"fill_bins must be integers in [0, B)"`
and `"fill_steps must be integers in [0, M)"` (with the number);
`"fill_qtys must be finite and >= 0"`; `"fill_prices must be finite and > 0"`;
`"executed quantity must be > 0"`; `"bins must be non-empty"`;
`"bin arrays must have equal length"`; `"bin_volumes must be finite and >= 0"`;
`"bin_vwaps must be finite and > 0"`; `"total market volume must be > 0"`;
`"market volume in traded bins must be > 0"`; `"mids must be non-empty"`,
`"mids must be finite"`; `"horizons must be non-empty"`,
`"horizons must be integers >= 0"`, `"fill step + horizon exceeds mid series"`;
`"fill volume exceeds market volume in a bin"`; `"a must have at least 2
values"` (likewise `b`), `"a must be finite"`; `"samples have zero variance"`;
`"t must be finite"`, `"df must be finite and > 0"`; `"a must be finite and >
0"`, `"b must be finite and > 0"`, `"x must be in [0, 1]"` (`betainc_reg`);
`"x must be finite and > 0"` (`ln_gamma`). Error types as Section 6; the
continued-fraction cap is reported like a solver failure (Python
`ValueError`, C++ `std::runtime_error`, Rust `SolverFailed`, Java
`IllegalStateException`). Typed ports take bins, steps and horizons as
unsigned or `int` indices; C++/Java also reject negative ones.

### 10.6 Golden cases

Kinds and expected keys: `slippage` (`executed_qty`, `avg_price`,
`market_vwap`, `interval_vwap`, `arrival_bps`, `vwap_bps`,
`interval_vwap_bps`, `close_bps`), `markout` (`per_fill_{k}`, `per_share_{k}`
per horizon), `participation` (`rate_{b}`, `overall`), `welch` (`mean_diff`,
`t_stat`, `df`, `p_value`), `t_cdf` (`cdf`). Inputs: `side_buy` (1 buy, 0
sell), `arrival_price`, `close_price`, `t`, `df`, and arrays as indexed keys
with a length key: `n_fills` with `bin_j`/`step_j`, `qty_j`, `px_j`; `n_bins`
with `vol_b`, `vwap_b`; `n_mids` with `mid_k`; `n_horizons` with `h_k`;
`n_a`/`a_i`, `n_b`/`b_i`. Tolerance `1e-12` with the rule of Section 5. Every
suite asserts every key of every case and that the count is 11.

## 11. Calibration from recorded data (pinned)

Estimators that measure the Section 7 and 8 inputs from plain arrays, so
they need no data access and every port returns the same numbers. Python
`etrade.calibration` is the reference; ports expose the same names in their
own style (C++ `etrade::volume_curve`, Rust `etrade::calibration`, Java
`Calibration`). Ingestion is tooling, not part of the contract:
`data/calibration/fit_itch.py` (Python only) turns one recorded NASDAQ ITCH
day into these arrays. Golden cases: `data/golden/calibration_golden.json`
(9 cases, generated by `data/generate_calibration.py`).

Validation messages name the offending argument, as in Section 7. Sums run
in index order with a single accumulator.

### 11.1 Binning rule

A session `[t0, t1]` (`t0` finite, `t1` finite and `> t0`) is split into an
integer `n_bins` in `[1, MAX_CALIBRATION_BINS = 10000]` equal bins. An
observation at `t` with `t0 <= t <= t1` falls in bin

```
j(t) = min(floor((t - t0) * n_bins / (t1 - t0)), n_bins - 1)
```

evaluated exactly in that order in double precision. The closing instant
belongs to the last bin; observations outside the session are ignored.

### 11.2 Volume curve

`volume_curve(times, sizes, t0, t1, n_bins)`: equal-length arrays, every
time finite, every size finite and `>= 0`. `V_j` is the sum of the sizes in
bin `j`; the result is `w_j = V_j / sum_j V_j`. A zero total is an error
(`"no volume inside the session"`). The result is a Section 8 weight vector.

### 11.3 Volatility curve

`volatility_curve(times, mids, t0, t1, n_bins)`: equal-length arrays, times
finite and non-decreasing, mids finite and `> 0`. For `i >= 1`, the return
`r_i = ln(m_i / m_{i-1})` counts when `t_{i-1} >= t0` and `t_i <= t1`, and
`r_i^2` is added to `RV_{j(t_i)}`. This is realised variance (Andersen,
Bollerslev, Diebold and Labys 2001, "The distribution of realized exchange
rate volatility", JASA 96(453)). Returns `bin_vol_j = sqrt(RV_j)`,
`daily_vol = sqrt(sum_j RV_j)` and `n_returns`, the number of returns
counted; zero is an error (`"no return inside the session"`). `daily_vol` is
the open-to-close volatility of the session; the overnight return is not
in it.

### 11.4 Spread statistics

`spread_stats(spreads, mids, tick)`: non-empty equal-length arrays, spreads
finite and `>= 0`, mids finite and `> 0`, `tick > 0`. Per observation
`ticks_i = s_i / tick` and `bps_i = 1e4 * s_i / m_i`; returns the mean and
the median of each, and `n`. The median is the middle element of the
ascending sort, or the mean of the two middle elements for an even count.
The quoted spread is `ask - bid` sampled on a clock. The effective spread is
`2 |p - m|` per trade, with `m` the mid before the trade (the definition in
SEC Rule 605, 17 CFR 242.605).

### 11.5 Fill intensity

`fill_intensity(distances, tick, n_levels, days = 1, sides = 2)`.
`distances` holds, for each market order, the largest distance from the
prevailing mid at which it printed (finite, `>= 0`). `tick > 0`;
`n_levels` is an integer in `[2, MAX_FILL_LEVELS = 1000]`; `days > 0` is the
sample length in trading days; `sides > 0` is the number of quote sides
pooled (2 for buys and sells together).

```
c_b      = #{ i : distances_i / tick >= b + 0.5 - 1e-9 }      b = 0 .. n_levels-1
delta_b  = (b + 0.5) * tick
y_b      = ln(c_b / (sides * days))                            over buckets with c_b > 0
w_b      = c_b
k        = -sum w (delta - dbar)(y - ybar) / sum w (delta - dbar)^2   (dbar, ybar weighted means)
A        = exp(ybar + k * dbar)
touch_fills_per_day = A * exp(-k * tick / 2)
halving_ticks       = ln 2 / (k * tick)
```

This fits the intensity `lambda(delta) = A exp(-k delta)` of Avellaneda and
Stoikov (2008, Section 2.3). The fit is weighted least squares on the log of
the rate at which market orders reach each distance. The weight is the
bucket count: for Poisson counts `Var(ln c_b) ~ 1/c_b` (delta method).
Without the weight, the sparse tail buckets, a handful of orders each, set
the slope. The `1e-9` slack keeps a print at exactly half a tick in
bucket 0. Fewer than two buckets with fills is an error
(`"fewer than two distance buckets have fills"`), and so is `k <= 0`
(`"fitted fill intensity does not decay with distance"`). The market units
are those of Section 7.2, so `as_params_from_market` with the fitted
`touch_fills_per_day` and `halving_ticks` returns the fitted `A` and `k`
(asserted).

### 11.6 What was fitted, from what, and the caveats

**Data.** The NASDAQ TotalView-ITCH 5.0 sample file `12302019.NASDAQ_ITCH50`
(Monday 2019-12-30), from the public directory `emi.nasdaq.com/ITCH/`. It
was ingested by the sibling project intraday-alpha-platform
(`iap.marketdata ingest`, dataset `ds3_20191230`, which rebuilds the AAPL,
MSFT and QQQ books with zero sequence gaps and a clean book check; the
earlier `ds_20191230` ingest has a tick-size bug and is not used). The
directory showed no licence text when the sibling project checked it on
2026-10-11. The Nasdaq Data Policies define "Derived Data" as aggregate
statistics that cannot reconstruct the messages. Accordingly the repository
holds only `data/calibration/fitted_20191230.json`: per-symbol aggregates,
curves and bucket counts. Raw and normalized data stay outside the
repository.

**Method** (`fit_itch.py`). The top of book is rebuilt from the order
events. The window is the continuous session, 09:30-16:00 ET; the opening
and closing crosses are excluded. Quoted spread: the NASDAQ best bid and
offer sampled every second (locked or crossed books skipped). Realised
variance: one-minute mids. Trades: every continuous-session print,
hidden-order prints included. The effective spread uses the mid before the
print's timestamp. For the fill intensity, prints with the same timestamp
and aggressor side form one market order, whose reach is its farthest
print from the prior mid; six buckets.

| | AAPL | MSFT | QQQ |
|---|---|---|---|
| volume-weighted mid ($) | 289.72 | 157.67 | 212.16 |
| NASDAQ continuous volume (shares) | 5,678,176 | 3,625,411 | 3,713,405 |
| prints / market orders | 67,077 / 45,121 | 40,012 / 22,734 | 23,607 / 15,907 |
| open-to-close realised vol | 1.157% | 0.743% | 0.574% |
| quoted spread mean / median (ticks) | 2.64 / 3 | 1.23 / 1 | 1.16 / 1 |
| quoted spread mean (bps) | 0.91 | 0.78 | 0.55 |
| effective spread mean / median (ticks) | 2.08 / 2 | 1.16 / 1 | 1.04 / 1 |
| orders reaching 0.5 / 1.5 / 2.5 ticks | 43,334 / 11,235 / 1,307 | 21,352 / 377 / 91 | 15,179 / 69 / 16 |
| `touch_fills_per_day` (fit, per side) | 22,070 | 10,507 | 7,532 |
| `halving_ticks` | 0.460 | 0.251 | 0.214 |

The presets `itch_aapl_20191230`, `itch_msft_20191230` and
`itch_qqq_20191230` in `data/presets.json` are built by
`data/generate_market.py`, each with a `source` field. They take from the
fit: `price`, `spread` (the mean quoted spread), `daily_vol`, `adv` (the
NASDAQ volume), `touch_fills_per_day` and `halving_ticks`. They do not fit,
and take from `us_large_cap` (AAPL, MSFT) or `us_etf` (QQQ): the order size
(5% of the `adv` used), the A-S `skew_ticks`, `q_max`, `liq_ticks` and
`dt_seconds`, and `lambda_ra`, set so that `lambda_ra * sigma^2 / eta`
equals the template's value. `FITTED_VOLUME_CURVE_13` is the equal-weighted
mean of the three volume curves, rounded to `1e-4`, with the rounding
residual added to the largest bin so that it sums to 1:
`0.1987, 0.1247, 0.0742, 0.0680, 0.0589, 0.0604, 0.0414, 0.0430, 0.0484,
0.0466, 0.0376, 0.0535, 0.1446`.
It is an addition: `US_EQUITY_VOLUME_CURVE_13` stays as pinned in 8.3.

**Caveats.** Read these before using the numbers.

- *One day, three names.* Every estimate comes from a single session and
  carries sampling error that this release does not quantify. Each
  thirty-minute realised volatility rests on 30 one-minute returns.
- *An unrepresentative day.* 2019-12-30 falls in the week between Christmas
  and New Year, when US volume is typically light.
- *One venue.* NASDAQ matches only part of US consolidated volume in these
  names, so `adv` is a lower bound on consolidated ADV, and impact
  calibrated on it (Sections 7.1, 8.3) is pessimistic. Likewise the quoted
  spread is the NASDAQ BBO, which is never tighter than the NBBO, and the
  fill counts cover NASDAQ market orders only.
- *Open-to-close volatility.* It excludes the overnight return and the
  crosses, so it understates close-to-close volatility.
- *The fill intensity is not exponential.* The counts fall faster than
  exponentially from the touch to 1.5 ticks and then flatten (table above).
  The weighted fit matches the observed touch rate to within 2% and
  understates the tail. A reach also counts every order that printed at or
  beyond a distance, whatever the spread was at that moment.
- *Sampling choices.* The one-second spread clock, the one-minute variance
  sampling and the timestamp grouping of market orders are tooling
  choices, not part of the pinned estimators, and they move the fitted
  numbers.

### 11.7 Golden cases

`data/golden/calibration_golden.json`, schema as Section 5, tolerance
`1e-12` with the rule of Section 5. Arrays are flattened as `<name>_<i>`
with the common length in `n`; integer arguments are stored as floats.

- `cal_volume_curve` (inputs `times_*`, `sizes_*`, `session_start`,
  `session_end`, `n_bins`): expect `frac_00 ..`.
- `cal_volatility` (`times_*`, `mids_*`, session, `n_bins`): `daily_vol`,
  `n_returns`, `bin_vol_00 ..`.
- `cal_spread` (`spreads_*`, `mids_*`, `tick`): `mean_ticks`,
  `median_ticks`, `mean_bps`, `median_bps`, `n`.
- `cal_fill_intensity` (`distances_*`, `tick`, `n_levels`, `days`, `sides`):
  `a_rate`, `k`, `touch_fills_per_day`, `halving_ticks`, `n_buckets`.
- `cal_fitted_curve` (no inputs): `w_00 .. w_12`, the pinned
  `FITTED_VOLUME_CURVE_13`.

Nine cases: two per estimator plus the fitted curve. Every suite asserts
every expected key of every case and that the count is nine. The three
fitted presets add six cases to `data/golden/market_golden.json`
(regenerated by `data/generate_market.py`; the nine existing cases are
unchanged). This supersedes the count in 7.4: there are now fifteen market
cases, one `market_ac` per preset and one `market_as` per preset with an A-S
block.

## 12. Execution extensions: transient impact, adaptive re-planning, internalisation (pinned)

Python `etrade.execution` is the reference; ports expose the same names in
their own style (C++ `etrade::TransientParams`, `etrade::simulate_adaptive`,
Rust `etrade::execution`, Java `Execution`). Every entry point bounds
`n_intervals` to `[1, MAX_EXEC_INTERVALS = 1000]` and each order list to
`MAX_ORDERS = 10000`. Golden cases: `data/golden/execution_golden.json`
(8 cases, generated by `data/generate_execution.py`).

### 12.1 Transient impact (Obizhaeva and Wang 2013, discrete time)

Parameters: `total_shares X >= 0`, integer `N`, `tau > 0`, `kappa > 0`
(price displacement per share traded: inverse book depth), resilience
`rho > 0` finite, `gamma_p >= 0`. With `a = exp(-rho*tau)`, validity
(pinned) is `gamma_p < kappa*(1 - a)/(1 + a)`; violation is an input error.
For a fully liquidating holdings path (length `N + 1`, finite,
`|x_N| <= 1e-9 max(1, |x_0|)`; trades `n_j = x_{j-1} - x_j` may have either
sign), trade `j` happens at the start of interval `j`:

```
D_1 = 0,   D_{j+1} = a * (D_j + kappa * n_j)                       (recursion, pinned order)
E[IS] = gamma_p x_0^2 / 2 + sum_j [ (kappa - gamma_p)/2 * n_j^2 + n_j * D_j ]
      = gamma_p X^2 / 2 + n' A n / 2,   A = kappa K - gamma_p I,  K_ij = a^|i-j|
```

**Static optimum (pinned):** minimise over `1'n = X`: `n = X y / sum(y)`
with `A y = 1`. `K` is Kac-Murdock-Szego, `(1 - a^2) K^{-1}` is tridiagonal,
so `y` solves exactly, by the Thomas algorithm, the system with `b =
-expm1(-rho*tau)` (`= 1 - a`):

```
diag_1 = diag_N = b(1+a) kappa - gamma_p,   diag_k = b(1+a) kappa - gamma_p (1 + a^2)  (1 < k < N)
off_k  = gamma_p * a,                       rhs_1 = rhs_N = b,  rhs_k = b^2
```

It is strictly diagonally dominant exactly when the validity condition
holds. Holdings `x_j = x_{j-1} - X y_j / sum(y)`, `x_N = 0` exactly; `N = 1`
or `X = 0` skip the solve. With `gamma_p = 0` the solution is
`n ~ [1, 1-a, ..., 1-a, 1]`: equal blocks at the start and end and a
constant rate in between (Obizhaeva and Wang 2013; Alfonsi, Fruth and Schied
2010), asserted by every suite. **Reduction (asserted):** with
`kappa = 2 eta / tau` and `rho*tau` large enough that `a == 0` (the suites
use `rho = 1e3`), `E[IS]` equals Section 1.3 for any schedule and the
optimum is the `lambda_ra = 0` Almgren-Chriss (TWAP) trajectory, i.e.
Section 8 with `alpha = 1`.

### 12.2 Adaptive re-planning (pinned)

Section 1 dynamics (`ACParams`) plus an observable AR(1) drift signal and
random partial fills: `phi` in `[0, 1)`, `signal_sd >= 0`, `fill_risk` in
`[0, 1)`. Over interval `j` the price drifts by `s_{j-1} tau`;
`s_0 = 0`, `s_j = phi s_{j-1} + sqrt(1 - phi^2) signal_sd eps_j`.

**Re-plan (pinned):** at decision index `k` (0-based) with holding `h`,
signal `s`, `m = N - k` intervals left: `m = 1` returns `[h, 0]`;
otherwise, with `e = eta_tilde/tau`, solve by Thomas the `(m-1)`-system
`diag = 2e + lambda_ra sigma^2 tau`, `off = -e`,
`rhs_i = tau phi^(i-1) s / 2 + (i == 1 ? e h : 0)` (`i = 1..m-1`), the
stationarity condition of
`sum (eta_tilde/tau) n^2 + lambda_ra sigma^2 tau sum x^2 - sum_i x_{k+i} tau phi^(i-1) s`.
The re-plan is linear in `(h, s)`; `replan_coefficients` returns `c[m]`,
`d[m]` (first-step responses to the two right-hand sides, `c[1] = 1`,
`d[1] = 0`) so the next child order is `c[m] h - d[m] s`. The static plan is
the re-plan at `k = 0`, `h = X`, `s = 0`; it equals the Section 1.4
trajectory (asserted, `1e-9 X`).

**Simulation (`simulate_adaptive`, loop and draw order pinned):** per path,
two strategies (static, adaptive) sharing the draws; `S_0 = 0`. For
`j = 1..N`: child orders `q` (static: the plan's `n_j`; adaptive:
`c[N-j+1] x - d[N-j+1] s_{j-1}`; at `j = N` both send their whole
remainder); draw `u ~ U(0,1)`, then `xi ~ N(0,1)`, then `eps ~ N(0,1)`
(always all three); executed `e = f q` with `f = 1 - fill_risk u` for
`j < N` and `f = 1` at `j = N`; proceeds `+= e (S - eta e / tau)`;
`S += s_{j-1} tau + sigma sqrt(tau) xi - gamma_p e` (each strategy has its
own `S`); `s_j` from `eps`; holdings `-= e`. `IS = -proceeds`. Report per
strategy the mean, sample std (ddof 1) and `SE = std/sqrt(M)`, and the
paired saving `static - adaptive` (mean and SE). Python draws vectors
across paths per step (Section 4); ports loop path by path with the same
per-path order. `2 <= n_paths <= MAX_PATHS`.

**Where adaptivity cannot help, and where it does (asserted):** with
`signal_sd = 0` and `fill_risk = 0` the two strategies are identical path by
path (`|saving| <= 1e-9 * static_mean`): the Almgren-Chriss optimum is time
consistent, so re-solving from the realised state reproduces the tail of the
plan whatever the price did. With a signal (`phi = 0.7`, `signal_sd = 0.5`)
or fill risk (`0.5`) the saving exceeds 4 SE. Almgren and Lorenz (2007) show
a different gain this model excludes: under a single mean-variance criterion
for the whole trade, a non-time-consistent adaptive rule can beat the static
optimum.

### 12.3 Internalisation (pinned)

`internalise(buys, sells, price, spread, adv, daily_vol, n_intervals, tau,
alpha = 0.5, y = 1, perm_frac = 0.10, weights = default)`: order sizes
finite `> 0`, at least one order. `B = sum(buys)`, `S = sum(sells)`;
crossed `min(B, S)` at the mid; residual `|B - S|` on the larger side
(`"none"` if equal). `C(Q)` = Section 8 `E[IS]` of `vwap_holdings(Q, w)`
under `impact_params_from_market(total_shares = Q, lambda_ra = 0, ...)`,
`C(0) = 0`. Baseline `C(B) + C(S)`; internalised `C(|B - S|)`; saving the
difference, `saved_bps = bps(saving, (B + S) price)`. Per order `q` on a side
with total `Q_side`: baseline `C(Q_side) q / Q_side`, internalised
`C(|B-S|) q / Q_side` on the residual side and `0` otherwise, reported as
`bps(baseline - internalised, q price)`. `C` is increasing, so crossing never
increases total cost (asserted on random order sets).

### 12.4 Golden cases

Kinds: `transient_schedule` (exact; `xfrac_01..xfrac_{N-1}` = `x_j / X` of
the optimum, `e_cost`, `twap_e_cost`), `internalise` (exact;
`crossed_shares`, `residual_shares`, `baseline_cost`, `internal_cost`,
`saved_cost`, `saved_bps`, `buy{i}_saved_bps`, `sell{i}_saved_bps`; inputs
carry `n_buys`, `buy_{i}`, `n_sells`, `sell_{i}`, `volume_curve` as in 8.5),
`adaptive_sim` (statistical, Section 4 rule with `tol = 4`; `static_mean`,
`adaptive_mean`, and `saving_mean` where re-planning can help, each with
`_se`; inputs are the `ac_*` keys plus `phi`, `signal_sd`, `fill_risk`,
`seed`, `n_paths_ref = 20000`). Exact tolerance `1e-9` with the Section 5
rule. Every suite asserts every key of every case and that the count is 8.

References: Obizhaeva and Wang, Journal of Financial Markets 16(1), 2013;
Alfonsi, Fruth and Schied, Quantitative Finance 10(2), 2010; Almgren and
Chriss, Journal of Risk 3(2), 2000; Almgren and Lorenz, Trading 2007(1).

## 13. Limit-order placement: queue fill probability, passive or aggressive, liquidation with limit and market orders (pinned)

Python `etrade.placement` is the reference; ports expose the same names in
their own style (C++ `etrade::queue_fill_probability`,
`etrade::LiquidationParams`, Rust `etrade::placement`, Java `Placement`).
Golden cases: `data/golden/placement_golden.json` (11 cases, generated by
`data/generate_placement.py`). Pinned bounds: `MAX_QUEUE_UNITS = 5000`,
`MAX_UNIFORMISATION_STEPS = 200000`, `MAX_LIQ_UNITS = 1000`,
`MAX_LIQ_PERIODS = 1000` (C++ `kMaxQueueUnits`, `kMaxUniformisationSteps`,
`kMaxLiqUnits`, `kMaxLiqPeriods`; Rust `etrade::placement::` and Java
`Placement.` with the Python names).

### 13.1 Fill probability by queue position

`queue_fill_probability(q_ahead, size, mu, theta, horizon)`. A FIFO queue
holds integer `q_ahead >= 0` units ahead of our resting order of integer
`size >= 1` units, `q_ahead + size <= MAX_QUEUE_UNITS`. Market orders of one
unit arrive as a Poisson process of rate `mu > 0` (units per time unit) and
execute the front of the queue; each unit ahead is cancelled independently
at rate `theta >= 0`; our units are never cancelled. With `horizon >= 0`
(all finite) the result is the probability of a **complete** fill by
`horizon`:

```
T = sum_{k=1}^{q_ahead} Exp(mu + theta k) + sum_{i=1}^{size} Exp(mu)      P = P(T <= horizon)
```

This is the pure-death queue of Cont, Stoikov and Talreja (2010) with
proportional cancellations; Cont and de Larrard (2013) use the same Poisson
order flow. `theta = 0` gives the Poisson tail
`P(N(horizon) >= q_ahead + size)`, and the depletion of the queue ahead
alone is `I_{1-exp(-theta t)}(q_ahead, mu/theta + 1)` (regularised
incomplete beta); both are asserted by the Python suite. The
partial-fraction closed form alternates in sign for deep queues and is not
used.

**Evaluation (pinned uniformisation).** `Lambda = mu + theta q_ahead`,
`L = Lambda horizon` (a non-finite `L` is an input error); `L = 0` returns
0. Stages `s = 0..S-1`, `S = q_ahead + size`, with rates
`r_s = mu + theta (q_ahead - s)` for `s < q_ahead` and `r_s = mu`
otherwise; `go_s = r_s / Lambda`, `stay_s = 1 - go_s`.

1. Window: `m = floor(L)`, `half = ceil(10 sqrt(L) + 10)`,
   `k_lo = max(0, m - half)`, `k_hi = m + half`;
   `k_hi > MAX_UNIFORMISATION_STEPS` is an input error ("horizon too long").
2. Poisson weights (unnormalised): `w_m = 1`; `w_k = w_{k-1} L / k` for
   `k = m+1..k_hi`; `w_k = w_{k+1} (k+1) / L` for `k = m-1` down to `k_lo`.
3. Chain: `p = e_0` (length `S`), `a = 0`. For `k = 0..k_hi`: if `k > 0`,
   `move_s = p_s go_s` and `p_s <- p_s stay_s + move_{s-1}` (`move_{-1} = 0`),
   `a <- a + move_{S-1}`; then, if `k >= k_lo`, `acc += w_k a` and
   `W += w_k` (both in increasing `k`).
4. `P = min(1, max(0, acc / W))`.

The Poisson mass outside the window is below `exp(-50)`. Tolerance `1e-10`.

### 13.2 Passive or aggressive for one child order

`place_or_cross(size, spread, fill_prob, adverse_move, eta, alpha, weight)`
for a sell child order of `size > 0` shares; `spread >= 0`,
`fill_prob` in `[0, 1]` (normally 13.1 over the posting horizon),
`adverse_move >= 0` (expected adverse mid move per share by the time an
unfilled order is completed), `eta >= 0`, `alpha` in `(0, 1]`, `weight > 0`
(the Section 8 parameters of the interval), all finite. Costs per share
relative to the current mid, `hs = spread / 2`:

```
impact     = eta * weight^(-alpha) * size^alpha       (= Section 8 E[IS] of [size, 0], / size)
aggressive = hs + impact
passive    = fill_prob (impact - hs) + (1 - fill_prob) (adverse_move + hs + impact)
breakeven  = adverse_move / (adverse_move + spread)    (1 when the denominator is 0)
passive chosen  iff  passive < aggressive              (ties cross)
```

The impact is charged to passive fills too: the square-root law measures
the price pressure of a metaorder's volume and holds for both order types
(Toth et al. 2011), so posting saves the spread, not the impact. Hence
`passive - aggressive = (1 - fill_prob) adverse_move - fill_prob spread`:
posting wins exactly when `fill_prob > breakeven`, and with a negligible
spread crossing is always (weakly) preferred (asserted by every suite).
Moallemi and Yuan (2016) value queue position on the same terms: the half
spread earned against the delay and adverse selection a resting order
bears. Returned: `passive_cost`, `aggressive_cost`, `impact_cost`,
`breakeven_fill_prob`, `passive` (bool).

### 13.3 Liquidation with limit and market orders

A discrete version of Cartea and Jaimungal (2015).
`LiquidationParams(total_units Q, n_periods N, dt, sigma, spread, kappa,
p_touch, beta, phi = 0)`: integers `Q` in `[1, MAX_LIQ_UNITS]`, `N` in
`[1, MAX_LIQ_PERIODS]`; `dt > 0`, `sigma >= 0`, `spread >= 0`, `kappa > 0`,
`p_touch` in `(0, 1]`, `beta >= 0`, `phi >= 0`, all finite. Each period
with inventory `q > 0` the trader either posts one unit at depth
`delta >= 0` beyond the touch, filled with probability
`p_touch exp(-kappa delta)` and earning `hs + delta` over the mid, or sends
a one-unit market order costing `hs + beta`. The remainder at the end is
swept at `hs + beta q` per unit; a running penalty `phi q^2 dt` per period
expresses urgency.

**Policy (`liquidation_policy`, pinned backward induction).**
`C_N(q) = q (hs + beta q)`; for `j = N-1` down to `0`: `C_j(0) = 0` and, for
`q = 1..Q`,

```
g      = hs + (C_{j+1}(q) - C_{j+1}(q-1))
d      = max(0, 1/kappa - g)
post   = C_{j+1}(q) - p_touch exp(-kappa d) (d + g)
market = hs + beta + C_{j+1}(q-1)
market chosen iff market < post (depth recorded as 0); otherwise depth d
C_j(q) = phi q^2 dt + min(post, market)
```

`d` is the exact maximiser of `exp(-kappa delta) (delta + g)` over
`delta >= 0`, so the depth falls as the marginal cost of holding one more
unit rises (asserted). `aggressive_schedule(Q, N)`:
`n_j = floor(Q (j+1)/N) - floor(Q j/N)` in integer arithmetic;
`aggressive_cost = sum_j n_j (hs + beta n_j)`.

**Simulation (`simulate_liquidation`, loop and draw order pinned).** Per
path, `S_0 = 0`; three strategies share the draws. For `j = 0..N-1`: draw
`u ~ U(0,1)` then `z ~ N(0,1)` (always both); at the current mid `S`:
optimal, with inventory `q > 0`: either the market order, proceeds
`+= S - hs - beta`, or the post at `d = depth[j][q]`, filled iff
`u < p_touch exp(-kappa d)`, proceeds `+= S + hs + d`; passive-then-sweep:
post at the touch, filled iff `u < p_touch` (and `q > 0`), proceeds
`+= S + hs`; aggressive: proceeds `+= n_j (S - hs - beta n_j)`. Each fill
or market order removes one unit. Then `S += sigma sqrt(dt) z`. Finally
each remainder `q` is swept, proceeds `+= q (S_N - hs - beta q)`, and
`IS = -proceeds`. Report per strategy the mean, sample std (ddof 1) and
`SE = std/sqrt(M)`, and the paired savings `aggressive - optimal` and
`passive - optimal` (mean and SE); `2 <= n_paths <= MAX_PATHS`. With
`phi = 0` the policy's mean equals `C_0(Q)` within 4 SE and beats both
baselines by more than 4 SE on the patient set (asserted); with `phi > 0`
it gives up expected cost for lower dispersion (the urgent golden set,
where posting at the touch and sweeping is cheaper on average). The model
has no adverse selection on fills and no permanent impact. Python draws
vectors across paths per period (Section 4); ports loop path by path with
the same per-path order.

### 13.4 Golden cases

Kinds: `fill_probability` (exact; `fill_prob`; inputs `q_ahead`, `size`,
`mu`, `theta`, `horizon`), `placement_decision` (exact; `fill_prob` from the
queue inputs, then `passive_cost`, `aggressive_cost`, `impact_cost`,
`breakeven_fill_prob` and `passive` as 1 or 0; inputs add `child_shares`
(the `size` of 13.2), `spread`, `adverse_move`, `eta`, `alpha`, `weight`),
`liquidation_policy` (exact; `dp_cost = C_0(Q)`, `depth_first = depth[0][Q]`,
`depth_last = depth[N-1][1]`, `n_market` = number of market states with
`q >= 1`, `aggressive_cost`), `liquidation_sim` (statistical, Section 4
rule with `tol = 4`; `optimal_mean`, `aggressive_mean`, `passive_mean`,
`saving_vs_aggressive_mean`, `saving_vs_passive_mean`, each with `_se`;
inputs add `seed` and `n_paths_ref = 20000`). Integer inputs are stored as
JSON numbers. Exact tolerance `1e-10` with the Section 5 rule. Every suite
asserts every key of every case and that the count is 11.

References: Cont, Stoikov and Talreja, Operations Research 58(3), 2010;
Cont and de Larrard, SIAM Journal on Financial Mathematics 4(1), 2013;
Moallemi and Yuan, "A model for queue position valuation in a limit order
book", working paper, Columbia Business School, 2016; Cartea and Jaimungal,
Quantitative Finance 15(8), 2015; Toth, Lemperiere, Deremble, de Lataillade,
Kockelkoren and Bouchaud, Physical Review X 1(2), 2011.

## 14. Hedged market making and risk controls (pinned)

The Section 9 maker can shed inventory only by skewing its own quotes. A desk
that makes markets in an ETF also trades the future or the basket: it lays
inventory risk off in a correlated instrument, pays that instrument's half
spread and fee, and keeps the basis risk. The same desk runs hard risk
controls outside the quoting model. Section 14 adds both to the Section 9
loop in a separate simulator; Section 9's loop and goldens are unchanged.
Python `etrade.hedging` is the reference; ports expose the same names in
their own style (C++ `etrade::simulate_hedged`, `etrade::HedgeConfig`,
`etrade::RiskConfig`; Rust `etrade::hedging`; Java `Hedging`). Golden cases:
`data/golden/hedging_golden.json` (18 cases: 8 exact, 10 statistical),
generated by `data/generate_hedging.py`.

### 14.1 Configuration

`simulate_hedged(params, n_paths, seed, config, hedge, risk)` takes the
Section 2 `ASParams`, the Section 9 `MMConfig` (9.1, same validation) and:

`HedgeConfig` (all finite): `rho` in `[-1, 1]` (correlation of B's move with
A's), `hedge_ratio >= 0` (B shares per A share; `0` = no hedging),
`hedge_cost >= 0` (per B share traded: half spread plus fee),
`hedge_threshold >= 0`, `hedge_target` in `[0, 1)`. Defaults all `0`.

`RiskConfig`: `loss_limit >= 0` finite (`0` = off), integer
`position_limit >= 0` (`0` = off), `flatten_penalty >= 0` finite (per share
flattened by a control), integer `throttle_updates >= 0` (`0` = off),
integer `throttle_window` in `[1, n_steps]` (default 1), integer
`kill_step` in `[-1, n_steps]` (`-1` = off, the default). Defaults are all
off. `n_paths` integer in `[2, MAX_PATHS]`; `latency_steps <= n_steps`.

### 14.2 Hedge sizing (pinned, exact)

`hedge_trade(q, y, hedge_ratio, hedge_threshold, hedge_target)`: with
`hedge_ratio = 0` return `0`; else `u = q + y/hedge_ratio` (A shares not
covered at the chosen ratio) and

```
dy = -hedge_ratio * (1 - hedge_target) * u   if |u| > hedge_threshold,   else 0
```

so a fraction `hedge_target` of `u` is left uncovered. `q`, `y` finite; the
bounds of 14.1 apply.

### 14.3 Instrument B and the skew exposure

B moves `sigma*sqrt(dt)*(rho*z + sqrt(1 - rho^2)*z_b)` per step: it is quoted
in A-equivalent units (same per-step volatility), so the minimum-variance
hedge ratio is `rho` (Ederington 1979). B starts at `s0`. The book's
per-step PnL variance is `sigma^2 dt (q^2 + 2 rho q y + y^2)`, whose
half-derivative in `q` is `sigma^2 dt x` with

```
x = q + rho*y        (risk-equivalent exposure in A shares)
```

and the quoting agents of 9.4 use `x` in place of `q` (the multi-asset
reservation price, Guéant 2017). Without a hedge `y = 0` and `x = q`. A full
hedge at `hedge_ratio = rho` leaves `x = (1 - rho^2) q`. The skew of the
desired quotes is `skew = x gamma sigma^2 (T - t)` (`as`), `x c2` (`glft`)
and `0` (`fixed`). The `q_max` cap of 9.4 still applies to `q`.

### 14.4 Simulation loop (ORDER PINNED)

State per path: the Section 9 state (`s`, `q`, `cash`, latency history), B
mid `b = s0`, B position `y = 0`, `active = true`, standing quotes
`bid_q = ask_q = s0`, the steps of the last `throttle_updates` quote updates
(all `-inf`), `mark_prev = 0`. Per step `i`, `t = i*dt`:

1. **Draws, always all six, in this order:** the five of 9.4 (`z`, `u_sig`,
   `u_ask`, `u_bid`, `u_inf`), then `z_b ~ N(0,1)`.
2. **Stops.** `mark = cash + q s + y b`. `kill = active and i == kill_step`;
   `loss = active and not kill and loss_limit > 0 and mark < -loss_limit`.
   On a loss stop record `overshoot = max(0, -loss_limit - mark)` and
   `stop_move = |mark - mark_prev|` (both 0 on paths that never stop on
   loss). On either stop flatten both legs at their mids:
   `cash += q s - flatten_penalty |q|`, `risk_costs += flatten_penalty |q|`,
   `q = 0`; `cash += y b - hedge_cost |y|`, `hedge_costs += hedge_cost |y|`,
   `hedge_trades += (y != 0)`, `y = 0`; then `active = false` for the rest of
   the horizon. `kill_stop` and `loss_stop` record the stop (0 or 1). Then
   `mark_prev = mark`.
3. **Desired quotes.** 9.4 steps 2 and 3 with `x = q + rho y` in place of
   `q`, then 9.2 rounding when `tick > 0`. If `active`,
   `sum_skew2 += skew^2`.
4. **Throttle.** `allow = active` when `throttle_updates = 0`; else
   `allow = active and (step of the throttle_updates-th most recent update)
   <= i - throttle_window`, i.e. at most `throttle_updates` updates in any
   `throttle_window` consecutive steps. If `allow`, the desired quotes become
   the standing quotes, the step is recorded and `quote_updates += 1`;
   otherwise the previous standing quotes stand (stale: fills and informed
   traders see them against the true mid).
5. **Fills.** 9.4 steps 4 to 6 against the standing quotes, both sides gated
   by `active` as well as `q_max`, with the same accumulators.
6. **Position limit.** If `active`, `position_limit > 0` and
   `|q| >= position_limit`: `cash += q s - flatten_penalty |q|`,
   `risk_costs += flatten_penalty |q|`, `pos_flattens += 1`, `q = 0`.
7. **Hedge.** If `active` and `hedge_ratio > 0`: `u = q + y/hedge_ratio`; if
   `|u| > hedge_threshold`, `dy = -hedge_ratio (1 - hedge_target) u`,
   `cash -= dy b + hedge_cost |dy|`, `hedge_costs += hedge_cost |dy|`,
   `hedge_trades += 1`, `y += dy`.
8. **Accounting and moves.** `move = sigma sqrt(dt) z`,
   `move_b = sigma sqrt(dt) (rho z + sqrt(1 - rho^2) z_b)`;
   `inventory += q move`, `markout += (fb - fa) move`,
   `hedge_pnl += y move_b`, `sum_q2 += q^2`, `sum_x2 += (q + rho y)^2`;
   `s += move`, `b += move_b`, push `s` into the history.

Terminal: `liquidation = eta_liq |q_T|`; the hedge is unwound,
`hedge_costs += hedge_cost |y_T|`, `hedge_trades += (y_T != 0)`;
`pnl = cash + q_T s_T + y_T b_T - liquidation - hedge_cost |y_T|`;
`rms_q = sqrt(sum_q2/n_steps)`, `rms_x = sqrt(sum_x2/n_steps)`,
`rms_skew = sqrt(sum_skew2/n_steps)`.

A stale quote deep through the mid has fill probability 1 (the exponential
overflows to infinity and `-expm1(-inf) = 1`).

With the default `HedgeConfig` and `RiskConfig` the loop is Section 9.4 plus
one unused draw, so the result equals Section 9 **in distribution** (not
sample by sample); every suite asserts this within the Section 4 rule.

### 14.5 Attribution and statistics (pinned)

Per path, exactly (relative to `max(1, |pnl|)`, tolerance `1e-9`, asserted
by every suite):

```
pnl = spread + fees + inventory - liquidation + hedge_pnl - hedge_costs - risk_costs
```

Statistics, each a per-path array: the nine of 9.5, then `hedge_pnl,
hedge_costs, hedge_trades, risk_costs, rms_x, rms_skew, quote_updates,
loss_stop, kill_stop, pos_flattens, overshoot, stop_move`. `mean` and `se`
as 9.5. `var(stat) = m2 = mean((v - mean)^2)` (population) with the
delta-method standard error `var_se = sqrt(max(0, m4 - m2^2)/n_paths)`,
`m4 = mean((v - mean)^4)`. `quantile(stat, p)` is 14.6 over paths. Unknown
names are input errors. The result reports the largest identity residual.

The loss limit never lets the mark at the stop fall below the limit by more
than that step's change of the mark: `overshoot <= stop_move` on every path
(asserted). The overshoot is the gap risk of a control checked once per
step; the flatten penalty and B's unwind cost come on top of it.

### 14.6 Quantile (pinned, exact)

`quantile(values, p)`: Hyndman and Fan (1996) type 7, the default of NumPy
and R. Sort ascending to `v` (length `n >= 1`, finite), `h = (n - 1) p`,
`j = floor(h)`; return `v[n-1]` if `j >= n - 1`, else
`v[j] + (h - j)(v[j+1] - v[j])`. `p` in `[0, 1]`.

### 14.7 Golden cases

Exact kinds (`tol 1e-12`, Section 5 rule): `hedge_trade` (inputs `q`, `y`,
`hedge_ratio`, `hedge_threshold`, `hedge_target`; expect `dy`) and
`hedge_quantile` (inputs `v_0..v_9`, `p`; expect `q`). Statistical kind
`hedge_sim` (`tol = 4.0`, Section 4 rule): inputs are the `mm_sim` inputs of
9.7 plus the 14.1 fields (integers stored as JSON numbers), `seed` and
`n_paths_ref` (20000); expect `pnl, spread, fees, inventory, hedge_pnl,
hedge_costs, risk_costs, fills, rms_q, rms_x, rms_skew, quote_updates,
loss_stop, kill_stop, pos_flattens`, each with `_se`, and `pnl_var` with
`pnl_var_se` (14.5). A key whose two standard errors are both 0 must match
within `1e-12`. Ports run their own RNG (Section 4) with at least 2000
paths. Every suite asserts every key of every case, that the count is 18,
and the attribution identity on every simulated path.

What the goldens show (Python, 20000 paths, `us_large_cap`, one-tick grid,
0.2-cent rebate, 15% informed arrivals, B at `rho = 0.9`, hedge cost half a
cent; PnL per session, 5% quantile by 14.6):

| case | mean pnl | sd | 5% quantile | rms skew |
|---|---|---|---|---|
| A-S, unhedged | 4.00 | 1.45 | 1.59 | 0.0067 |
| A-S, full hedge | 7.83 | 1.36 | 5.59 | 0.0031 |
| fixed spread, unhedged | 3.83 | 25.80 | -40.41 | 0 |
| fixed spread, full hedge | 7.92 | 11.39 | -11.18 | 0 |
| fixed, loss limit 5 | 2.17 | 14.22 | -5.83 | 0 |
| fixed, position limit 10 | 4.16 | 6.04 | -5.86 | 0 |
| A-S, throttle 1 per 3 steps | -32.86 | 2.98 | -37.98 | 0.0135 |
| fixed, kill at 3/4 of the session | 3.13 | 19.86 | -30.68 | 0 |

The hedge halves the A-S skew: the book stops giving away edge to shed
inventory, earns more spread and pays about 3 per session in hedge costs for
it. For the fixed-spread agent, which cannot skew, it cuts the PnL standard
deviation by more than half. The loss and position limits lift the 5%
quantile from -40 to about -6, the loss limit at a cost in mean; the
throttle turns the book into a loss because informed traders pick off stale
quotes; the kill switch gives up the last quarter's spread for less
inventory risk. These are illustrative parameters, not a calibration.

References: Ederington, Journal of Finance 34(1), 1979 (minimum-variance
hedge ratio); Guéant, *The Financial Mathematics of Market Liquidity*,
Chapman and Hall/CRC, 2017 (multi-asset market making); Hyndman and Fan,
The American Statistician 50(4), 1996 (sample quantiles).
