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

Names per language: Python `et.MAX_AC_INTERVALS` etc.; C++
`etrade::kMaxAcIntervals`, `kMaxPovIntervals`, `kMaxPaths`, `kMaxAsSteps`,
`kMaxAsPaths`; Rust `etrade::MAX_AC_INTERVALS`, `MAX_POV_INTERVALS`,
`MAX_PATHS`, `MAX_AS_STEPS`, `MAX_AS_PATHS`; Java
`AlmgrenChriss.MAX_AC_INTERVALS`, `MAX_POV_INTERVALS`, `MAX_PATHS`,
`AvellanedaStoikov.MAX_AS_STEPS`, `MAX_AS_PATHS`.

Units are abstract but must be consistent within a parameter set; nothing
ties `tau`/`dt` to a clock or calendar (no day-count, no holidays, no
intraday volume curve):

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
  `M = ceil(X/c - 1e-12)`, holdings `x_i = max(X - i*c, 0)` for `i = 0..M`,
  `x_M = 0` exactly.  **Bound (pinned):** `X/c <= MAX_POV_INTERVALS` is
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
| Java | `java.util.SplittableRandom(seed)` (`nextDouble`, `nextGaussian`) | bit-stable within one JVM build and platform; `nextDouble` is specified, but `nextGaussian` is a `RandomGenerator` default and `Math.*` may differ by 1 ulp across platforms, so cross-JDK bit-stability is not guaranteed |

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

No public function returns NaN or infinity from finite valid inputs, and no
loop or allocation is unbounded (Section 0).

Non-errors (documented behavior): negative prices (arithmetic walks);
`kappa = 0` (TWAP branch, half-life = infinity); a quote through the mid
(`delta < 0`); `q_max = 0` (no quoting, zero PnL); `x_j` underflowing to
exactly `0` far out on a very urgent schedule; a participation rate of
exactly 1.
