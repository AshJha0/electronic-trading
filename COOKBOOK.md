# COOKBOOK — `etrade` recipes

Task-oriented recipes in all four languages. Python snippets run as-is
against the reference package (`cd python && PYTHONPATH=src python3`, then
paste; `import etrade as et` is assumed once per session). Every C++, Rust
and Java snippet below has been compiled and run against the shipped
library (C++17 with `-I cpp/include` linking `libetrade.a`; Rust as a
`main()` body returning `Result<(), Box<dyn std::error::Error>>` so `?`
works; Java inside a class in package `io.github.ashjha0.etrade` or with the
imports shown). The Rust snippets assume

```rust
use etrade::{ACParams, ASParams, Agent, SimConfig, Side};
```

once per file, and the Java ones

```java
import io.github.ashjha0.etrade.*;
import io.github.ashjha0.etrade.AlmgrenChriss.FrontierPoint;
import io.github.ashjha0.etrade.AlmgrenChriss.ISSimResult;
import io.github.ashjha0.etrade.Analytics.ISDecomposition;
import io.github.ashjha0.etrade.Analytics.OLSResult;
```

Sample simulation numbers in comments are the Python demo's 2,000-path
values; the native ports reproduce them statistically (4-SE bands), not
bit-for-bit. Deterministic values are shown to 15 significant digits: the
16th and 17th can differ by an ulp between platforms and languages (for
example `kappa` prints as `...561` or `...562`), which the golden tolerances
absorb.

Parameter values throughout are the liquid large-cap set
(`data/params_liquid.json`): X = 1e6 shares, N = 5, tau = 1, sigma = 0.95,
eta = 2.5e-6, gamma_p = 2.5e-7, lambda_ra = 2e-6; A-S: s0 = 100, sigma = 2,
T = 1, dt = 0.005, gamma = 0.1, k = 1.5, A = 140, q_max = 20, eta_liq = 0.5.

---

## 1. How do I compute the optimal execution schedule for a sell order?

Build `ACParams`, then ask for `kappa`, the holdings trajectory and the
trade list.

**Python**

```python
import etrade as et

p = et.ACParams(total_shares=1_000_000.0, n_intervals=5, tau=1.0,
                sigma=0.95, eta=2.5e-6, gamma_p=2.5e-7, lambda_ra=2e-6)
print(et.ac_kappa(p))        # 0.846297134501256
print(et.half_life(p))       # 1.18161808569673
print(et.ac_trajectory(p))   # [1000000, 428598.85, 182932.81, 76295.72, 27643.38, 0]
print(et.ac_trade_list(p))   # [571401.15, 245666.03, 106637.09, 48652.34, 27643.38], sums to X
```

**C++**

```cpp
#include <etrade/almgren_chriss.hpp>

etrade::ACParams p{1'000'000.0, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6};
double kappa = etrade::ac_kappa(p);            // 0.846297134501256
double theta = etrade::half_life(p);           // 1.1816...
std::vector<double> x = etrade::ac_trajectory(p);   // size N+1, x[0]=X, x[N]=0
std::vector<double> n = etrade::ac_trade_list(p);   // size N, sums to X
```

**Rust**

```rust
let p = ACParams::new(1_000_000.0, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6)?;
let kappa = etrade::ac_kappa(&p);              // 0.846297134501256
let theta = etrade::half_life(&p);             // 1.1816... (f64::INFINITY if kappa == 0)
let x: Vec<f64> = etrade::ac_trajectory(&p);   // len N+1, x[0] == X, x[N] == 0
let n: Vec<f64> = etrade::ac_trade_list(&p);   // len N, sums to X
```

**Java**

```java
import io.github.ashjha0.etrade.ACParams;
import io.github.ashjha0.etrade.AlmgrenChriss;

ACParams p = new ACParams(1_000_000.0, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6);
double kappa = AlmgrenChriss.kappa(p);         // 0.846297134501256
double theta = AlmgrenChriss.halfLife(p);      // 1.1816... (Double.POSITIVE_INFINITY if 0)
double[] x = AlmgrenChriss.trajectory(p);      // length N+1
double[] n = AlmgrenChriss.tradeList(p);       // length N
```

---

## 2. How do I get E[cost] and Var[cost] of a schedule (optimal or TWAP)?

`expected_cost` / `variance_cost` evaluate the pinned discrete sums; pass no
holdings for the optimal trajectory, or any fully liquidating path.

**Python**

```python
e_opt, v_opt = et.expected_cost(p), et.variance_cost(p)     # 1078215.167..., 2.0193e11
twap = et.twap_holdings(p.total_shares, p.n_intervals)
e_tw, v_tw = et.expected_cost(p, twap), et.variance_cost(p, twap)  # 600000.0, 1.083e12
```

**C++**

```cpp
double e_opt = etrade::expected_cost(p);                 // optimal trajectory
double v_opt = etrade::variance_cost(p);
std::vector<double> twap = etrade::twap_holdings(p.total_shares, p.n_intervals);
double e_tw = etrade::expected_cost(p, twap);            // 600000.0
double v_tw = etrade::variance_cost(p, twap);            // 1.083e12
```

**Rust** (fields are private — use the accessor methods; both cost
functions return `Result`)

```rust
let e_opt = etrade::expected_cost(&p, None)?;
let v_opt = etrade::variance_cost(&p, None)?;
let twap = etrade::twap_holdings(p.total_shares(), p.n_intervals())?;
let e_tw = etrade::expected_cost(&p, Some(&twap))?;      // 600000.0
let v_tw = etrade::variance_cost(&p, Some(&twap))?;      // 1.083e12
```

**Java**

```java
double eOpt = AlmgrenChriss.expectedCost(p);
double vOpt = AlmgrenChriss.varianceCost(p);
double[] twap = AlmgrenChriss.twapHoldings(p.totalShares(), p.nIntervals());
double eTw = AlmgrenChriss.expectedCost(p, twap);        // 600000.0
double vTw = AlmgrenChriss.varianceCost(p, twap);        // 1.083e12
```

---

## 3. How do I sweep the efficient frontier of execution?

Each point carries `(lambda_ra, kappa, E, Var, sqrt(Var))`. The grid must be
non-empty (input error otherwise).

**Python**

```python
grid = [0.0, 1e-8, 1e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5]
for fp in et.efficient_frontier(p, grid):
    print(f"{fp.lambda_ra:.1e}  E={fp.e_cost:,.0f}  std={fp.std_cost:,.0f}  k={fp.kappa:.4f}")
```

**C++**

```cpp
std::vector<double> grid{0.0, 1e-8, 1e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5};
for (const etrade::FrontierPoint& fp : etrade::efficient_frontier(p, grid))
    std::printf("%.1e  E=%.0f  std=%.0f  k=%.4f\n",
                fp.lambda_ra, fp.e_cost, fp.std_cost, fp.kappa);
```

**Rust**

```rust
let grid = [0.0, 1e-8, 1e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5];
for fp in etrade::efficient_frontier(&p, &grid)? {
    println!("{:.1e}  E={:.0}  std={:.0}  k={:.4}",
             fp.lambda_ra, fp.e_cost, fp.std_cost, fp.kappa);
}
```

**Java**

```java
double[] grid = {0.0, 1e-8, 1e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5};
for (FrontierPoint fp : AlmgrenChriss.efficientFrontier(p, grid))
    System.out.printf("%.1e  E=%.0f  std=%.0f  k=%.4f%n",
                      fp.lambdaRa(), fp.eCost(), fp.stdCost(), fp.kappa());
```

---

## 4. How do I build and cost a POV (percentage-of-volume) schedule?

POV determines its own duration: clip `c = p*V*tau`, `M = ceil(X/c)`
intervals, last clip partial. Cost it with the same E/Var functions. A clip
too small for the order (`X/c > 10,000,000`) is an input error, not a huge
allocation.

**Python**

```python
pov = et.pov_holdings(1_000_000.0, tau=1.0, market_volume_rate=2_500_000.0,
                      participation=0.15)
print(len(pov) - 1)                      # 3 intervals used
print(et.expected_cost(p, pov))          # 941406.25
print(et.variance_cost(p, pov))          # 408945312500.0
```

**C++**

```cpp
std::vector<double> pov = etrade::pov_holdings(1'000'000.0, 1.0, 2'500'000.0, 0.15);
std::size_t intervals = pov.size() - 1;        // 3
double e = etrade::expected_cost(p, pov);      // 941406.25
double v = etrade::variance_cost(p, pov);      // 4.08945e11
```

**Rust**

```rust
let pov = etrade::pov_holdings(1_000_000.0, 1.0, 2_500_000.0, 0.15)?;
let intervals = pov.len() - 1;                     // 3
let e = etrade::expected_cost(&p, Some(&pov))?;    // 941406.25
let v = etrade::variance_cost(&p, Some(&pov))?;    // 4.08945e11
```

**Java**

```java
double[] pov = AlmgrenChriss.povHoldings(1_000_000.0, 1.0, 2_500_000.0, 0.15);
int intervals = pov.length - 1;                    // 3
double e = AlmgrenChriss.expectedCost(p, pov);     // 941406.25
double v = AlmgrenChriss.varianceCost(p, pov);     // 4.08945e11
```

---

## 5. How do I verify the closed forms with Monte Carlo?

Simulate the arithmetic walk and check the sample mean/variance land within
a few standard errors of the closed forms. The schedule must fully liquidate
(a partial schedule is an input error, as for `expected_cost`).

**Python**

```python
mc = et.simulate_is(p, n_paths=20_000, seed=123)
assert abs(mc.mean - et.expected_cost(p)) < 4 * mc.se_mean
assert abs(mc.var - et.variance_cost(p)) < 4 * mc.se_var
print(mc.mean, mc.se_mean)               # 1075336.37..., 3176.97...
```

**C++**

```cpp
etrade::ISSimResult mc = etrade::simulate_is(p, 20'000, /*seed=*/123);
assert(std::abs(mc.mean - etrade::expected_cost(p)) < 4.0 * mc.se_mean);
assert(std::abs(mc.var - etrade::variance_cost(p)) < 4.0 * mc.se_var);
```

**Rust**

```rust
let mc = etrade::simulate_is(&p, None, 20_000, 123)?;
assert!((mc.mean - etrade::expected_cost(&p, None)?).abs() < 4.0 * mc.se_mean);
assert!((mc.var - etrade::variance_cost(&p, None)?).abs() < 4.0 * mc.se_var);
```

**Java**

```java
ISSimResult mc = AlmgrenChriss.simulateIS(p, 20_000, 123L);
assert Math.abs(mc.mean() - AlmgrenChriss.expectedCost(p)) < 4 * mc.seMean();
assert Math.abs(mc.var() - AlmgrenChriss.varianceCost(p)) < 4 * mc.seVar();
```

Note: only Python's PCG64 sample values are stored in `golden.json`; other
languages use their own RNG (C++ `std::mt19937_64`, Rust `ChaCha12Rng`,
Java `SplittableRandom`) and must agree *statistically* (4-SE band).

---

## 6. How do I quote like Avellaneda-Stoikov?

Reservation price, optimal spread, and the (bid, ask) pair for a given
state `(s, q, t)`; `s` and `q` must be finite and `t` in `[0, T]`.

**Python**

```python
ap = et.ASParams(s0=100.0, sigma=2.0, horizon=1.0, dt=0.005, gamma=0.1,
                 k=1.5, a_rate=140.0, q_max=20, eta_liq=0.5)
r = et.reservation_price(ap, s=100.0, q=3.0, t=0.3)   # 99.16
sp = et.optimal_spread(ap, t=0.3)                     # 1.57077042275142
bid, ask = et.quotes(ap, 100.0, 3.0, 0.3)             # (98.3746..., 99.9454...)
```

**C++**

```cpp
etrade::ASParams ap{100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, 20, 0.5};
double r  = etrade::reservation_price(ap, 100.0, 3.0, 0.3);   // 99.16
double sp = etrade::optimal_spread(ap, 0.3);                  // 1.57077...
auto [bid, ask] = etrade::quotes(ap, 100.0, 3.0, 0.3);
```

**Rust**

```rust
let ap = ASParams::new(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, 20, 0.5)?;
let r  = etrade::reservation_price(&ap, 100.0, 3.0, 0.3)?;    // 99.16
let sp = etrade::optimal_spread(&ap, 0.3)?;                   // 1.57077...
let (bid, ask) = etrade::quotes(&ap, 100.0, 3.0, 0.3)?;
```

**Java**

```java
ASParams ap = new ASParams(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, 20, 0.5);
double r  = AvellanedaStoikov.reservationPrice(ap, 100.0, 3.0, 0.3);  // 99.16
double sp = AvellanedaStoikov.optimalSpread(ap, 0.3);                 // 1.57077...
AvellanedaStoikov.Quotes qq = AvellanedaStoikov.quotes(ap, 100.0, 3.0, 0.3);
double bid = qq.bid();                                                // 98.3746...
double ask = qq.ask();                                                // 99.9454...
```

---

## 7. How do I get the fill probability of a quote at distance delta?

`lambda(delta) = A*exp(-k*delta)`, `P = 1 - exp(-lambda*dt)`. Delta is
signed — a quote through the mid (`delta < 0`) is valid and fills faster.

**Python**

```python
lam = et.intensity(ap, delta=0.6)          # 56.9197523636839
pf  = et.fill_probability(ap, delta=0.6)   # 0.247683948306426
```

**C++**

```cpp
double lam = etrade::intensity(ap, 0.6);          // 56.9197523636839
double pf  = etrade::fill_probability(ap, 0.6);   // 0.247683948306426
```

**Rust**

```rust
let lam = etrade::intensity(&ap, 0.6)?;           // 56.9197523636839
let pf  = etrade::fill_probability(&ap, 0.6)?;    // 0.247683948306426
```

**Java**

```java
double lam = AvellanedaStoikov.intensity(ap, 0.6);        // 56.9197...
double pf  = AvellanedaStoikov.fillProbability(ap, 0.6);  // 0.2476839...
```

---

## 8. How do I run the market-making simulation and compare A-S vs a fixed spread?

Same parameters, same seed, two agents. The fixed agent defaults to the
time-averaged A-S spread, so the comparison isolates the value of skew.
`mean_fills` is the total number of fills per path, both sides counted.

**Python**

```python
res_as = et.simulate(ap, n_paths=2000, seed=42, agent="as")
res_fx = et.simulate(ap, n_paths=2000, seed=42, agent="fixed")
print(res_as.mean_pnl, res_as.std_pnl, res_as.rms_q, res_as.mean_fills)  # ~55.9, ~6.4, ~1.45, ~85.6
print(res_fx.mean_pnl, res_fx.std_pnl, res_fx.rms_q, res_fx.mean_fills)  # ~57.8, ~12.8, ~4.96, ~81.8
assert res_as.rms_q < res_fx.rms_q                      # inventory control works
assert not res_as.cap_violated                          # |q| <= q_max always
```

**C++** (the agent is an enum; the fixed spread defaults to the NaN
sentinel `kDefaultFixedSpread`, which selects `average_as_spread(ap)`)

```cpp
etrade::ASSimResult a = etrade::simulate(ap, 2000, 42, etrade::Agent::kAS);
etrade::ASSimResult f = etrade::simulate(ap, 2000, 42, etrade::Agent::kFixed);
assert(a.rms_q() < f.rms_q());
assert(!a.cap_violated);
double fills = a.mean_fills();                 // ~85 (both sides counted)
```

**Rust** (configuration goes through `SimConfig`; `fixed_spread: None`
selects `average_as_spread`)

```rust
let a = etrade::simulate(&ap, 2000, 42, &SimConfig::default())?;
let fixed = SimConfig { agent: Agent::Fixed, ..SimConfig::default() };
let f = etrade::simulate(&ap, 2000, 42, &fixed)?;
assert!(a.rms_q() < f.rms_q());
assert!(!a.cap_violated);
let fills = a.mean_fills();                    // ~85 (both sides counted)
```

**Java** (the agent is an enum; there is no NaN sentinel — the short
overloads use `averageAsSpread(ap)` for the fixed agent)

```java
ASSimResult a = AvellanedaStoikov.simulate(ap, 2000, 42L, AvellanedaStoikov.Agent.AS);
ASSimResult f = AvellanedaStoikov.simulate(ap, 2000, 42L, AvellanedaStoikov.Agent.FIXED);
if (!(a.rmsQ() < f.rmsQ())) throw new AssertionError("inventory control");
if (a.capViolated()) throw new AssertionError("cap");
double fills = a.meanFills();                  // ~85 (both sides counted)
```

---

## 9. How do I stress-test the market maker with a mid-simulation volatility shock?

`stress_factor` multiplies the *realised* volatility from
`t >= stress_time_frac * T`; the agent keeps quoting with the base sigma
(model misspecification scenario).

**Python**

```python
base   = et.simulate(ap, 2000, 42, agent="as")
stress = et.simulate(ap, 2000, 42, agent="as", stress_factor=2.0,
                     stress_time_frac=0.5)
print(base.std_pnl, "->", stress.std_pnl)      # ~6.37 -> ~7.90
```

**C++**

```cpp
etrade::ASSimResult base   = etrade::simulate(ap, 2000, 42, etrade::Agent::kAS);
etrade::ASSimResult stress = etrade::simulate(ap, 2000, 42, etrade::Agent::kAS,
                                              etrade::kDefaultFixedSpread,
                                              /*stress_factor=*/2.0,
                                              /*stress_time_frac=*/0.5);
assert(stress.std_pnl() > base.std_pnl());
```

**Rust**

```rust
let base = etrade::simulate(&ap, 2000, 42, &SimConfig::default())?;
let shock = SimConfig { stress_factor: 2.0, stress_time_frac: 0.5, ..SimConfig::default() };
let stress = etrade::simulate(&ap, 2000, 42, &shock)?;
assert!(stress.std_pnl() > base.std_pnl());
```

**Java** (the full overload needs an explicit fixed spread — pass
`averageAsSpread(ap)`; `Double.NaN` is rejected)

```java
ASSimResult base   = AvellanedaStoikov.simulate(ap, 2000, 42L);
ASSimResult stress = AvellanedaStoikov.simulate(ap, 2000, 42L, AvellanedaStoikov.Agent.AS,
                                                AvellanedaStoikov.averageAsSpread(ap),
                                                2.0, 0.5);
if (!(stress.stdPnl() > base.stdPnl())) throw new AssertionError("stress");
```

---

## 10. How do I decompose an order's implementation shortfall?

Perold delay / trading / opportunity; the components sum to the total
exactly.

**Python**

```python
d = et.decompose_shortfall("sell", decision_price=100.0, arrival_price=99.5,
                           close_price=98.8, total_qty=1000.0,
                           fill_qtys=[300.0, 400.0, 200.0],
                           fill_prices=[99.4, 99.2, 99.1])
print(d.delay, d.trading, d.opportunity, d.total)
# 450.0 229.9999999999983 120.00000000000028 799.9999999999986 (230, 120, 800 up to rounding)
assert abs(d.total - (d.delay + d.trading + d.opportunity)) < 1e-10
```

**C++** (the side is an enum)

```cpp
etrade::ISDecomposition d = etrade::decompose_shortfall(
    etrade::Side::kSell, 100.0, 99.5, 98.8, 1000.0,
    {300.0, 400.0, 200.0}, {99.4, 99.2, 99.1});
// d.delay == 450, d.trading == 230, d.opportunity == 120, d.total == 800
```

**Rust** (the side is an enum; `"sell".parse::<Side>()?` also works)

```rust
let d = etrade::decompose_shortfall(Side::Sell, 100.0, 99.5, 98.8, 1000.0,
                                    &[300.0, 400.0, 200.0],
                                    &[99.4, 99.2, 99.1])?;
assert!((d.total - (d.delay + d.trading + d.opportunity)).abs() < 1e-10);
```

**Java**

```java
ISDecomposition d = Analytics.decomposeShortfall(
    "sell", 100.0, 99.5, 98.8, 1000.0,
    new double[]{300.0, 400.0, 200.0}, new double[]{99.4, 99.2, 99.1});
// d.delay() == 450, d.trading() == 230, d.opportunity() == 120, d.total() == 800
```

---

## 11. How do I estimate market impact from fills?

Regress per-share cost on participation rate with the tiny native OLS. The
pinned synthetic fills are deterministic and identical in every language.
`participation_rate` requires `executed <= market` (market volume includes
your own fills), so a rate above 1 is an input error.

**Python**

```python
parts, costs = et.synthetic_impact_fills(25, 0.5, 8.0, 0.05)
reg = et.estimate_impact(parts, costs)
print(reg.slope, reg.intercept, reg.r_squared)
# 7.97066529117945 0.503539513916228 0.99342097759679
part = et.participation_rate(120_000.0, 1_500_000.0)   # 0.08
```

**C++**

```cpp
auto [parts, costs] = etrade::synthetic_impact_fills(25, 0.5, 8.0, 0.05);
etrade::OLSResult reg = etrade::estimate_impact(parts, costs);
double part = etrade::participation_rate(120'000.0, 1'500'000.0);   // 0.08
```

**Rust**

```rust
let (parts, costs) = etrade::synthetic_impact_fills(25, 0.5, 8.0, 0.05)?;
let reg = etrade::estimate_impact(&parts, &costs)?;   // slope ~7.9707
let part = etrade::participation_rate(120_000.0, 1_500_000.0)?;     // 0.08
```

**Java**

```java
double[][] pc = Analytics.syntheticImpactFills(25, 0.5, 8.0, 0.05);
OLSResult reg = Analytics.estimateImpact(pc[0], pc[1]);   // slope ~7.9707
double part = Analytics.participationRate(120_000.0, 1_500_000.0);  // 0.08
```

---

## 12. How do I handle invalid inputs in each language?

All validation is at construction/call time; nothing is silently clamped.
The canonical example is the AC model-validity failure
`gamma_p*tau >= 2*eta` (`eta_tilde <= 0`).

**Python** — catch `ValueError`:

```python
try:
    et.ACParams(1e6, 5, 1.0, 0.95, eta=1e-7, gamma_p=1e-6, lambda_ra=0.0)
except ValueError as e:
    print("rejected:", e)     # model invalid: gamma_p*tau >= 2*eta ...
```

**C++** — catch `std::invalid_argument` (general) / `std::domain_error`
(model validity):

```cpp
try {
    etrade::ACParams bad{1e6, 5, 1.0, 0.95, 1e-7, 1e-6, 0.0};
} catch (const std::domain_error& e) {
    std::cerr << "rejected: " << e.what() << '\n';
}
```

**Rust** — constructors and fallible functions return
`Result<_, EtradeError>`; no panics on bad input:

```rust
match ACParams::new(1e6, 5, 1.0, 0.95, 1e-7, 1e-6, 0.0) {
    Err(e) => eprintln!("rejected: {e}"),
    Ok(_) => unreachable!(),
}
```

**Java** — catch `IllegalArgumentException`:

```java
try {
    new ACParams(1e6, 5, 1.0, 0.95, 1e-7, 1e-6, 0.0);
} catch (IllegalArgumentException ex) {
    System.err.println("rejected: " + ex.getMessage());
}
```

Other input errors, identical in every language: any NaN/inf (parameters,
holdings paths, quote state `s`/`q`, volumes, `fixed_spread`), a schedule
that does not liquidate (`expected_cost` *and* `simulate_is`), an empty or
non-finite `lambda_ra` grid, participation above 1, and any size beyond the
pinned bounds (`n_intervals`, POV clips `X/clip`, steps `T/dt`, `n_paths`
all capped at 10,000,000 — checked before anything is allocated).

Documented **non-errors** in every language: negative prices (arithmetic
walks), `kappa = 0` (TWAP branch, half-life = infinity), a quote through
the mid (`delta < 0`), `q_max = 0` (no quoting, zero PnL), and `x_j`
underflowing to exactly 0 far out on a very urgent schedule.

---

## 13. How do I set parameters in market units and report in bps?

Describe the instrument the way a desk quotes it and let the market layer
derive model units (API_SPEC Section 7). Execution impact uses the
Almgren-Chriss calibration: one spread of temporary impact per 1% of ADV
traded per day, one spread of permanent impact per 10% of ADV.

**Python**

```python
from etrade import market as mk

p = mk.ac_params_from_market(price=100.0, spread=0.01, adv=5e6, daily_vol=0.015,
                             total_shares=250_000.0, n_intervals=13, tau=1/13,
                             lambda_ra=1e-7)
notional = 250_000.0 * 100.0
print(mk.bps(et.expected_cost(p), notional))            # 5.345 (bps)
print(mk.bps(mk.half_spread_cost(0.01, 250_000.0), notional))   # 0.5 (bps)
a = mk.as_params_from_market(price=100.0, tick=0.01, daily_vol=0.015,
                             touch_fills_per_day=1170.0, halving_ticks=0.462,
                             skew_ticks=2.25, q_max=50, liq_ticks=1.0, dt_seconds=5.0)
print(mk.bps(et.optimal_spread(a, 0.0), a.s0))          # 3.583 (bps, at the open)
```

**C++**

```cpp
#include <etrade/market.hpp>

etrade::ACMarket m;
m.price = 100.0; m.spread = 0.01; m.adv = 5e6; m.daily_vol = 0.015;
m.total_shares = 250'000.0; m.n_intervals = 13; m.tau = 1.0 / 13.0; m.lambda_ra = 1e-7;
etrade::ACParams p = etrade::ac_params_from_market(m);
double e_bps = etrade::bps(etrade::expected_cost(p), m.total_shares * m.price);  // 5.35
```

**Rust**

```rust
use etrade::market::{ac_params_from_market, bps, ACMarket};

let m = ACMarket { price: 100.0, spread: 0.01, adv: 5e6, daily_vol: 0.015,
                   total_shares: 250_000.0, n_intervals: 13, tau: 1.0 / 13.0,
                   lambda_ra: 1e-7, ..ACMarket::default() };
let p = ac_params_from_market(&m)?;
let e_bps = bps(etrade::expected_cost(&p, None)?, m.total_shares * m.price)?;  // 5.35
```

**Java**

```java
Market.ACMarket m = Market.ACMarket.of(100.0, 0.01, 5e6, 0.015, 250_000.0, 13, 1.0 / 13.0, 1e-7);
ACParams p = Market.acParamsFromMarket(m);
double eBps = Market.bps(AlmgrenChriss.expectedCost(p), m.totalShares() * m.price());  // 5.35
```

The presets in `data/presets.json` (US large cap, US small cap, US ETF,
EUR/USD and the Almgren-Chriss paper example) use exactly these inputs; the
paper preset reproduces `params_liquid.json`. They are illustrative, not
fitted to data, and the linear impact under-prices small orders relative to
the square-root law (recipe 14).

---

## 14. How do I get a pre-trade cost estimate with square-root impact?

Build the order in market units, pick the impact exponent (`alpha = 0.5`
is the square-root law, the default) and ask for the pre-trade report or an
optimal schedule (API_SPEC Section 8). A volume curve makes VWAP and TWAP
differ; a drift in bps per day makes the schedule account for the
information behind the order.

**Python**

```python
from etrade import impact as im, market as mk

order = dict(price=100.0, spread=0.01, adv=5e6, daily_vol=0.015,
             total_shares=250_000.0, n_intervals=13, tau=1/13,
             weights=im.US_EQUITY_VOLUME_CURVE_13)
rep = mk.pretrade_report(lambdas=[0.0, 1e-6], **order)
print(rep.sqrt_law_bps)                   # 33.541
print([(r.label, round(r.impact_bps, 2)) for r in rep.rows])
# [('twap', 34.96), ('vwap', 33.77), ('optimal', 33.77), ('optimal', 36.95)]
p = mk.impact_params_from_market(lambda_ra=0.0, drift_bps_per_day=50.0, **order)
s = im.optimal_schedule(p)                # drift-aware schedule
print(mk.bps(im.impact_expected_cost(p, s.holdings), 2.5e7))   # 52.130
```

**C++**

```cpp
#include <etrade/market.hpp>

etrade::ImpactMarket m;
m.price = 100.0; m.spread = 0.01; m.adv = 5e6; m.daily_vol = 0.015;
m.total_shares = 250'000.0; m.n_intervals = 13; m.tau = 1.0 / 13.0;
m.weights.assign(etrade::kUsEquityVolumeCurve13.begin(), etrade::kUsEquityVolumeCurve13.end());
etrade::PretradeReport rep = etrade::pretrade_report(m, {0.0, 1e-6});
double vwap_bps = rep.rows[1].impact_bps;                          // 33.77
```

**Rust**

```rust
use etrade::impact::US_EQUITY_VOLUME_CURVE_13;
use etrade::market::{pretrade_report, ImpactMarket};

let m = ImpactMarket { price: 100.0, spread: 0.01, adv: 5e6, daily_vol: 0.015,
                       total_shares: 250_000.0, n_intervals: 13, tau: 1.0 / 13.0,
                       weights: Some(US_EQUITY_VOLUME_CURVE_13.to_vec()),
                       ..ImpactMarket::default() };
let rep = pretrade_report(&m, &[0.0, 1e-6])?;
let vwap_bps = rep.rows[1].impact_bps;                             // 33.77
```

**Java**

```java
Market.ImpactMarket m = Market.ImpactMarket.of(100.0, 0.01, 5e6, 0.015, 250_000.0, 13, 1.0 / 13.0)
        .withWeights(Impact.usEquityVolumeCurve13());
Market.PretradeReport rep = Market.pretradeReport(m, new double[] {0.0, 1e-6});
double vwapBps = rep.rows().get(1).impactBps();                    // 33.77
```

`alpha = 1` with uniform weights reproduces the Almgren-Chriss trajectory of
recipe 1 exactly. The volume curve is illustrative, not fitted to data.

---

Recipes 15 to 23 cover the post-trade, calibration and execution-extension
modules (API_SPEC Sections 10, 11 and 12). They were checked differently
from recipes 1 to 14: each one was run as a whole program in all four
languages (Python from `python/` with `src` on the path; C++ against the
Release `etrade.lib` built with MSVC, `/W4 /WX`; Rust as a binary crate with
a path dependency on `rust/`; Java against `java/target/classes`), and every
number in a comment is what that program printed. Where the four ports agree
to 15 significant digits only one value is shown. The Rust snippets use the
module paths shown in each recipe; the Java ones need only
`import io.github.ashjha0.etrade.*;`.

The Section 10 example order used in recipes 15 and 23: a **sell** of 70,000
shares, arrival mid 100.00, close 99.80, filled in the first three of four
bins of a falling market.

| bin | market volume | market VWAP | fills (qty @ price) |
|---|---|---|---|
| 0 | 400,000 | 99.95 | 20,000 @ 99.96 |
| 1 | 300,000 | 99.90 | 15,000 @ 99.91, 10,000 @ 99.89 |
| 2 | 300,000 | 99.85 | 25,000 @ 99.84 |
| 3 | 500,000 | 99.82 | none |

---

## 15. How do I measure slippage against arrival, VWAP and close, and the markouts?

`slippage` reports the cost in bps against four benchmarks, positive meaning
the order did worse than the benchmark (API_SPEC 10.1). `participation`
shows how much of each bin the order took; `markouts` measures how the mid
moved after each fill (10.2, 10.3). The markouts need a mid series indexed
by step; here ten steps, with the four fills at steps 0, 2, 4 and 6.

**Python**

```python
from etrade import tca

fill_bins = [0, 1, 1, 2]
fill_qtys = [20_000.0, 15_000.0, 10_000.0, 25_000.0]
fill_prices = [99.96, 99.91, 99.89, 99.84]
bin_volumes = [400_000.0, 300_000.0, 300_000.0, 500_000.0]
bin_vwaps = [99.95, 99.90, 99.85, 99.82]
r = tca.slippage("sell", 100.00, 99.80, fill_bins, fill_qtys, fill_prices,
                 bin_volumes, bin_vwaps)
print(r.avg_price, r.market_vwap, r.interval_vwap)   # 99.8964285714286 99.8766666666667 99.905
print(r.arrival_bps, r.vwap_bps)        # 10.3571428571428 -1.97863078749529
print(r.interval_vwap_bps, r.close_bps) # 0.857957917164196 -9.66218150586925
part = tca.participation(fill_bins, fill_qtys, bin_volumes)
print(part.per_bin, part.overall)       # (0.05, 0.0833.., 0.0833.., 0.0) 0.0466666666666667
mids = [100.00, 99.98, 99.95, 99.93, 99.90, 99.91, 99.88, 99.86, 99.85, 99.84]
mo = tca.markouts("sell", [0, 2, 4, 6], fill_qtys, mids, [1, 3])
print(mo.per_fill, mo.per_share)        # (0.0125, 0.0475) (0.0157142857142836, 0.048571428571426)
```

**C++**

```cpp
#include <etrade/tca.hpp>

std::vector<int> fill_bins{0, 1, 1, 2};
std::vector<double> fill_qtys{20'000.0, 15'000.0, 10'000.0, 25'000.0};
std::vector<double> fill_prices{99.96, 99.91, 99.89, 99.84};
std::vector<double> bin_volumes{400'000.0, 300'000.0, 300'000.0, 500'000.0};
std::vector<double> bin_vwaps{99.95, 99.90, 99.85, 99.82};
etrade::SlippageReport r = etrade::slippage(etrade::Side::kSell, 100.0, 99.80, fill_bins,
                                            fill_qtys, fill_prices, bin_volumes, bin_vwaps);
// r.arrival_bps 10.3571428571428, r.interval_vwap_bps 0.857957917164196
etrade::ParticipationResult part = etrade::participation(fill_bins, fill_qtys, bin_volumes);
std::vector<double> mids{100.00, 99.98, 99.95, 99.93, 99.90, 99.91, 99.88, 99.86, 99.85, 99.84};
etrade::MarkoutResult mo = etrade::markouts(etrade::Side::kSell, {0, 2, 4, 6}, fill_qtys, mids, {1, 3});
// part.overall 0.0466666666666667, mo.per_share {0.0157142857142836, 0.048571428571426}
```

**Rust**

```rust
use etrade::tca::{markouts, participation, slippage};
use etrade::Side;

let fill_bins = [0, 1, 1, 2];
let fill_qtys = [20_000.0, 15_000.0, 10_000.0, 25_000.0];
let fill_prices = [99.96, 99.91, 99.89, 99.84];
let bin_volumes = [400_000.0, 300_000.0, 300_000.0, 500_000.0];
let bin_vwaps = [99.95, 99.90, 99.85, 99.82];
let r = slippage(Side::Sell, 100.0, 99.80, &fill_bins, &fill_qtys, &fill_prices,
                 &bin_volumes, &bin_vwaps)?;                       // r.vwap_bps -1.97863078749529
let part = participation(&fill_bins, &fill_qtys, &bin_volumes)?;    // part.overall 0.0466666666666667
let mids = [100.00, 99.98, 99.95, 99.93, 99.90, 99.91, 99.88, 99.86, 99.85, 99.84];
let mo = markouts(Side::Sell, &[0, 2, 4, 6], &fill_qtys, &mids, &[1, 3])?;  // mo.per_fill [0.0125, 0.0475]
```

**Java**

```java
int[] fillBins = {0, 1, 1, 2};
double[] fillQtys = {20_000.0, 15_000.0, 10_000.0, 25_000.0};
double[] fillPrices = {99.96, 99.91, 99.89, 99.84};
double[] binVolumes = {400_000.0, 300_000.0, 300_000.0, 500_000.0};
double[] binVwaps = {99.95, 99.90, 99.85, 99.82};
Tca.SlippageReport r = Tca.slippage("sell", 100.0, 99.80, fillBins, fillQtys, fillPrices,
        binVolumes, binVwaps);                                 // r.closeBps() -9.66218150586925
Tca.ParticipationResult part = Tca.participation(fillBins, fillQtys, binVolumes);
double[] mids = {100.00, 99.98, 99.95, 99.93, 99.90, 99.91, 99.88, 99.86, 99.85, 99.84};
Tca.MarkoutResult mo = Tca.markouts("sell", new int[] {0, 2, 4, 6}, fillQtys, mids, new int[] {1, 3});
```

Reading the numbers. Against arrival the order cost 10.4 bps, but the whole
market fell during the day, so against the full-day VWAP it looks 2.0 bps
*better* than the market and against the close 9.7 bps better. Neither of
those is a fair score: the order finished in bin 2 and never faced bin 3's
lower prices. The interval VWAP uses only the bins the order traded in, and
says the order paid 0.86 bps more than the market did while it was trading.
That is the number to compare across algorithms (LEARN Section 4.2).

The markouts are positive: the mid kept falling after each sale (by 1.6
cents per share after one step, 4.9 cents after three). For a seller that is
favourable in the sign convention of 10.2, but on an order this size some of
the fall is the order's own impact, so a positive markout is not by itself
evidence of skill.

---

## 16. How do I tell whether algo A really beat algo B?

Collect one cost per parent order (bps against the same benchmark), then
run Welch's t-test. It does not assume equal variances, which matters
because algorithms with different urgency have different cost dispersion
(API_SPEC 10.4).

**Python**

```python
from etrade import tca

a = [1.2, 3.4, 2.2, 5.1, 0.7, 2.9, 4.0, 3.3]   # algo A, bps per order
b = [0.4, 1.1, -0.2, 0.9, 1.6, 0.3]            # algo B
w = tca.welch_ttest(a, b)
print(w.mean_diff, w.t_stat)        # 2.16666666666667 3.76751684064683
print(w.df, w.p_value)              # 10.1837133791222 0.00356077027255458
print(tca.student_t_cdf(2.0, 5.0))  # 0.949030260585071
```

**C++**

```cpp
#include <etrade/tca.hpp>

etrade::WelchResult w = etrade::welch_ttest({1.2, 3.4, 2.2, 5.1, 0.7, 2.9, 4.0, 3.3},
                                            {0.4, 1.1, -0.2, 0.9, 1.6, 0.3});
// w.t_stat 3.76751684064683, w.df 10.1837133791222, w.p_value 0.00356077027255458
double cdf = etrade::student_t_cdf(2.0, 5.0);   // 0.949030260585071
```

**Rust**

```rust
use etrade::tca::{student_t_cdf, welch_ttest};

let w = welch_ttest(&[1.2, 3.4, 2.2, 5.1, 0.7, 2.9, 4.0, 3.3],
                    &[0.4, 1.1, -0.2, 0.9, 1.6, 0.3])?;   // w.p_value 0.0035607702725545835
let cdf = student_t_cdf(2.0, 5.0)?;                       // 0.9490302605850708
```

**Java**

```java
Tca.WelchResult w = Tca.welchTTest(new double[] {1.2, 3.4, 2.2, 5.1, 0.7, 2.9, 4.0, 3.3},
        new double[] {0.4, 1.1, -0.2, 0.9, 1.6, 0.3});    // w.pValue() 0.0035607702725545835
double cdf = Tca.studentTCdf(2.0, 5.0);                    // 0.9490302605850708
```

This is golden case `welch_unequal_variance`; SciPy's
`ttest_ind(equal_var=False)` gives the same `t`, `df` and `p`. A p-value of
0.0036 says a 2.2 bps gap this large would be rare if the algorithms had
the same mean cost. It says nothing about orders unlike these fourteen;
LEARN Section 4.4 covers what the test assumes and how many orders a
realistic gap needs.

---

## 17. How do I calibrate the model inputs from my own arrays?

The four estimators of API_SPEC Section 11 take plain arrays, so the data
can come from any source. The example below is a toy session of 390 minutes
split into four bins: twelve trades, fourteen mids sampled every half hour,
six spread snapshots and the reach of 100 market orders (60 printed at the
touch, 25 one tick further out, then 10 and 5).

**Python**

```python
from etrade import calibration as cal, market as mk

times = [0.0, 30.0, 60.0, 95.0, 130.0, 180.0, 210.0, 260.0, 300.0, 340.0, 370.0, 390.0]
sizes = [900.0, 400.0, 300.0, 250.0, 200.0, 150.0, 150.0, 200.0, 250.0, 300.0, 500.0, 1200.0]
print(cal.volume_curve(times, sizes, 0.0, 390.0, 4))
# [0.385416666666667, 0.0729166666666667, 0.0729166666666667, 0.46875]
mid_t = [30.0 * i for i in range(14)]
mids = [100.0, 100.4, 99.9, 100.2, 100.1, 100.3, 100.2, 100.0, 99.8, 99.9, 100.1, 99.7, 99.9, 100.2]
vc = cal.volatility_curve(mid_t, mids, 0.0, 390.0, 4)
print(vc.daily_vol, vc.n_returns)        # 0.00989296735734774 13
ss = cal.spread_stats([0.01, 0.02, 0.01, 0.03, 0.01, 0.02],
                      [100.0, 100.1, 99.9, 100.0, 100.2, 100.0], tick=0.01)
print(ss.mean_ticks, ss.median_ticks, ss.mean_bps)   # 1.66666666666667 1.5 1.66616783183649
dist = [0.005] * 60 + [0.015] * 25 + [0.025] * 10 + [0.035] * 5
fi = cal.fill_intensity(dist, tick=0.01, n_levels=4, days=1.0, sides=2.0)
print(fi.a_rate, fi.k)                            # 81.6598656851043 96.1051103763025
print(fi.touch_fills_per_day, fi.halving_ticks)   # 50.5032196251289 0.721238629086326
a = mk.as_params_from_market(price=100.0, tick=0.01, daily_vol=vc.daily_vol,
                             touch_fills_per_day=fi.touch_fills_per_day,
                             halving_ticks=fi.halving_ticks, skew_ticks=2.25, q_max=50,
                             liq_ticks=1.0, dt_seconds=5.0)
print(a.k, a.a_rate)                              # 96.1051103763025 81.6598656851043 (round trip)
```

**C++**

```cpp
#include <etrade/calibration.hpp>
#include <etrade/market.hpp>

std::vector<double> w = etrade::volume_curve(
    {0, 30, 60, 95, 130, 180, 210, 260, 300, 340, 370, 390},
    {900, 400, 300, 250, 200, 150, 150, 200, 250, 300, 500, 1200}, 0.0, 390.0, 4);
std::vector<double> t{0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390};
std::vector<double> m{100.0, 100.4, 99.9, 100.2, 100.1, 100.3, 100.2,
                      100.0, 99.8, 99.9, 100.1, 99.7, 99.9, 100.2};
etrade::VolatilityCurve vc = etrade::volatility_curve(t, m, 0.0, 390.0, 4);  // daily_vol 0.00989296735734774
etrade::SpreadStats ss = etrade::spread_stats({0.01, 0.02, 0.01, 0.03, 0.01, 0.02},
                                              {100.0, 100.1, 99.9, 100.0, 100.2, 100.0}, 0.01);
std::vector<double> dist;
dist.insert(dist.end(), 60, 0.005);
dist.insert(dist.end(), 25, 0.015);
dist.insert(dist.end(), 10, 0.025);
dist.insert(dist.end(), 5, 0.035);
etrade::FillIntensity fi = etrade::fill_intensity(dist, 0.01, 4);  // touch 50.5032196251289
etrade::ASMarket am;
am.price = 100.0; am.tick = 0.01; am.daily_vol = vc.daily_vol;
am.touch_fills_per_day = fi.touch_fills_per_day; am.halving_ticks = fi.halving_ticks;
am.skew_ticks = 2.25; am.q_max = 50; am.liq_ticks = 1.0; am.dt_seconds = 5.0;
etrade::ASParams ap = etrade::as_params_from_market(am);           // ap.k 96.1051103763025
```

**Rust**

```rust
use etrade::calibration::{fill_intensity, spread_stats, volatility_curve, volume_curve};
use etrade::market::{as_params_from_market, ASMarket};

let w = volume_curve(&[0.0, 30.0, 60.0, 95.0, 130.0, 180.0, 210.0, 260.0, 300.0, 340.0, 370.0, 390.0],
                     &[900.0, 400.0, 300.0, 250.0, 200.0, 150.0, 150.0, 200.0, 250.0, 300.0, 500.0, 1200.0],
                     0.0, 390.0, 4)?;                               // [0.3854166666666667, ...]
let t: Vec<f64> = (0..14).map(|i| 30.0 * i as f64).collect();
let m = [100.0, 100.4, 99.9, 100.2, 100.1, 100.3, 100.2, 100.0, 99.8, 99.9, 100.1, 99.7, 99.9, 100.2];
let vc = volatility_curve(&t, &m, 0.0, 390.0, 4)?;                 // vc.n_returns 13
let ss = spread_stats(&[0.01, 0.02, 0.01, 0.03, 0.01, 0.02],
                      &[100.0, 100.1, 99.9, 100.0, 100.2, 100.0], 0.01)?;
let mut dist = vec![0.005; 60];
dist.extend([0.015; 25]);
dist.extend([0.025; 10]);
dist.extend([0.035; 5]);
let fi = fill_intensity(&dist, 0.01, 4, 1.0, 2.0)?;                 // fi.halving_ticks 0.7212386290863264
let ap = as_params_from_market(&ASMarket {
    price: 100.0, tick: 0.01, daily_vol: vc.daily_vol,
    touch_fills_per_day: fi.touch_fills_per_day, halving_ticks: fi.halving_ticks,
    skew_ticks: 2.25, q_max: 50, liq_ticks: 1.0, dt_seconds: 5.0, ..ASMarket::default()
})?;                                                                // ap.a_rate() 81.65986568510425
```

**Java**

```java
double[] w = Calibration.volumeCurve(new double[] {0, 30, 60, 95, 130, 180, 210, 260, 300, 340, 370, 390},
        new double[] {900, 400, 300, 250, 200, 150, 150, 200, 250, 300, 500, 1200}, 0.0, 390.0, 4);
double[] t = new double[14];
for (int i = 0; i < 14; i++) t[i] = 30.0 * i;
double[] m = {100.0, 100.4, 99.9, 100.2, 100.1, 100.3, 100.2, 100.0, 99.8, 99.9, 100.1, 99.7, 99.9, 100.2};
Calibration.VolatilityCurve vc = Calibration.volatilityCurve(t, m, 0.0, 390.0, 4);
Calibration.SpreadStats ss = Calibration.spreadStats(new double[] {0.01, 0.02, 0.01, 0.03, 0.01, 0.02},
        new double[] {100.0, 100.1, 99.9, 100.0, 100.2, 100.0}, 0.01);
double[] dist = new double[100];
java.util.Arrays.fill(dist, 0, 60, 0.005);
java.util.Arrays.fill(dist, 60, 85, 0.015);
java.util.Arrays.fill(dist, 85, 95, 0.025);
java.util.Arrays.fill(dist, 95, 100, 0.035);
Calibration.FillIntensity fi = Calibration.fillIntensity(dist, 0.01, 4);   // days 1, sides 2
ASParams ap = Market.asParamsFromMarket(new Market.ASMarket(100.0, 0.01, vc.dailyVol(),
        fi.touchFillsPerDay(), fi.halvingTicks(), 2.25, 50, 1.0, 5.0, 23_400.0, 1.0));
```

Things to know. The last trade, at exactly the session end, falls in the
last bin (11.1). The volatility is open-to-close realised volatility over
13 returns, so `0.99%` here is a daily figure only if the session is the
whole trading day. The trade at minute 95 sits in bin 0 because the bins are
97.5 minutes wide. Fourteen observations are far too few for real use; LEARN
Section 3.10 discusses how much one day of data can say.

---

## 18. How do I load a fitted preset and use it?

`data/presets.json` holds eight presets; three of them
(`itch_aapl_20191230`, `itch_msft_20191230`, `itch_qqq_20191230`) are fitted
to one NASDAQ ITCH day (API_SPEC 11.6). Each has an `ac` block (the
arguments of `ac_params_from_market`) and an `as` block (the arguments of
`as_params_from_market`). Python reads the file directly. The native
libraries have no JSON dependency by design, so the C++, Rust and Java
snippets copy the AAPL values from the file; the repository's test suites
parse it with their own small JSON readers.

**Python**

```python
import json, math
import etrade as et
from etrade import market as mk

with open("../data/presets.json") as f:
    preset = json.load(f)["itch_aapl_20191230"]
p = mk.ac_params_from_market(**preset["ac"])
notional = preset["ac"]["total_shares"] * preset["ac"]["price"]
print(mk.bps(et.expected_cost(p), notional))                # 4.87072935815202
print(mk.bps(math.sqrt(et.variance_cost(p)), notional))     # 58.198878708374
a = mk.as_params_from_market(**preset["as"])
print(a.k, a.a_rate)                       # 150.684169686945 46882.2935867192
print(mk.bps(et.optimal_spread(a, 0.0), a.s0))              # 1.23473374934756
```

**C++**

```cpp
#include <etrade/market.hpp>

etrade::ACMarket m;  // data/presets.json, itch_aapl_20191230 "ac"
m.price = 289.72; m.spread = 0.026401; m.adv = 5'678'176.0; m.daily_vol = 0.01157;
m.total_shares = 283'900.0; m.n_intervals = 13; m.tau = 1.0 / 13.0; m.lambda_ra = 4.66e-8;
etrade::ACParams p = etrade::ac_params_from_market(m);
double e_bps = etrade::bps(etrade::expected_cost(p), m.total_shares * m.price);   // 4.87072935815202
etrade::ASMarket a;  // itch_aapl_20191230 "as"
a.price = 289.72; a.tick = 0.01; a.daily_vol = 0.01157; a.touch_fills_per_day = 22'070.0;
a.halving_ticks = 0.46; a.skew_ticks = 2.25; a.q_max = 50; a.liq_ticks = 1.0; a.dt_seconds = 5.0;
etrade::ASParams ap = etrade::as_params_from_market(a);
double open_bps = etrade::bps(etrade::optimal_spread(ap, 0.0), ap.s0);            // 1.23473374934756
```

**Rust**

```rust
use etrade::market::{ac_params_from_market, as_params_from_market, bps, ACMarket, ASMarket};

let m = ACMarket { price: 289.72, spread: 0.026401, adv: 5_678_176.0, daily_vol: 0.01157,
                   total_shares: 283_900.0, n_intervals: 13, tau: 1.0 / 13.0,
                   lambda_ra: 4.66e-8, ..ACMarket::default() };
let p = ac_params_from_market(&m)?;
let e_bps = bps(etrade::expected_cost(&p, None)?, m.total_shares * m.price)?;    // 4.870729358152021
let a = as_params_from_market(&ASMarket {
    price: 289.72, tick: 0.01, daily_vol: 0.01157, touch_fills_per_day: 22_070.0,
    halving_ticks: 0.46, skew_ticks: 2.25, q_max: 50, liq_ticks: 1.0, dt_seconds: 5.0,
    ..ASMarket::default()
})?;
let open_bps = bps(etrade::optimal_spread(&a, 0.0)?, a.s0())?;                   // 1.2347337493475625
```

**Java**

```java
Market.ACMarket m = Market.ACMarket.of(289.72, 0.026401, 5_678_176.0, 0.01157, 283_900.0, 13,
        1.0 / 13.0, 4.66e-8);
ACParams p = Market.acParamsFromMarket(m);
double eBps = Market.bps(AlmgrenChriss.expectedCost(p), m.totalShares() * m.price());  // 4.870729358152021
ASParams a = Market.asParamsFromMarket(new Market.ASMarket(289.72, 0.01, 0.01157, 22_070.0, 0.46,
        2.25, 50, 1.0, 5.0, 23_400.0, 1.0));
double openBps = Market.bps(AvellanedaStoikov.optimalSpread(a, 0.0), a.s0());           // 1.2347337493475625
```

The `ac` block feeds the linear Almgren-Chriss model, which charges 4.9 bps
for this 5%-of-volume order; the square-root model in recipe 22 charges
about 26 bps for the same order. Treat the gap as model uncertainty, not
as a precise number. Every fitted value carries the caveats of API_SPEC
11.6: one day, one venue, the week between Christmas and New Year.

---

## 19. How do I schedule an order when impact decays (transient impact)?

In the Obizhaeva-Wang model each trade pushes the price by `kappa` per
share and the push decays at rate `rho` (API_SPEC 12.1). The optimal
schedule trades a block at the start, a constant rate in the middle and a
block at the end.

**Python**

```python
import etrade as et
from etrade import execution as ex

tp = ex.TransientParams(total_shares=1e6, n_intervals=10, tau=0.1, kappa=1e-6, rho=5.0)
x = ex.transient_optimal_schedule(tp)
print([round(x[j] - x[j + 1]) for j in range(10)])
# [194259, 76435, 76435, 76435, 76435, 76435, 76435, 76435, 76435, 194259]
print(ex.transient_expected_cost(tp, x))                         # 156041.881012075
print(ex.transient_expected_cost(tp, et.twap_holdings(1e6, 10)))  # 165236.399784175
tp2 = ex.TransientParams(1e6, 10, 0.1, 1e-6, 5.0, gamma_p=1e-7)   # add permanent impact
print(ex.transient_expected_cost(tp2, ex.transient_optimal_schedule(tp2)))  # 199689.45841113
```

**C++**

```cpp
#include <etrade/almgren_chriss.hpp>
#include <etrade/execution.hpp>

etrade::TransientParams tp{1e6, 10, 0.1, 1e-6, 5.0};            // gamma_p defaults to 0
std::vector<double> x = etrade::transient_optimal_schedule(tp);  // x[0]-x[1] = 194259.449788511
double e_opt = etrade::transient_expected_cost(tp, x);           // 156041.881012075
double e_twap = etrade::transient_expected_cost(tp, etrade::twap_holdings(1e6, 10));  // 165236.399784175
```

**Rust**

```rust
use etrade::execution::{transient_expected_cost, transient_optimal_schedule, TransientParams};

let tp = TransientParams::new(1e6, 10, 0.1, 1e-6, 5.0, 0.0)?;
let x = transient_optimal_schedule(&tp);                         // x[1]-x[2] = 76435.13755287218
let e_opt = transient_expected_cost(&tp, &x)?;                   // 156041.88101207488
let e_twap = transient_expected_cost(&tp, &etrade::twap_holdings(1e6, 10)?)?;  // 165236.3997841754
```

**Java**

```java
Execution.TransientParams tp = new Execution.TransientParams(1e6, 10, 0.1, 1e-6, 5.0, 0.0);
double[] x = Execution.transientOptimalSchedule(tp);
double eOpt = Execution.transientExpectedCost(tp, x);                                   // 156041.88101207488
double eTwap = Execution.transientExpectedCost(tp, AlmgrenChriss.twapHoldings(1e6, 10)); // 165236.3997841754
```

With `rho * tau = 0.5`, a fraction `a = exp(-0.5) = 0.607` of each push
survives to the next interval. The middle trades are exactly `1 - a = 0.393`
of the end blocks (76,435 / 194,259), the closed form quoted in API_SPEC
12.1, and the optimum saves 5.6% against TWAP. Adding permanent impact
(`gamma_p = 1e-7`) breaks the symmetry between the interior trades and
raises the cost; `gamma_p` must stay below `kappa (1 - a)/(1 + a)` (here
`2.45e-7`) or the constructor rejects the parameters (recipe 23).

---

## 20. When does re-planning beat a static schedule?

`simulate_adaptive` runs the static Almgren-Chriss plan and a strategy that
re-solves every interval from the realised holding and an observed drift
signal, on the same random draws (API_SPEC 12.2). Three cases on the liquid
parameter set: no signal and full fills, an AR(1) signal, and random
partial fills.

**Python**

```python
import etrade as et
from etrade import execution as ex

p = et.ACParams(1_000_000.0, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6)
for sig in (ex.SignalParams(), ex.SignalParams(phi=0.7, signal_sd=0.5),
            ex.SignalParams(fill_risk=0.5)):
    s = ex.simulate_adaptive(p, sig, n_paths=20_000, seed=1)
    print(round(s.static_mean), round(s.adaptive_mean), round(s.saving_mean), round(s.saving_se))
# 1081939 1081939 0 0
# 1082614 1035683 46931 815
# 880416 830959 49457 1716
sig = ex.SignalParams(phi=0.7, signal_sd=0.5)
print(ex.replan(p, sig, holding=600_000.0, signal=0.2, k=1))
# [600000.0, 280638.099154863, 132455.890509526, 55466.474440904, 0.0]
```

**C++**

```cpp
#include <etrade/execution.hpp>

etrade::ACParams p{1'000'000.0, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6};
etrade::AdaptiveSimResult s = etrade::simulate_adaptive(p, etrade::SignalParams{0.7, 0.5}, 20'000, 1);
// saving_mean 46449.9270684647, saving_se 812.331591614749
// SignalParams{0.0, 0.0, 0.5}: saving 48603.0383284152 +- 1701.56503456007; SignalParams{}: exactly 0
auto [c, d] = etrade::replan_coefficients(p, etrade::SignalParams{0.7, 0.5});
// c = {0, 1, 0.63768115942029, 0.582930367504836, ...}, d = {0, 0, 76277.65064836, ...}
```

**Rust**

```rust
use etrade::execution::{simulate_adaptive, SignalParams};
use etrade::ACParams;

let p = ACParams::new(1_000_000.0, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6)?;
let s = simulate_adaptive(&p, &SignalParams::new(0.7, 0.5, 0.0)?, 20_000, 1)?;
// saving_mean 47193.56255104075, saving_se 811.3802461749913
// fill_risk 0.5: 49832.01789591557 +- 1730.285316162344; no signal, no fill risk: exactly 0
```

**Java**

```java
ACParams p = new ACParams(1_000_000.0, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6);
Execution.AdaptiveSimResult s = Execution.simulateAdaptive(p,
        new Execution.SignalParams(0.7, 0.5, 0.0), 20_000, 1L);
// savingMean 47228.07206833507, savingSe 823.4842665906858
// fillRisk 0.5: 50652.76385541127 +- 1743.5939167455688; no signal, no fill risk: exactly 0.0
```

The deterministic parts (`replan`, `replan_coefficients`) agree across the
four ports to 15 digits. The simulations do not: Python draws one vector
per step across all paths, while the native ports loop path by path
(API_SPEC 4), so each language samples its own paths. The savings agree
within about one standard error (46,931, 46,450, 47,194 and 47,228 with SEs
near 815).

The first row is the point of the recipe. With no signal and full fills the
saving is zero on every path, not just on average, because the
Almgren-Chriss optimum is time-consistent (LEARN Section 3.12). Re-planning
earns its keep only when the trader learns something the plan did not
know: a forecastable drift, or a fill shortfall to make up.

---

## 21. How much does internalising opposite orders save?

When the desk holds buy and sell orders in the same name, it can cross them
at the mid and send only the net to the market (API_SPEC 12.3). The saving
is the square-root impact the crossed shares no longer pay.

**Python**

```python
from etrade import execution as ex, impact as im

r = ex.internalise([200_000.0, 50_000.0], [150_000.0], price=100.0, spread=0.01, adv=5e6,
                   daily_vol=0.015, n_intervals=13, tau=1 / 13,
                   weights=im.US_EQUITY_VOLUME_CURVE_13)
print(r.crossed_shares, r.residual_shares, r.residual_side)   # 150000.0 100000.0 buy
print(r.baseline_cost, r.internal_cost)   # 123600.889826542 21304.6384355964
print(r.saved_cost, r.saved_bps)          # 102296.251390945 25.5740628477364
print(r.buy_saved_bps, r.sell_saved_bps)  # (25.2477517882583, 25.2477517882583) (26.1179146135332,)
```

**C++**

```cpp
#include <etrade/execution.hpp>
#include <etrade/impact.hpp>

std::vector<double> w(etrade::kUsEquityVolumeCurve13.begin(), etrade::kUsEquityVolumeCurve13.end());
etrade::InternalisationResult r = etrade::internalise(
    {200'000.0, 50'000.0}, {150'000.0}, 100.0, 0.01, 5e6, 0.015, 13, 1.0 / 13.0, 0.5, 1.0, 0.10, w);
// r.residual_side "buy", r.saved_cost 102296.251390945, r.saved_bps 25.5740628477364
```

**Rust**

```rust
use etrade::execution::internalise;
use etrade::impact::US_EQUITY_VOLUME_CURVE_13;
use etrade::market::ImpactMarket;

let m = ImpactMarket { price: 100.0, spread: 0.01, adv: 5e6, daily_vol: 0.015,
                       n_intervals: 13, tau: 1.0 / 13.0,
                       weights: Some(US_EQUITY_VOLUME_CURVE_13.to_vec()),
                       ..ImpactMarket::default() };   // total_shares and lambda_ra are ignored
let r = internalise(&[200_000.0, 50_000.0], &[150_000.0], &m)?;   // r.saved_bps 25.574062847736357
```

**Java**

```java
Execution.InternalisationResult r = Execution.internalise(new double[] {200_000.0, 50_000.0},
        new double[] {150_000.0}, 100.0, 0.01, 5e6, 0.015, 13, 1.0 / 13.0, 0.5, 1.0, 0.10,
        Impact.usEquityVolumeCurve13());                  // r.savedBps() 25.574062847736357
```

Of the 400,000 shares only 100,000 (the net buy) reach the market, and the
desk's impact bill falls from 123,601 to 21,305: 25.6 bps of the gross
notional. The saving is larger than the crossed fraction alone suggests
because under square-root impact the cost per share grows with size, so the total
`C(Q)` grows like `Q^1.5` and netting first always helps (LEARN Section
3.13). The per-order figures share the saving pro rata within each side, the
allocation the contract pins; LEARN discusses why a desk might allocate
differently.

---

## 22. End to end: fitted preset, pre-trade, schedule, execution, TCA

This recipe chains the modules the way a desk would: take the fitted AAPL
inputs (recipe 18) and the fitted volume curve, produce a pre-trade report,
pick a schedule, execute it against a scripted market, and score the
result with TCA. The "market" is deliberately simple and deterministic, so
every port gets the same fills: the mid falls linearly by 4 bps over the
day, each child order trades at the mid minus its own temporary impact
`eta_j n_j^alpha` per share, and each bin's market volume is
`adv * w_j`. A real simulation would add noise and permanent impact.

**Python**

```python
import json
from etrade import calibration as cal, impact as im, market as mk, tca

with open("../data/presets.json") as f:
    ac = json.load(f)["itch_aapl_20191230"]["ac"]
order = dict(price=ac["price"], spread=ac["spread"], adv=ac["adv"], daily_vol=ac["daily_vol"],
             total_shares=ac["total_shares"], n_intervals=13, tau=1 / 13,
             weights=cal.FITTED_VOLUME_CURVE_13)

# 1. Pre-trade
rep = mk.pretrade_report(lambdas=[0.0, 1e-6], **order)
print(rep.sqrt_law_bps, rep.half_spread_bps)   # 25.8709055447508 0.455629573381196
for r in rep.rows:
    print(r.label, r.lambda_ra, round(r.impact_bps, 2), round(r.sd_bps, 2))
# twap 0.0 28.8 62.93 / vwap 0.0 26.07 51.65 / optimal 0.0 26.07 51.62 / optimal 1e-06 31.03 26.26

# 2. Schedule: the risk-averse optimum
ip = mk.impact_params_from_market(lambda_ra=1e-6, **order)
sched = im.optimal_schedule(ip)
trades = [sched.holdings[j] - sched.holdings[j + 1] for j in range(13)]

# 3. Execution against the scripted market
w = cal.FITTED_VOLUME_CURVE_13
mid = [ac["price"] * (1 - 0.0004 * j / 12) for j in range(13)]
prices = [mid[j] - ip.eta * w[j] ** -ip.alpha * trades[j] ** ip.alpha for j in range(13)]
vols = [ac["adv"] * w[j] for j in range(13)]

# 4. Post-trade
t = tca.slippage("sell", ac["price"], mid[12], list(range(13)), trades, prices, vols, mid)
print(t.arrival_bps, t.interval_vwap_bps)       # 31.5909176590214 29.9058388111938
print(t.close_bps)                              # 27.6019584423989
print(tca.participation(list(range(13)), trades, vols).overall)   # 0.0499984502065452
```

**C++**

```cpp
#include <etrade/calibration.hpp>
#include <etrade/impact.hpp>
#include <etrade/market.hpp>
#include <etrade/tca.hpp>
#include <cmath>

etrade::ImpactMarket m;  // itch_aapl_20191230 with the fitted volume curve
m.price = 289.72; m.spread = 0.026401; m.adv = 5'678'176.0; m.daily_vol = 0.01157;
m.total_shares = 283'900.0; m.n_intervals = 13; m.tau = 1.0 / 13.0;
m.weights.assign(etrade::FITTED_VOLUME_CURVE_13.begin(), etrade::FITTED_VOLUME_CURVE_13.end());
etrade::PretradeReport rep = etrade::pretrade_report(m, {0.0, 1e-6});  // rows[3].impact_bps 31.0252906877883
m.lambda_ra = 1e-6;
etrade::ImpactParams ip = etrade::impact_params_from_market(m);
etrade::ScheduleResult s = etrade::optimal_schedule(ip);
std::vector<int> bins;
std::vector<double> trades, prices, vols, mids;
for (int j = 0; j < 13; ++j) {
    double n = s.holdings[j] - s.holdings[j + 1];
    double mid = m.price * (1.0 - 0.0004 * j / 12.0);
    bins.push_back(j);
    trades.push_back(n);
    mids.push_back(mid);
    prices.push_back(mid - ip.etas()[j] * std::pow(n, ip.alpha()));
    vols.push_back(m.adv * m.weights[j]);
}
etrade::SlippageReport t = etrade::slippage(etrade::Side::kSell, m.price, mids[12], bins, trades,
                                            prices, vols, mids);  // t.arrival_bps 31.5909176590214
```

**Rust**

```rust
use etrade::calibration::FITTED_VOLUME_CURVE_13;
use etrade::impact::optimal_schedule;
use etrade::market::{impact_params_from_market, pretrade_report, ImpactMarket};
use etrade::tca::slippage;
use etrade::Side;

let mut m = ImpactMarket { price: 289.72, spread: 0.026401, adv: 5_678_176.0,
                           daily_vol: 0.01157, total_shares: 283_900.0, n_intervals: 13,
                           tau: 1.0 / 13.0, weights: Some(FITTED_VOLUME_CURVE_13.to_vec()),
                           ..ImpactMarket::default() };
let rep = pretrade_report(&m, &[0.0, 1e-6])?;          // rep.sqrt_law_bps 25.870905544750755
m.lambda_ra = 1e-6;
let ip = impact_params_from_market(&m)?;
let s = optimal_schedule(&ip)?;
let bins: Vec<usize> = (0..13).collect();
let trades: Vec<f64> = (0..13).map(|j| s.holdings[j] - s.holdings[j + 1]).collect();
let mids: Vec<f64> = (0..13).map(|j| m.price * (1.0 - 0.0004 * j as f64 / 12.0)).collect();
let prices: Vec<f64> = (0..13).map(|j| mids[j] - ip.etas()[j] * trades[j].powf(ip.alpha())).collect();
let vols: Vec<f64> = FITTED_VOLUME_CURVE_13.iter().map(|w| m.adv * w).collect();
let t = slippage(Side::Sell, m.price, mids[12], &bins, &trades, &prices, &vols, &mids)?;
// t.interval_vwap_bps 29.90583881119377
```

**Java**

```java
double[] w13 = Calibration.fittedVolumeCurve13();
Market.ImpactMarket im = Market.ImpactMarket.of(289.72, 0.026401, 5_678_176.0, 0.01157, 283_900.0,
        13, 1.0 / 13.0).withWeights(w13);
Market.PretradeReport rep = Market.pretradeReport(im, new double[] {0.0, 1e-6});
Impact.Params ip = Market.impactParamsFromMarket(im.withLambdaRa(1e-6));
Impact.ScheduleResult s = Impact.optimalSchedule(ip);
int[] bins = new int[13];
double[] trades = new double[13], prices = new double[13], vols = new double[13], mid = new double[13];
for (int j = 0; j < 13; j++) {
    bins[j] = j;
    trades[j] = s.holdings()[j] - s.holdings()[j + 1];
    mid[j] = im.price() * (1.0 - 0.0004 * j / 12.0);
    prices[j] = mid[j] - ip.etas()[j] * Math.pow(trades[j], ip.alpha());
    vols[j] = im.adv() * w13[j];
}
Tca.SlippageReport t = Tca.slippage("sell", im.price(), mid[12], bins, trades, prices, vols, mid);
// t.closeBps() 27.60195844239889
```

How to read it. The pre-trade report prices the risk-averse schedule at
31.0 bps of impact with a 26.3 bps standard deviation, against 26.1 bps and
51.6 bps for VWAP: paying 5 bps for half the risk. Post-trade, the order
cost 31.6 bps against arrival. Against the interval VWAP it cost 29.9 bps,
which is its own temporary impact; the 1.7 bps between the two is the
scripted fall of the mid, a cost of timing that no schedule fully avoids.
The interval VWAP equals the full-day VWAP here because the order traded in
every bin. Participation came out at 5.0%, the size of the order against
the day's volume, as planned.

---

## 23. Troubleshooting: what do the validation errors mean?

Every port rejects bad input with the same message (API_SPEC 6, 10.5, 11,
12); only the exception type differs.

| | input error | model invalid | solver failure |
|---|---|---|---|
| Python | `ValueError` | `ValueError` | `ValueError` |
| C++ | `std::invalid_argument` | `std::domain_error` (Almgren-Chriss), else `std::invalid_argument` | `std::runtime_error` |
| Rust | `EtradeError::InvalidInput` | `EtradeError::ModelInvalid` (Almgren-Chriss), else `InvalidInput` | `EtradeError::SolverFailed` |
| Java | `IllegalArgumentException` | `IllegalArgumentException` | `IllegalStateException` |

The messages most people meet first, each checked in all four languages
unless marked (Python only):

| message | cause | fix |
|---|---|---|
| `side must be 'buy' or 'sell'` | a typo such as `"hold"` (Python, Java; C++ and Rust use an enum) | pass the side string exactly |
| `fill volume exceeds market volume in a bin` | the order's fills in a bin exceed the bin's market volume; market volume must include your own fills | add your fills to the tape volume, or check the bin assignment |
| `samples have zero variance` | both A/B samples are constant, so the t-statistic is undefined | collect real per-order costs; a constant cost usually means a units bug |
| `no volume inside the session` (Python only) | every trade time falls outside `[session_start, session_end]` | check the time units (seconds vs minutes) and the session bounds |
| `fewer than two distance buckets have fills` | all market orders printed at the touch, so no slope can be fitted | use more levels or a longer sample; for a one-tick name the reach may never pass 0.5 ticks |
| `model invalid: gamma_p >= kappa*(1-a)/(1+a) (cost not convex in the trades)` | permanent impact too large for the transient model | lower `gamma_p` or raise `kappa`, `rho`; the bound is printed in recipe 19 |
| `phi must be in [0, 1)` | a unit-root or negative signal | the AR(1) signal must be stationary and non-oscillating |
| `at least one order is required` (Python only) | both order lists empty in `internalise` | pass at least one buy or sell |
| `model invalid: temporary impact not convex enough against permanent impact ...` (Python only) | in Section 8, `perm_frac` so small that permanent impact dominates | use a realistic `perm_frac` (default 0.10) |

**Python**

```python
from etrade import tca

try:
    tca.welch_ttest([1.0, 1.0], [2.0, 2.0])
except ValueError as e:
    print(e)        # samples have zero variance
```

**C++**

```cpp
try {
    etrade::TransientParams bad{1e6, 10, 0.1, 1e-6, 5.0, 1e-6};
} catch (const std::invalid_argument& e) {
    std::cout << e.what() << '\n';   // model invalid: gamma_p >= kappa*(1-a)/(1+a) (cost not convex in the trades)
}
```

**Rust**

```rust
use etrade::calibration::fill_intensity;
use etrade::EtradeError;

match fill_intensity(&[0.005, 0.005], 0.01, 4, 1.0, 2.0) {
    Err(EtradeError::InvalidInput(msg)) => eprintln!("{msg}"),  // fewer than two distance buckets have fills
    other => println!("{other:?}"),
}
```

**Java**

```java
try {
    Tca.participation(new int[] {0}, new double[] {20}, new double[] {10});
} catch (IllegalArgumentException ex) {
    System.err.println(ex.getMessage());   // fill volume exceeds market volume in a bin
}
```

---

## 24. How do I run the market maker with frictions and read its PnL attribution?

`simulate_mm` runs the A-S (or GLFT, or fixed-spread) market maker with a
price grid, maker fees or rebates, stale quotes, informed traders and a
fair-value signal, each switched on in `MMConfig` (API_SPEC 9.1-9.4). Every
path's PnL splits exactly into spread, fees, inventory and liquidation; the
markout is the part of inventory earned in the step after each fill (9.5).
The example is the informed-flow row of the friction ledger: the US
large-cap preset with a 1-cent tick, a 0.2-cent rebate and informed traders
present 15% of the time.

**Python**

```python
import json
from etrade import market as mk, mm

with open("../data/presets.json") as f:
    a = mk.as_params_from_market(**json.load(f)["us_large_cap"]["as"])
print(mm.glft_half_spreads(a, 3.0))          # (0.00766899516689547, 0.00594791025950275)
print(mm.weighted_mid(99.99, 100.01, 900.0, 100.0))   # 100.008
cfg = mm.MMConfig(tick=0.01, maker_fee=0.002, informed_prob=0.15)
r = mm.simulate_mm(a, 20_000, 103, cfg)
print([round(r.mean(s), 2) for s in ("pnl", "spread", "fees", "inventory", "markout")])
# [4.01, 10.25, 1.86, -8.07, -8.06]
print(round(r.se("pnl"), 3), r.max_identity_error < 1e-9)   # 0.01 True
```

**C++**

```cpp
#include <etrade/market.hpp>
#include <etrade/mm.hpp>

etrade::ASMarket m;  // data/presets.json, us_large_cap "as"
m.price = 100.0; m.tick = 0.01; m.daily_vol = 0.015; m.touch_fills_per_day = 1170.0;
m.halving_ticks = 0.462; m.skew_ticks = 2.25; m.q_max = 50; m.liq_ticks = 1.0; m.dt_seconds = 5.0;
etrade::ASParams a = etrade::as_params_from_market(m);
auto [db, da] = etrade::glft_half_spreads(a, 3.0);   // 0.00766899516689547 0.00594791025950274
etrade::MMConfig cfg;
cfg.tick = 0.01; cfg.maker_fee = 0.002; cfg.informed_prob = 0.15;
etrade::MMResult r = etrade::simulate_mm(a, 5'000, 103, cfg);
// r.mean(etrade::MMStat::kPnl) 3.9994 (se 0.0209), kSpread 10.2537, kMarkout -8.0799,
// kInformedFills 289.044, r.max_identity_error 4.3e-12
```

**Rust**

```rust
use etrade::market::{as_params_from_market, ASMarket};
use etrade::mm::{glft_half_spreads, simulate_mm, MMConfig, MMStat};

let a = as_params_from_market(&ASMarket {
    price: 100.0, tick: 0.01, daily_vol: 0.015, touch_fills_per_day: 1170.0,
    halving_ticks: 0.462, skew_ticks: 2.25, q_max: 50, liq_ticks: 1.0, dt_seconds: 5.0,
    ..ASMarket::default()
})?;
let (db, da) = glft_half_spreads(&a, 3.0)?;   // 0.007668995166895468 0.005947910259502745
let cfg = MMConfig { tick: 0.01, maker_fee: 0.002, informed_prob: 0.15, ..MMConfig::default() };
let r = simulate_mm(&a, 5_000, 103, &cfg)?;
// r.mean(MMStat::Pnl) 4.0046 (se 0.0205), Spread 10.2621, Markout -8.0717, InformedFills 289.0858
```

**Java**

```java
ASParams a = Market.asParamsFromMarket(new Market.ASMarket(100.0, 0.01, 0.015, 1170.0, 0.462,
        2.25, 50, 1.0, 5.0, 23_400.0, 1.0));
MarketMaking.HalfSpreads h = MarketMaking.glftHalfSpreads(a, 3.0);   // 0.00766899516689547, 0.00594791025950274
MarketMaking.Config cfg = new MarketMaking.Config(MarketMaking.Agent.AS, 0.01, 0.002, 0, 0.15,
        0.5, null);
MarketMaking.Result r = MarketMaking.simulate(a, 5_000, 103L, cfg);
// r.mean(MarketMaking.Stat.PNL) 3.9959 (se 0.0205), SPREAD 10.2492, MARKOUT -8.0762
```

The GLFT half spreads agree across the four ports to 15 digits; the
simulations agree within their standard errors, since each port samples its
own paths (API_SPEC 4). Reading the attribution: the spread column is the
largest it is anywhere in the ledger (10.25 a day) and the PnL is among the
smallest (4.01), because the markout, -8.06, takes back most of it. Almost
all of the inventory term is in the markout: the loss comes from the step
right after the informed fills, not from carrying a position. The other
ledger rows come from the same call with the other `MMConfig` switches:
`latency_steps=1` for stale quotes, `signal_accuracy=0.6` for a fair-value
signal, `agent="glft"` for GLFT quotes, a negative `maker_fee` for a fee
(LEARN Sections 6.7-6.12 discuss each).

---

## 25. Should a child order rest in the queue or cross, and how do I liquidate with both?

`queue_fill_probability` gives the probability that a resting order is
completely filled within a horizon, from its place in the queue and the
rates of market orders and cancellations (API_SPEC 13.1). `place_or_cross`
compares the expected cost of posting with that of crossing (13.2).
`liquidation_policy` solves the multi-period problem with limit and market
orders by backward induction, and `simulate_liquidation` checks it against
two simple strategies on shared paths (13.3).

**Python**

```python
from etrade import placement as pl

p = pl.queue_fill_probability(5, 2, 10.0, 0.1, 1.0)   # 0.882592233494604
d = pl.place_or_cross(2000.0, 0.02, p, 0.01, 1e-4, 0.5, 1 / 13)
print(d.passive, d.passive_cost, d.aggressive_cost)   # True 0.00964674849175898 0.0261245154965971
print(d.breakeven_fill_prob)                          # 0.333333333333333
lp = pl.LiquidationParams(10, 50, 0.02, 0.5, 0.02, 30.0, 0.3, 0.005)
pol = pl.liquidation_policy(lp)
print(pol.expected_cost, pol.depth[0, 10], pl.aggressive_cost(lp))
# -0.205095993155636 0.0174145888001820 0.15
s = pl.simulate_liquidation(lp, n_paths=20_000, seed=201)
print(s.optimal_mean, s.saving_vs_aggressive_mean, s.saving_vs_aggressive_se)
# -0.187424605121854 0.353065036162625 0.00477327948214878
```

**C++**

```cpp
#include <etrade/placement.hpp>

double pf = etrade::queue_fill_probability(5, 2, 10.0, 0.1, 1.0);   // 0.882592233494604
etrade::PlacementDecision d =
    etrade::place_or_cross(2000.0, 0.02, pf, 0.01, 1e-4, 0.5, 1.0 / 13.0);   // d.passive true
etrade::LiquidationParams lp(10, 50, 0.02, 0.5, 0.02, 30.0, 0.3, 0.005);
etrade::LiquidationPolicy pol = etrade::liquidation_policy(lp);
// pol.expected_cost() -0.205095993155636, pol.depth[0][10] 0.0174145888001820
etrade::LiquidationSimResult s = etrade::simulate_liquidation(lp, 20'000, 201);
// s.optimal_mean -0.224640119504025, s.saving_vs_aggressive_mean 0.359874733664059 (se 0.00483)
```

**Rust**

```rust
use etrade::placement::{
    liquidation_policy, place_or_cross, queue_fill_probability, simulate_liquidation,
    LiquidationParams,
};

let pf = queue_fill_probability(5, 2, 10.0, 0.1, 1.0)?;                  // 0.882592233494604
let d = place_or_cross(2000.0, 0.02, pf, 0.01, 1e-4, 0.5, 1.0 / 13.0)?;  // d.passive_cost 0.009646748491758977
let lp = LiquidationParams::new(10, 50, 0.02, 0.5, 0.02, 30.0, 0.3, 0.005, 0.0)?;
let pol = liquidation_policy(&lp);                                        // pol.expected_cost() -0.2050959931556364
let s = simulate_liquidation(&lp, 20_000, 201)?;
// s.optimal_mean -0.2105849605671755, s.saving_vs_aggressive_mean 0.35881962044547183 (se 0.0048)
```

**Java**

```java
double pf = Placement.queueFillProbability(5, 2, 10.0, 0.1, 1.0);   // 0.882592233494604
Placement.PlacementDecision d = Placement.placeOrCross(2000.0, 0.02, pf, 0.01, 1e-4, 0.5,
        1.0 / 13.0);                                                 // d.passive() true
Placement.LiquidationParams lp = new Placement.LiquidationParams(10, 50, 0.02, 0.5, 0.02, 30.0,
        0.3, 0.005, 0.0);
Placement.LiquidationPolicy pol = Placement.liquidationPolicy(lp);  // pol.expectedCost() -0.2050959931556364
Placement.LiquidationSimResult s = Placement.simulateLiquidation(lp, 20_000, 201L);
// s.optimalMean() -0.17096924370659827, s.savingVsAggressiveMean() 0.35690147613272427 (se 0.0048)
```

Reading the numbers. Five units ahead of a two-unit order, with ten market
orders per unit of time and some cancellation, fill within one time unit
with probability 0.88. A miss costs one cent of adverse move, so posting
pays whenever the fill probability exceeds one third; here it saves 1.65
cents a share against crossing. The liquidation policy for ten units over
fifty periods expects to sell 2.05 cents a unit above the starting mid,
against 1.5 cents a unit below it for an even schedule of market orders.
The simulated means differ between ports by about one or two standard
errors (0.0217 per port), as samples of the same distribution should; the
paired saving against all-market orders has a much smaller error (0.0048)
because both strategies see the same prices (LEARN Sections 3.14-3.16).
