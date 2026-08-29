# COOKBOOK — `etrade` recipes

Task-oriented recipes in all four languages. Python snippets run as-is
against the reference package (`cd python && PYTHONPATH=src python3`, then
paste; `import etrade as et` is assumed once per session). Native snippets
follow the `API_SPEC.md` contract and naming conventions (C++ namespace
`etrade`, Rust crate `etrade`, Java package `com.quant.etrade`); exact
signatures are in each port's public headers / `lib.rs` / javadoc.

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
print(et.ac_kappa(p))        # 0.8462971345012561
print(et.half_life(p))       # 1.1816180856967273
print(et.ac_trajectory(p))   # [1000000, 428598.85, 182932.81, 76303.52, 27600.15, 0]
print(et.ac_trade_list(p))   # [571401.15, 245666.03, ...]  sums to X
```

**C++**

```cpp
#include <etrade/almgren_chriss.hpp>

etrade::ACParams p{1'000'000.0, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6};
double kappa = etrade::ac_kappa(p);            // 0.8462971345012561
double theta = etrade::half_life(p);           // 1.1816...
std::vector<double> x = etrade::ac_trajectory(p);   // size N+1, x[0]=X, x[N]=0
std::vector<double> n = etrade::ac_trade_list(p);   // size N, sums to X
```

**Rust**

```rust
use etrade::AcParams;

let p = AcParams::new(1_000_000.0, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6)?;
let kappa = etrade::ac_kappa(&p);              // 0.8462971345012561
let theta = etrade::half_life(&p);             // 1.1816... (f64::INFINITY if kappa == 0)
let x: Vec<f64> = etrade::ac_trajectory(&p);   // len N+1
let n: Vec<f64> = etrade::ac_trade_list(&p);   // len N
```

**Java**

```java
import com.quant.etrade.ACParams;
import com.quant.etrade.AlmgrenChriss;

ACParams p = new ACParams(1_000_000.0, 5, 1.0, 0.95, 2.5e-6, 2.5e-7, 2e-6);
double kappa = AlmgrenChriss.kappa(p);         // 0.8462971345012561
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

**Rust**

```rust
let (e_opt, v_opt) = (etrade::expected_cost(&p, None)?, etrade::variance_cost(&p, None));
let twap = etrade::twap_holdings(p.total_shares, p.n_intervals)?;
let e_tw = etrade::expected_cost(&p, Some(&twap))?;      // 600000.0
let v_tw = etrade::variance_cost(&p, Some(&twap));       // 1.083e12
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
intervals, last clip partial. Cost it with the same E/Var functions.

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
let v = etrade::variance_cost(&p, Some(&pov));     // 4.08945e11
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
a few standard errors of the closed forms.

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
assert!((mc.var - etrade::variance_cost(&p, None)).abs() < 4.0 * mc.se_var);
```

**Java**

```java
ISSimResult mc = AlmgrenChriss.simulateIS(p, 20_000, 123L);
assert Math.abs(mc.mean() - AlmgrenChriss.expectedCost(p)) < 4 * mc.seMean();
assert Math.abs(mc.var() - AlmgrenChriss.varianceCost(p)) < 4 * mc.seVar();
```

Note: only Python's PCG64 sample values are stored in `golden.json`; other
languages use their own RNG and must agree *statistically* (4-SE band).

---

## 6. How do I quote like Avellaneda-Stoikov?

Reservation price, optimal spread, and the (bid, ask) pair for a given
state `(s, q, t)`.

**Python**

```python
ap = et.ASParams(s0=100.0, sigma=2.0, horizon=1.0, dt=0.005, gamma=0.1,
                 k=1.5, a_rate=140.0, q_max=20, eta_liq=0.5)
r = et.reservation_price(ap, s=100.0, q=3.0, t=0.3)   # 99.16
sp = et.optimal_spread(ap, t=0.3)                     # 1.5707704227514234
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
let ap = AsParams::new(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, 20, 0.5)?;
let r  = etrade::reservation_price(&ap, 100.0, 3.0, 0.3)?;    // 99.16
let sp = etrade::optimal_spread(&ap, 0.3)?;                   // 1.57077...
let (bid, ask) = etrade::quotes(&ap, 100.0, 3.0, 0.3)?;
```

**Java**

```java
ASParams ap = new ASParams(100.0, 2.0, 1.0, 0.005, 0.1, 1.5, 140.0, 20, 0.5);
double r  = AvellanedaStoikov.reservationPrice(ap, 100.0, 3.0, 0.3);  // 99.16
double sp = AvellanedaStoikov.optimalSpread(ap, 0.3);                 // 1.57077...
double[] ba = AvellanedaStoikov.quotes(ap, 100.0, 3.0, 0.3);          // {bid, ask}
```

---

## 7. How do I get the fill probability of a quote at distance delta?

`lambda(delta) = A*exp(-k*delta)`, `P = 1 - exp(-lambda*dt)`. Delta is
signed — a quote through the mid (`delta < 0`) is valid and fills faster.

**Python**

```python
lam = et.intensity(ap, delta=0.6)          # 56.919752363683884
pf  = et.fill_probability(ap, delta=0.6)   # 0.24768394830642568
```

**C++**

```cpp
double lam = etrade::intensity(ap, 0.6);          // 56.919752363683884
double pf  = etrade::fill_probability(ap, 0.6);   // 0.24768394830642568
```

**Rust**

```rust
let lam = etrade::intensity(&ap, 0.6)?;           // 56.919752363683884
let pf  = etrade::fill_probability(&ap, 0.6)?;    // 0.24768394830642568
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

**Python**

```python
res_as = et.simulate(ap, n_paths=2000, seed=42, agent="as")
res_fx = et.simulate(ap, n_paths=2000, seed=42, agent="fixed")
print(res_as.mean_pnl, res_as.std_pnl, res_as.rms_q)   # ~55.9, ~6.4, ~1.45
print(res_fx.mean_pnl, res_fx.std_pnl, res_fx.rms_q)   # ~57.8, ~12.8, ~4.96
assert res_as.rms_q < res_fx.rms_q                      # inventory control works
assert not res_as.cap_violated                          # |q| <= q_max always
```

**C++**

```cpp
etrade::ASSimResult a = etrade::simulate(ap, 2000, 42, "as");
etrade::ASSimResult f = etrade::simulate(ap, 2000, 42, "fixed");
assert(a.rms_q() < f.rms_q());
assert(!a.cap_violated);
```

**Rust**

```rust
let a = etrade::simulate(&ap, 2000, 42, "as", None, 1.0, 0.5)?;
let f = etrade::simulate(&ap, 2000, 42, "fixed", None, 1.0, 0.5)?;
assert!(a.rms_q() < f.rms_q());
assert!(!a.cap_violated);
```

**Java**

```java
ASSimResult a = AvellanedaStoikov.simulate(ap, 2000, 42L, "as");
ASSimResult f = AvellanedaStoikov.simulate(ap, 2000, 42L, "fixed");
assert a.rmsQ() < f.rmsQ();
assert !a.capViolated();
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
auto base   = etrade::simulate(ap, 2000, 42, "as");
auto stress = etrade::simulate(ap, 2000, 42, "as",
                               /*fixed_spread=*/std::nullopt,
                               /*stress_factor=*/2.0, /*stress_time_frac=*/0.5);
```

**Rust**

```rust
let base   = etrade::simulate(&ap, 2000, 42, "as", None, 1.0, 0.5)?;
let stress = etrade::simulate(&ap, 2000, 42, "as", None, 2.0, 0.5)?;
```

**Java**

```java
ASSimResult base   = AvellanedaStoikov.simulate(ap, 2000, 42L, "as");
ASSimResult stress = AvellanedaStoikov.simulate(ap, 2000, 42L, "as",
                                                Double.NaN, 2.0, 0.5);
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
print(d.delay, d.trading, d.opportunity, d.total)   # 450.0 230.0 120.0 800.0
assert abs(d.total - (d.delay + d.trading + d.opportunity)) < 1e-10
```

**C++**

```cpp
etrade::ISDecomposition d = etrade::decompose_shortfall(
    "sell", 100.0, 99.5, 98.8, 1000.0,
    {300.0, 400.0, 200.0}, {99.4, 99.2, 99.1});
// d.delay == 450, d.trading == 230, d.opportunity == 120, d.total == 800
```

**Rust**

```rust
let d = etrade::decompose_shortfall("sell", 100.0, 99.5, 98.8, 1000.0,
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

**Python**

```python
parts, costs = et.synthetic_impact_fills(25, 0.5, 8.0, 0.05)
reg = et.estimate_impact(parts, costs)
print(reg.slope, reg.intercept, reg.r_squared)
# 7.970665291179456 0.503539513916228 0.9934209775967902
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
match AcParams::new(1e6, 5, 1.0, 0.95, 1e-7, 1e-6, 0.0) {
    Err(e) => eprintln!("rejected: {e}"),
    Ok(_) => unreachable!(),
}
```

**Java** — catch `IllegalArgumentException`:

```java
try {
    new ACParams(1e6, 5, 1.0, 0.95, 1e-7, 1e-6, 0.0);
} catch (IllegalArgumentException e) {
    System.err.println("rejected: " + e.getMessage());
}
```

Documented **non-errors** in every language: negative prices (arithmetic
walks), `kappa = 0` (TWAP branch, half-life = infinity), a quote through
the mid (`delta < 0`), and `q_max = 0` (no quoting, zero PnL).
