# LEARN — Electronic Trading: Optimal Execution and Market Making

This document teaches the theory behind the `etrade` package: how modern
electronic markets work at the microstructure level, why executing a large
order is a genuine optimization problem, the Almgren-Chriss solution to it,
the Perold implementation-shortfall framework used to measure execution
quality, and the Avellaneda-Stoikov model of an inventory-averse market
maker. Every worked number below is reproduced bit-for-bit by the code and
pinned in `data/golden/golden.json`.

---

## 1. Market microstructure in five minutes

### 1.1 The limit order book

Modern electronic markets are driven by a **central limit order book** (LOB).
Participants post **limit orders** — "buy 500 at 99.98 or better" — which
rest in the book, sorted by price and then by arrival time (price-time
priority). The best resting buy price is the **bid**, the best resting sell
price is the **ask** (or offer), and their difference is the **bid-ask
spread**. The midpoint `(bid + ask)/2` is the conventional "fair value"
reference, the **mid**.

A **market order** ("buy 500 now, at whatever price") crosses the spread and
executes immediately against the resting limit orders, walking through
successive price levels if it is larger than the quantity at the best level.
The two order types embody the fundamental trade-off of trading:

- **Limit orders** earn the spread but may never execute (and tend to execute
  exactly when you least want them to — see adverse selection, Section 6).
- **Market orders** guarantee execution but pay the spread, plus **market
  impact** if they are large.

### 1.2 Market impact: temporary vs permanent

When a large order consumes liquidity, the price moves against the trader.
Empirically this movement has two components:

- **Temporary (transient) impact**: the immediate concession paid for
  demanding liquidity *now* — you walk the book, and the book refills after
  you stop trading. It affects *your* execution price but decays.
- **Permanent impact**: the lasting shift in the price level. Part of what a
  large seller reveals is information ("someone informed is selling"), and
  the market re-prices accordingly. This affects every subsequent trade,
  yours and everyone else's.

Almgren-Chriss makes both linear: selling at rate `v` costs `h(v) = eta*v`
per share temporarily and depresses the price permanently by `g(v)*tau =
gamma_p * n` per interval. Linearity is a modelling choice, not an empirical
fact (real-world impact is closer to square-root in size, Section 12), but it
is what makes the problem solvable in closed form.

### 1.3 Which price did you actually get?

Execution quality is judged against a benchmark. The most important ones:

| Benchmark | Definition | Used for |
|---|---|---|
| **Arrival / IS** | mid at the moment the order started | measuring total cost of the decision (this project) |
| **TWAP** | time-weighted average price over the window | schedules that trade evenly in time |
| **VWAP** | volume-weighted average price over the window | schedules that trade pro-rata with market volume |
| **Close** | official closing price | funds benchmarked to the close |

**Implementation shortfall (IS)** — the arrival-price benchmark — is the
economically honest one: it charges you for everything that happens between
"I decided to trade" and "I finished", including the impact you caused and
the drift you sat through. TWAP/VWAP benchmarks are easier to beat but can
hide cost (you can match VWAP perfectly while moving the VWAP itself).

---

## 2. The trader's dilemma

Suppose you must sell X = 1,000,000 shares. Two extreme strategies:

- **Sell everything immediately.** You pay maximal temporary impact — you
  demand a million shares of liquidity in one interval — but you bear no
  price risk: the cost is locked in. In the model below this costs
  E = \$2,500,000 with **zero** variance.
- **Sell slowly** (say evenly over the day, TWAP). Each child order is small,
  so impact cost is minimal (E = \$600,000 below), but you remain exposed to
  the market for hours: if the price trends down while you hold, the loss
  can dwarf the impact saving. Below, the standard deviation of the TWAP
  cost is \$1,040,673 — larger than its mean.

Neither extreme is "right". Fast execution buys certainty with impact cost;
slow execution buys cheapness with **timing risk**. The trader's dilemma is a
mean-variance trade-off, and Almgren-Chriss resolves it exactly the way
Markowitz resolves portfolio choice: pick your risk aversion, get an optimal
point on an **efficient frontier of execution**.

---

## 3. Almgren-Chriss optimal execution

### 3.1 Model setup (discrete, as in the 2000 paper)

Sell `X >= 0` shares over `N` intervals of length `tau` (horizon `T = N*tau`,
times `t_j = j*tau`). A **trajectory** is the holdings path
`x_0 = X, x_1, ..., x_N = 0`; the **trade list** is `n_j = x_{j-1} - x_j`.

The mid follows an arithmetic random walk with linear permanent impact, and
each trade executes at the pre-trade mid less linear temporary impact:

```
S_j    = S_{j-1} + sigma*sqrt(tau)*xi_j - gamma_p*n_j      (xi_j iid N(0,1))
S~_j   = S_{j-1} - eta*n_j/tau                             (execution price)
IS     = X*S_0 - sum_{j=1}^{N} n_j*S~_j                    (capture shortfall)
```

Note the model is **arithmetic**: prices can go negative. That is a
documented property (fine over short horizons where `sigma*sqrt(T) << S_0`),
not a bug and not an error condition.

### 3.2 Exact discrete moments

Take expectations of IS for any fully liquidating schedule. The temporary
impact contributes `(eta/tau) * sum n_j^2` directly. The permanent impact of
trade `i` lowers the price seen by all later trades `j > i`; summing the
cross terms and using `sum_j n_j sum_{i<j} n_i = (X^2 - sum_j n_j^2)/2`:

$$
E[IS] = \frac{1}{2}\gamma_p X^2 + \frac{\tilde\eta}{\tau}\sum_{j=1}^{N} n_j^2,
\qquad
\tilde\eta = \eta\left(1 - \frac{\gamma_p \tau}{2\eta}\right)
           = \eta - \frac{\gamma_p \tau}{2}.
$$

Two readings of this formula:

- `(1/2)*gamma_p*X^2` is **schedule-independent**: however you sell X shares,
  permanent impact costs you half the total price displacement times X (you
  ride your own impact down, on average halfway).
- Only the `sum n_j^2` term is controllable, and it is minimized by equal
  clips — TWAP minimizes pure expected cost.

The variance comes only from the volatility term acting on what you still
hold:

$$
\mathrm{Var}[IS] = \sigma^2 \tau \sum_{j=1}^{N} x_j^2 .
$$

**Validity condition (paper footnote):** the adjusted impact must satisfy
`eta_tilde > 0`, i.e. `gamma_p * tau < 2*eta`. Otherwise the quadratic form
is not positive definite and the "optimal" schedule oscillates (sell, buy
back, sell...) to farm its own permanent impact — an artefact, so the code
rejects such parameters at construction time (Python `ValueError`, C++
`std::domain_error`, Rust `Err(EtradeError)`, Java
`IllegalArgumentException`).

### 3.3 The optimization and its closed form

Minimize the mean-variance objective over trajectories:

$$
U(x) = E[IS] + \lambda \,\mathrm{Var}[IS], \qquad \lambda \ge 0 .
$$

`U` is a strictly convex quadratic in the interior holdings `x_1..x_{N-1}`,
so first-order conditions `dU/dx_j = 0` are necessary and sufficient. They
give a linear second-order difference equation:

$$
\frac{x_{j-1} - 2x_j + x_{j+1}}{\tau^2} = \tilde\kappa^2\, x_j,
\qquad
\tilde\kappa^2 = \frac{\lambda\,\sigma^2}{\tilde\eta}.
$$

This is a discrete analogue of `x'' = kappa^2 x`; with boundary conditions
`x_0 = X, x_N = 0` the solution is the **sinh trajectory**

$$
x_j = X\,\frac{\sinh\!\big(\kappa\,(T - t_j)\big)}{\sinh(\kappa T)},
\qquad
n_j = \frac{2X\sinh(\kappa\tau/2)}{\sinh(\kappa T)}
      \cosh\!\big(\kappa\,(T - t_{j-1/2})\big),
$$

where the **discrete** rate `kappa` solves
`(2/tau^2)(cosh(kappa*tau) - 1) = kappa_tilde^2`, i.e.

$$
\kappa = \frac{1}{\tau}\,\operatorname{arccosh}\!\left(1 + \frac{\tilde\kappa^2 \tau^2}{2}\right)
\;\xrightarrow[\tau\to 0]{}\; \tilde\kappa .
$$

The implementation computes `n_j` as differences of `x_j`; the cosh
expression is the identity the differences must satisfy (a good property
test). E and Var of the optimum are then evaluated through the **exact
discrete sums** of Section 3.2 — never through the continuous-time
hyperbolic approximations, which differ at the 4th-5th significant digit for
small N and would break the 1e-10 golden tolerances.

### 3.4 Reading kappa: urgency and half-life

`kappa` has units of 1/time and is the **urgency** of the schedule. The
trajectory decays roughly like `exp(-kappa*t)`, so `theta = 1/kappa` is the
**characteristic time** ("half-life" in trader speak): how long the schedule
effectively takes regardless of the nominal horizon T. High risk aversion,
high volatility, or cheap liquidity (small `eta_tilde`) all raise kappa —
you trade faster when waiting is risky and trading is cheap.

Limits:

- `lambda -> 0` (or `sigma = 0`): `kappa -> 0` and
  `sinh(kappa(T-t))/sinh(kappa T) -> (T-t)/T` — the trajectory degenerates to
  the **linear TWAP schedule** `x_j = X(N-j)/N`, which minimizes pure
  expected cost. `theta = +inf` (reported as infinity, not an error).
- `lambda -> inf`: everything is sold in the first interval — the
  **instantaneous liquidation** corner of the frontier, with
  `E = (1/2)gamma_p X^2 + eta_tilde X^2/tau` and `Var = 0`.

Numerically the ratio `sinh(kappa(T-t))/sinh(kappa T)` is 0/0 as
`kappa -> 0`, so the code switches to the exact linear branch whenever
`kappa*T < KAPPA_T_LIMIT = 1e-6`. The two branches agree to ~1e-12 at the
threshold (golden case `ac_lambda_zero_limit_liquid` pins the `lambda = 0`
trajectory to TWAP exactly, tolerance 1e-12).

### 3.5 The efficient frontier of execution

Sweeping `lambda` traces a curve in the `(sqrt(Var), E)` plane — the
**efficient frontier of execution**. Every point is optimal for someone;
where you sit depends on your risk aversion. From the demo (liquid set:
X = 1e6, N = 5, tau = 1, sigma = 0.95, eta = 2.5e-6, gamma_p = 2.5e-7):

| lambda_ra | kappa | E[cost] \$ | sqrt(Var) \$ |
|---|---|---|---|
| 0 (TWAP) | 0.0000 | 600,000 | 1,040,673 |
| 5e-7 | 0.4325 | 705,633 | 750,597 |
| 2e-6 | 0.8463 | 1,078,215 | 449,368 |
| 1e-5 | 1.7267 | 1,782,711 | 171,712 |
| infinity (instant) | — | 2,500,000 | 0 |

Moving down the table you pay more expected cost for less risk. The frontier
is convex: the first units of risk reduction are cheap, the last are ruinous.

### 3.6 Worked example (matches golden case `ac_liquid_solution`)

Liquid set, `lambda_ra = 2e-6`:

1. `eta_tilde = 2.5e-6 - 2.5e-7 * 1.0 / 2 = 2.375e-6` (valid: > 0).
2. `kappa_tilde^2 = 2e-6 * 0.95^2 / 2.375e-6 = 0.76`.
3. `kappa = arccosh(1 + 0.76 * 1 / 2) / 1 = arccosh(1.38)
   = ln(1.38 + sqrt(1.38^2 - 1)) = 0.8462971345012561`;
   half-life `1/kappa = 1.1816180856967273`.
4. `x_2 = 1e6 * sinh(0.84630*3)/sinh(0.84630*5) = 182,932.81426176784`.
5. `n_1 = X - x_1 = 571,401.1542529825` — more than half the order in the
   first of five intervals: risk-averse schedules are front-loaded.
6. Through the exact sums: `E[IS] = 1,078,215.167049785`,
   `Var[IS] = 201,931,287,150.5245` (std = \$449,368).

TWAP on the same parameters: `n_j = 200,000` each, so
`E = 0.5*2.5e-7*1e12 + 2.375e-6*5*200000^2 = 125,000 + 475,000 = 600,000`
and `Var = 0.9025 * (800k^2 + 600k^2 + 400k^2 + 200k^2) = 1.083e12`
(golden `ac_twap_liquid`). The optimal schedule pays \$478k more in
expectation to cut the cost standard deviation by \$591k.

### 3.7 TWAP, VWAP, POV baselines

- **TWAP** (time-weighted): equal clips, `x_j = X(N-j)/N`. It is the
  `lambda = 0` optimum of this model. Real desks use it when volume patterns
  are unreliable or the benchmark itself is TWAP.
- **VWAP** (volume-weighted): clips proportional to the (forecast) market
  volume curve — heavier at the open and close, following the U-shaped
  intraday volume profile. In the AC model with flat volume, VWAP and TWAP
  coincide; the package therefore ships TWAP and POV, the two shapes that
  differ in this framework.
- **POV** (percentage of volume): trade a fixed participation `p` of market
  volume `V` per unit time until done — clip `c = p*V*tau`, duration
  endogenous: `M = ceil(X/c)` intervals, last clip partial. Golden
  `ac_pov_liquid`: p = 15% of 2.5M sh/interval gives clips of 375,000, done
  in M = 3 intervals, `E = 941,406.25`, `Var = 4.0895e11`. POV finishes
  faster than TWAP here, so it costs more impact but carries less risk —
  a mid-frontier heuristic that real desks like because it self-adjusts to
  volume.

**Monte Carlo cross-check.** `simulate_is` simulates the arithmetic walk
path by path and re-measures IS. With 20,000 seeded paths the sample mean
and variance must land within 4 standard errors of the closed forms — golden
case `ac_mc_liquid` stores the Python reference plus SEs so ports with
different RNGs can verify distributional agreement.

---

## 4. Implementation-shortfall decomposition (Perold)

Perold (1988) compared a real portfolio to a **paper portfolio** that
executes instantly, for free, at the decision price. The difference — the
implementation shortfall — decomposes into where the money actually leaked.
Inputs: side, decision price `P_d`, arrival price `P_0` (when trading
began), close `P_T`, parent size X, fills `(n_j, p_j)`, executed
`Q = sum n_j`, remainder `R = X - Q`. For a sell (`sign = +1`; a buy mirrors
every difference with `sign = -1`):

$$
\begin{aligned}
\text{delay} &= Q\,(P_d - P_0)
&&\text{price moved before you started trading}\\
\text{trading} &= \textstyle\sum_j n_j\,(P_0 - p_j)
&&\text{slippage of fills vs arrival (impact + spread)}\\
\text{opportunity} &= R\,(P_d - P_T)
&&\text{cost of the shares you never traded}\\
\text{total} &= X P_d - \textstyle\sum_j n_j p_j - R\,P_T
&&\text{(identity: total = delay + trading + opportunity)}
\end{aligned}
$$

The identity is algebraic, not approximate — the tests assert it to 1e-10,
and it is the first sanity check to run on any TCA vendor's numbers.

**Worked example** (golden `is_decomposition_sell`): sell 1,000 shares,
`P_d = 100.00`, `P_0 = 99.50`, `P_T = 98.80`; fills 300 @ 99.40,
400 @ 99.20, 200 @ 99.10 (Q = 900, R = 100):

- delay = 900 × 0.50 = **\$450** — half a dollar of adverse drift before the
  first fill; the biggest leak, and it happened before the algo even started.
- trading = 300×0.10 + 400×0.30 + 200×0.40 = **\$230** — execution slippage.
- opportunity = 100 × 1.20 = **\$120** — the unfilled tail rode the price
  down.
- total = **\$800** = 80 bps of the \$100 decision value... exactly
  `1000×100 − (sum of fill proceeds) − 100×98.80`.

The buy case (`is_decomposition_buy`) mirrors the prices and yields the same
components — a deliberate symmetry check.

---

## 5. Market making: the other side of the book

The execution trader *takes* liquidity; the **market maker** provides it,
quoting both a bid and an ask and earning the spread on round trips. Two
risks dominate the business:

- **Inventory risk.** Fills are asymmetric by chance (or by flow), so the
  maker accumulates a position `q`. While held, `q` is exposed to
  `sigma` — a maker long 10 units in a falling market loses 10 × the move,
  spread income notwithstanding. Inventory risk grows with `|q|`,
  `sigma`, and the remaining holding time.
- **Adverse selection.** The counterparties who most eagerly hit your quotes
  are the informed ones. A maker's bid gets lifted *because* the price is
  about to fall. In intensity language: your fills are correlated with
  subsequent price moves against you. A-S models arrivals as
  price-insensitive Poisson flow (no adverse selection); real makers add
  protections — see Section 12.

The maker's controls are the **skew** (where to centre the quotes relative
to the mid) and the **width** (how much spread to demand). Avellaneda-
Stoikov derives both from first principles.

---

## 6. Avellaneda-Stoikov market making

### 6.1 Setup

Mid price: arithmetic Brownian motion `dS = sigma dW` over a finite horizon
`[0, T]`. The maker has exponential utility `-exp(-gamma * W_T)` over
terminal wealth and quotes ask/bid at distances `delta_a, delta_b` from the
mid. Market buy/sell orders lift the quotes with Poisson intensity

$$
\lambda(\delta) = A\,e^{-k\delta},
$$

so quoting further away trades less often: `A` scales overall flow, `k`
measures how price-sensitive the flow is (deep books / patient flow = high
`k`). Each fill is one unit.

### 6.2 Reservation price

Solving the HJB equation for a maker with frozen inventory yields the
**indifference (reservation) price** — the effective mid from the maker's
point of view:

$$
r(s, q, t) = s - q\,\gamma\,\sigma^2\,(T - t).
$$

A long maker (`q > 0`) values the asset *below* the market mid: every extra
unit held adds variance `sigma^2 (T-t)` to terminal wealth, and exponential
utility charges `gamma/2` per unit of variance. So the maker shades both
quotes down — the ask becomes more aggressive (sell inventory faster), the
bid less attractive (accumulate slower). This is **inventory skew**, and it
is the entire mechanism of inventory control in the model: no explicit
targets, just a price adjustment linear in `q`.

### 6.3 Optimal spread

A first-order expansion of the HJB optimality conditions in the fill
intensities gives the total quoted spread

$$
\text{spread}(t) = \gamma\,\sigma^2\,(T - t)
                 + \frac{2}{\gamma}\ln\!\Big(1 + \frac{\gamma}{k}\Big),
$$

quoted symmetrically around `r`: `ask = r + spread/2`, `bid = r − spread/2`.
Two terms, two stories:

- `gamma*sigma^2*(T−t)`: **risk compensation** — a fill saddles you with
  inventory for the remaining `T − t`, so charge for it; shrinks linearly to
  zero at the terminal time (**risk-off into the close**).
- `(2/gamma)*ln(1 + gamma/k)`: the **dealer markup** against the exponential
  demand curve — the monopolist's optimal margin, and the terminal floor of
  the spread. As `gamma -> 0` it tends to `2/k` analytically, but the
  utility derivation degenerates, so the code requires `gamma > 0` strictly
  (validated, per spec).

Note what the *distances* `delta_a = ask − s`, `delta_b = s − bid` do under
skew: they are signed, and a sufficiently long maker's ask can cross the mid
(`delta_a < 0` — quoting *through* the market to dump inventory). The
intensity formula still applies; it is not an error.

### 6.4 Fills in discrete time

In a step `dt`, a quoted side fills with the thinned-Poisson probability

$$
P(\text{fill}) = 1 - e^{-\lambda(\delta)\,dt} \in [0, 1).
$$

**Worked example** (liquid set: `A = 140, k = 1.5, dt = 0.005`; golden
`as_fill_prob_liquid`): at `delta = 0.6`,
`lambda = 140*exp(-0.9) = 56.919752363683884` and
`P = 1 - exp(-56.9198*0.005) = 0.24768394830642568`.

**Worked quotes** (golden `as_reservation_liquid`, `as_spread_liquid`):
`s = 100, q = +3, t = 0.3, T = 1, sigma = 2, gamma = 0.1, k = 1.5`:

- `r = 100 − 3*0.1*4*0.7 = 99.16` — the long maker prices 84 cents below
  mid.
- `spread(0.3) = 0.1*4*0.7 + 20*ln(1 + 0.1/1.5) = 0.28 + 1.29077
  = 1.5707704227514234`; terminal floor `1.2907704227514234`.
- ask = 99.9454, bid = 98.3746: the ask sits close to the mid (eager to
  sell), the bid far below (reluctant to buy more).

### 6.5 The simulation, inventory caps, and stress

The simulation (loop order pinned in `API_SPEC.md` Section 2.2) runs
`n_steps = T/dt` steps per path: quote from the current `(s, q, t)`; draw
`u_ask` then `u_bid` (always both — so the random stream does not depend on
cap state); fill quoted sides with `u < P(fill)`; update `q` and cash; then
move the mid. A **hard inventory cap** `q_max` suppresses the bid when
`q >= q_max` and the ask when `q <= −q_max` (one-sided quoting at the cap;
`q_max = 0` quotes nothing), so `|q| <= q_max` holds at every step by
construction — and the simulation asserts it. After the last step the
residual inventory is liquidated at the final mid with penalty
`eta_liq * |q|`:

```
PnL = cash + q*S_T - eta_liq*|q|
```

A **stress mode** doubles the realised volatility from
`t >= stress_time_frac*T` while the agent keeps quoting with the base
`sigma` — a model-misspecification scenario: in the liquid demo, PnL std
rises from 6.37 to 7.90 while the quoting logic stays blissfully unaware.

### 6.6 Why the skew matters: A-S vs fixed spread

The comparison agent quotes the *same average spread*
(`avg_spread = gamma*sigma^2*T/2 + (2/gamma)ln(1+gamma/k)` — the time
average of the A-S spread) symmetrically around the **mid**, with no skew,
same fill mechanics, same RNG. From the demo (2,000 paths, liquid set):

| agent | mean PnL | std PnL | Sharpe | rms q | mean max abs q |
|---|---|---|---|---|---|
| A-S | 55.92 | 6.37 | 8.78 | 1.45 | 4.10 |
| fixed-spread | 57.84 | 12.81 | 4.51 | 4.96 | 9.50 |

Nearly identical expected PnL (the fixed agent even earns slightly more —
it never shades away from flow), but the A-S agent runs **~3.4x smaller RMS
inventory** and half the PnL volatility, roughly doubling the Sharpe-like
ratio. Inventory skew costs almost nothing in expectation and buys an
enormous variance reduction — the central practical lesson of the model.
Golden case `as_fixed_comparison_liquid` pins both RMS inventories with SE
bands.

---

## 7. Model assumptions — and where they break in real life

| Assumption | Reality |
|---|---|
| Linear impact `eta*v`, `gamma_p*n` (AC) | Empirical impact is concave — roughly square-root in participation ("square-root law"). Linear over-penalizes big clips, under-penalizes small ones. |
| Impact parameters known and constant | `eta` must be estimated (Section 3.3 of the code does a toy version by OLS); it varies by name, time of day, and regime. |
| Arithmetic random walk, no drift | Fine intraday; over days, drift and mean reversion matter. Negative prices possible (documented). No fat tails, no jumps — variance understates tail risk. |
| Static AC trajectory | The 2000 solution is deterministic (pre-committed). With linear impact and no signals, the static path is also dynamically optimal — but real algos adapt to fills, news, and volume surprises. |
| Poisson arrivals `A e^{-k delta}` (A-S) | No adverse selection: fills are independent of future price moves. Real fills are informative — makers add alpha signals and quote-fade protection. |
| Unit fill size, one venue, continuous quoting | Real fills are lumpy, markets fragmented, quoting throttled by messaging limits and tick sizes. |
| Exponential utility, terminal-wealth horizon | Gives clean closed forms; real desks think in drawdown limits, VaR budgets and overnight caps rather than gamma. |
| Frictionless quote updates | Every A-S quote reprice is free; in reality cancels/replacements lose queue position (Section 12). |

---

## 8. Practical market conventions: equity vs FX

The three bundled parameter sets are calibrated caricatures:

- **Liquid large-cap equity** (`params_liquid.json`): sigma = 0.95 \$/sqrt
  (interval) on a \$100 stock, eta = 2.5e-6 (selling 100k shares in one
  interval costs 25 cents/share temporary), plentiful flow (A = 140,
  k = 1.5). Impact is small relative to volatility — risk dominates.
- **Small-cap illiquid** (`params_illiquid.json`): eta 4x larger, less flow
  (A = 20), gamma bigger. Impact dominates; optimal schedules stretch out,
  and the maker quotes wide.
- **FX-like** (`params_fx.json`): tight spreads and steep intensity decay
  (k = 2000 — quoting 5 pips wide kills your flow), tiny volatility per
  step, price 1.1000. Note the *units* discipline: gamma = 100 and k = 2000
  look huge but are per-unit-of-price on an asset quoted to 4 decimals —
  always sanity-check `gamma/k` (here 0.05) and `gamma*sigma^2*T` (here
  0.0016, ~1.6 pips of skew per unit of inventory) rather than raw numbers.
  Also FX conventions: two-way prices are quoted as bid/ask outright, sizes
  in base currency, and "spread" is measured in pips = 1e-4.

Equity conventions worth knowing: order sizes in shares with round lots,
prices on a tick grid (\$0.01), volume follows a U-shaped intraday curve
(open/close auctions heaviest), short-sale rules constrain the sell side.
None of these are modelled here — the models live in a friction-free
mid-price world — which is exactly why they are teachable in closed form.

---

## 9. Common pitfalls and numerical issues

1. **The 0/0 at small kappa.** `sinh(kappa(T-t))/sinh(kappa T)` is
   numerically fine down to surprisingly small kappa, but at
   `lambda_ra = 0` exactly it is 0/0. The pinned fix: branch to the exact
   linear TWAP when `kappa*T < 1e-6`. Do **not** branch on
   `lambda_ra == 0` only — `sigma == 0` also gives kappa = 0.
2. **Using continuous-time formulas for E/Var.** The paper's hyperbolic
   closed forms for E and Var are continuous-time *approximations* of the
   discrete sums. At N = 5 they differ from the exact sums well above
   1e-10. All golden values use the pinned discrete sums.
3. **Forgetting `eta_tilde`.** Using raw `eta` in the cost sum double-counts
   permanent impact (the `gamma_p*tau/2` correction removes the overlap
   between the within-interval permanent move and the temporary cost). And
   `gamma_p*tau >= 2*eta` must be an error, not a warning.
4. **`log(1+x)` and `exp(x)-1` naively.** The A-S markup uses
   `log1p(gamma/k)` and the fill probability uses `-expm1(-lambda*dt)`;
   with FX-scale numbers (`lambda*dt ~ 1e-3`, `gamma/k = 0.05`) the naive
   forms lose digits and break the 1e-12 golden cases.
5. **RNG draw-order drift.** The pinned order is: per step, `u_ask` vector,
   then `u_bid` vector, then `z` vector — and both uniforms are drawn even
   when a side is suppressed by the cap. Skip a suppressed side's draw and
   your stream (hence your regression against your own goldens) silently
   forks the first time any path touches the cap.
6. **ddof confusion.** All sample variances/stds here are `ddof = 1`
   (unbiased). Mixing ddof = 0 into `std_pnl` shifts results by ~1/2M —
   invisible in a demo, fatal to a 4-SE statistical golden with M = 20,000.
7. **Comparing MC across languages bit-for-bit.** Only Python's PCG64
   stream is stored. Ports must compare statistically:
   `|stat − ref| <= 4*sqrt(se_ref^2 + se_own^2)` — and must run enough paths
   (`M >= 2000`) for their own SE to be meaningful.
8. **Terminal edge cases.** At `t = T`: spread = terminal floor, `r = s` —
   valid inputs, not errors. `q_max = 0`: quotes nothing, PnL identically 0.
   `N = 1`: single clip = instantaneous liquidation. `X = 0`: empty
   trajectory, `E = Var = 0`. All are tested, none should throw.
9. **Signed deltas.** Under heavy skew a quote crosses the mid
   (`delta < 0`, fill probability rises toward `1 − exp(−A dt)`). Clamping
   delta at 0 "to be safe" changes the dynamics and breaks the golden
   simulation statistics.

---

## 10. How real e-trading desks differ

The models here are the intellectual core of real systems, wrapped in layers
this project deliberately omits:

- **Order types & venues.** Real child orders are limit orders with
  time-in-force flags (IOC, FOK, GTC), peg orders, icebergs/reserve, and
  auction orders — not idealized mid-price trades. Marketable flow is
  routed by a **smart order router** across fragmented venues (a dozen-plus
  lit exchanges in US equities, plus dark pools and single-dealer
  platforms), each with its own fees/rebates and latency.
- **Queues.** At a given price level, execution is FIFO: **queue position**
  is an asset. A maker who reprices on every tick (as A-S implies) donates
  queue priority; real makers trade off skew accuracy against staying near
  the front of the queue.
- **Impact models.** Desks fit concave (square-root) impact with decay
  kernels from their own fill data, often with **ML models** on top —
  gradient-boosted or neural models over book-shape, volume, volatility and
  news features to predict short-horizon cost and adverse selection. The
  OLS-on-participation recipe in this package is the "hello world" of that
  pipeline.
- **Adaptivity.** Production algos re-optimize on fills, volume surprise
  and signals ("aggress when ahead of schedule and the book is thick");
  AC gives the baseline schedule the adaptive layer perturbs.
- **Risk plumbing.** Kill switches, fat-finger and price-collar checks,
  self-match prevention, messaging throttles, and exchange-mandated market
  maker obligations (max spread, min size, min uptime).

## 11. Regulatory context

- **Best execution.** Brokers must take *all sufficient steps* (MiFID II
  wording; FINRA: *reasonable diligence*) to obtain the best possible result
  for clients — price, cost, speed, likelihood of execution and settlement,
  size, nature. This is a process obligation: firms must define an execution
  policy, monitor outcomes, and evidence them. The IS decomposition of
  Section 4 is exactly the kind of **TCA** (transaction cost analysis)
  evidence regulators and clients expect.
- **MiFID II** (EU, 2018): algorithmic-trading firms must have tested,
  monitored algos with kill functionality (Art. 17); market makers running
  continuous strategies take on formal quoting obligations; execution
  quality reporting (the RTS 27/28 regime, since scaled back) pushed
  standardized venue statistics; unbundling separated research payments
  from execution commissions.
- **US:** Reg NMS (order protection rule, best displayed prices), SEC Rules
  605/606 (execution-quality and routing disclosure), the Market Access
  Rule 15c3-5 (pre-trade risk checks on every order).
- Almgren-Chriss-style frameworks are the lingua franca of best-execution
  committees: an E-V frontier is a defensible way to document *why* an
  order was traded at a given urgency.

---

## 12. Interview-style Q&A

**Q1. Why is the optimal AC trajectory front-loaded for lambda > 0?**
Variance comes from inventory held (`sigma^2 tau sum x_j^2`), cost from
trading speed (`sum n_j^2`). A risk-averse trader accepts extra impact early
to shrink the inventory that generates risk later. The trade-off equalizes
at the sinh shape; urgency `kappa = sqrt(lambda sigma^2/eta_tilde)` (in the
tau -> 0 limit) grows with risk aversion and volatility, falls with impact
cost.

**Q2. Why does permanent impact contribute `gamma_p X^2 / 2` regardless of
the schedule?** Each share you sell pushes the price down `gamma_p` for all
*later* sales. Summed over ordered pairs, the total cost is
`gamma_p * sum_{i<j} n_i n_j = gamma_p (X^2 − sum n_j^2)/2`; the
schedule-dependent `sum n_j^2` piece is absorbed into `eta_tilde`, leaving a
fixed `gamma_p X^2/2`. Consequence: permanent impact does not affect the
*shape* of the optimal trajectory in this model.

**Q3. When is TWAP exactly optimal?** When `lambda_ra*sigma^2 = 0`: risk
neutrality or zero volatility. Then only `sum n_j^2` matters, minimized by
equal clips (convexity/Jensen). This is also the code's `kappa*T < 1e-6`
limit branch — and golden case `ac_lambda_zero_limit_liquid` asserts the
identity to 1e-12.

**Q4. Your AC model says sell 1M shares in 5 minutes. What do you check
before believing it?** That `eta, gamma_p` were estimated on comparable
sizes (linear impact extrapolates badly — square-root law), that
participation per interval is sane vs typical volume, that
`gamma_p*tau < 2*eta` holds, that sigma isn't stale (regime change), and
that the horizon vs half-life `1/kappa` makes sense — if `1/kappa << T` the
nominal horizon is irrelevant and you are effectively doing a fast
liquidation.

**Q5. Decompose implementation shortfall and state the identity.**
Delay `Q(P_d − P_0)` + trading `sum n_j (P_0 − p_j)` + opportunity
`R(P_d − P_T)` (sell signs), summing *exactly* to
`X P_d − sum n_j p_j − R P_T`. Interview follow-up: a desk with great
"trading" numbers but chronic delay cost has a decision-to-desk latency
problem, not an execution problem.

**Q6. Why does the A-S reservation price sit below the mid when long?**
Exponential utility charges `gamma/2 * sigma^2 (T−t)` of certainty
equivalent per unit of inventory variance. Long inventory makes additional
buys risk-increasing and sells risk-reducing, so the maker's indifference
price drops by `q gamma sigma^2 (T−t)`: the ask gets aggressive, the bid
backs off, and flow mean-reverts the inventory.

**Q7. What are the two terms of the A-S optimal spread?**
`gamma sigma^2 (T−t)` — inventory-risk compensation, decaying to zero at
the horizon — plus `(2/gamma) ln(1 + gamma/k)` — the demand-curve markup
against `A e^{−k delta}` arrivals, the terminal floor. Higher `k` (more
price-elastic flow) tightens the markup; higher `gamma sigma^2` widens the
risk term.

**Q8. Why does the fixed-spread agent match A-S PnL but with far worse
inventory?** Expected spread income depends mostly on the average quoted
width, which is identical by construction. But without skew, inventory is a
random walk reflected only at `q_max`, so `E[q^2]` grows large (rms q 4.96
vs 1.45 in the demo); the extra `q` variance feeds straight into PnL
variance via `sigma`. Skew converts inventory from a random walk into a
mean-reverting process at essentially zero expected cost.

**Q9. What is adverse selection for a market maker, and where does A-S
ignore it?** The tendency of fills to precede adverse price moves — you get
lifted before rallies, hit before sell-offs — because some counterparties
are informed. A-S arrivals are Poisson, independent of future `dW`, so fills
carry no information; the model prices only inventory risk. Real makers add
short-horizon alpha to the reservation price and fade quotes on toxic-flow
signals.

**Q10. Your port's simulated `mean_pnl` misses the Python golden by 2.5
combined SEs. Bug?** Not evidence of one: with the 4-SE acceptance band,
|z| = 2.5 is within tolerance and expected occasionally across many
statistics. Investigate only if it exceeds 4 SE, drifts systematically with
M, or co-occurs with other near-band misses; first suspects are draw order,
ddof, and the cap's always-draw rule — not the RNG itself.

---

## 13. Further reading

- R. Almgren, N. Chriss, "Optimal Execution of Portfolio Transactions",
  *Journal of Risk* 3(2), 2000 — the AC model, exactly as pinned here.
- M. Avellaneda, S. Stoikov, "High-frequency trading in a limit order
  book", *Quantitative Finance* 8(3), 2008 — reservation price and spread.
- A. Perold, "The Implementation Shortfall: Paper versus Reality", *Journal
  of Portfolio Management*, 1988 — the decomposition of Section 4.
- R. Kissell, *The Science of Algorithmic Trading and Portfolio
  Management* — practitioner treatment of benchmarks, POV/VWAP, TCA.
- L. Harris, *Trading and Exchanges* — the microstructure classic; order
  types, dealers, adverse selection.
- Cartea, Jaimungal, Penalva, *Algorithmic and High-Frequency Trading* —
  rigorous stochastic-control treatment generalizing both models here.
- Guéant, Lehalle, Fernandez-Tapia, "Dealing with the inventory risk"
  (2013) — the A-S model solved with inventory limits (closed forms for the
  capped case).
- J.-P. Bouchaud et al., *Trades, Quotes and Prices* — empirical impact,
  the square-root law, order-book dynamics.
- ESMA MiFID II best-execution guidance and SEC Rules 605/606 — the
  regulatory texts behind Section 11.
