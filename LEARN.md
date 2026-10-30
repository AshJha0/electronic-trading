# LEARN — Electronic Trading: Optimal Execution and Market Making

This document teaches the theory behind the `etrade` research platform: how modern
electronic markets work at the microstructure level, why executing a large
order is a genuine optimization problem, the Almgren-Chriss solution to it,
the Perold implementation-shortfall framework used to measure execution
quality, and the Avellaneda-Stoikov model of an inventory-averse market
maker. It also covers stating parameters in market units (Section 3.8),
the square-root law of impact with intraday volume and alpha-aware schedules
(Section 3.9), calibration from recorded data (3.10), transient impact
(3.11), adaptive re-planning (3.12), internalisation (3.13) and post-trade
analysis: benchmarks, markouts and comparing algorithms (4.2-4.4). A
glossary is at the end (Section 15). The closed-form numbers below are
reproduced by the code and
pinned in `data/golden/golden.json` (tolerance 1e-10..1e-12); the market-unit
and power-law figures are pinned in `data/golden/market_golden.json` and
`data/golden/impact_golden.json` and printed by every demo; the TCA,
calibration and execution-extension figures come from the golden files of
`API_SPEC.md` Sections 10-12 or from COOKBOOK recipes 15-22, run in all four
languages. The simulation
numbers (Section 6.5–6.6) are 2,000-path samples from the Python demo that
the C++/Rust/Java ports reproduce *statistically* (within the 4-SE bands of
the golden file), not bit-for-bit; those of Section 3.12 are 20,000-path
Python samples.

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
fact (real-world impact is closer to square-root in size, Section 3.9), but it
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
test). E and Var of the optimum are evaluated through the **exact discrete
sums** of Section 3.2. These coincide with the paper's hyperbolic closed
forms for E and V (evaluated with the *discrete* kappa above) to machine
precision — on the liquid set `E_paper = 1,078,215.1670497851` vs the sum
`1,078,215.167049785` (relative difference 2e-16) — and every test suite
asserts that identity at 1e-12. The sums are pinned because one code path
also covers TWAP, POV and any hand-built schedule.

Two numerical details are pinned as well (`API_SPEC.md` Section 1.4):
`kappa` is evaluated as `log1p(x + sqrt(x(2+x)))/tau` with
`x = kappa_tilde^2 tau^2/2`, which is `arccosh(1+x)/tau` without the
cancellation that costs `acosh(1+x)` about `eps/x` relative digits when
`tau` is a second rather than a day; and the sinh ratio is evaluated as
`exp(-kappa t_j) * expm1(-2 kappa (T - t_j)) / expm1(-2 kappa T)`, which is
the same number for every `kappa > 0` but cannot overflow (`sinh` overflows
at `kappa T > 710`, i.e. a horizon of more than ~710 characteristic times).
Note that `kappa T` does not depend on the grid: refining `tau` at a fixed
horizon leaves it essentially unchanged (0.8497 at N = 390 and at
N = 23,400 on the liquid set). It is reached by very urgent schedules or very
long horizons, not by fine grids.

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
   = ln(1.38 + sqrt(1.38^2 - 1)) = 0.846297134501256`;
   half-life `1/kappa = 1.18161808569673`.
4. `x_2 = 1e6 * sinh(0.84630*3)/sinh(0.84630*5) = 182,932.814261768`.
5. `n_1 = X - x_1 = 571,401.154252983`: more than half the order in the
   first of five intervals, because risk-averse schedules are front-loaded.
6. Through the exact sums: `E[IS] = 1,078,215.16704979`,
   `Var[IS] = 201,931,287,150.524` (std = \$449,368).

(Golden case values to 15 significant digits; the last one or two of the 17
printed digits differ by an ulp between platforms.)

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
  coincide; they separate once liquidity varies through the day, which the
  power-law model of Section 3.9 supports (`vwap_holdings` plus a volume
  curve).
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

### 3.8 Parameters in market units

**The plain idea.** Nobody on a desk knows their `eta` or `gamma_p`. They
know the price, the quoted spread, the average daily volume (ADV) and the
daily volatility. The market-unit layer turns those into model parameters
with a stated rule, so that every result can be read in basis points of the
order's notional.

**The formal statement** (`API_SPEC.md` Section 7.1, which follows the
calibration in Section 4 of Almgren and Chriss 2000): trading 1% of ADV per
day costs one spread of temporary impact, and 10% of ADV moves the price
permanently by one spread:

```
sigma   = price * daily_vol
eta     = spread / (0.01 * ADV)
gamma_p = spread / (0.10 * ADV)
```

With the paper's own inputs (price 50, spread 0.125, ADV 5M, daily vol 1.9%)
this gives exactly the liquid parameter set used above (`eta = 2.5e-6`,
`gamma_p = 2.5e-7`, `sigma = 0.95`). The A-S side (Section 7.2) is set from
the tick: a touch fill rate, the distance in ticks over which the arrival
rate halves (`k = ln 2 / (halving_ticks * tick)`) and the opening skew per
unit of inventory.

**Worked example** (preset `us_large_cap` in `data/presets.json`, golden
cases in `market_golden.json`): \$100 stock, one-cent spread (1 bp), ADV 5M,
daily vol 1.5%; sell 250,000 shares (5% of ADV) over one day in thirteen
30-minute intervals at `lambda_ra = 1e-7`. Then `sigma = 1.5`,
`eta = 2e-7`, `gamma_p = 2e-8`, and the optimal schedule costs
`E[IS] = 5.35 bps` with a standard deviation of 75.5 bps; the half-spread
adds 0.50 bps. The A-S block of the same preset quotes 3.58 bps wide at the
open and 1.33 bps at the close.

The five presets (US large cap, US small cap, US ETF, EUR/USD and the paper
example) are illustrative, not fitted to data. Because the calibration is
linear it under-prices small orders against the square-root law, which is
the subject of the next section.

### 3.9 Beyond linear impact: the square-root law and alpha-aware schedules

**The plain idea.** Linear impact says that trading twice as fast costs
twice as much per share. Measured trading costs do not behave like that: the
cost per share of an order grows roughly with the *square root* of its size
relative to daily volume. Doubling an order raises its cost per share by
about 40%, not 100%. This is one of the most robust regularities in market
microstructure (Almgren et al. 2005; Tóth et al. 2011; Bouchaud et al.
2018). The practical consequence is that linear models under-price small
orders, which are most orders, and over-price very large ones.

**The formal statement.** The `impact` module replaces the temporary cost
`(eta/tau) n^2` per interval with

$$
c_j(n) = \eta_j\, n^{1+\alpha} - \tfrac12 \gamma_p n^2,
\qquad \eta_j = \eta\, w_j^{-\alpha},
$$

where `w_j` is the fraction of daily volume traded in interval `j`. With
`alpha = 1` and uniform weights this is exactly Almgren-Chriss; with
`alpha = 1/2` it is the square-root law. In market units
`eta = Y * sigma / ADV^alpha`, so trading at participation `p` costs
`Y * sigma * p^alpha` per share, and a one-day TWAP of `Q` shares costs

$$
\text{cost (fraction of price)} = Y \cdot \sigma_{\text{daily}} \sqrt{Q/\text{ADV}},
$$

with `Y` of order one. There is no closed-form optimum any more, so the
schedule is computed by a pinned Newton method (`API_SPEC.md` Section 8.2).

**Worked example.** US large cap, \$100, daily volatility 1.5%, ADV 5M,
sell 250,000 shares (5% of ADV) in one day: the square-root law gives
`1.5% * sqrt(0.05) = 33.5 bps`. The linear calibration of Section 3.8 gives
5.35 bps for the same order, more than six times less. On a U-shaped volume
curve, VWAP costs 33.8 bps and TWAP 35.0 bps: trading against the curve
costs more, because it trades hardest when the market is thinnest.

**Alpha-aware scheduling.** A client usually sells because they expect the
price to fall. If the price is expected to drift against the order by `mu`
per share per day, every share still held costs `mu * tau` per interval,
which adds `mu * tau * sum x_k` to expected cost. The optimum then trades
faster, and if the drift is strong enough it finishes before the horizon;
the solver detects that and certifies the result with the KKT condition.
In the worked example a 50 bps/day drift makes VWAP cost 57.3 bps while the
drift-aware schedule costs 52.1 bps. The urgency of an order is a business
decision about the information behind it; this is where the model lets a
desk price that decision.

**What it still leaves out.** Volatility is constant through the day (real
intraday volatility is also U-shaped) and impact is instantaneous rather
than decaying over time. Section 3.10 measures the volume and volatility
curves on recorded data; Section 3.11 adds impact that decays.

### 3.10 Calibration: measuring the inputs, and what one day can tell you

Claims in this and the following sections carry an evidence class:
*asserted* (checked by every test suite), *measured* (computed from data
in this repository), *published* (a peer-reviewed result, cited),
*derived* (arithmetic shown in the text) or *judgement* (the author's
reading, open to dispute).

**The plain idea.** Every model above needs numbers: how volume is spread
over the day, how volatile the price is, how wide the spread is, and how
often a resting quote gets hit at a given distance from the mid. Each of
these can be counted from a day of market data. Counting is easy. The hard
part is knowing how far one day's count generalises.

**The formal statement** (`API_SPEC.md` Section 11). A session `[t0, t1]`
is cut into `B` equal bins, and an observation at `t` falls in bin
`min(floor((t - t0) B / (t1 - t0)), B - 1)`. Then:

- the **volume curve** is `w_j = V_j / sum V`, the share of the session's
  volume traded in bin `j`;
- the **realised variance** of bin `j` is the sum of squared log returns of
  the mid ending in it, `RV_j = sum r_i^2`, and the daily volatility is
  `sqrt(sum_j RV_j)` (Andersen, Bollerslev, Diebold and Labys 2001,
  published);
- the **spread statistics** are the mean and median of the quoted spread in
  ticks and in bps;
- the **fill intensity** fits `lambda(delta) = A exp(-k delta)` to the rate
  at which market orders reach distance `delta` from the mid, by least
  squares on `ln(count)` weighted by the count. The weight is there because
  a Poisson count `c` has `Var(ln c) ~ 1/c` (delta method, derived): without
  it the sparse tail buckets set the slope.

**Worked example** (measured; `data/calibration/fitted_20191230.json`,
table in `API_SPEC.md` 11.6). On the NASDAQ ITCH sample day 2019-12-30, AAPL
traded 5,678,176 shares on NASDAQ in the continuous session, with an
open-to-close realised volatility of 1.157% and a mean quoted spread of
2.64 ticks (0.91 bps). Its market orders reached 0.5, 1.5 and 2.5 ticks from
the mid 43,334, 11,235 and 1,307 times. The fit gives
`touch_fills_per_day = 22,070` per side and `halving_ticks = 0.460`, which
`as_params_from_market` turns into `k = 150.684169686945` per dollar and
`A = 46,882.2935867192` per day (COOKBOOK recipe 18, asserted round trip).
The fitted volume curve puts 19.9% of the day's volume in the first half
hour and 14.5% in the last, against 12% and 14% in the illustrative curve
of Section 3.9.

**What one day and three names can tell you** (judgement, with the
arithmetic derived):

- *Orders of magnitude and shapes.* The spread is a few ticks for AAPL and
  about one tick for MSFT and QQQ; mean and median agree, so the estimate
  is not driven by a few wide moments. Volume is U-shaped. These are
  stable features of large US names and one day is enough to see them.
- *Not the sampling error.* Under i.i.d. Gaussian returns the relative
  standard error of a realised volatility built from `n` returns is about
  `1/sqrt(2n)` (derived): 12.9% for a thirty-minute bin of 30 one-minute
  returns and 3.6% for the whole day of 390. That is the *best* case; real
  returns have fat tails and volatility clustering, so the true error is
  larger. Day-to-day variation, the larger effect, cannot be measured from
  a single day at all.
- *Not impact.* The day contains trades, not parent orders, so the
  square-root prefactor `Y` and exponent `alpha` of Section 3.9 are not
  fitted; they come from the literature. The fitted presets combine
  measured market inputs with assumed impact parameters.
- *Not the consolidated market.* NASDAQ is one venue. Its volume is a lower
  bound on consolidated ADV, so impact calibrated on it is pessimistic; its
  best quotes are never tighter than the NBBO.
- *Not a typical day.* The day falls between Christmas and New Year, when
  US volume is usually light.
- *Not an exponential.* The fill counts fall by a factor of 3.9 from 0.5 to
  1.5 ticks and by 8.6 from 1.5 to 2.5 ticks for AAPL (measured): the decay
  is not a constant ratio, so `A exp(-k delta)` is a convenient fit, not a
  law. The weighted fit gets the touch rate right to within 2% and
  understates the tail (`API_SPEC.md` 11.6).

The honest use of these numbers is as a sanity anchor: a model whose spread
is ten times the measured one, or whose volume curve is flat, is wrong.
Using them as precise inputs would need many days, the consolidated tape and
a standard error on every estimate.

### 3.11 Transient impact and resilience (Obizhaeva-Wang)

**The plain idea.** When you sell into the book you eat the resting bids,
and the price drops. Then new bids arrive and the price recovers part of the
way: the book is *resilient*. Almgren-Chriss treats the temporary part of
impact as gone by the next interval and the permanent part as there forever.
Real impact sits in between and decays over minutes. If it decays, the
timing of trades matters in a new way: a trade made right after another one
pays for the first one's unrecovered push.

**The formal statement** (Obizhaeva and Wang 2013, published; discrete form
pinned in `API_SPEC.md` 12.1). Trading `n` shares moves the price by
`kappa n` (the inverse of book depth). The displacement decays by a factor
`a = exp(-rho tau)` per interval, so before trade `j` it is
`D_j = a (D_{j-1} + kappa n_{j-1})`, and

$$
E[\text{IS}] = \tfrac12 \gamma_p X^2 + \sum_j \Big[ \tfrac12(\kappa - \gamma_p)\, n_j^2 + n_j D_j \Big]
             = \tfrac12 \gamma_p X^2 + \tfrac12\, n^\top A\, n,
\qquad A = \kappa K - \gamma_p I,\; K_{ij} = a^{|i-j|}.
$$

`K` is a Kac-Murdock-Szegő matrix, whose inverse is tridiagonal, so the
optimum `n = X y / sum(y)` with `A y = 1` comes from one Thomas solve, with
no iteration (`docs/ARCHITECTURE.md` Section 3). The cost is convex in the
trades exactly when `gamma_p < kappa (1 - a)/(1 + a)` (asserted). With
`gamma_p = 0` the optimal trades are proportional to
`[1, 1-a, ..., 1-a, 1]`: a block at the start, a constant rate in between,
a block at the end (Obizhaeva and Wang 2013; Alfonsi, Fruth and Schied 2010,
published; asserted by every suite).

**Worked example** (COOKBOOK recipe 19). Sell 1,000,000 shares in 10
intervals of `tau = 0.1` with `kappa = 1e-6` and `rho = 5`, so
`a = exp(-0.5) = 0.607`. The optimum trades 194,259 shares at each end and
76,435 in each of the eight middle intervals; `76,435 / 194,259 = 0.393 =
1 - a` (derived). Its expected cost is 156,041.881012075 against
165,236.399784175 for TWAP, 5.6% less. The first block is cheap because the
book is untouched; the middle trades are small because each one pays for
the decaying push of the ones before; the last block is cheap because no
later trade will pay for its push.

**The link to Almgren-Chriss** (asserted). With `kappa = 2 eta / tau` and
resilience so fast that `a` underflows to 0, every push is gone by the
next interval, the cost equals Section 3.2's expected cost with
`lambda_ra = 0`, and the optimum is TWAP. Almgren-Chriss is the
instant-resilience limit of this model.

**What it leaves out.** The model is risk-neutral (no variance term), the
decay is exponential, and the book shape is flat (linear impact). Gatheral
(2010) shows that some combinations of decay kernel and impact shape admit
price manipulation; the exponential-linear pair used here does not
(published).

### 3.12 When does re-planning help? Time consistency

**The plain idea.** An execution algorithm can follow a schedule fixed at
the start, or re-solve the problem every interval from wherever it now is.
It seems obvious that re-solving should do better. In the Almgren-Chriss
model, with nothing new learned along the way, it does exactly as well and
no better: re-solving from the realised state gives back the remaining
part of the original plan. Re-planning pays only when the trader learns
something the plan did not know.

**The formal statement** (`API_SPEC.md` 12.2). The re-plan at decision `k`
with holding `h` and an observed drift signal `s` minimises the remaining
mean-variance cost minus the signal's expected gain. The solution is linear
in `(h, s)`: the next child order is `c_m h - d_m s` with `m` intervals left.
With `s = 0` the re-plan from the planned holding reproduces the planned
tail exactly. That is time consistency: the plan made at time 0 is still
optimal when evaluated at any later time, from the state it predicted, and
from any other state too, because the problem from `(k, h)` has the same
shape as the original (asserted to `1e-9 X`).

**Worked example** (COOKBOOK recipe 20). Liquid parameters, 20,000 paths,
seed 1:

| signal and fills | static mean IS | adaptive mean IS | paired saving (SE) |
|---|---|---|---|
| none, full fills | 1,081,939 | 1,081,939 | 0 (0) |
| AR(1) drift, `phi = 0.7`, `signal_sd = 0.5` | 1,082,614 | 1,035,683 | 46,931 (815) |
| random fills, `fill_risk = 0.5` | 880,416 | 830,959 | 49,457 (1,716) |

The first row is zero on every path, not just on average (asserted). The
signal row saves 4.3% of the static cost at more than 50 standard errors:
the adaptive trader sells faster when the signal says the price is about to
fall. The fill-risk row saves because the static plan, having fallen short,
dumps the shortfall in the last interval, while the re-planner spreads it
over the remaining ones. The coefficients make this concrete: with four
intervals left, `c_4 = 0.573184071592799`, so a holding of 428,598.85 shares
(the static plan's) gives a child order of 245,666, exactly the static
plan's second trade (derived from `replan_coefficients`). With
`d_4 = 122,742.71`, a signal of `s = +0.2` (the price expected to rise)
cuts that child order by 24,549 shares, and `s = -0.2` adds as many.

**What this does and does not show.** Almgren and Lorenz (2007, published)
find a different gain that this model excludes: if the trader is judged on
one mean-variance criterion for the whole trade, rather than re-optimising
the same criterion each period, a rule that sells faster after gains and
slower after losses beats the static optimum. That rule is not time
consistent and is not implemented here.

### 3.13 Internalisation economics

**The plain idea.** A broker that holds a client buy order and a client
sell order in the same stock can cross them against each other at the mid
and send only the difference to the market. Both clients avoid the impact
the crossed shares would have caused. Because impact grows faster than
linearly in size, the saving is larger than the crossed fraction alone
suggests.

**The formal statement** (`API_SPEC.md` 12.3). Let `C(Q)` be the Section
3.9 expected cost of working `Q` shares on the VWAP schedule. With buys
summing to `B` and sells to `S`, the baseline sends both, costing
`C(B) + C(S)`; internalising sends only `|B - S|`, costing `C(|B - S|)`.
`C` is increasing and `C(0) = 0`, so the saving is never negative
(asserted on random order sets). Under the square-root law the temporary
cost per share grows like `Q^(1/2)`, so the total grows like `Q^(3/2)`.

**Worked example** (COOKBOOK recipe 21). Buys of 200,000 and 50,000, a sell
of 150,000, on the US large-cap inputs with the illustrative volume curve.
150,000 shares cross; 100,000 net buy go to the market. The impact bill
falls from 123,600.89 to 21,304.64, a saving of 102,296.25, or 25.57 bps of
the 400,000 gross shares. A pure `Q^(3/2)` law predicts the ratio
`100^1.5 / (250^1.5 + 150^1.5) = 0.1727` (derived); the model gives
0.1724, the difference being the permanent-impact term.

**Who gets the saving** (judgement). The contract allocates pro rata within
each side: every buy saves 25.25 bps and the sell 26.12 bps. Other rules are
defensible (all to the smaller side, which made the cross possible; or
split by notional), and regulation cares about this. In the EU a firm that
crosses client orders still owes each client best execution (MiFID II,
Directive 2014/65/EU, Article 27); in the US, execution quality and order
routing are disclosed under SEC Rules 605 and 606. Crossing also leaks
information if done carelessly, and a client who learns that the other side
was a large seller may judge the "saving" differently. The numbers here
are the impact arithmetic only.

---

## 4. Post-trade analysis: implementation shortfall, benchmarks, markouts

### 4.1 Implementation-shortfall decomposition (Perold)

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
- total = **\$800** = 80 bps of the \$100,000 decision value (1,000 shares ×
  \$100)... exactly `1000×100 − (sum of fill proceeds) − 100×98.80`.

The buy case (`is_decomposition_buy`) mirrors the prices and yields the same
components — a deliberate symmetry check.

### 4.2 Benchmarks, and why interval VWAP

**The plain idea.** "How much did this order cost?" needs a reference
price, and each common reference answers a different question. The
*arrival* price (the mid when the order reached the desk) measures the
whole cost of trading, including the market moving while you traded. The
*VWAP* of the day asks whether you traded at the market's average price.
The *close* asks how you did against the price at which the order's
performance is often marked. A benchmark that the trader influences, or
that includes prices the order could never have traded at, gives a
misleading score.

**The formal statement** (`API_SPEC.md` 10.1). With fills `(n_j, p_j)`,
average price `pbar` and `sign = +1` for a sell, the cost against a
benchmark `B` is `1e4 * sign * (B - pbar) / B` bps, positive meaning
worse than the benchmark. The *interval VWAP* restricts the market VWAP to
the bins in which the order traded:

$$
\text{interval VWAP} = \frac{\sum_{b \in S} V_b P_b}{\sum_{b \in S} V_b},
\qquad S = \{\, b : \text{the order filled in bin } b \,\}.
$$

**Worked example** (COOKBOOK recipe 15). A sell of 70,000 shares in the
first three of four bins of a falling market:

| benchmark | price | cost (bps) |
|---|---|---|
| arrival | 100.00 | 10.3571428571428 |
| full-day VWAP | 99.8766666666667 | -1.97863078749529 |
| interval VWAP | 99.905 | 0.857957917164196 |
| close | 99.80 | -9.66218150586925 |

Against the day's VWAP and the close the order seems to have beaten the
market, only because it finished before the last, lowest-priced bin. The
interval VWAP compares the order with the market over the time it was
actually trading and shows a small cost.

**Why interval VWAP** (judgement, with published support). Berkowitz, Logue
and Noser (1988) proposed the day's VWAP as a benchmark because it needs no
model and cannot be gamed by choosing a single print. Its weakness is the
one in the example: an order that trades over part of the day is scored
against prices it could not reach, so the score mostly measures the
market's drift. The interval VWAP keeps the model-free property and removes
the timing luck. It has its own flaw: for an order that is a large part of
the volume in its bins, the benchmark contains the order's own fills and
impact, and flatters it (a 100%-participation order matches its interval
VWAP exactly). Use it next to arrival cost, which charges the order for
everything it did, and alongside participation (10.3).

### 4.3 Markouts

**The plain idea.** A markout asks what the mid did after each fill. If
prices keep moving against the side you traded on (they rise after you buy),
your counterparties were trading on information you lacked, or the fill was
too early. If they move your way, you were either well informed or pushing
the price yourself.

**The formal statement** (`API_SPEC.md` 10.2). For fill `j` at step `t_j`
and horizon `h`, `mk_j(h) = d (m_{t_j + h} - m_{t_j})` with `d = +1` for a
buy and `-1` for a sell, in currency per share. Negative is adverse. The
module reports the average per fill and the quantity-weighted average per
share. Fill prices are not inputs: a markout measures the information in
the timing of the fill, not the spread paid.

**Worked example** (COOKBOOK recipe 15). The four sells, marked one and
three steps later, average +1.6 and +4.9 cents per share (per share,
`0.0157142857142836` and `0.048571428571426`). The mid kept falling after
each sale, favourable to a seller. On an order that is 4.7% of the volume,
part of that fall is the order's own impact (Section 3.11). A market maker
reads the same statistic the other way: negative markouts on its passive
fills are adverse selection, the cost Section 6 does not model
(Glosten and Milgrom 1985, published, for why informed flow makes it
unavoidable).

### 4.4 Comparing algorithms: is the difference real?

**The plain idea.** Algo A cost 2.9 bps on average and algo B 0.7 bps. Is A
worse, or did it get the harder orders, or was it unlucky? Per-order costs
are noisy (a standard deviation of 10 bps or more is common for liquid
names, judgement), so a few orders prove little.

**The formal statement** (`API_SPEC.md` 10.4). Welch's t-test (Welch 1947,
published) compares two means without assuming equal variances:
`t = (mean_a - mean_b) / sqrt(v_a/n_a + v_b/n_b)` with the
Welch-Satterthwaite degrees of freedom
`df = (v_a/n_a + v_b/n_b)^2 / ((v_a/n_a)^2/(n_a-1) + (v_b/n_b)^2/(n_b-1))`.
The two-sided p-value uses the Student t distribution, computed through the
regularised incomplete beta function (pinned so that every port gives the
same p-value; `docs/ARCHITECTURE.md` Section 3).

**Worked example** (golden `welch_unequal_variance`, COOKBOOK recipe 16).
Eight orders of A with mean 2.85 bps, six of B with mean 0.683 bps:
`t = 3.76751684064683`, `df = 10.1837133791222`,
`p = 0.00356077027255458` (SciPy agrees).

**How many orders a comparison needs** (derived). To detect a true
difference `delta` with 80% power at the 5% level, each arm needs about
`n = 2 (z_0.975 + z_0.8)^2 (sigma/delta)^2 = 15.7 (sigma/delta)^2` orders.
With `sigma = 10` bps and `delta = 1` bp that is 1,570 orders per algorithm.

**What the test assumes** (judgement). Orders are independent (they are
not, when several are worked at once in the same name), the two samples
face the same order mix (randomise the assignment of orders to algorithms,
or the test compares order mixes), and the cost is measured against the
same benchmark. With costs skewed by a few large orders, also look at the
median and a trimmed mean.

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
  protections — see Section 11.

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

This exact thinned probability is the pinned choice. The paper's own
simulation (as commonly reproduced) uses the first-order `lambda*dt`
instead; with `lambda*dt ~ 0.28` at these parameters that over-fills by
~12%, which is why the mean PnL in Section 6.6 (~56) sits below the paper's
Table 1 value (~65 at `gamma = 0.1`) while the PnL standard deviations (6.4
vs 6.4; symmetric strategy 12.8 vs 12.7) agree.

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

| agent | mean PnL | std PnL | Sharpe | rms q | mean max abs q | fills |
|---|---|---|---|---|---|---|
| A-S | 55.92 | 6.37 | 8.78 | 1.45 | 4.10 | 85.6 |
| fixed-spread | 57.84 | 12.81 | 4.51 | 4.96 | 9.50 | 81.8 |

(`fills` is the total number of fills per path, both sides counted — a step
in which both the bid and the ask fill contributes two; ~4–5% of steps here.
The 20,000-path golden reference is 85.53 ± 0.06.)

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
| Linear impact `eta*v`, `gamma_p*n` (AC) | Empirical impact is concave, roughly square-root in participation ("square-root law"). Linear over-penalizes big clips and under-penalizes small ones. Section 3.9 adds the power-law model, with instantaneous impact; Section 3.11 adds exponential decay, but only with linear impact. |
| Impact parameters known and constant | `eta` must be estimated (`estimate_impact` in `analytics` does a toy version by OLS, Section 10); it varies by name, time of day, and regime. Five presets and the pinned volume curve are illustrative; three presets and a second curve are fitted to one ITCH day (Section 3.10), and even those take their impact prefactor from the literature. |
| Arithmetic random walk, no drift | Fine intraday; over days, drift and mean reversion matter. Negative prices possible (documented). No fat tails, no jumps — variance understates tail risk. |
| Static AC trajectory | The 2000 solution is deterministic (pre-committed). With linear impact and no signals, the static path is also dynamically optimal (Section 3.12) — but real algos adapt to fills, news, and volume surprises; the `execution` module measures what re-planning gains from a drift signal or partial fills. |
| Poisson arrivals `A e^{-k delta}` (A-S) | No adverse selection: fills are independent of future price moves. Real fills are informative — makers add alpha signals and quote-fade protection. |
| Unit fill size, one venue, continuous quoting | Real fills are lumpy, markets fragmented, quoting throttled by messaging limits and tick sizes. |
| Exponential utility, terminal-wealth horizon | Gives clean closed forms; real desks think in drawdown limits, VaR budgets and overnight caps rather than gamma. |
| Frictionless quote updates | Every A-S quote reprice is free; in reality cancels/replacements lose queue position (Section 11). |

---

## 8. Practical market conventions: equity vs FX

The three bundled parameter sets below are calibrated caricatures in model
units; `data/presets.json` states eight instruments in market units instead
(Section 3.8), three of them fitted to recorded data (Section 3.10):

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
  `100 × 0.004² × 1 = 0.0016`, i.e. 16 pips of skew per unit of inventory)
  rather than raw numbers.
  Also FX conventions: two-way prices are quoted as bid/ask outright, sizes
  in base currency, and "spread" is measured in pips = 1e-4.

Equity conventions worth knowing: order sizes in shares with round lots,
prices on a tick grid (\$0.01), volume follows a U-shaped intraday curve
(open/close auctions heaviest), short-sale rules constrain the sell side.
Only the volume curve is modelled here (as liquidity weights in the
power-law model of Section 3.9, with an illustrative pinned curve and one
fitted in Section 3.10); lots,
the tick grid and short-sale rules are not. The models live in a
friction-free mid-price world, which is exactly why the core of them is
teachable in closed form.

---

## 9. Common pitfalls and numerical issues

1. **The 0/0 at small kappa.** `sinh(kappa(T-t))/sinh(kappa T)` is
   numerically fine down to surprisingly small kappa, but at
   `lambda_ra = 0` exactly it is 0/0. The pinned fix: branch to the exact
   linear TWAP when `kappa*T < 1e-6`. Do **not** branch on
   `lambda_ra == 0` only — `sigma == 0` also gives kappa = 0.
2. **Mixing the continuous and discrete kappa.** The paper's hyperbolic
   closed forms for E and Var are *exact* — but only with the discrete
   `kappa` from `cosh(kappa tau)`; plugging in the continuous
   `kappa_tilde = sqrt(lambda sigma^2/eta_tilde)` instead is off well above
   1e-10 at N = 5. All golden values use the pinned discrete sums, and the
   suites assert the closed forms against them at 1e-12.
   Two related traps: `sinh(kappa(T-t))/sinh(kappa T)` overflows to
   `inf/inf = NaN` for `kappa T > 710` (use the pinned `exp/expm1` form),
   and `acosh(1 + x)` for `kappa` loses `eps/x` digits when `tau` is small
   (use the pinned `log1p` form) — the first bites on very urgent or very long schedules, the second on fine grids.
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

## 10. How a desk would use this

A concrete workflow for the pieces that are here, in the order a desk would
run them:

1. **Calibrate impact.** From a history of child-order fills, compute
   per-share cost versus participation rate and fit `estimate_impact` (the
   OLS in `analytics`). The slope is a first estimate of the temporary
   impact coefficient; the intercept absorbs half-spread and fees (the
   paper's `epsilon`, which the AC cost functions deliberately omit — add
   `epsilon*X` back when reporting total cost). Check `r_squared` and the
   size range of the fills before trusting the slope for a bigger order.
2. **Choose the grid and units.** Pick `tau` (the rebalancing interval), set
   `sigma` as an *arithmetic* volatility in currency per share per
   `sqrt(tau unit)`, and make sure `gamma_p * tau < 2 * eta` — the
   constructor rejects the parameter set otherwise. `N = T/tau` is bounded
   at 10 million; a one-second grid over a day is `N = 23,400`, which the
   trajectory handles without overflow at any urgency.
3. **Sweep the frontier.** `efficient_frontier` over a `lambda_ra` grid gives
   `(E, sqrt(Var), kappa)` per point; `half_life` (= `1/kappa`) tells you
   whether the nominal horizon is even relevant (`1/kappa << T` means you
   are effectively liquidating fast). The best-execution committee sees this
   table: it documents *why* an order was traded at a given urgency.
4. **Pick a schedule, compare baselines.** `ac_trajectory` /
   `ac_trade_list` for the chosen `lambda_ra`; `twap_holdings` and
   `pov_holdings` for the benchmarks; `expected_cost` / `variance_cost`
   price any of them on the same footing (`X` is taken from the schedule).
   Round the fractional `n_j` to lots downstream — the library does not.
5. **Sanity-check with Monte Carlo.** `simulate_is` on the chosen schedule
   must land within a few SE of the closed forms; if it does not, the
   parameters are outside the model (typically `eta_tilde` near zero).
6. **After the fact: TCA.** `decompose_shortfall` with the decision price,
   arrival price, fills and close splits the realised shortfall into delay,
   trading and opportunity cost — the identity is exact, so any mismatch
   with a vendor's number is a convention difference, not rounding.
   `participation_rate` (market volume *including* own fills, so never above
   1) labels the fills for the next calibration round.
7. **Pre-trade, with concave impact.** `pretrade_report` in `market` costs
   TWAP, VWAP and the optimal schedule per `lambda_ra` under square-root
   impact and a volume curve, next to the closed-form square-root-law
   estimate; `drift_bps_per_day` prices the information behind the order
   (Section 3.9). Treat `Y` and the curve as assumptions until they are
   fitted to your own fills.
8. **Market-making side.** `quotes(params, s, q, t)` gives the reservation
   price and skewed bid/ask for the live state; `simulate` with the A-S and
   fixed agents on the same seed quantifies how much inventory variance the
   skew removes, and the `stress_factor` run shows what a volatility regime
   change does to the PnL distribution before it happens.

Everything above is deterministic per seed and validated at the entry
points, so it can sit inside a larger system; what the library does *not*
do — calendars, lots, ticks, venues, adaptive re-optimisation, adverse
selection — is listed in the README's real-world usage notes.

---

## 11. How real e-trading desks differ

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

## 12. Regulatory context

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

## 13. Knowledge check

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
`X P_d − sum n_j p_j − R P_T`. Follow-up: a desk with great
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

**Q11. An algo beat the day's VWAP by 2 bps on a sell order that finished
by noon in a falling market. Is that good execution?** Not necessarily.
The day's VWAP includes the afternoon's lower prices, which the order never
faced, so the 2 bps may be the market's drift. Compare against the interval
VWAP over the bins it traded, and against arrival (Section 4.2; the recipe
15 example beats the day's VWAP by 1.98 bps and loses 0.86 bps to its
interval VWAP).

**Q12. Your sell fills show positive markouts: the mid keeps falling after
each fill. Should you be pleased?** Only partly. It means you sold ahead of
a fall, which is what an informed seller wants, but on an order that is a
noticeable share of volume some of the fall is your own impact still
working through (Section 4.3 and 3.11). Compare markouts on fills of
different sizes: impact grows with size, information does not.

**Q13. Algo A averaged 1 bp less than B over 200 orders each, with a
per-order standard deviation of 10 bps. Switch?** The standard error of the
difference is `10 sqrt(2/200) = 1` bp, so `t` is about 1 and the
difference is noise. Detecting 1 bp at 80% power needs about 1,570 orders
per algo (Section 4.4), with orders randomised between the two.

**Q14. Why can a realised volatility from one day not be trusted to two
digits?** Even in the ideal i.i.d. case the relative standard error with
390 one-minute returns is `1/sqrt(780) = 3.6%`, and 12.9% for a half-hour
bin of 30 returns; fat tails make it worse, and one day says nothing about
how volatility varies between days (Section 3.10).

**Q15. In the transient-impact model with no permanent impact, why does the
optimum trade blocks at the start and end?** The first trade meets an
untouched book; the last trade's push is never paid for by a later trade.
Every middle trade pays for the decaying push of its predecessors, so the
middle runs at a constant, smaller rate, `1 - a` of the end blocks
(Section 3.11).

**Q16. You re-plan an Almgren-Chriss schedule every interval with no signal
and full fills. What do you gain?** Nothing, on every path: the optimum is
time-consistent and the re-plan reproduces the remaining plan (Section
3.12; the simulation saving is exactly 0). Gains come from information the
plan lacked, such as a drift signal or partial fills.

**Q17. Crossing a 150,000-share buy against a 150,000-share sell saves more
than twice the cost of a 75,000-share order. Why?** Under the square-root
law the total cost of an order grows like `Q^(3/2)`; avoiding two
150,000-share orders saves `2 C(150,000)`, about `2 * 2^(3/2) = 5.7` times
`C(75,000)`; the model with the US large-cap inputs gives 5.67 (Section
3.13).

---

## 14. Further reading

- R. Almgren, N. Chriss (2000), "Optimal execution of portfolio
  transactions", *Journal of Risk* 3(2), 5–39 — the AC model, exactly as
  pinned here (with the fixed-cost `epsilon` term omitted).
- M. Avellaneda, S. Stoikov (2008), "High-frequency trading in a limit order
  book", *Quantitative Finance* 8(3), 217–224 — reservation price and
  spread.
- A. F. Perold (1988), "The implementation shortfall: paper versus reality",
  *Journal of Portfolio Management* 14(3), 4–9 — the decomposition of
  Section 4; W. H. Wagner, M. Edwards (1993), "Best execution", *Financial
  Analysts Journal* 49(1), 65–71 — the delay/trading/opportunity naming.
- R. Kissell, *The Science of Algorithmic Trading and Portfolio
  Management* — practitioner treatment of benchmarks, POV/VWAP, TCA.
- L. Harris, *Trading and Exchanges* — the microstructure classic; order
  types, dealers, adverse selection.
- Cartea, Jaimungal, Penalva, *Algorithmic and High-Frequency Trading* —
  rigorous stochastic-control treatment generalizing both models here.
- O. Guéant, C.-A. Lehalle, J. Fernandez-Tapia (2013), "Dealing with the
  inventory risk: a solution to the market making problem", *Mathematics
  and Financial Economics* 7(4), 477–507 — the A-S model solved with
  inventory limits (closed forms for the capped case).
- J.-P. Bouchaud, J. Bonart, J. Donier, M. Gould (2018), *Trades, Quotes and
  Prices*, Cambridge University Press: empirical impact, the square-root law,
  order-book dynamics.
- R. Almgren, C. Thum, E. Hauptmann, H. Li (2005), "Direct estimation of
  equity market impact", *Risk* 18(7), 58–62: impact exponents fitted to
  institutional orders (the source of the square-root regularity cited in
  Section 3.9).
- B. Tóth, Y. Lempérière, C. Deremble, J. de Lataillade, J. Kockelkoren,
  J.-P. Bouchaud (2011), "Anomalous price impact and the critical nature of
  liquidity in financial markets", *Physical Review X* 1, 021006.
- ESMA MiFID II best-execution guidance and SEC Rules 605/606 — the
  regulatory texts behind Section 12.
- A. A. Obizhaeva, J. Wang (2013), "Optimal trading strategy and
  supply/demand dynamics", *Journal of Financial Markets* 16(1), 1–32: the
  transient-impact model of Section 3.11.
- A. Alfonsi, A. Fruth, A. Schied (2010), "Optimal execution strategies in
  limit order books with general shape functions", *Quantitative Finance*
  10(2), 143–157: the discrete-time solution with its end blocks.
- J. Gatheral (2010), "No-dynamic-arbitrage and market impact",
  *Quantitative Finance* 10(7), 749–759: which decay kernels and impact
  shapes are consistent.
- R. Almgren, J. Lorenz (2007), "Adaptive arrival price", *Trading*
  2007(1), 59–66: the non-time-consistent adaptive gain of Section 3.12.
- T. G. Andersen, T. Bollerslev, F. X. Diebold, P. Labys (2001), "The
  distribution of realized exchange rate volatility", *Journal of the
  American Statistical Association* 96(453), 42–55: realised variance
  (Section 3.10).
- S. A. Berkowitz, D. E. Logue, E. A. Noser (1988), "The total cost of
  transactions on the NYSE", *Journal of Finance* 43(1), 97–112: VWAP as an
  execution benchmark (Section 4.2).
- L. R. Glosten, P. R. Milgrom (1985), "Bid, ask and transaction prices in
  a specialist market with heterogeneously informed traders", *Journal of
  Financial Economics* 14(1), 71–100: adverse selection (Section 4.3).
- B. L. Welch (1947), "The generalization of 'Student's' problem when
  several different population variances are involved", *Biometrika*
  34(1–2), 28–35; F. E. Satterthwaite (1946), *Biometrics Bulletin* 2(6),
  110–114: the test and degrees of freedom of Section 4.4.
- W. H. Press, S. A. Teukolsky, W. T. Vetterling, B. P. Flannery,
  *Numerical Recipes*, 3rd ed., Sections 6.1 and 6.4: the incomplete beta
  function behind the pinned t CDF.

---

## 15. Glossary

| term | meaning | where |
|---|---|---|
| ADV | average daily volume, shares per day; the fitted presets use one day's NASDAQ volume | 3.8, 3.10 |
| arrival price | the mid when the order reaches the desk; the benchmark of implementation shortfall | 4.1, 4.2 |
| basis point (bp) | one hundredth of a percent of notional; `bps(x, N) = 1e4 x / N` | 3.8 |
| child order | one slice of a parent order, sent in one interval | 3.7 |
| efficient frontier | the set of schedules with the least expected cost for each variance | 3.5 |
| fill intensity | the rate `A exp(-k delta)` at which a quote `delta` from the mid is hit | 6.1, 3.10 |
| half-life | the time `ln 2 / kappa` over which an optimal schedule halves its holding | 3.4 |
| implementation shortfall (IS) | paper-portfolio value minus realised value; delay + trading + opportunity | 4.1 |
| internalisation | crossing opposite client orders at the mid and sending only the net | 3.13 |
| interval VWAP | market VWAP over only the bins in which the order traded | 4.2 |
| kappa (Almgren-Chriss) | urgency of the optimal schedule, from `cosh(kappa tau)` | 3.3, 3.4 |
| kappa (transient) | price displacement per share traded, the inverse of book depth | 3.11 |
| markout | the mid's move after a fill, signed so that negative is adverse | 4.3 |
| participation | the order's share of market volume, per bin or overall | 3.7, 4.2 |
| permanent impact | the part of a trade's price move that never decays (`gamma_p`) | 1.2 |
| POV | percentage-of-volume schedule | 3.7 |
| realised variance | the sum of squared intraday log returns | 3.10 |
| reservation price | the market maker's indifference price given its inventory | 6.2 |
| resilience | the rate `rho` at which the book refills after a trade | 3.11 |
| square-root law | cost per share grows like the square root of order size over ADV | 3.9 |
| temporary impact | the part of a trade's price move that lasts only while trading (`eta`) | 1.2 |
| time consistency | a plan stays optimal when re-solved later from the state it reaches | 3.12 |
| TWAP | equal slices over time | 3.7 |
| VWAP | volume-weighted average price, or the schedule that trades in proportion to volume | 3.7, 4.2 |
| Welch's t-test | a two-sample test of means that does not assume equal variances | 4.4 |

