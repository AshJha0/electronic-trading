"""Almgren-Chriss discrete-time optimal execution (Almgren & Chriss, 2000).

Model
-----
Sell ``X`` shares over ``N`` intervals of length ``tau`` (horizon ``T = N*tau``,
times ``t_j = j*tau``).  Holdings ``x_0 = X, ..., x_N = 0``; trade list
``n_j = x_{j-1} - x_j`` (shares sold in interval ``j``, ``j = 1..N``).

The mid price follows an arithmetic random walk with permanent impact,

    S_j = S_{j-1} + sigma * sqrt(tau) * xi_j - tau * g(n_j / tau),

with linear permanent impact ``g(v) = gamma_p * v`` (so the price drop from
trade ``j`` is ``gamma_p * n_j``).  The trade in interval ``j`` executes at the
effective price

    S~_j = S_{j-1} - h(n_j / tau),    h(v) = eta * v   (linear temporary impact).

Implementation shortfall (capture shortfall) of a schedule is

    IS = X * S_0 - sum_j n_j * S~_j.

The paper's fixed cost term ``epsilon * sum_j |n_j|`` (half-spread and fees)
is deliberately omitted: it is schedule-independent for a one-sided program
and does not change the optimal trajectory.  Add ``epsilon * X`` to E[IS]
yourself if you need it.

Discrete closed forms (the exact paper formulas)
------------------------------------------------
Taking expectations and collecting the permanent-impact cross terms
(``sum_j n_j * sum_{i<j} n_i = (X^2 - sum_j n_j^2) / 2``) gives the *exact
discrete* moments of IS for any schedule with ``x_N = 0``:

    E[IS]   = (1/2) * gamma_p * X^2 + (eta_tilde / tau) * sum_{j=1}^{N} n_j^2,
    Var[IS] = sigma^2 * tau * sum_{j=1}^{N} x_j^2,

where

    eta_tilde = eta * (1 - gamma_p * tau / (2 * eta)) = eta - gamma_p * tau / 2.

The model is well posed only for ``eta_tilde > 0``, i.e. ``gamma_p * tau <
2 * eta`` (paper footnote); otherwise a ``ValueError`` is raised.

The mean-variance objective ``U(x) = E[IS] + lambda_ra * Var[IS]`` is a
strictly convex quadratic in the interior holdings ``x_1..x_{N-1}``.  Setting
``dU/dx_j = 0`` yields the linear difference equation

    (x_{j-1} - 2*x_j + x_{j+1}) / tau^2 = kappa_tilde^2 * x_j,

with

    kappa_tilde^2 = lambda_ra * sigma^2 / eta_tilde.

Its solution with boundary conditions ``x_0 = X, x_N = 0`` is the sinh
trajectory

    x_j = X * sinh(kappa * (T - t_j)) / sinh(kappa * T),

where the *discrete* rate ``kappa`` solves

    (2 / tau^2) * (cosh(kappa * tau) - 1) = kappa_tilde^2,
    i.e.  kappa = arccosh(1 + kappa_tilde^2 * tau^2 / 2) / tau,

(``kappa -> kappa_tilde`` as ``tau -> 0``).  The associated trade list is

    n_j = (2 * X * sinh(kappa * tau / 2) / sinh(kappa * T))
          * cosh(kappa * (T - t_{j-1/2})),      t_{j-1/2} = (j - 1/2) * tau.

Substituting the optimal trajectory into the exact sums above gives E and Var
of the optimal strategy; this module always evaluates the moments through the
pinned discrete sums, which coincide with the paper's closed-form hyperbolic
expressions (evaluated with the discrete ``kappa``) to machine precision.
The sums are used because they also cover TWAP, POV and arbitrary schedules.

Numerical evaluation (pinned in API_SPEC.md)
--------------------------------------------
* ``kappa`` is evaluated as ``log1p(x + sqrt(x*(2 + x))) / tau`` with
  ``x = kappa_tilde^2 * tau^2 / 2`` — algebraically ``arccosh(1 + x)`` but
  free of cancellation for small ``x`` (the ``acosh(1 + x)`` form loses
  ``~eps/x`` relative digits by cancellation, e.g. for ``tau`` in seconds).
* The sinh ratio is evaluated in the overflow-free form

      x_j = X * exp(-kappa*t_j) * expm1(-2*kappa*(T - t_j)) / expm1(-2*kappa*T)

  which is exactly ``sinh(kappa*(T - t_j))/sinh(kappa*T)`` for every
  ``kappa > 0`` and never overflows (``sinh`` overflows at ``kappa*T > 710``,
  reached by very urgent or very long schedules; ``kappa*T`` is essentially
  independent of the grid at a fixed horizon).

Small-``kappa`` limit (``lambda_ra -> 0``)
------------------------------------------
As ``lambda_ra -> 0``, ``kappa -> 0`` and ``sinh(kappa*(T-t_j))/sinh(kappa*T)
-> (T - t_j)/T``: the trajectory degenerates to the linear (TWAP) schedule
``x_j = X * (N - j) / N``, which minimises pure expected cost.  To avoid
0/0 the implementation switches to the exact linear branch when
``kappa * T < KAPPA_T_LIMIT = 1e-6`` (this includes ``lambda_ra == 0`` and
``sigma == 0``).  The expm1 expression is numerically smooth down to the
threshold, so the branch is continuous to ~1e-12.

Other conventions: the "half-life" of the trade reported here is the paper's
characteristic time ``theta = 1 / kappa`` (an e-folding time, *not*
``ln 2 / kappa``; infinite for the TWAP limit).  Prices may go negative under
the arithmetic walk; this is a documented model property, not a bug.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

#: A holdings path: any 1-D float sequence or NumPy array.
Holdings = Sequence[float] | npt.NDArray[np.float64]

#: Threshold on kappa*T below which the exact linear (TWAP) limit branch is used.
KAPPA_T_LIMIT: float = 1e-6

#: Upper bound on the number of AC trading intervals N (pinned, API_SPEC §0).
MAX_AC_INTERVALS: int = 10_000_000

#: Upper bound on the number of POV intervals M = ceil(X/clip) (pinned, API_SPEC §0).
MAX_POV_INTERVALS: int = 10_000_000

#: Upper bound on Monte Carlo path counts (pinned, API_SPEC §0).
MAX_PATHS: int = 10_000_000


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _is_int(v: object) -> bool:
    """True for Python/NumPy integers, False for bool and floats."""
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


@dataclass(frozen=True)
class ACParams:
    """Parameters of the discrete Almgren-Chriss problem (sell ``X`` shares).

    Attributes
    ----------
    total_shares : X >= 0, shares to sell (X = 0 gives the empty trajectory).
    n_intervals  : N, integer with 1 <= N <= MAX_AC_INTERVALS.
    tau          : interval length (> 0).
    sigma        : arithmetic volatility per sqrt(time unit) (>= 0).
    eta          : linear temporary impact coefficient (> 0).
    gamma_p      : linear permanent impact coefficient (>= 0).
    lambda_ra    : risk aversion (>= 0); 0 gives the TWAP limit.

    Units are abstract but must be consistent: ``tau`` in time units,
    ``sigma`` in currency/share/sqrt(time unit), ``eta`` in
    currency*time/share^2, ``gamma_p`` in currency/share^2, ``lambda_ra`` in
    1/currency.
    """

    total_shares: float
    n_intervals: int
    tau: float
    sigma: float
    eta: float
    gamma_p: float
    lambda_ra: float

    def __post_init__(self) -> None:
        _require(
            math.isfinite(self.total_shares) and self.total_shares >= 0.0,
            "total_shares must be finite and >= 0",
        )
        _require(
            _is_int(self.n_intervals) and 1 <= self.n_intervals <= MAX_AC_INTERVALS,
            f"n_intervals must be an integer in [1, {MAX_AC_INTERVALS}]",
        )
        _require(math.isfinite(self.tau) and self.tau > 0.0, "tau must be > 0")
        _require(math.isfinite(self.sigma) and self.sigma >= 0.0, "sigma must be >= 0")
        _require(math.isfinite(self.eta) and self.eta > 0.0, "eta must be > 0")
        _require(math.isfinite(self.gamma_p) and self.gamma_p >= 0.0, "gamma_p must be >= 0")
        _require(math.isfinite(self.lambda_ra) and self.lambda_ra >= 0.0, "lambda_ra must be >= 0")
        # Paper validity condition: eta_tilde = eta - gamma_p*tau/2 must be > 0.
        _require(
            self.gamma_p * self.tau < 2.0 * self.eta,
            "model invalid: gamma_p*tau >= 2*eta (eta_tilde <= 0); "
            "see Almgren-Chriss (2000) footnote",
        )

    @property
    def horizon(self) -> float:
        """Total horizon T = N * tau."""
        return self.n_intervals * self.tau

    @property
    def eta_tilde(self) -> float:
        """Adjusted temporary impact eta_tilde = eta*(1 - gamma_p*tau/(2*eta))."""
        return self.eta * (1.0 - self.gamma_p * self.tau / (2.0 * self.eta))

    def with_lambda(self, lambda_ra: float) -> ACParams:
        """Copy of these params with a different risk aversion (revalidated)."""
        return ACParams(
            self.total_shares,
            self.n_intervals,
            self.tau,
            self.sigma,
            self.eta,
            self.gamma_p,
            lambda_ra,
        )


def ac_kappa(params: ACParams) -> float:
    """Discrete urgency parameter kappa.

    Solves (2/tau^2)*(cosh(kappa*tau) - 1) = kappa_tilde^2 with
    kappa_tilde^2 = lambda_ra*sigma^2/eta_tilde; returns 0.0 in the
    lambda_ra*sigma^2 = 0 limit.  Evaluated as
    ``log1p(x + sqrt(x*(2 + x)))/tau`` with ``x = kappa_tilde^2*tau^2/2``
    (identical to ``arccosh(1 + x)/tau`` but free of cancellation for small
    ``x``; pinned in API_SPEC.md Section 1.4).
    """
    kt2 = params.lambda_ra * params.sigma**2 / params.eta_tilde
    if kt2 == 0.0:
        return 0.0
    x = 0.5 * kt2 * params.tau**2
    return math.log1p(x + math.sqrt(x * (2.0 + x))) / params.tau


def half_life(params: ACParams) -> float:
    """Characteristic decay time theta = 1/kappa of the optimal trajectory.

    This is the paper's e-folding time (the trajectory decays like
    ``exp(-t/theta)``), *not* ``ln(2)/kappa``.  Returns ``inf`` in the TWAP
    limit (kappa == 0).  ``characteristic_time`` is an alias.
    """
    kappa = ac_kappa(params)
    return math.inf if kappa == 0.0 else 1.0 / kappa


def characteristic_time(params: ACParams) -> float:
    """Alias of :func:`half_life` with the paper's name (theta = 1/kappa)."""
    return half_life(params)


def ac_trajectory(params: ACParams) -> np.ndarray:
    """Optimal holdings path ``x_0..x_N`` (length N+1, x_0 = X, x_N = 0).

    Uses the sinh solution in the overflow-free ``exp/expm1`` ratio form;
    switches to the exact linear (TWAP) branch when ``kappa*T <
    KAPPA_T_LIMIT`` (covers lambda_ra = 0 and sigma = 0).  Every entry is
    finite for any valid parameters (no ``sinh`` overflow).
    """
    n, x0 = params.n_intervals, params.total_shares
    j = np.arange(n + 1, dtype=float)
    if x0 == 0.0:
        return np.zeros(n + 1)
    kappa = ac_kappa(params)
    t_j = j * params.tau
    big_t = params.horizon
    if kappa * big_t < KAPPA_T_LIMIT:
        x = x0 * (n - j) / n  # exact TWAP limit, no 0/0
    else:
        # sinh(k(T-t))/sinh(kT) = exp(-k t) * expm1(-2k(T-t)) / expm1(-2kT);
        # the ratio is formed first so that x_0 == X exactly (1 * 1 * X).
        ratio = np.expm1(-2.0 * kappa * (big_t - t_j)) / np.expm1(-2.0 * kappa * big_t)
        x = x0 * np.exp(-kappa * t_j) * ratio
    x[-1] = 0.0
    return x


def ac_trade_list(params: ACParams) -> np.ndarray:
    """Optimal trade list ``n_1..n_N`` (shares sold per interval, sums to X)."""
    return trades_from_holdings(ac_trajectory(params))


def _holdings_array(holdings: Holdings) -> np.ndarray:
    x = np.asarray(holdings, dtype=float)
    _require(x.ndim == 1 and x.size >= 2, "holdings must be a 1-D array, length >= 2")
    _require(bool(np.all(np.isfinite(x))), "holdings must be finite")
    return x


def _require_liquidating(x: np.ndarray) -> None:
    _require(
        abs(float(x[-1])) <= 1e-9 * max(1.0, abs(float(x[0]))),
        "holdings path must fully liquidate (x_N = 0)",
    )


def trades_from_holdings(holdings: Holdings) -> np.ndarray:
    """Trade list n_j = x_{j-1} - x_j from a (finite) holdings path."""
    return -np.diff(_holdings_array(holdings))


def twap_holdings(total_shares: float, n_intervals: int) -> np.ndarray:
    """Linear TWAP holdings path x_j = X*(N-j)/N."""
    _require(
        math.isfinite(total_shares) and total_shares >= 0.0, "total_shares must be finite and >= 0"
    )
    _require(
        _is_int(n_intervals) and 1 <= n_intervals <= MAX_AC_INTERVALS,
        f"n_intervals must be an integer in [1, {MAX_AC_INTERVALS}]",
    )
    j = np.arange(n_intervals + 1, dtype=float)
    return total_shares * (n_intervals - j) / n_intervals


def pov_holdings(
    total_shares: float, tau: float, market_volume_rate: float, participation: float
) -> np.ndarray:
    """POV (percentage-of-volume) holdings path.

    Sells ``participation * market_volume_rate * tau`` shares per interval
    until the parent order is done (last clip partial).  The path length is
    determined by the order size, not by a fixed N: ``M = ceil(X/clip -
    1e-12)`` intervals, with ``X/clip <= MAX_POV_INTERVALS`` required (a clip
    too small for the order is an input error, never a huge allocation).
    """
    _require(
        math.isfinite(total_shares) and total_shares >= 0.0, "total_shares must be finite and >= 0"
    )
    _require(math.isfinite(tau) and tau > 0.0, "tau must be > 0")
    _require(
        math.isfinite(market_volume_rate) and market_volume_rate > 0.0,
        "market_volume_rate must be > 0",
    )
    _require(
        math.isfinite(participation) and 0.0 < participation < 1.0,
        "participation must be in (0, 1)",
    )
    clip = participation * market_volume_rate * tau
    if total_shares == 0.0:
        return np.zeros(2)
    ratio = total_shares / clip
    _require(
        ratio <= MAX_POV_INTERVALS,
        f"POV clip too small: would need > {MAX_POV_INTERVALS} intervals",
    )
    n_int = math.ceil(ratio - 1e-12)
    x = total_shares - clip * np.arange(n_int + 1, dtype=float)
    x = np.maximum(x, 0.0)
    x[-1] = 0.0
    return x


def expected_cost(params: ACParams, holdings: Holdings | None = None) -> float:
    """E[IS] = 0.5*gamma_p*X^2 + (eta_tilde/tau)*sum n_j^2 (exact discrete sum).

    ``holdings`` defaults to the optimal trajectory; any fully-liquidating
    path (x_N = 0) may be supplied (TWAP, POV, ...).  The order size ``X``
    in the permanent-impact term is taken from the schedule (``x_0``), not
    from ``params.total_shares`` (pinned; API_SPEC.md Section 1.3).
    """
    x = ac_trajectory(params) if holdings is None else _holdings_array(holdings)
    _require_liquidating(x)
    n = -np.diff(x)
    x0 = float(x[0])
    return 0.5 * params.gamma_p * x0**2 + params.eta_tilde / params.tau * float(np.sum(n**2))


def variance_cost(params: ACParams, holdings: Holdings | None = None) -> float:
    """Var[IS] = sigma^2 * tau * sum_{j=1}^{N} x_j^2 (exact discrete sum).

    ``holdings`` (length >= 2, finite) defaults to the optimal trajectory.
    """
    x = ac_trajectory(params) if holdings is None else _holdings_array(holdings)
    return params.sigma**2 * params.tau * float(np.sum(x[1:] ** 2))


def instantaneous_cost(params: ACParams) -> tuple[float, float]:
    """(E, Var) of instantaneous liquidation: sell all X in the first interval.

    E = 0.5*gamma_p*X^2 + eta_tilde*X^2/tau,  Var = 0 (no residual exposure).
    """
    x0 = params.total_shares
    e = 0.5 * params.gamma_p * x0**2 + params.eta_tilde * x0**2 / params.tau
    return e, 0.0


@dataclass(frozen=True)
class FrontierPoint:
    """One point on the efficient frontier of execution."""

    lambda_ra: float
    kappa: float
    e_cost: float
    var_cost: float
    std_cost: float


def efficient_frontier(params: ACParams, lambdas: Sequence[float]) -> list[FrontierPoint]:
    """Sweep a risk-aversion grid; return (lambda, kappa, E, Var, sqrt(Var)).

    The grid must be non-empty and every entry finite and >= 0; order is
    preserved (duplicates and unsorted grids are allowed and reproduced).
    """
    _require(len(lambdas) > 0, "lambdas grid must be non-empty")
    out: list[FrontierPoint] = []
    for lam in lambdas:
        p = params.with_lambda(float(lam))
        e = expected_cost(p)
        v = variance_cost(p)
        out.append(FrontierPoint(float(lam), ac_kappa(p), e, v, math.sqrt(v)))
    return out


@dataclass(frozen=True)
class ISSimResult:
    """Monte Carlo implementation-shortfall summary (arithmetic-walk sim)."""

    mean: float
    var: float
    se_mean: float
    se_var: float
    n_paths: int


def simulate_is(
    params: ACParams, holdings: Holdings | None = None, n_paths: int = 10_000, seed: int = 0
) -> ISSimResult:
    """Simulate IS of a schedule under the arithmetic random walk.

    Per path: S_j = S_{j-1} + sigma*sqrt(tau)*xi_j - gamma_p*n_j, trade j
    executes at S_{j-1} - eta*n_j/tau, IS = X*S_0 - proceeds (independent of
    S_0, which is taken as 0).  The schedule must fully liquidate (same check
    as ``expected_cost``) so that the simulated IS is comparable to the
    closed forms.  Verifies the closed forms: E[IS] and Var[IS] should match
    `expected_cost` / `variance_cost` within a few standard errors.  SE of
    the sample variance uses the normal approximation ``var * sqrt(2/(M-1))``
    (exact here: IS is Gaussian in this model).
    """
    _require(
        _is_int(n_paths) and 2 <= n_paths <= MAX_PATHS,
        f"n_paths must be an integer in [2, {MAX_PATHS}]",
    )
    x = ac_trajectory(params) if holdings is None else _holdings_array(holdings)
    _require_liquidating(x)
    n = -np.diff(x)
    rng = np.random.default_rng(seed)
    n_int = n.size
    xi = rng.standard_normal((n_paths, n_int))
    s_prev = np.zeros(n_paths)
    proceeds = np.zeros(n_paths)
    srt = params.sigma * math.sqrt(params.tau)
    for j in range(n_int):
        exec_price = s_prev - params.eta * n[j] / params.tau
        proceeds += n[j] * exec_price
        s_prev = s_prev + srt * xi[:, j] - params.gamma_p * n[j]
    shortfall = x[0] * 0.0 - proceeds  # S_0 = 0
    mean = float(np.mean(shortfall))
    var = float(np.var(shortfall, ddof=1))
    se_mean = math.sqrt(var / n_paths)
    se_var = var * math.sqrt(2.0 / (n_paths - 1))
    return ISSimResult(mean, var, se_mean, se_var, n_paths)
