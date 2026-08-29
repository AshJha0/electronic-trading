/// \file avellaneda_stoikov.hpp
/// \brief Avellaneda-Stoikov market making (2008 paper, finite horizon).
///
/// The mid price is an arithmetic Brownian motion `dS = sigma dW` (prices may
/// go negative — documented model property).  A maker with exponential
/// utility quotes around the *reservation price* — the mid value at which it
/// is indifferent to holding one more/fewer share:
///
///     r(s, q, t) = s - q*gamma*sigma^2*(T - t)
///
/// (a long maker shades down to attract buyers of its inventory), with the
/// optimal total spread
///
///     spread(t) = gamma*sigma^2*(T - t) + (2/gamma)*ln(1 + gamma/k),
///
/// quotes `ask = r + spread/2`, `bid = r - spread/2`.  The first term is
/// inventory-risk compensation (vanishes at T), the second the monopolist-
/// dealer markup (terminal floor).  `gamma > 0` strictly: the gamma -> 0
/// limit degenerates the utility derivation and is an input error per spec.
///
/// Market orders lift a quote at distance delta from the mid with Poisson
/// intensity `lambda(delta) = A*exp(-k*delta)`; per discrete step of length
/// dt each quoted side fills (unit size) with the thinned probability
/// `P(fill) = 1 - exp(-lambda(delta)*dt)`.  delta is signed: a quote pushed
/// through the mid has delta < 0 and a higher intensity — not an error.
///
/// Simulation loop order is pinned (see API_SPEC.md section 2.2): per step
/// quote from current state (inventory cap: bid quoted only if q < q_max,
/// ask only if q > -q_max), draw u_ask then u_bid (always, even for a
/// suppressed side), apply fills, then move the mid with z ~ N(0,1) using
/// `sigma*stress_factor` for `t >= stress_time_frac*T` (the agent keeps
/// quoting with the base sigma — a model-misspecification stress).
/// Terminal: `PnL = cash + q*S_T - eta_liq*|q|`.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace etrade {

/// Avellaneda-Stoikov model parameters.
///
/// The constructor validates every field with std::invalid_argument,
/// including `T/dt` integral (tolerance 1e-9).
struct ASParams {
    double s0;       ///< Initial mid price (any real; arithmetic model).
    double sigma;    ///< Arithmetic mid vol per sqrt(time unit), >= 0.
    double horizon;  ///< Terminal time T > 0.
    double dt;       ///< Simulation step, 0 < dt <= T, T/dt integral.
    double gamma;    ///< Risk aversion, strictly > 0.
    double k;        ///< Intensity decay, > 0.
    double a_rate;   ///< Intensity scale A, > 0.
    int q_max;       ///< Hard inventory cap (units), >= 0 (0 quotes nothing).
    double eta_liq;  ///< Terminal liquidation penalty per unit, >= 0.

    ASParams(double s0, double sigma, double horizon, double dt, double gamma,
             double k, double a_rate, int q_max, double eta_liq);

    /// Number of simulation steps T/dt.
    int n_steps() const;
};

/// Reservation price r(s, q, t) = s - q*gamma*sigma^2*(T - t).
/// Requires t in [0, T].
double reservation_price(const ASParams& params, double s, double q, double t);

/// Total optimal spread gamma*sigma^2*(T - t) + (2/gamma)*ln(1 + gamma/k):
/// strictly positive, decreasing linearly in t to the terminal floor.
/// Requires t in [0, T].
double optimal_spread(const ASParams& params, double t);

/// (bid, ask) = reservation price -/+ half the optimal spread.
std::pair<double, double> quotes(const ASParams& params, double s, double q, double t);

/// Arrival intensity lambda(delta) = A*exp(-k*delta); delta signed, finite.
double intensity(const ASParams& params, double delta);

/// Per-step fill probability 1 - exp(-lambda(delta)*dt), in [0, 1).
double fill_probability(const ASParams& params, double delta);

/// Time average over [0, T] of the A-S spread (the fixed agent's default):
/// gamma*sigma^2*T/2 + (2/gamma)*ln(1 + gamma/k).
double average_as_spread(const ASParams& params);

/// Quoting agent for the simulation.
enum class Agent {
    kAS,     ///< A-S reservation price +/- half the optimal spread.
    kFixed,  ///< Mid +/- half a fixed spread, no inventory skew.
};

/// Per-path arrays and aggregate statistics of a market-making simulation.
///
/// Aggregates are means/stds over per-path values; standard errors via
/// se(): SE of a per-path-mean statistic = sample std / sqrt(M);
/// SE(std_pnl) = std_pnl / sqrt(2(M-1)) (normal approximation).
struct ASSimResult {
    std::vector<double> pnl;              ///< Terminal PnL per path.
    std::vector<double> mean_abs_q_path;  ///< Time average of |q| per path.
    std::vector<double> max_abs_q_path;   ///< Path max of |q|.
    std::vector<double> rms_q_path;       ///< sqrt(time average of q^2).
    /// Fill count per path, pinned to the reference semantics: the number
    /// of steps with at least one fill (a both-sides step counts once).
    std::vector<double> fills_path;
    double overall_max_abs_q;             ///< max over paths of max |q|.
    bool cap_violated;                    ///< Cap assertion flag (never true).

    /// Number of simulated paths M.
    int n_paths() const { return static_cast<int>(pnl.size()); }

    double mean_pnl() const;        ///< Mean terminal PnL.
    double std_pnl() const;         ///< Sample std of PnL (ddof = 1).
    double sharpe() const;          ///< mean/std of PnL; +inf when std == 0.
    double mean_abs_q() const;      ///< Mean over paths of time-avg |q|.
    double mean_max_abs_q() const;  ///< Mean over paths of max |q|.
    double rms_q() const;           ///< Mean over paths of per-path rms(q).
    double mean_fills() const;      ///< Mean fill count per path (see fills_path).

    /// Standard error of an aggregate statistic: one of "mean_pnl",
    /// "std_pnl", "mean_abs_q", "mean_max_abs_q", "rms_q", "mean_fills".
    /// Throws std::invalid_argument for an unknown name.
    double se(const std::string& stat) const;
};

/// Sentinel (NaN) selecting `average_as_spread` as the fixed agent's spread.
extern const double kDefaultFixedSpread;

/// Simulate the market maker over `n_paths` seeded paths (std::mt19937_64;
/// own stream — statistical agreement with the Python reference, not
/// bit-for-bit).  Loop and draw order pinned in the file docs / API_SPEC.
///
/// \param fixed_spread spread of the kFixed agent; a value <= 0 is an
///        error; NaN (the default, `kDefaultFixedSpread`) selects
///        `average_as_spread(params)`.
/// \param stress_factor multiplies the realised mid vol (> 0) for
///        `t >= stress_time_frac*T`; the agent keeps the base sigma.
ASSimResult simulate(const ASParams& params, int n_paths, std::uint64_t seed,
                     Agent agent = Agent::kAS,
                     double fixed_spread = kDefaultFixedSpread,
                     double stress_factor = 1.0, double stress_time_frac = 0.5);

}  // namespace etrade
