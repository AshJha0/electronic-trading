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
/// dt each quoted side fills (unit size) with the exact thinned probability
/// `P(fill) = 1 - exp(-lambda(delta)*dt)` (pinned; the paper's own simulation
/// uses the first-order `lambda*dt`, which over-fills by ~12% at
/// `lambda*dt ~ 0.28` — hence the mean PnL below the paper's Table 1).
/// delta is signed: a quote pushed through the mid has delta < 0 and a
/// higher intensity — not an error.
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
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace etrade {

/// Upper bound on the number of simulation steps T/dt (pinned, API_SPEC 0).
inline constexpr int kMaxAsSteps = 10'000'000;

/// Upper bound on Monte Carlo path counts (pinned, API_SPEC 0).
inline constexpr int kMaxAsPaths = 10'000'000;

/// Avellaneda-Stoikov model parameters.
///
/// The constructor validates every field with std::invalid_argument,
/// including `T/dt` integral (tolerance 1e-9).
struct ASParams {
    const double s0;       ///< Initial mid price (any real; arithmetic model).
    const double sigma;    ///< Arithmetic mid vol per sqrt(time unit), >= 0.
    const double horizon;  ///< Terminal time T > 0.
    const double dt;       ///< Simulation step, 0 < dt <= T, T/dt integral and <= kMaxAsSteps.
    const double gamma;    ///< Risk aversion, strictly > 0.
    const double k;        ///< Intensity decay, > 0.
    const double a_rate;   ///< Intensity scale A, > 0.
    const int q_max;       ///< Hard inventory cap (units), >= 0 (0 quotes nothing).
    const double eta_liq;  ///< Terminal liquidation penalty per unit, >= 0.

    ASParams(double s0, double sigma, double horizon, double dt, double gamma,
             double k, double a_rate, int q_max, double eta_liq);

    /// Number of simulation steps T/dt.
    [[nodiscard]] int n_steps() const;
};

/// Reservation price r(s, q, t) = s - q*gamma*sigma^2*(T - t).
/// Requires finite s and q, and t in [0, T].
[[nodiscard]] double reservation_price(const ASParams& params, double s, double q, double t);

/// Total optimal spread gamma*sigma^2*(T - t) + (2/gamma)*ln(1 + gamma/k):
/// strictly positive, decreasing linearly in t to the terminal floor.
/// Requires t in [0, T].
[[nodiscard]] double optimal_spread(const ASParams& params, double t);

/// (bid, ask) = reservation price -/+ half the optimal spread.  Requires
/// finite s and q, and t in [0, T].
[[nodiscard]] std::pair<double, double> quotes(const ASParams& params, double s, double q, double t);

/// Arrival intensity lambda(delta) = A*exp(-k*delta); delta signed, finite.
[[nodiscard]] double intensity(const ASParams& params, double delta);

/// Per-step fill probability 1 - exp(-lambda(delta)*dt), in [0, 1).
[[nodiscard]] double fill_probability(const ASParams& params, double delta);

/// Time average over [0, T] of the A-S spread (the fixed agent's default):
/// gamma*sigma^2*T/2 + (2/gamma)*ln(1 + gamma/k).
[[nodiscard]] double average_as_spread(const ASParams& params);

/// Quoting agent for the simulation.
enum class Agent {
    kAS,     ///< A-S reservation price +/- half the optimal spread.
    kFixed,  ///< Mid +/- half a fixed spread, no inventory skew.
};

/// Aggregate statistics of ASSimResult with a standard error.
enum class ASStat {
    kMeanPnl,      ///< mean_pnl()
    kStdPnl,       ///< std_pnl()
    kMeanAbsQ,     ///< mean_abs_q()
    kMeanMaxAbsQ,  ///< mean_max_abs_q()
    kRmsQ,         ///< rms_q()
    kMeanFills,    ///< mean_fills()
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
    /// Total number of fills per path, both sides counted (a step where the
    /// bid and the ask both fill contributes two).
    std::vector<double> fills_path;
    double overall_max_abs_q = 0.0;           ///< max over paths of max |q|.
    bool cap_violated = false;                ///< Cap assertion flag (never true).

    /// Number of simulated paths M.
    [[nodiscard]] int n_paths() const noexcept { return static_cast<int>(pnl.size()); }

    [[nodiscard]] double mean_pnl() const;        ///< Mean terminal PnL.
    [[nodiscard]] double std_pnl() const;         ///< Sample std of PnL (ddof = 1).
    [[nodiscard]] double sharpe() const;          ///< mean/std of PnL; +inf when std == 0.
    [[nodiscard]] double mean_abs_q() const;      ///< Mean over paths of time-avg |q|.
    [[nodiscard]] double mean_max_abs_q() const;  ///< Mean over paths of max |q|.
    [[nodiscard]] double rms_q() const;           ///< Mean over paths of per-path rms(q).
    [[nodiscard]] double mean_fills() const;      ///< Mean total fills per path (both sides).

    /// Standard error of an aggregate statistic.  The aggregates are
    /// undefined for fewer than two paths (std::invalid_argument).
    [[nodiscard]] double se(ASStat stat) const;

    /// String form of se(ASStat): one of "mean_pnl", "std_pnl", "mean_abs_q",
    /// "mean_max_abs_q", "rms_q", "mean_fills"; std::invalid_argument for an
    /// unknown name.
    [[deprecated("use se(ASStat)")]] [[nodiscard]] double se(const std::string& stat) const;
};

/// Former NaN sentinel selecting `average_as_spread` as the fixed agent's
/// spread; pass std::nullopt instead.  A NaN fixed_spread is still accepted
/// as "default" for one release.
[[deprecated("pass std::nullopt for the default fixed spread")]] extern const double
    kDefaultFixedSpread;

/// Simulate the market maker over `n_paths` seeded paths (std::mt19937_64
/// with std::uniform_real_distribution / std::normal_distribution; own
/// stream — statistical agreement with the Python reference, not
/// bit-for-bit, and deterministic per seed within one standard-library
/// implementation since those distributions are implementation-defined).
/// Loop and draw order pinned in the file docs / API_SPEC.
///
/// \param n_paths number of paths, in [2, kMaxAsPaths].
/// \param fixed_spread spread of the kFixed agent; empty (the default)
///        selects `average_as_spread(params)`; a non-finite or <= 0 value is
///        an error, except NaN which is still read as empty (deprecated).
/// \param stress_factor multiplies the realised mid vol (> 0) for
///        `t >= stress_time_frac*T`; the agent keeps the base sigma.
[[nodiscard]] ASSimResult simulate(const ASParams& params, int n_paths, std::uint64_t seed,
                     Agent agent = Agent::kAS,
                     std::optional<double> fixed_spread = std::nullopt,
                     double stress_factor = 1.0, double stress_time_frac = 0.5);

}  // namespace etrade
