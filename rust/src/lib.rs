//! `etrade` — electronic trading models: Almgren-Chriss optimal execution,
//! Avellaneda-Stoikov market making, and execution analytics.
//!
//! Rust port of the validated Python reference; see `API_SPEC.md` at the
//! project root for the pinned cross-language contract (exact discrete
//! formulas, simulation loop order, error behavior and golden tolerances).
//!
//! Modules:
//! * [`almgren_chriss`] — discrete optimal execution: sinh trajectory with
//!   exact small-kappa TWAP branch, exact discrete E/Var sums, efficient
//!   frontier, TWAP/POV/instantaneous baselines, Monte Carlo IS check;
//! * [`avellaneda_stoikov`] — reservation price, optimal spread, thinned
//!   Poisson fills, seeded market-making simulation with inventory cap,
//!   stress scenario and fixed-spread comparison agent;
//! * [`analytics`] — implementation-shortfall decomposition, participation
//!   rate, native OLS market-impact regression.
//!
//! All fallible operations return [`Result`]`<_, `[`EtradeError`]`>`; no
//! function panics on bad input.  Prices are arithmetic and may go negative —
//! a documented model property of both models, never an error.

pub mod almgren_chriss;
pub mod analytics;
pub mod avellaneda_stoikov;
mod error;

pub use almgren_chriss::{
    ac_kappa, ac_trade_list, ac_trajectory, efficient_frontier, expected_cost, half_life,
    instantaneous_cost, pov_holdings, simulate_is, trades_from_holdings, twap_holdings,
    variance_cost, ACParams, FrontierPoint, ISSimResult, KAPPA_T_LIMIT,
};
pub use analytics::{
    decompose_shortfall, estimate_impact, ols, participation_rate, synthetic_impact_fills,
    ISDecomposition, OLSResult, Side,
};
pub use avellaneda_stoikov::{
    average_as_spread, fill_probability, intensity, optimal_spread, quotes, reservation_price,
    simulate, ASParams, ASSimResult, ASStat, Agent, SimConfig,
};
pub use error::EtradeError;
