//! Error type shared by all fallible `etrade` operations.

use std::fmt;

/// Error returned by `etrade` functions on invalid input.
///
/// No function in this crate panics on bad input; every documented domain
/// violation surfaces as an `Err(EtradeError)` (see API_SPEC.md Section 6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EtradeError {
    /// A parameter violated its documented domain (wrong sign, non-finite,
    /// inconsistent array lengths, ...).
    InvalidInput(String),
    /// The Almgren-Chriss model-validity condition failed:
    /// `gamma_p * tau >= 2 * eta` (`eta_tilde <= 0`, paper footnote).
    ModelInvalid(String),
    /// A numerical solver failed to find a certified optimum (API_SPEC 8.2).
    SolverFailed(String),
}

impl fmt::Display for EtradeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EtradeError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            EtradeError::ModelInvalid(msg) => write!(f, "model invalid: {msg}"),
            EtradeError::SolverFailed(msg) => write!(f, "solver failed: {msg}"),
        }
    }
}

impl std::error::Error for EtradeError {}

/// Internal helper: return `Err(EtradeError::InvalidInput(msg))` unless `cond`.
pub(crate) fn require(cond: bool, msg: &str) -> Result<(), EtradeError> {
    if cond {
        Ok(())
    } else {
        Err(EtradeError::InvalidInput(msg.to_string()))
    }
}
