/// \file detail/require.hpp
/// \brief Internal input-validation helpers shared by every translation unit.
///
/// Not part of the public API: include only from `cpp/src`.
#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

namespace etrade::detail {

/// Throw std::invalid_argument unless `cond` holds.
inline void require(bool cond, const char* msg) {
    if (!cond) throw std::invalid_argument(msg);
}

/// Throw std::invalid_argument unless `cond` holds (built message).
inline void require(bool cond, const std::string& msg) {
    if (!cond) throw std::invalid_argument(msg);
}

/// Throw std::invalid_argument unless `v` is finite and > 0.
inline void require_positive(const char* name, double v) {
    if (!(std::isfinite(v) && v > 0.0)) {
        throw std::invalid_argument(std::string(name) + " must be finite and > 0");
    }
}

}  // namespace etrade::detail
