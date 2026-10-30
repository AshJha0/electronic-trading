#include "etrade/analytics.hpp"
#include "etrade/detail/require.hpp"

#include <cmath>
#include <stdexcept>

namespace etrade {

namespace {

using detail::require;

}  // namespace

ISDecomposition decompose_shortfall(Side side, double decision_price,
                                    double arrival_price, double close_price,
                                    double total_qty,
                                    const std::vector<double>& fill_qtys,
                                    const std::vector<double>& fill_prices) {
    require(std::isfinite(decision_price), "decision_price must be finite");
    require(std::isfinite(arrival_price), "arrival_price must be finite");
    require(std::isfinite(close_price), "close_price must be finite");
    require(std::isfinite(total_qty) && total_qty >= 0.0,
            "total_qty must be >= 0");
    require(fill_qtys.size() == fill_prices.size(),
            "fill_qtys and fill_prices must have equal length");
    double executed = 0.0;
    for (std::size_t j = 0; j < fill_qtys.size(); ++j) {
        require(std::isfinite(fill_qtys[j]) && fill_qtys[j] >= 0.0,
                "fill quantities must be >= 0");
        require(std::isfinite(fill_prices[j]), "fills must be finite");
        executed += fill_qtys[j];
    }
    require(executed <= total_qty * (1.0 + 1e-12) + 1e-12,
            "executed quantity exceeds total_qty");
    const double remaining = total_qty - executed;
    const double sign = side == Side::kSell ? 1.0 : -1.0;
    const double delay = executed * sign * (decision_price - arrival_price);
    double trading = 0.0;
    for (std::size_t j = 0; j < fill_qtys.size(); ++j) {
        trading += fill_qtys[j] * sign * (arrival_price - fill_prices[j]);
    }
    const double opportunity = remaining * sign * (decision_price - close_price);
    const double total = delay + trading + opportunity;
    return {total, delay, trading, opportunity, executed, remaining};
}

double participation_rate(double executed_volume, double market_volume) {
    require(std::isfinite(executed_volume) && executed_volume >= 0.0,
            "executed_volume must be >= 0");
    require(std::isfinite(market_volume) && market_volume > 0.0,
            "market_volume must be > 0");
    require(executed_volume <= market_volume,
            "executed volume exceeds market volume - market volume must include own fills");
    return executed_volume / market_volume;
}

OLSResult ols(const std::vector<double>& x, const std::vector<double>& y) {
    require(x.size() == y.size(), "x and y must have equal length");
    require(x.size() >= 2, "need at least 2 observations");
    const std::size_t n = x.size();
    double xm = 0.0, ym = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        require(std::isfinite(x[i]) && std::isfinite(y[i]),
                "x and y must be finite");
        xm += x[i];
        ym += y[i];
    }
    xm /= static_cast<double>(n);
    ym /= static_cast<double>(n);
    double sxx = 0.0, sxy = 0.0, sst = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sxx += (x[i] - xm) * (x[i] - xm);
        sxy += (x[i] - xm) * (y[i] - ym);
        sst += (y[i] - ym) * (y[i] - ym);
    }
    require(sxx > 0.0, "x is degenerate (zero variance)");
    const double slope = sxy / sxx;
    const double intercept = ym - slope * xm;
    double ssr = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double resid = y[i] - (intercept + slope * x[i]);
        ssr += resid * resid;
    }
    const double r2 = sst == 0.0 ? 1.0 : 1.0 - ssr / sst;
    return {slope, intercept, r2, static_cast<int>(n)};
}

std::pair<std::vector<double>, std::vector<double>> synthetic_impact_fills(
    int n_obs, double intercept, double slope, double noise_amp) {
    require(n_obs >= 2, "n_obs must be an integer >= 2");
    std::vector<double> part(static_cast<std::size_t>(n_obs));
    std::vector<double> cost(static_cast<std::size_t>(n_obs));
    for (int i = 0; i < n_obs; ++i) {
        part[i] = 0.02 + 0.18 * i / (n_obs - 1);
        cost[i] = intercept + slope * part[i] + noise_amp * std::sin(2.5 * i + 0.3);
    }
    return {part, cost};
}

OLSResult estimate_impact(const std::vector<double>& participations,
                          const std::vector<double>& costs) {
    return ols(participations, costs);
}

}  // namespace etrade
