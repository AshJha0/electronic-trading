/// \file test_trajectory_golden.cpp
/// \brief Whole-trajectory goldens (schema v2, API_SPEC 5.1): every indexed
/// key of every case of data/golden/trajectory_golden.json.
#ifndef ETRADE_DATA_DIR
#define ETRADE_DATA_DIR "../data"
#endif

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "etrade/almgren_chriss.hpp"
#include "etrade/impact.hpp"
#include "etrade/market.hpp"
#include "mini_json.hpp"

namespace {

using mini_json::Value;

const Value& traj_golden() {
    static const Value doc =
        mini_json::parse_file(std::string(ETRADE_DATA_DIR) + "/golden/trajectory_golden.json");
    return doc;
}

double in(const Value& c, const std::string& key) { return c.at("inputs").at(key).number; }

void check(const Value& c, const std::string& key, double actual) {
    const double expected = c.at("expect").at(key).number;
    const double tol = c.at("tol").number;
    EXPECT_LE(std::abs(actual - expected), tol * std::max(1.0, std::abs(expected)))
        << c.at("name").str << ": key '" << key << "' actual " << actual << " expected "
        << expected;
}

void check_vector(const Value& c, const std::string& prefix, const std::vector<double>& v) {
    check(c, "len", static_cast<double>(v.size()));
    ASSERT_EQ(static_cast<double>(v.size()), c.at("expect").at("len").number) << c.at("name").str;
    for (std::size_t j = 0; j < v.size(); ++j) check(c, prefix + "_" + std::to_string(j), v[j]);
}

etrade::ACParams ac_params(const Value& c) {
    return etrade::ACParams(in(c, "ac_total_shares"), static_cast<int>(in(c, "ac_n_intervals")),
                            in(c, "ac_tau"), in(c, "ac_sigma"), in(c, "ac_eta"),
                            in(c, "ac_gamma_p"), in(c, "ac_lambda_ra"));
}

std::vector<double> impact_fracs(const Value& c) {
    std::vector<double> weights;
    if (in(c, "volume_curve") == 1.0) {
        weights.assign(etrade::kUsEquityVolumeCurve13.begin(), etrade::kUsEquityVolumeCurve13.end());
    }
    const etrade::ImpactMarket m{in(c, "price"),
                                 in(c, "spread"),
                                 in(c, "adv"),
                                 in(c, "daily_vol"),
                                 in(c, "total_shares"),
                                 static_cast<int>(in(c, "n_intervals")),
                                 in(c, "tau"),
                                 in(c, "lambda_ra"),
                                 in(c, "alpha"),
                                 in(c, "y"),
                                 etrade::kDefaultPermFrac,
                                 in(c, "drift_bps_per_day"),
                                 weights};
    const etrade::ImpactParams p = etrade::impact_params_from_market(m);
    std::vector<double> x = etrade::optimal_schedule(p).holdings;
    for (double& v : x) v /= p.total_shares();
    return x;
}

}  // namespace

TEST(TrajectoryGolden, SchemaAndCaseCount) {
    EXPECT_EQ(traj_golden().at("schema").number, 2.0);
    EXPECT_EQ(traj_golden().at("cases").arr.size(), 7u);
}

TEST(TrajectoryGolden, EveryCaseEveryKey) {
    int checked = 0;
    for (const Value& c : traj_golden().at("cases").arr) {
        const std::string& kind = c.at("kind").str;
        SCOPED_TRACE(c.at("name").str);
        if (kind == "ac_trajectory") {
            check_vector(c, "x", etrade::ac_trajectory(ac_params(c)));
        } else if (kind == "twap") {
            check_vector(c, "x", etrade::twap_holdings(in(c, "ac_total_shares"),
                                                       static_cast<int>(in(c, "ac_n_intervals"))));
        } else if (kind == "pov") {
            check_vector(c, "x",
                         etrade::pov_holdings(in(c, "ac_total_shares"), in(c, "ac_tau"),
                                              in(c, "pov_market_volume_rate"),
                                              in(c, "pov_participation")));
        } else if (kind == "frontier") {
            const int n = static_cast<int>(in(c, "n_lambdas"));
            std::vector<double> grid;
            for (int i = 0; i < n; ++i) grid.push_back(in(c, "lambda_" + std::to_string(i)));
            const auto fr = etrade::efficient_frontier(ac_params(c), grid);
            check(c, "len", static_cast<double>(fr.size()));
            for (std::size_t i = 0; i < fr.size(); ++i) {
                const std::string s = std::to_string(i);
                check(c, "kappa_" + s, fr[i].kappa);
                check(c, "e_cost_" + s, fr[i].e_cost);
                check(c, "std_cost_" + s, fr[i].std_cost);
            }
        } else if (kind == "impact_schedule") {
            check_vector(c, "xfrac", impact_fracs(c));
        } else {
            ADD_FAILURE() << "unknown kind " << kind;
        }
        // Every stored key was checked: len plus len values (x 3 for frontier).
        const double len = c.at("expect").at("len").number;
        const double per = kind == "frontier" ? 3.0 : 1.0;
        EXPECT_EQ(static_cast<double>(c.at("expect").obj.size()), 1.0 + per * len);
        ++checked;
    }
    EXPECT_EQ(checked, 7);
}
