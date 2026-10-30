"""Write the market presets and the market-unit golden cases.

Run:  python data/generate_market.py

Outputs
  data/presets.json              named presets in market units
  data/golden/market_golden.json cross-language golden cases for API_SPEC s7

The five hand-written presets are illustrative, chosen inside typical public
ranges for each instrument type (one-tick spreads for liquid US names and
ETFs, wider for small caps, sub-bp for EUR/USD); the paper preset reproduces
the Almgren-Chriss (2000) example exactly.  The ``itch_*`` presets are
fitted to the NASDAQ TotalView-ITCH sample day 2019-12-30 from the committed
aggregate ``data/calibration/fitted_20191230.json`` (written by
``data/calibration/fit_itch.py``; API_SPEC Section 11.6 states the rules).
Every preset carries a ``source`` field.

Golden schema matches data/golden/golden.json: {"cases": [{name, kind,
inputs, expect, tol}]}, flat scalar keys.  kind is "market_ac" or
"market_as"; every port evaluates every case with one generic routine.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA_DIR.parent / "python" / "src"))

import etrade as et
from etrade import market as mk

TOL = 1e-12

PRESETS: dict[str, dict] = {
    "paper_almgren_chriss_2000": {
        "description": "Almgren and Chriss (2000) Section 4 example: price 50, spread 1/8, "
        "ADV 5M, 30% annual vol; reproduces params_liquid.json (AC) exactly.",
        "source": "Almgren and Chriss (2000), Section 4",
        "ac": {
            "price": 50.0,
            "spread": 0.125,
            "adv": 5.0e6,
            "daily_vol": 0.019,
            "total_shares": 1.0e6,
            "n_intervals": 5,
            "tau": 1.0,
            "lambda_ra": 2.0e-6,
        },
    },
    "us_large_cap": {
        "description": "Liquid US large cap: $100, one-tick (1 bp) spread, 1.5% daily vol, "
        "ADV 5M; sell 5% of ADV over one day in 30-minute bins.",
        "source": "illustrative, not fitted to data",
        "ac": {
            "price": 100.0,
            "spread": 0.01,
            "adv": 5.0e6,
            "daily_vol": 0.015,
            "total_shares": 250_000.0,
            "n_intervals": 13,
            "tau": 1.0 / 13.0,
            "lambda_ra": 1.0e-7,
        },
        "as": {
            "price": 100.0,
            "tick": 0.01,
            "daily_vol": 0.015,
            "touch_fills_per_day": 1170.0,
            "halving_ticks": 0.462,
            "skew_ticks": 2.25,
            "q_max": 50,
            "liq_ticks": 1.0,
            "dt_seconds": 5.0,
            "session_seconds": 23_400.0,
            "horizon_days": 1.0,
        },
    },
    "us_small_cap": {
        "description": "US small cap: $20, five-tick (25 bp) spread, 3.5% daily vol, "
        "ADV 300k; sell 10% of ADV over one day.",
        "source": "illustrative, not fitted to data",
        "ac": {
            "price": 20.0,
            "spread": 0.05,
            "adv": 3.0e5,
            "daily_vol": 0.035,
            "total_shares": 30_000.0,
            "n_intervals": 13,
            "tau": 1.0 / 13.0,
            "lambda_ra": 3.0e-5,
        },
        "as": {
            "price": 20.0,
            "tick": 0.01,
            "daily_vol": 0.035,
            "touch_fills_per_day": 195.0,
            "halving_ticks": 2.0,
            "skew_ticks": 5.0,
            "q_max": 10,
            "liq_ticks": 5.0,
            "dt_seconds": 5.0,
            "session_seconds": 23_400.0,
            "horizon_days": 1.0,
        },
    },
    "us_etf": {
        "description": "Very liquid US ETF: $500, one-tick (0.2 bp) spread, 1% daily vol, "
        "ADV 60M; sell 1% of ADV over one day.",
        "source": "illustrative, not fitted to data",
        "ac": {
            "price": 500.0,
            "spread": 0.01,
            "adv": 6.0e7,
            "daily_vol": 0.01,
            "total_shares": 600_000.0,
            "n_intervals": 13,
            "tau": 1.0 / 13.0,
            "lambda_ra": 7.0e-10,
        },
        "as": {
            "price": 500.0,
            "tick": 0.01,
            "daily_vol": 0.01,
            "touch_fills_per_day": 11_700.0,
            "halving_ticks": 0.5,
            "skew_ticks": 1.0,
            "q_max": 200,
            "liq_ticks": 1.0,
            "dt_seconds": 1.0,
            "session_seconds": 23_400.0,
            "horizon_days": 1.0,
        },
    },
    "eurusd": {
        "description": "EUR/USD on an electronic venue: 1.10, 0.5 pip spread (about 0.45 bp), "
        "0.5% daily vol, 2bn EUR daily venue volume, 24-hour session; sell 100M EUR "
        "(5% of volume) over one day in hourly bins.",
        "source": "illustrative, not fitted to data",
        "ac": {
            "price": 1.10,
            "spread": 5.0e-5,
            "adv": 2.0e9,
            "daily_vol": 0.005,
            "total_shares": 1.0e8,
            "n_intervals": 24,
            "tau": 1.0 / 24.0,
            "lambda_ra": 8.0e-8,
        },
        "as": {
            "price": 1.10,
            "tick": 1.0e-5,
            "daily_vol": 0.005,
            "touch_fills_per_day": 8640.0,
            "halving_ticks": 2.0,
            "skew_ticks": 3.0,
            "q_max": 50,
            "liq_ticks": 2.0,
            "dt_seconds": 5.0,
            "session_seconds": 86_400.0,
            "horizon_days": 1.0,
        },
    },
}


FITTED = DATA_DIR / "calibration" / "fitted_20191230.json"

#: Inputs the data cannot fit, taken from the illustrative preset of the same
#: instrument type: order size as a fraction of ADV, intervals, and the A-S
#: skew, inventory limit, liquidation penalty and step.
FITTED_TEMPLATE = {"AAPL": "us_large_cap", "MSFT": "us_large_cap", "QQQ": "us_etf"}
ORDER_FRAC_OF_ADV = 0.05


def _sig(x: float, digits: int = 4) -> float:
    return float(f"{x:.{digits}g}")


def fitted_presets() -> dict[str, dict]:
    """Presets fitted to one ITCH day (API_SPEC 11.6)."""
    fit = json.loads(FITTED.read_text())
    out: dict[str, dict] = {}
    for sym, f in fit["symbols"].items():
        template = PRESETS[FITTED_TEMPLATE[sym]]
        tick = f["tick"]
        price = round(f["volume_weighted_mid"], 2)
        spread = round(f["quoted_spread"]["mean_ticks"] * tick, 6)
        adv = float(round(f["volume_shares"]))
        daily_vol = _sig(f["daily_vol"])
        # lambda_ra is not fitted: keep the template's lambda_ra * sigma^2 / eta
        t_ac = template["ac"]
        t_ratio = (
            t_ac["lambda_ra"]
            * (t_ac["price"] * t_ac["daily_vol"]) ** 2
            / (t_ac["spread"] / (mk.DEFAULT_TEMP_FRAC * t_ac["adv"]))
        )
        eta = spread / (mk.DEFAULT_TEMP_FRAC * adv)
        lambda_ra = _sig(t_ratio * eta / (price * daily_vol) ** 2, 3)
        fi = f["fill_intensity"]
        as_block = dict(template["as"])
        as_block.update(
            price=price,
            tick=tick,
            daily_vol=daily_vol,
            touch_fills_per_day=float(round(fi["touch_fills_per_day"])),
            halving_ticks=round(fi["halving_ticks"], 3),
        )
        out[f"itch_{sym.lower()}_20191230"] = {
            "description": f"{sym} fitted to one NASDAQ ITCH day (2019-12-30): volume-weighted "
            f"mid {price}, mean quoted spread {f['quoted_spread']['mean_ticks']:.2f} ticks, "
            f"open-to-close realised vol {100 * daily_vol:.3f}%, NASDAQ continuous volume "
            f"{adv:.0f} shares as ADV; sell {ORDER_FRAC_OF_ADV:.0%} of it over one day. "
            f"Order size, lambda_ra, skew, q_max, liq_ticks and dt follow {FITTED_TEMPLATE[sym]}.",
            "source": fit["source"],
            "ac": {
                "price": price,
                "spread": spread,
                "adv": adv,
                "daily_vol": daily_vol,
                "total_shares": float(round(ORDER_FRAC_OF_ADV * adv / 100.0) * 100),
                "n_intervals": 13,
                "tau": 1.0 / 13.0,
                "lambda_ra": lambda_ra,
            },
            "as": as_block,
        }
    return out


def ac_expect(c: dict) -> dict:
    p = mk.ac_params_from_market(**c)
    notional = c["total_shares"] * c["price"]
    return {
        "sigma": p.sigma,
        "eta": p.eta,
        "gamma_p": p.gamma_p,
        "kappa": et.ac_kappa(p),
        "e_cost_bps": mk.bps(et.expected_cost(p), notional),
        "sd_cost_bps": mk.bps(math.sqrt(et.variance_cost(p)), notional),
        "spread_cost_bps": mk.bps(mk.half_spread_cost(c["spread"], c["total_shares"]), notional),
    }


def as_expect(c: dict) -> dict:
    a = mk.as_params_from_market(**c)
    return {
        "sigma": a.sigma,
        "k": a.k,
        "a_rate": a.a_rate,
        "gamma": a.gamma,
        "eta_liq": a.eta_liq,
        "dt": a.dt,
        "n_steps": float(a.n_steps),
        "spread_open_bps": mk.bps(et.optimal_spread(a, 0.0), c["price"]),
        "spread_close_bps": mk.bps(et.optimal_spread(a, a.horizon), c["price"]),
    }


def build_cases() -> list[dict]:
    cases = []
    for name, preset in PRESETS.items():
        if "ac" in preset:
            inputs = {k: float(v) for k, v in preset["ac"].items()}
            cases.append(
                {
                    "name": f"market_ac_{name}",
                    "kind": "market_ac",
                    "inputs": inputs,
                    "expect": ac_expect(preset["ac"]),
                    "tol": TOL,
                }
            )
        if "as" in preset:
            inputs = {k: float(v) for k, v in preset["as"].items()}
            cases.append(
                {
                    "name": f"market_as_{name}",
                    "kind": "market_as",
                    "inputs": inputs,
                    "expect": as_expect(preset["as"]),
                    "tol": TOL,
                }
            )
    return cases


def main() -> None:
    PRESETS.update(fitted_presets())
    (DATA_DIR / "presets.json").write_text(json.dumps(PRESETS, indent=2) + "\n")
    cases = build_cases()
    (DATA_DIR / "golden" / "market_golden.json").write_text(
        json.dumps({"cases": cases}, indent=2) + "\n"
    )
    print(f"wrote presets.json ({len(PRESETS)} presets), market_golden.json ({len(cases)} cases)")
    for c in cases:
        e = c["expect"]
        if c["kind"] == "market_ac":
            print(
                f"  {c['name']:38s} E {e['e_cost_bps']:7.2f} bps  sd {e['sd_cost_bps']:7.2f} bps"
                f"  half-spread {e['spread_cost_bps']:5.2f} bps  kappa*T "
                f"{e['kappa'] * c['inputs']['n_intervals'] * c['inputs']['tau']:.2f}"
            )
        else:
            print(
                f"  {c['name']:38s} A-S spread {e['spread_close_bps']:6.2f}-"
                f"{e['spread_open_bps']:6.2f} bps  steps {e['n_steps']:.0f}"
            )


if __name__ == "__main__":
    main()
