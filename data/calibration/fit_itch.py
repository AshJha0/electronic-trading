"""Fit market presets to one recorded NASDAQ TotalView-ITCH day.

Run:  python data/calibration/fit_itch.py <dataset_dir>

``<dataset_dir>`` is a dataset written by the ``iap.marketdata ingest``
command of the sibling intraday-alpha-platform project (here the re-ingest
``data/real/ds3_20191230`` of the NASDAQ sample file 12302019.NASDAQ_ITCH50;
not the earlier ``ds_20191230``, which carries a tick-size bug).  It reads
``normalized/events.parquet`` and ``reference/security_master.json`` and
needs ``pyarrow``, which the etrade package itself does not depend on.

This is tooling, Python only.  It rebuilds the top of book from the order
events, turns the day into plain arrays and calls the ported estimators of
``etrade.calibration`` (API_SPEC Section 11).  The only output is
``data/calibration/fitted_20191230.json``: aggregate statistics and fitted
parameters, which cannot reconstruct the underlying messages.  Raw and
normalized data are never written into this repository.

Event schema (iap ``core/events.py``): event_type 1 ADD, 2 MODIFY (qty is
the remaining size), 3 CANCEL, 4 EXECUTE (qty executed at the resting
order's price; an order filled to zero gets no further event), 5 TRADE (one
tape print per executed order, continuous session only, hidden ``P``
prints included, cross prints excluded); side 0 bid, 1 ask (the aggressor
side on a TRADE); price_ticks in units of the security-master tick;
exchange_ts in nanoseconds since the Unix epoch, UTC.
"""

from __future__ import annotations

import heapq
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

DATA_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DATA_DIR.parent / "python" / "src"))

from etrade import calibration as cal

DAY = "2019-12-30"
SOURCE = "NASDAQ TotalView-ITCH 5.0 sample day 2019-12-30 (12302019.NASDAQ_ITCH50)"
# 09:30 and 16:00 US Eastern (EST, UTC-5) on 2019-12-30, in ns since the epoch.
MIDNIGHT_UTC_NS = 1_577_664_000 * 10**9
SESSION_START_NS = MIDNIGHT_UTC_NS + (14 * 3600 + 30 * 60) * 10**9
SESSION_END_NS = MIDNIGHT_UTC_NS + 21 * 3600 * 10**9
N_BINS = 13
QUOTE_SAMPLE_S = 1  # quoted spread sampled every second
MID_SAMPLE_S = 60  # realised variance from one-minute mids
FILL_LEVELS = 6

ADD, MODIFY, CANCEL, EXECUTE, TRADE = 1, 2, 3, 4, 5


class TopOfBook:
    """Order-level book keeping only what is needed for the best prices."""

    def __init__(self) -> None:
        self.orders: dict[int, list[int]] = {}  # id -> [side, price, qty]
        self.levels: tuple[dict[int, int], dict[int, int]] = ({}, {})
        self.heaps: tuple[list[int], list[int]] = ([], [])  # bid as -price

    def _change(self, side: int, price: int, dq: int) -> None:
        lv = self.levels[side]
        q = lv.get(price, 0) + dq
        if q > 0:
            if price not in lv:
                heapq.heappush(self.heaps[side], -price if side == 0 else price)
            lv[price] = q
        else:
            lv.pop(price, None)

    def apply(self, etype: int, side: int, price: int, qty: int, oid: int) -> None:
        if etype == ADD:
            self.orders[oid] = [side, price, qty]
            self._change(side, price, qty)
        elif etype == MODIFY:
            o = self.orders.get(oid)
            if o is not None:
                self._change(o[0], o[1], qty - o[2])
                o[2] = qty
        elif etype == CANCEL:
            o = self.orders.pop(oid, None)
            if o is not None:
                self._change(o[0], o[1], -o[2])
        elif etype == EXECUTE:
            o = self.orders.get(oid)
            if o is not None:
                q = min(qty, o[2])
                self._change(o[0], o[1], -q)
                o[2] -= q
                if o[2] == 0:
                    del self.orders[oid]

    def best(self, side: int) -> int | None:
        h, lv = self.heaps[side], self.levels[side]
        while h:
            p = -h[0] if side == 0 else h[0]
            if p in lv:
                return p
            heapq.heappop(h)
        return None


def symbol_arrays(table: object, inst: int, tick: float) -> dict[str, list[float]]:
    ids = table.column("instrument_id").to_numpy()  # type: ignore[attr-defined]
    mask = ids == inst
    sub = {
        name: table.column(name).to_numpy()[mask].tolist()  # type: ignore[attr-defined]
        for name in ("exchange_ts", "event_type", "side", "price_ticks", "qty", "order_id")
    }
    book = TopOfBook()
    quote_spread: list[float] = []
    quote_mid: list[float] = []
    mid_t: list[float] = []
    mid_v: list[float] = []
    trade_t: list[float] = []
    trade_q: list[float] = []
    eff_spread: list[float] = []
    eff_mid: list[float] = []
    reach: list[float] = []
    next_quote = SESSION_START_NS
    next_mid = SESSION_START_NS
    mid_before: float | None = None  # mid after the previous timestamp
    group_key: tuple[int, int] | None = None
    group_reach = 0.0
    last_ts = -1
    n = len(sub["exchange_ts"])

    def current_mid() -> tuple[float, float] | None:
        b, a = book.best(0), book.best(1)
        if b is None or a is None or a <= b:
            return None
        return (a + b) * 0.5 * tick, (a - b) * tick

    for i in range(n + 1):
        ts = sub["exchange_ts"][i] if i < n else SESSION_END_NS + 1
        if ts != last_ts:
            # state as of the end of last_ts: sample the clocks up to ts
            state = current_mid()
            while next_quote < ts and next_quote <= SESSION_END_NS:
                if state is not None:
                    quote_mid.append(state[0])
                    quote_spread.append(state[1])
                next_quote += QUOTE_SAMPLE_S * 10**9
            while next_mid < ts and next_mid <= SESSION_END_NS:
                if state is not None:
                    mid_t.append((next_mid - SESSION_START_NS) / 1e9)
                    mid_v.append(state[0])
                next_mid += MID_SAMPLE_S * 10**9
            mid_before = state[0] if state is not None else None
            last_ts = ts
        if i == n:
            break
        et = sub["event_type"][i]
        if et == TRADE:
            if SESSION_START_NS <= ts <= SESSION_END_NS and mid_before is not None:
                px = sub["price_ticks"][i] * tick
                q = float(sub["qty"][i])
                trade_t.append((ts - SESSION_START_NS) / 1e9)
                trade_q.append(q)
                eff_spread.append(2.0 * abs(px - mid_before))
                eff_mid.append(mid_before)
                key = (ts, sub["side"][i])
                d = abs(px - mid_before)
                if key == group_key:
                    group_reach = max(group_reach, d)
                else:
                    if group_key is not None:
                        reach.append(group_reach)
                    group_key, group_reach = key, d
            continue
        book.apply(et, sub["side"][i], sub["price_ticks"][i], sub["qty"][i], sub["order_id"][i])
    if group_key is not None:
        reach.append(group_reach)
    return {
        "quote_spread": quote_spread,
        "quote_mid": quote_mid,
        "mid_t": mid_t,
        "mid_v": mid_v,
        "trade_t": trade_t,
        "trade_q": trade_q,
        "eff_spread": eff_spread,
        "eff_mid": eff_mid,
        "reach": reach,
    }


def fit_symbol(arr: dict[str, list[float]], tick: float) -> dict[str, object]:
    span = (SESSION_END_NS - SESSION_START_NS) / 1e9
    curve = cal.volume_curve(arr["trade_t"], arr["trade_q"], 0.0, span, N_BINS)
    vc = cal.volatility_curve(arr["mid_t"], arr["mid_v"], 0.0, span, N_BINS)
    qs = cal.spread_stats(arr["quote_spread"], arr["quote_mid"], tick)
    es = cal.spread_stats(arr["eff_spread"], arr["eff_mid"], tick)
    fi = cal.fill_intensity(arr["reach"], tick, FILL_LEVELS)
    volume = sum(arr["trade_q"])
    vwap_num = 0.0
    for p_mid, q in zip(arr["eff_mid"], arr["trade_q"], strict=True):
        vwap_num += p_mid * q
    return {
        "volume_shares": volume,
        "n_trades": len(arr["trade_q"]),
        "n_market_orders": len(arr["reach"]),
        "volume_weighted_mid": vwap_num / volume,
        "volume_curve": curve,
        "daily_vol": vc.daily_vol,
        "bin_vol": vc.bin_vol,
        "n_returns": vc.n_returns,
        "quoted_spread": vars(qs),
        "effective_spread": vars(es),
        "fill_intensity": vars(fi),
        "reach_counts": [
            sum(1 for d in arr["reach"] if d / tick >= b + 0.5 - cal.FILL_EDGE_SLACK)
            for b in range(FILL_LEVELS)
        ],
    }


def main() -> None:
    ds = Path(sys.argv[1])
    master = json.loads((ds / "reference" / "security_master.json").read_text())
    table = pq.read_table(
        ds / "normalized" / "events.parquet",
        columns=[
            "instrument_id",
            "exchange_ts",
            "event_type",
            "side",
            "price_ticks",
            "qty",
            "order_id",
        ],
    )
    out: dict[str, object] = {"source": SOURCE, "day": DAY, "symbols": {}}
    curves = []
    for rec in master["records"]:
        sym, tick = rec["symbol"], float(rec["tick_size"])
        arr = symbol_arrays(table, rec["instrument_id"], tick)
        fit = fit_symbol(arr, tick)
        fit["tick"] = tick
        out["symbols"][sym] = fit  # type: ignore[index]
        curves.append(fit["volume_curve"])
        fi = fit["fill_intensity"]
        print(
            f"{sym}: vol {fit['volume_shares']:.0f} sh, mid {fit['volume_weighted_mid']:.2f}, "
            f"daily vol {fit['daily_vol']:.4%}, "
            f"quoted {fit['quoted_spread']['mean_ticks']:.2f} ticks, "
            f"effective {fit['effective_spread']['mean_ticks']:.2f} ticks, "
            f"touch {fi['touch_fills_per_day']:.0f}/day, halving {fi['halving_ticks']:.3f} ticks"
        )
    pooled = [sum(c[j] for c in curves) / len(curves) for j in range(N_BINS)]
    rounded = [round(w, 4) for w in pooled]
    j_max = max(range(N_BINS), key=lambda j: rounded[j])
    rounded[j_max] = round(rounded[j_max] + 1.0 - sum(rounded), 4)
    out["pooled_volume_curve"] = pooled
    out["fitted_volume_curve_13"] = rounded
    print("FITTED_VOLUME_CURVE_13 =", rounded, "sum", sum(rounded))
    path = Path(__file__).resolve().parent / "fitted_20191230.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
