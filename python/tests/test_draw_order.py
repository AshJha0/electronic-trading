"""Draw-order guards for the pinned simulation loops (Sections 2.2 and 9.4).

Each test re-implements the loop with the pinned draw order and checks it
reproduces the library bit for bit on the same seed, then perturbs only the
draw order and checks the result changes. If the library ever drifts from
the pinned order, the first assertion fails; if the order stopped mattering
(for example a case where no fill can happen), the second one does.
"""

from __future__ import annotations

import math

import numpy as np

import etrade as et
from conftest import as_from_cfg, load_json
from etrade import mm

# Small and fast, with a cap that binds (q_max = 1) so that suppressed sides
# are common and the "always draw both" rule is exercised.
P = as_from_cfg({**load_json("params_liquid.json"), "as_dt": 0.02, "as_q_max": 1})
N_PATHS = 16
SEED = 123


def as_reference(order: str) -> np.ndarray:
    """Section 2.2 loop for the 'as' agent; returns per-path PnL.

    order: 'pinned' (u_ask, u_bid, both always), 'bid_first' (u_bid drawn
    before u_ask), 'skip_suppressed' (a side's uniform is drawn only for
    paths where that side is quoted), 'z_first' (z drawn before the fills).
    """
    p = P
    rng = np.random.default_rng(SEED)
    g, sig2, big_t, dt = p.gamma, p.sigma**2, p.horizon, p.dt
    markup = 2.0 / g * math.log1p(g / p.k)
    s = np.full(N_PATHS, p.s0)
    q = np.zeros(N_PATHS)
    cash = np.zeros(N_PATHS)
    for i in range(p.n_steps):
        t = i * dt
        r = s - q * g * sig2 * (big_t - t)
        half = 0.5 * (g * sig2 * (big_t - t) + markup)
        ask, bid = r + half, r - half
        quote_ask, quote_bid = q > -p.q_max, q < p.q_max
        z = rng.standard_normal(N_PATHS) if order == "z_first" else None
        if order == "bid_first":
            u_bid = rng.random(N_PATHS)
            u_ask = rng.random(N_PATHS)
        elif order == "skip_suppressed":
            u_ask = np.ones(N_PATHS)
            u_ask[quote_ask] = rng.random(int(quote_ask.sum()))
            u_bid = np.ones(N_PATHS)
            u_bid[quote_bid] = rng.random(int(quote_bid.sum()))
        else:
            u_ask = rng.random(N_PATHS)
            u_bid = rng.random(N_PATHS)
        p_ask = -np.expm1(-p.a_rate * np.exp(-p.k * (ask - s)) * dt)
        p_bid = -np.expm1(-p.a_rate * np.exp(-p.k * (s - bid)) * dt)
        fill_ask = quote_ask & (u_ask < p_ask)
        fill_bid = quote_bid & (u_bid < p_bid)
        q += fill_bid.astype(float) - fill_ask.astype(float)
        cash += np.where(fill_ask, ask, 0.0) - np.where(fill_bid, bid, 0.0)
        if z is None:
            z = rng.standard_normal(N_PATHS)
        s = s + p.sigma * math.sqrt(dt) * z
    pnl: np.ndarray = cash + q * s - p.eta_liq * np.abs(q)
    return pnl


def test_section2_pinned_order_reproduces_the_library_bitwise() -> None:
    lib = et.simulate(P, n_paths=N_PATHS, seed=SEED)
    assert np.array_equal(as_reference("pinned"), lib.pnl)


def test_section2_wrong_orders_diverge() -> None:
    lib = et.simulate(P, n_paths=N_PATHS, seed=SEED).pnl
    for order in ("bid_first", "skip_suppressed", "z_first"):
        assert not np.array_equal(as_reference(order), lib), order


PINNED_MM = ("z", "u_sig", "u_ask", "u_bid", "u_inf")


def mm_reference(order: tuple[str, ...]) -> np.ndarray:
    """Section 9.4 loop, 'as' agent with an informed flow and a signal;
    the five uniform/normal vectors are drawn in ``order``."""
    p = P
    cfg = mm.MMConfig(informed_prob=0.3, signal_accuracy=0.7)
    rng = np.random.default_rng(SEED)
    g, sig, big_t, dt, k, a_rate, qmax = (p.gamma, p.sigma, p.horizon, p.dt, p.k, p.a_rate, p.q_max)
    sig2, sqdt = sig * sig, math.sqrt(dt)
    markup = 2.0 / g * math.log1p(g / k)
    shift = (2.0 * cfg.signal_accuracy - 1.0) * sig * sqdt * math.sqrt(2.0 / math.pi)
    s = np.full(N_PATHS, p.s0)
    q = np.zeros(N_PATHS)
    cash = np.zeros(N_PATHS)
    for i in range(p.n_steps):
        t = i * dt
        d = {
            name: rng.standard_normal(N_PATHS) if name == "z" else rng.random(N_PATHS)
            for name in order
        }
        z = d["z"]
        sz = np.sign(z)
        pred = np.where(d["u_sig"] < cfg.signal_accuracy, sz, -sz)
        fair = s + shift * pred
        r = fair - q * g * sig2 * (big_t - t)
        half = 0.5 * (g * sig2 * (big_t - t) + markup)
        bid, ask = r - half, r + half
        can_ask, can_bid = q > -qmax, q < qmax
        p_ask = -np.expm1(-a_rate * np.exp(-k * (ask - s)) * dt)
        p_bid = -np.expm1(-a_rate * np.exp(-k * (s - bid)) * dt)
        fill_ask = can_ask & (d["u_ask"] < p_ask)
        fill_bid = can_bid & (d["u_bid"] < p_bid)
        move = sig * sqdt * z
        informed = d["u_inf"] < cfg.informed_prob
        inf_ask = informed & (z > 0.0) & can_ask & ~fill_ask & (ask < s + move)
        inf_bid = informed & (z < 0.0) & can_bid & ~fill_bid & (bid > s + move)
        fa, fb = fill_ask | inf_ask, fill_bid | inf_bid
        q = q + fb.astype(float) - fa.astype(float)
        cash = cash + np.where(fa, ask, 0.0) - np.where(fb, bid, 0.0)
        s = s + move
    pnl: np.ndarray = cash + q * s - p.eta_liq * np.abs(q)
    return pnl


def test_section9_pinned_order_reproduces_the_library_bitwise() -> None:
    cfg = mm.MMConfig(informed_prob=0.3, signal_accuracy=0.7)
    lib = mm.simulate_mm(P, n_paths=N_PATHS, seed=SEED, config=cfg)
    assert np.array_equal(mm_reference(PINNED_MM), lib.paths["pnl"])


def test_section9_wrong_orders_diverge() -> None:
    cfg = mm.MMConfig(informed_prob=0.3, signal_accuracy=0.7)
    lib = mm.simulate_mm(P, n_paths=N_PATHS, seed=SEED, config=cfg).paths["pnl"]
    wrong = [
        ("z", "u_sig", "u_bid", "u_ask", "u_inf"),  # bid before ask
        ("u_sig", "u_ask", "u_bid", "u_inf", "z"),  # z last
        ("z", "u_ask", "u_bid", "u_inf", "u_sig"),  # signal last
        ("z", "u_sig", "u_ask", "u_inf", "u_bid"),  # informed before bid
    ]
    for order in wrong:
        assert not np.array_equal(mm_reference(order), lib), order
