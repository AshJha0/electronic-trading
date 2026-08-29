"""Shared test helpers: locate bundled data, build params from config dicts."""

from __future__ import annotations

import json
from pathlib import Path

import etrade as et

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def load_json(name: str) -> dict:
    """Load a JSON file from the project data/ directory."""
    return json.loads((DATA_DIR / name).read_text())


def ac_from_cfg(cfg: dict, lambda_ra: float | None = None) -> et.ACParams:
    """Build ACParams from a params_*.json / golden-inputs style dict."""
    return et.ACParams(
        total_shares=float(cfg["ac_total_shares"]),
        n_intervals=int(cfg["ac_n_intervals"]),
        tau=float(cfg["ac_tau"]),
        sigma=float(cfg["ac_sigma"]),
        eta=float(cfg["ac_eta"]),
        gamma_p=float(cfg["ac_gamma_p"]),
        lambda_ra=float(cfg["ac_lambda_ra"]) if lambda_ra is None else lambda_ra,
    )


def as_from_cfg(cfg: dict) -> et.ASParams:
    """Build ASParams from a params_*.json / golden-inputs style dict."""
    return et.ASParams(
        s0=float(cfg["as_s0"]),
        sigma=float(cfg["as_sigma"]),
        horizon=float(cfg["as_horizon"]),
        dt=float(cfg["as_dt"]),
        gamma=float(cfg["as_gamma"]),
        k=float(cfg["as_k"]),
        a_rate=float(cfg["as_a_rate"]),
        q_max=int(cfg["as_q_max"]),
        eta_liq=float(cfg["as_eta_liq"]),
    )
