"""Synthetic market data with the same schema as the real downloads.

Purpose: run every notebook offline and in CI, and test that the pipeline
recovers known properties (e.g. GARCH persistence). Synthetic results are
NEVER research results; notebooks print a banner when this mode is active.

Model: GJR-GARCH(1,1) with Student-t shocks, an overnight/intraday variance
split, a 78-step intraday path for High/Low, quarterly dividends with
Yahoo-style backward adjustment, and implied-vol indices equal to the model's
expected future variance scaled by a variance risk premium.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OMEGA, ALPHA, GAMMA, BETA = 2.0e-6, 0.05, 0.15, 0.85
PERSIST = ALPHA + GAMMA / 2 + BETA
UNCOND = OMEGA / (1 - PERSIST)


def _sessions(start: str, end: str) -> pd.DatetimeIndex:
    try:
        import pandas_market_calendars as mcal
        s = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end).index
        return pd.DatetimeIndex(s).tz_localize(None).normalize()
    except ImportError:
        return pd.bdate_range(start, end)


def _expected_avg_var(h_next: np.ndarray, k: int) -> np.ndarray:
    """E_t[mean of h_{t+1..t+k}] under GJR-GARCH mean reversion."""
    w = (1 - PERSIST ** k) / ((1 - PERSIST) * k)
    return UNCOND + w * (h_next - UNCOND)


def generate(start: str, end: str, seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = _sessions(start, end)
    n = len(dates)
    nu = 6.0
    share_on = 0.22                                   # overnight share of daily variance
    z_on = rng.standard_t(nu, size=n) / np.sqrt(nu / (nu - 2))
    z_id = rng.standard_t(nu, size=n) / np.sqrt(nu / (nu - 2))

    h = np.empty(n)
    r = np.empty(n)
    on = np.empty(n)
    h[0] = UNCOND
    for t in range(n):
        if t > 0:
            neg = r[t - 1] < 0
            h[t] = OMEGA + (ALPHA + GAMMA * neg) * r[t - 1] ** 2 + BETA * h[t - 1]
        on[t] = 0.0001 + np.sqrt(h[t] * share_on) * z_on[t]
        r[t] = on[t] + 0.0002 + np.sqrt(h[t] * (1 - share_on)) * z_id[t]
    h_next = OMEGA + (ALPHA + GAMMA * (r < 0)) * r ** 2 + BETA * h   # known at close t

    # 78-step intraday path from the open, pinned to the day's open-to-close return
    intraday_total = r - on
    steps = 78
    paths = rng.normal(size=(n, steps))
    paths = np.cumsum(paths, axis=1) * np.sqrt(h * (1 - share_on) / steps)[:, None]
    drift = (intraday_total - paths[:, -1])[:, None] * np.arange(1, steps + 1)[None, :] / steps
    paths = paths + drift                                            # Brownian bridge to the close
    paths = np.concatenate([np.zeros((n, 1)), paths], axis=1)

    c_adj = 200.0 * np.exp(np.cumsum(r))
    prev_c = np.concatenate([[200.0], c_adj[:-1]])
    o_adj = prev_c * np.exp(on)
    hi_adj = o_adj * np.exp(paths.max(axis=1))
    lo_adj = o_adj * np.exp(paths.min(axis=1))

    # quarterly ex-dividend dates (third Friday-ish of Mar/Jun/Sep/Dec)
    ser = pd.Series(np.arange(n), index=dates)
    ex_pos = []
    for (y, m), grp in ser.groupby([dates.year, dates.month]):
        if m in (3, 6, 9, 12) and len(grp) > 14:
            ex_pos.append(int(grp.iloc[14]))
    ex_pos = np.array(ex_pos)
    y_q = 0.0035
    n_after = np.array([(ex_pos > t).sum() for t in range(n)])
    factor = (1 - y_q) ** n_after
    close_raw = c_adj / factor
    div = np.zeros(n)
    div[ex_pos[ex_pos > 0]] = y_q * close_raw[ex_pos[ex_pos > 0] - 1]

    vol = np.exp(18.2 + 8 * (np.sqrt(h) - np.sqrt(UNCOND)) / np.sqrt(UNCOND) * 0.1
                 + rng.normal(0, 0.25, n))

    spy = pd.DataFrame({
        "Open": o_adj / factor, "High": hi_adj / factor, "Low": lo_adj / factor,
        "Close": close_raw, "AdjClose": c_adj, "Volume": np.round(vol),
        "Dividends": div, "Splits": 0.0,
    }, index=dates)
    spy.index.name = "date"

    def iv_index(k_days: int, vrp: float, noise: float) -> np.ndarray:
        iv = 100 * np.sqrt(252 * _expected_avg_var(h_next, k_days) * vrp)
        return np.clip(iv * np.exp(rng.normal(0, noise, n)), 9.0, None)

    vix = pd.Series(iv_index(21, 1.25, 0.04), index=dates)
    vix9d = pd.Series(iv_index(6, 1.15, 0.06), index=dates)
    vix3m = pd.Series(iv_index(63, 1.35, 0.03), index=dates)
    vix.iloc[rng.choice(n, 6, replace=False)] = np.nan                  # FRED-style gaps

    # piecewise-linear 3m T-bill path (percent), FRED skips bond-market holidays
    knots = pd.Series({"2014-06-01": 0.03, "2016-12-31": 0.5, "2019-06-30": 2.3, "2020-04-01": 0.1,
                       "2022-01-01": 0.05, "2023-07-01": 5.4, "2024-09-01": 5.1, "2026-03-01": 3.8})
    knots.index = pd.to_datetime(knots.index)
    rate = knots.reindex(knots.index.union(dates)).interpolate("time").reindex(dates).round(2)
    rate_dates = dates.delete(rng.choice(n, 20, replace=False))

    return {
        "spy": spy,
        "vix": vix.to_frame("VIXCLS"),
        "vix9d": vix9d.to_frame("vix9d"),
        "vix3m": vix3m.to_frame("VXVCLS"),
        "rate_3m": rate.reindex(rate_dates).to_frame("DGS3MO"),
    }
