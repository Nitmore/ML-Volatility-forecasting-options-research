"""Forecast targets.

Timing: a forecast issued at the close of day t predicts realised variance over
the next h sessions, i.e. returns r_{t+1}, ..., r_{t+h}:

    RV_{t,h} = (A / h) * sum_{i=1..h} r_{t+i}^2        (annualised variance)

Targets are stored in VARIANCE space because QLIKE, the variance risk premium
(IV^2 - RV) and GARCH forecasts are all defined on variance. Volatility and
log-variance are derived columns. `label_end_{h}` records the last date whose
return enters the target; the purged splitter in Notebook 3 needs it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_realized_variance(r: pd.Series, h: int, annualization: int = 252) -> pd.Series:
    """Annualised realised variance of r_{t+1..t+h}, aligned to row t.

    Each window is summed independently (sliding_window_view) rather than with
    a running rolling-sum, so a change to one return cannot leak floating-point
    drift into windows that do not contain it. This makes the no-look-ahead
    perturbation test exact.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    vals = r.to_numpy(dtype=float) ** 2
    out = np.full(len(vals), np.nan)
    if len(vals) > h:
        out[: len(vals) - h] = sliding_window_view(vals[1:], h).sum(axis=1)
    return pd.Series(out * annualization / h, index=r.index, name=r.name)


def label_end_dates(index: pd.DatetimeIndex, h: int) -> pd.Series:
    """Date of the last return in each row's target window (NaT if beyond data)."""
    return pd.Series(index, index=index).shift(-h)


def build_targets(panel: pd.DataFrame, horizons: list[int], annualization: int = 252) -> pd.DataFrame:
    r = panel["ret"]
    out = pd.DataFrame(index=panel.index)
    for h in horizons:
        rv = forward_realized_variance(r, h, annualization)
        out[f"rv_var_{h}"] = rv
        out[f"rv_vol_{h}"] = np.sqrt(rv)
        # floor protects log() on a (theoretical) all-zero-return window
        out[f"rv_logvar_{h}"] = np.log(rv.clip(lower=1e-8))
        out[f"label_end_{h}"] = label_end_dates(panel.index, h)
    out.index.name = "date"
    return out


def brute_force_rv(r: pd.Series, h: int, annualization: int = 252) -> pd.Series:
    """Slow loop reference implementation, used only to verify the fast one."""
    vals = r.to_numpy()
    out = np.full(len(vals), np.nan)
    for t in range(len(vals) - h):
        window = vals[t + 1: t + 1 + h]
        if not np.isnan(window).any():
            out[t] = annualization / h * np.sum(window ** 2)
    return pd.Series(out, index=r.index)
