"""Forecast loss functions. Inputs are annualised VARIANCES unless stated.

QLIKE and MSE on variance are the two losses that are 'robust' in Patton's
(2011) sense: they rank forecasts correctly even though realised variance is
only a noisy proxy for true variance. QLIKE is the primary metric because it
penalises under-prediction of variance more than over-prediction, which is the
economically costly mistake for a short-volatility position.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FLOOR = 1e-8


def qlike(rv, f) -> np.ndarray:
    """Patton's QLIKE, normalised to 0 at a perfect forecast: x - log(x) - 1, x = RV/F."""
    x = np.asarray(rv, float) / np.clip(np.asarray(f, float), FLOOR, None)
    return x - np.log(x) - 1.0


def se_var(rv, f) -> np.ndarray:
    return (np.asarray(rv, float) - np.asarray(f, float)) ** 2


def se_vol(rv, f) -> np.ndarray:
    return (np.sqrt(np.asarray(rv, float)) - np.sqrt(np.clip(np.asarray(f, float), 0, None))) ** 2


def ae_vol(rv, f) -> np.ndarray:
    return np.abs(np.sqrt(np.asarray(rv, float)) - np.sqrt(np.clip(np.asarray(f, float), 0, None)))


def summary(rv, f, benchmark=None) -> dict:
    """QLIKE, RMSE/MAE on volatility, MSE on variance, bias, optional R2_OOS vs a benchmark."""
    rv, f = np.asarray(rv, float), np.asarray(f, float)
    out = {
        "QLIKE": float(np.mean(qlike(rv, f))),
        "MSE_var": float(np.mean(se_var(rv, f))),
        "RMSE_vol": float(np.sqrt(np.mean(se_vol(rv, f)))),
        "MAE_vol": float(np.mean(ae_vol(rv, f))),
        "bias_vol": float(np.mean(np.sqrt(f) - np.sqrt(rv))),
        "n": int(len(rv)),
    }
    if benchmark is not None:
        b = np.asarray(benchmark, float)
        out["R2oos_var"] = float(1 - np.sum(se_var(rv, f)) / np.sum(se_var(rv, b)))
    return out


def loss_table(df: pd.DataFrame, rv_col: str, model_cols: list[str], benchmark: str | None = None) -> pd.DataFrame:
    rows = {m: summary(df[rv_col], df[m], df[benchmark] if benchmark else None) for m in model_cols}
    return pd.DataFrame(rows).T
