"""Produce out-of-sample forecasts for every model over a list of folds.

One function serves both the validation folds (2019-2022) and, after the
configuration is frozen, the test folds (2023-2025), so the test run cannot
differ from the validation run in anything except the dates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import models as M
from .validation import Fold


def _long(series: pd.Series, model: str, h: int, fold: Fold) -> pd.DataFrame:
    return pd.DataFrame({"date": series.index, "horizon": h, "model": model, "split": fold.split,
                         "fold": fold.name, "forecast": series.to_numpy(dtype=float)})


def forecast_folds(folds_by_h: dict[int, list[Fold]], X: pd.DataFrame, targets: pd.DataFrame,
                   panel: pd.DataFrame, feature_sets: dict, frozen: dict, log=print):
    """Return (long forecast table, GARCH parameter path per monthly refit)."""
    seed = frozen["seed"]
    out = []
    ret = panel["ret"]
    horizons = sorted(folds_by_h)

    # ---- naive -------------------------------------------------------------------
    ewma = M.ewma_forecast(ret, frozen["ewma_lambda"])
    for h in horizons:
        hist = M.hist_forecast(ret, h)
        for f in folds_by_h[h]:
            out.append(_long(hist.loc[f.evaluate], "HistVol", h, f))
            out.append(_long(ewma.loc[f.evaluate], "EWMA", h, f))
    log("naive benchmarks done")

    # ---- GARCH family: one monthly-refit pass over all evaluation dates -------------
    eval_dates = pd.DatetimeIndex(sorted(set().union(*[f.evaluate for f in folds_by_h[horizons[0]]])))
    param_paths = []
    for spec in M.GARCH_SPECS:
        fc, params = M.garch_forecasts(ret, spec, eval_dates, horizons, frozen["garch_fit_start"],
                                  frozen["egarch_simulations"], seed)
        for h in horizons:
            for f in folds_by_h[h]:
                out.append(_long(fc.loc[f.evaluate, h], spec, h, f))
        param_paths.append(params.assign(model=spec))
        log(f"{spec} done")

    # ---- regressions on log variance ----------------------------------------------
    for h in horizons:
        y_log, label_end = targets[f"rv_logvar_{h}"], targets[f"label_end_{h}"]
        best = frozen["best_params"]
        specs = {
            "HAR": (M.make_ols(), feature_sets["HAR"]),
            "HAR_X": (M.make_ols(), feature_sets["HAR_X"]),
            "RF": (M.make_rf(best[f"RF_h{h}"], seed), feature_sets[frozen["primary_set"]]),
            "XGB": (M.make_xgb(best[f"XGB_h{h}"], seed), feature_sets[frozen["primary_set"]]),
            # ablations reuse the tuned hyper-parameters: only the inputs change
            "RF_noVIX": (M.make_rf(best[f"RF_h{h}"], seed), feature_sets["E_noVIX"]),
            **{f"XGB_{k}": (M.make_xgb(best[f"XGB_h{h}"], seed), feature_sets[k])
               for k in ("E1_hist_vol", "E2_ohlc", "E3_plus_implied", "E_noVIX")},
        }
        for f in folds_by_h[h]:
            for name, (factory, cols) in specs.items():
                pred = M.fit_predict(factory, X[cols], y_log, f, label_end, frozen["calib_rows"])
                out.append(_long(pred, name, h, f))
            q = M.quantile_forecasts(X[feature_sets[frozen["primary_set"]]], y_log, f,
                                     tuple(frozen["quantiles"]), seed)
            for c in q:
                out.append(_long(q[c], f"QGB_{c}", h, f))
        log(f"h={h}: regression and ML models done")

    df = pd.concat(out, ignore_index=True)
    assert df["forecast"].notna().all(), "missing forecasts"
    assert (df["forecast"] > 0).all(), "non-positive variance forecast"
    return df, pd.concat(param_paths)


def regimes_for_folds(folds: list[Fold], trailing_logvar: pd.Series, quantiles) -> tuple[pd.DataFrame, pd.DataFrame]:
    labs, thr = [], []
    for f in folds:
        lab, (lo, hi) = M.regime_labels(trailing_logvar, f, quantiles)
        labs.append(pd.DataFrame({"date": lab.index, "fold": f.name, "split": f.split, "regime": lab.to_numpy()}))
        thr.append({"fold": f.name, "low_upper_logvar": lo, "mid_upper_logvar": hi,
                    "low_upper_vol": float(np.sqrt(np.exp(lo))), "mid_upper_vol": float(np.sqrt(np.exp(hi)))})
    return pd.concat(labs, ignore_index=True), pd.DataFrame(thr).set_index("fold")
