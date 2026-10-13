"""Feature engineering.

Every feature at row t uses only data observable by the close of session t.
All rolling windows are trailing (pandas default: window ends at t, inclusive).
Variance-type features are stored as log annualised variance, the same scale
as the ML target `rv_logvar_h`, so linear models (HAR) and trees share inputs.

Features are registered in a catalog with a group label. The groups drive the
feature-ablation experiments in Notebook 4, so a feature's group is part of
the research design, not decoration.

Regime labels (low/medium/high vol) are NOT built here: their thresholds must
come from each training fold, so Notebook 3 computes them inside the loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ANN = 252
LOG_FLOOR = 1e-8          # floor on annualised variance before log (vol ~ 0.01%)
LN2 = np.log(2.0)

GROUPS = {
    "returns": "A. Return-based",
    "hist_vol": "B1. Close-to-close realised variance",
    "range_vol": "B2. Range-based and overnight/intraday estimators",
    "dynamics": "C. Volatility dynamics and HAR components",
    "implied": "D1. Implied volatility (VIX family)",
    "volume": "D2. Volume",
    "momentum": "E. Momentum and drawdown",
}


def _log_var(x: pd.Series) -> pd.Series:
    return np.log(x.clip(lower=LOG_FLOOR))


# --------------------------------------------------------------------------- #
# variance estimators (annualised variance over a trailing window of n days)
# --------------------------------------------------------------------------- #
def rv_close_to_close(r: pd.Series, n: int) -> pd.Series:
    """Zero-mean realised variance, same definition as the target."""
    return r.pow(2).rolling(n, min_periods=n).mean() * ANN


def parkinson(h: pd.Series, l: pd.Series, n: int) -> pd.Series:
    hl = np.log(h / l) ** 2
    return hl.rolling(n, min_periods=n).mean() / (4 * LN2) * ANN


def garman_klass(o, h, l, c, n: int) -> pd.Series:
    d = 0.5 * np.log(h / l) ** 2 - (2 * LN2 - 1) * np.log(c / o) ** 2
    return d.rolling(n, min_periods=n).mean() * ANN


def rogers_satchell(o, h, l, c, n: int) -> pd.Series:
    d = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    return d.rolling(n, min_periods=n).mean() * ANN


def yang_zhang(o, h, l, c, n: int) -> pd.Series:
    """Yang & Zhang (2000): overnight + open-to-close + Rogers-Satchell.

    Unbiased under drift and opening jumps, which matters here because the
    overnight session carries ~40% of SPY's variance.
    """
    ret_on = np.log(o / c.shift(1))
    ret_oc = np.log(c / o)
    var_on = ret_on.rolling(n, min_periods=n).var(ddof=1)
    var_oc = ret_oc.rolling(n, min_periods=n).var(ddof=1)
    rs = (np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)).rolling(n, min_periods=n).mean()
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    return (var_on + k * var_oc + (1 - k) * rs) * ANN


# --------------------------------------------------------------------------- #
# feature registry
# --------------------------------------------------------------------------- #
@dataclass
class FeatureSet:
    data: dict = field(default_factory=dict)
    meta: list = field(default_factory=list)

    def add(self, name: str, values: pd.Series, group: str, desc: str, lookback: int) -> None:
        assert group in GROUPS, group
        assert name not in self.data, f"duplicate feature {name}"
        self.data[name] = values.astype(float)
        self.meta.append({"feature": name, "group": group, "definition": desc, "lookback_days": lookback})

    def frame(self, index) -> tuple[pd.DataFrame, pd.DataFrame]:
        X = pd.DataFrame(self.data, index=index)
        X = X.replace([np.inf, -np.inf], np.nan)
        cat = pd.DataFrame(self.meta).set_index("feature")
        cat["group_label"] = cat["group"].map(GROUPS)
        return X, cat


def build_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (X, catalog). X is indexed like `panel`; warm-up rows contain NaN."""
    o, h, l, c = panel["open"], panel["high"], panel["low"], panel["close"]
    r, r_on, r_id = panel["ret"], panel["ret_on"], panel["ret_id"]
    fs = FeatureSet()

    # ---- A. returns ---------------------------------------------------------
    fs.add("ret_1", r, "returns", "close-to-close log return", 1)
    fs.add("ret_on_1", r_on, "returns", "overnight log return", 1)
    fs.add("ret_id_1", r_id, "returns", "open-to-close log return", 1)
    fs.add("abs_ret_1", r.abs(), "returns", "|r_t|", 1)
    fs.add("neg_ret_1", r.clip(upper=0), "returns", "min(r_t, 0): signed downside move (leverage)", 1)
    for n in (5, 10, 20):
        fs.add(f"cumret_{n}", r.rolling(n, min_periods=n).sum(), "returns", f"sum of last {n} log returns", n)

    # ---- B1. close-to-close realised variance ------------------------------
    for n in (5, 10, 20, 60):
        fs.add(f"lv_cc_{n}", _log_var(rv_close_to_close(r, n)), "hist_vol",
               f"log 252*mean(r^2) over {n}d (target definition, trailing)", n)

    # ---- B2. range-based, overnight/intraday, semivariance -----------------
    for n in (5, 20, 60):
        fs.add(f"lv_park_{n}", _log_var(parkinson(h, l, n)), "range_vol", f"log Parkinson variance, {n}d", n)
        fs.add(f"lv_gk_{n}", _log_var(garman_klass(o, h, l, c, n)), "range_vol", f"log Garman-Klass variance, {n}d", n)
        fs.add(f"lv_rs_{n}", _log_var(rogers_satchell(o, h, l, c, n)), "range_vol", f"log Rogers-Satchell variance, {n}d", n)
        fs.add(f"lv_yz_{n}", _log_var(yang_zhang(o, h, l, c, n)), "range_vol", f"log Yang-Zhang variance, {n}d", n + 1)
        fs.add(f"lv_on_{n}", _log_var(r_on.pow(2).rolling(n, min_periods=n).mean() * ANN), "range_vol",
               f"log overnight realised variance, {n}d", n)
        fs.add(f"lv_id_{n}", _log_var(r_id.pow(2).rolling(n, min_periods=n).mean() * ANN), "range_vol",
               f"log open-to-close realised variance, {n}d", n)
        fs.add(f"lv_semi_neg_{n}", _log_var((r.clip(upper=0) ** 2).rolling(n, min_periods=n).mean() * ANN),
               "range_vol", f"log negative realised semivariance, {n}d", n)
    on20 = r_on.pow(2).rolling(20, min_periods=20).sum()
    fs.add("on_share_20", on20 / (on20 + r_id.pow(2).rolling(20, min_periods=20).sum()), "range_vol",
           "overnight share of (overnight + intraday) squared returns, 20d", 20)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    for n in (14, 20):
        fs.add(f"log_atr_{n}", np.log(tr.rolling(n, min_periods=n).mean() / c), "range_vol",
               f"log(ATR_{n} / close)", n + 1)

    # ---- C. dynamics + HAR components ---------------------------------------
    gk_1 = 0.5 * np.log(h / l) ** 2 - (2 * LN2 - 1) * np.log(c / o) ** 2
    lv_gk_1 = _log_var(gk_1 * ANN)
    fs.add("lv_cc_1", _log_var(r.pow(2) * ANN), "dynamics", "log 252*r_t^2 (HAR daily, close-to-close)", 1)
    fs.add("lv_gk_1", lv_gk_1, "dynamics", "log daily Garman-Klass variance (HAR daily, range)", 1)
    fs.add("har_w", _log_var(rv_close_to_close(r, 5)), "dynamics", "HAR weekly: log RV over 5d", 5)
    fs.add("har_m", _log_var(rv_close_to_close(r, 22)), "dynamics", "HAR monthly: log RV over 22d", 22)
    lv5, lv20, lv60 = (_log_var(rv_close_to_close(r, n)) for n in (5, 20, 60))
    fs.add("vol_ratio_5_20", lv5 - lv20, "dynamics", "log(RV5/RV20): short vs medium vol", 20)
    fs.add("vol_ratio_20_60", lv20 - lv60, "dynamics", "log(RV20/RV60)", 60)
    fs.add("d_lv20_5", lv20 - lv20.shift(5), "dynamics", "5-day change in log RV20", 25)
    fs.add("accel_lv20", (lv20 - lv20.shift(5)) - (lv20.shift(5) - lv20.shift(10)), "dynamics",
           "change in the 5-day change of log RV20", 30)
    fs.add("volofvol_20", lv_gk_1.rolling(20, min_periods=20).std(), "dynamics",
           "std of daily log GK variance, 20d", 20)
    r2 = r.pow(2)
    fs.add("persist_60", r2.rolling(60, min_periods=60).corr(r2.shift(1)), "dynamics",
           "60d autocorrelation of r^2 at lag 1", 61)

    # ---- D1. implied volatility ----------------------------------------------
    lv_vix = np.log(panel["vix_var"])
    fs.add("lv_vix", lv_vix, "implied", "log VIX^2 (annualised implied variance)", 1)
    fs.add("d_lv_vix_1", lv_vix.diff(1), "implied", "1-day change in log VIX^2", 2)
    fs.add("d_lv_vix_5", lv_vix.diff(5), "implied", "5-day change in log VIX^2", 6)
    fs.add("vix_ma_ratio_20", np.log(panel["vix"] / panel["vix"].rolling(20, min_periods=20).mean()), "implied",
           "log(VIX / 20d mean VIX)", 20)
    fs.add("vix_vol_20", np.log(panel["vix"]).diff().rolling(20, min_periods=20).std(), "implied",
           "std of daily log VIX changes, 20d", 21)
    fs.add("vrp_proxy_20", lv_vix - lv20, "implied",
           "log VIX^2 - log trailing RV20: implied vs backward-looking realised", 20)
    if "vix9d" in panel:
        fs.add("ts_9d_30d", np.log(panel["vix9d"] / panel["vix"]), "implied",
               "log(VIX9D/VIX): > 0 = short-end stress (inverted)", 1)
    if "vix3m" in panel:
        fs.add("ts_30d_3m", np.log(panel["vix"] / panel["vix3m"]), "implied",
               "log(VIX/VIX3M): > 0 = backwardation", 1)

    # ---- D2. volume -----------------------------------------------------------
    lvol = np.log(panel["volume"].clip(lower=1))
    fs.add("lvol_ratio_20", lvol - lvol.rolling(20, min_periods=20).mean(), "volume",
           "log volume minus its 20d mean", 20)
    fs.add("d_lvol_1", lvol.diff(), "volume", "1-day change in log volume", 2)
    fs.add("lvol_ratio_5_60", lvol.rolling(5, min_periods=5).mean() - lvol.rolling(60, min_periods=60).mean(),
           "volume", "5d vs 60d mean log volume", 60)

    # ---- E. momentum / drawdown ----------------------------------------------
    fs.add("mom_60", r.rolling(60, min_periods=60).sum(), "momentum", "sum of last 60 log returns", 60)
    fs.add("dd_20", c / c.rolling(20, min_periods=20).max() - 1, "momentum", "drawdown from 20d high", 20)
    fs.add("dd_60", c / c.rolling(60, min_periods=60).max() - 1, "momentum", "drawdown from 60d high", 60)
    fs.add("trend_60", np.log(c / c.rolling(60, min_periods=60).mean()), "momentum", "log(close / 60d MA)", 60)

    return fs.frame(panel.index)


# --------------------------------------------------------------------------- #
# leakage audit
# --------------------------------------------------------------------------- #
def audit_leakage(builder, panel: pd.DataFrame, n_dates: int = 40, seed: int = 0,
                  start_pos: int = 120) -> pd.DataFrame:
    """Two independent look-ahead tests at randomly chosen dates t.

    1. Truncation: features built on panel[:t] must equal features at t built
       on the full panel.
    2. Perturbation: scramble every numeric value after t (multiply by
       lognormal noise); features at t and earlier must not change.

    Any difference means some feature reads data from after t.
    """
    rng = np.random.default_rng(seed)
    X_full, _ = builder(panel)
    num_cols = panel.select_dtypes("number").columns
    positions = np.sort(rng.choice(np.arange(start_pos, len(panel) - 2), size=n_dates, replace=False))
    rows = []
    for pos in positions:
        t = panel.index[pos]
        X_trunc, _ = builder(panel.iloc[: pos + 1])
        a, b = X_full.iloc[pos].to_numpy(), X_trunc.iloc[pos].to_numpy()
        trunc_diff = np.nanmax(np.abs(a - b)) if np.isfinite(a - b).any() else 0.0
        trunc_nan_mismatch = int((np.isnan(a) != np.isnan(b)).sum())

        pert = panel.copy()
        noise = np.exp(rng.normal(0, 0.2, size=(len(panel) - pos - 1, len(num_cols))))
        pert.iloc[pos + 1:, [panel.columns.get_loc(cn) for cn in num_cols]] *= noise
        X_pert, _ = builder(pert)
        past_a = X_full.iloc[: pos + 1].to_numpy()
        past_b = X_pert.iloc[: pos + 1].to_numpy()
        d = np.abs(past_a - past_b)
        pert_diff = np.nanmax(d) if np.isfinite(d).any() else 0.0
        pert_nan_mismatch = int((np.isnan(past_a) != np.isnan(past_b)).sum())

        bad_cols = []
        if trunc_diff > 1e-12 or pert_diff > 1e-12 or trunc_nan_mismatch or pert_nan_mismatch:
            col_d = np.nanmax(np.abs(past_a - past_b), axis=0)
            bad_cols = list(X_full.columns[(col_d > 1e-12) | (np.abs(a - b) > 1e-12)])
        rows.append({"date": t, "truncation_max_diff": trunc_diff, "perturbation_max_diff": pert_diff,
                     "nan_mismatches": trunc_nan_mismatch + pert_nan_mismatch, "leaking_features": bad_cols})
    out = pd.DataFrame(rows).set_index("date")
    out["passed"] = out["leaking_features"].str.len() == 0
    return out
