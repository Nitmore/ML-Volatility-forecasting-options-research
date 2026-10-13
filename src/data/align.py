"""Build the aligned daily panel on the SPY (NYSE) trading calendar.

Timing convention (documented in the data dictionary):
  * Row t holds information observable at the close of session t.
  * SPY close is 16:00 ET; VIX/VIX9D/VIX3M closes are 16:15 ET. The 15-minute
    gap is tiny relative to a forecast horizon that starts at the t+1 close,
    and it is recorded as a known limitation rather than ignored.
  * DGS3MO is an end-of-day Treasury yield, available after the close.
Secondary series are only ever forward-filled (never back-filled), so no row
can see a value published after its own date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def adjust_ohlc(spy: pd.DataFrame) -> pd.DataFrame:
    """Dividend-adjust OHLC with the AdjClose/Close factor.

    Range-based estimators (log H/L, log C/O) are invariant to the factor, but
    the overnight return log(O_t / C_{t-1}) is not: on an ex-dividend date the
    raw open gaps down by the dividend. Using adjusted prices removes that
    artificial gap.
    """
    f = spy["AdjClose"] / spy["Close"]
    out = pd.DataFrame(index=spy.index)
    for col in ("Open", "High", "Low", "Close"):
        out[col.lower()] = spy[col] * f
    out["close_raw"] = spy["Close"]
    out["adj_factor"] = f
    out["volume"] = spy["Volume"]
    out["dividend"] = spy["Dividends"]
    return out


def build_panel(raw: dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    spy = raw["spy"].sort_index()
    # pandas 3 keeps the parsing resolution (s/ms/us); normalise to ns so the
    # parquet round trip and every downstream join see one index dtype
    spy.index = pd.DatetimeIndex(spy.index).as_unit("ns")
    spy = spy[~spy.index.duplicated(keep="first")]
    panel = adjust_ohlc(spy)
    idx = panel.index
    lim = cfg["data"]["ffill_limit"]

    # --- returns (log, dividend-adjusted) ---------------------------------
    panel["ret"] = np.log(panel["close"]).diff()                       # close-to-close
    panel["ret_on"] = np.log(panel["open"] / panel["close"].shift(1))  # overnight
    panel["ret_id"] = np.log(panel["close"] / panel["open"])           # open-to-close
    # identity check: ret = ret_on + ret_id
    gap = (panel["ret"] - panel["ret_on"] - panel["ret_id"]).abs().max()
    assert gap < 1e-10, f"return decomposition broken ({gap})"

    # --- implied-vol indices ----------------------------------------------
    for key in ("vix", "vix9d", "vix3m"):
        if key not in raw:
            continue
        s = raw[key].iloc[:, 0].sort_index()
        s.index = pd.DatetimeIndex(s.index).as_unit("ns")
        s = s[~s.index.duplicated(keep="last")].dropna()
        on_cal = s.reindex(idx)
        filled = on_cal.ffill(limit=lim["vix"])
        panel[key] = filled
        panel[f"{key}_ffilled"] = on_cal.isna() & filled.notna()
        # annualised implied variance in decimal units, e.g. VIX 20 -> 0.04
        panel[f"{key}_var"] = (filled / 100.0) ** 2

    # --- risk-free rate -----------------------------------------------------
    r = raw["rate_3m"].iloc[:, 0].sort_index().dropna()
    r.index = pd.DatetimeIndex(r.index).as_unit("ns")
    r = r[~r.index.duplicated(keep="last")]
    r_cal = r.reindex(idx)
    r_filled = r_cal.ffill(limit=lim["rate"])
    panel["rate"] = r_filled / 100.0                                    # decimal p.a.
    panel["rate_ffilled"] = r_cal.isna() & r_filled.notna()
    # continuously-compounded per-trading-day rate for financing in backtests
    panel["rate_daily"] = np.log1p(panel["rate"]) / cfg["targets"]["annualization"]

    d = cfg["data"]
    panel["in_sample"] = (idx >= d["sample_start"]) & (idx <= d["sample_end"])
    panel.index.name = "date"
    return panel
