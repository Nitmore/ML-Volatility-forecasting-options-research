"""Variance-risk-premium research and a forecast-timed variance-swap backtest.

Units: implied and realised quantities are annualised VARIANCE (decimal),
e.g. VIX 20 -> 0.04. Trade P&L is expressed in VOL POINTS per unit of vega
notional, the market convention for variance swaps:

    P&L_short = (K_var - RV) / (2 * sqrt(K_var))      K_var = IV^2 at entry

For RV near K_var this equals (IV - RV_vol) to first order, so "+0.02" means
two vol points earned. A cost of c vol points is charged once per trade.

Proxy caveat (stated wherever results are reported): VIX^2 approximates the
30-day SPX variance-swap rate. It is not an executable SPY quote; it ignores
the variance-swap convexity/jump adjustment and SPX/SPY basis. Results are a
research proxy for the volatility risk premium, not a live trading record.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

EULER = 0.5772156649015329


# --------------------------------------------------------------------------- #
# H3: is there information in the forecast beyond implied variance?
# --------------------------------------------------------------------------- #
def encompassing_regression(rv, iv_var, forecast, h: int) -> dict:
    """log RV = a + b_iv log IV^2 + b_f log F  (HAC, 3h lags).

    b_f significantly > 0 => the forecast carries information the option market
    has not already priced. Also reports the univariate efficiency regressions.
    """
    import statsmodels.api as sm
    df = pd.DataFrame({"y": np.log(rv), "iv": np.log(iv_var), "f": np.log(forecast)}).dropna()
    kw = dict(cov_type="HAC", cov_kwds={"maxlags": max(3 * h, 5)})
    both = sm.OLS(df.y, sm.add_constant(df[["iv", "f"]])).fit(**kw)
    iv_only = sm.OLS(df.y, sm.add_constant(df[["iv"]])).fit(**kw)
    f_only = sm.OLS(df.y, sm.add_constant(df[["f"]])).fit(**kw)
    return {"b_iv": both.params["iv"], "t_iv": both.tvalues["iv"], "b_f": both.params["f"], "t_f": both.tvalues["f"],
            "p_f": both.pvalues["f"], "R2_both": both.rsquared, "R2_iv_only": iv_only.rsquared,
            "R2_f_only": f_only.rsquared, "b_iv_univariate": iv_only.params["iv"], "n": int(len(df))}


def spread_regression(rv, iv_var, forecast, h: int) -> dict:
    """log(RV/IV^2) = a + b log(F/IV^2): does forecast-minus-implied predict realised-minus-implied?"""
    import statsmodels.api as sm
    y = np.log(np.asarray(rv) / np.asarray(iv_var))
    x = np.log(np.asarray(forecast) / np.asarray(iv_var))
    res = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": max(3 * h, 5)})
    return {"a": res.params[0], "b": res.params[1], "t_b": res.tvalues[1], "p_b": res.pvalues[1], "R2": res.rsquared}


# --------------------------------------------------------------------------- #
# H4: out-of-sample predictability of the VRP
# --------------------------------------------------------------------------- #
def vrp_oos_r2(vrp: pd.Series, vrp_hat: pd.Series, h: int, min_hist: int = 60) -> tuple[float, pd.Series]:
    """R2_OOS of a VRP forecast against the expanding historical-mean VRP.

    The historical mean at t uses only VRPs whose windows had closed by t
    (shifted by h). Returns (R2_OOS, benchmark series).
    """
    known = vrp.shift(h)
    bench = known.expanding(min_periods=min_hist).mean()
    m = vrp.notna() & vrp_hat.notna() & bench.notna()
    r2 = 1 - np.sum((vrp[m] - vrp_hat[m]) ** 2) / np.sum((vrp[m] - bench[m]) ** 2)
    return float(r2), bench


# --------------------------------------------------------------------------- #
# H5: forecast-timed variance-swap proxy
# --------------------------------------------------------------------------- #
def trade_pnl_vol(iv_var: pd.Series, rv_var: pd.Series) -> pd.Series:
    """P&L (vol points per vega notional) of a SHORT variance swap entered at t."""
    return (iv_var - rv_var) / (2 * np.sqrt(iv_var))


def positions(signal: pd.Series, short_above: float, long_below: float | None) -> pd.Series:
    """-1 = short variance, +1 = long variance, 0 = flat. signal = log(IV^2 / forecast)."""
    pos = pd.Series(0.0, index=signal.index)
    pos[signal > short_above] = -1.0
    if long_below is not None:
        pos[signal < long_below] = 1.0
    return pos


@dataclass
class BacktestResult:
    trades: pd.DataFrame          # one row per (offset, entry date): position, pnl, exit date
    per_offset: pd.DataFrame      # summary statistics per start offset
    ladder: pd.Series             # daily P&L of the 1/h ladder, booked at exit dates
    summary: dict


def backtest(pos: pd.Series, short_pnl: pd.Series, exit_date: pd.Series, h: int, cost_vol: float = 0.005,
             size: pd.Series | None = None, ann_periods: float | None = None) -> BacktestResult:
    """Non-overlapping h-day trades for each of the h possible start offsets.

    Offset k trades on entry days k, k+h, k+2h, ... Statistics are computed per
    offset (so each series has independent, non-overlapping trades) and then
    averaged, which removes dependence on an arbitrary start date. The ladder
    (1/h of capital entering every day) gives the equity curve.
    """
    idx = pos.index
    size = pd.Series(1.0, index=idx) if size is None else size.reindex(idx).fillna(0.0)
    sign_pnl = -pos * short_pnl                    # short pos (-1) earns short_pnl; long (+1) earns -short_pnl
    gross = sign_pnl * size
    cost = cost_vol * pos.abs() * size
    net = gross - cost
    ann_periods = ann_periods or 252 / h
    rows, trades = [], []
    for k in range(h):
        sel = idx[k::h]
        t = pd.DataFrame({"offset": k, "entry": sel, "exit": exit_date.reindex(sel).to_numpy(),
                          "pos": pos.reindex(sel).to_numpy(), "size": size.reindex(sel).to_numpy(),
                          "gross": gross.reindex(sel).to_numpy(), "net": net.reindex(sel).to_numpy()})
        t = t.dropna(subset=["net", "exit"])
        trades.append(t)
        rows.append({"offset": k, **perf_stats(t["net"].to_numpy(), ann_periods),
                     "n_trades_active": int((t["pos"] != 0).sum())})
    trades = pd.concat(trades, ignore_index=True)
    per_offset = pd.DataFrame(rows).set_index("offset")
    ladder = (trades.groupby("exit")["net"].sum() / h).sort_index()
    eq = ladder.cumsum()
    summary = {
        "sharpe_mean": float(per_offset["sharpe"].mean()), "sharpe_min": float(per_offset["sharpe"].min()),
        "sharpe_max": float(per_offset["sharpe"].max()),
        "mean_pnl_per_period": float(per_offset["mean"].mean()), "hit_rate": float(per_offset["hit_rate"].mean()),
        "cvar5_per_period": float(per_offset["cvar5"].mean()),
        "worst_trade": float(trades["net"].min()),
        "max_drawdown_ladder": float((eq - eq.cummax()).min()),
        "time_in_market": float((trades["pos"] != 0).mean()),
        "share_short": float((trades["pos"] < 0).sum() / max((trades["pos"] != 0).sum(), 1)),
        "total_pnl_ladder": float(eq.iloc[-1]) if len(eq) else 0.0,
    }
    return BacktestResult(trades, per_offset, ladder, summary)


def perf_stats(x: np.ndarray, ann_periods: float) -> dict:
    import warnings
    warnings.filterwarnings("ignore", message="Precision loss occurred in moment calculation")
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or np.std(x, ddof=1) == 0:
        return {"mean": float(np.mean(x)) if len(x) else 0.0, "sharpe": 0.0, "sortino": 0.0, "hit_rate": 0.0,
                "cvar5": 0.0, "skew": 0.0, "kurt": 3.0, "n": len(x)}
    sd = np.std(x, ddof=1)
    down = np.sqrt(np.mean(np.minimum(x, 0) ** 2))
    q = np.quantile(x, 0.05)
    return {"mean": float(x.mean()), "sharpe": float(x.mean() / sd * np.sqrt(ann_periods)),
            "sortino": float(x.mean() / down * np.sqrt(ann_periods)) if down > 0 else np.inf,
            "hit_rate": float(np.mean(x[x != 0] > 0)) if np.any(x != 0) else 0.0,
            "cvar5": float(x[x <= q].mean()), "skew": float(stats.skew(x)),
            "kurt": float(stats.kurtosis(x, fisher=False)), "n": int(len(x))}


# --------------------------------------------------------------------------- #
# Probabilistic and Deflated Sharpe Ratio (Bailey & Lopez de Prado 2012, 2014)
# --------------------------------------------------------------------------- #
def psr(sr: float, n: int, skew: float, kurt: float, sr_star: float = 0.0) -> float:
    """P(true per-period Sharpe > sr_star). sr is PER PERIOD (not annualised); kurt is raw (normal = 3)."""
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    return float(stats.norm.cdf((sr - sr_star) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(sr_var: float, n_trials: int) -> float:
    """Expected maximum per-period Sharpe among n_trials skill-less strategies."""
    if n_trials <= 1:
        return 0.0
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))
    return float(np.sqrt(sr_var) * ((1 - EULER) * z1 + EULER * z2))


def dsr(sr: float, n: int, skew: float, kurt: float, trial_srs: np.ndarray) -> dict:
    """Deflated Sharpe: PSR against the Sharpe expected from the best of N skill-less trials."""
    trial_srs = np.asarray(trial_srs, float)
    sr0 = expected_max_sharpe(float(np.var(trial_srs, ddof=1)), len(trial_srs))
    return {"sr_star": sr0, "dsr": psr(sr, n, skew, kurt, sr0), "n_trials": int(len(trial_srs))}
