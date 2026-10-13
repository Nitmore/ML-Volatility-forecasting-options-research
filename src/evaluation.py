"""Forecast evaluation: significance tests, combinations, calibration, attribution.

All tests respect the overlap in h-day targets: forecasts made on consecutive
days share up to h-1 returns, so loss differentials are serially correlated
by construction. Every standard error is HAC (Newey-West, Bartlett kernel)
with at least h lags, and every bootstrap uses blocks longer than h.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import qlike


# --------------------------------------------------------------------------- #
# HAC helpers
# --------------------------------------------------------------------------- #
def nw_variance(x: np.ndarray, lags: int) -> float:
    """Newey-West long-run variance of the mean of x (Bartlett kernel)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    s = x @ x / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        s += 2 * w * (x[k:] @ x[:-k]) / n
    return max(s, 1e-300) / n


def _fixed_b_cv(b: float) -> float:
    """Kiefer-Vogelsang (2005) 5% two-sided critical value for a Bartlett HAC t-test, b = bandwidth / n."""
    return 1.9600 + 2.9694 * b + 0.4160 * b ** 2 - 0.5324 * b ** 3


def diebold_mariano(loss_a, loss_b, h: int) -> dict:
    """DM test of equal expected loss, robust to overlapping h-day targets.

    d = loss_a - loss_b; negative mean => model a is more accurate.
    Long-run variance: Newey-West (Bartlett) with bandwidth 3h (min 5).
    Inference: fixed-b critical values (Kiefer-Vogelsang 2005); the statistic is
    rescaled so that |stat| > 1.96 <=> rejection at 5%, and p-values come from
    N(0,1) on the rescaled statistic (exact at 5%, approximate elsewhere).
    Simulated size with MA(19) differentials and n = 750 is ~5-6.5%,
    versus ~11% for plain HAC with h lags and ~68% for an iid t-test
    (see tests/test_evaluation.py).
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    bw = max(3 * h, 5)
    t = d.mean() / np.sqrt(nw_variance(d, bw))
    stat = t * 1.96 / _fixed_b_cv(bw / n)
    p = 2 * stats.norm.sf(abs(stat))
    return {"mean_diff": float(d.mean()), "dm_stat": float(stat), "p_value": float(p), "n": n, "hac_lags": bw}


def dm_table(losses: pd.DataFrame, benchmark: str, h: int) -> pd.DataFrame:
    rows = {}
    for m in losses.columns:
        if m == benchmark:
            continue
        r = diebold_mariano(losses[m], losses[benchmark], h)
        r["loss_ratio"] = float(losses[m].mean() / losses[benchmark].mean())
        rows[m] = r
    out = pd.DataFrame(rows).T
    out["verdict"] = np.select(
        [(out.p_value < 0.05) & (out.mean_diff < 0), (out.p_value < 0.05) & (out.mean_diff > 0)],
        [f"better than {benchmark}", f"worse than {benchmark}"], "no significant difference")
    return out.sort_values("loss_ratio")


# --------------------------------------------------------------------------- #
# Model Confidence Set and bootstrap intervals
# --------------------------------------------------------------------------- #
def model_confidence_set(losses: pd.DataFrame, alpha: float = 0.10, block: int = 40,
                         reps: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Hansen-Lunde-Nason MCS (range statistic, stationary bootstrap)."""
    from arch.bootstrap import MCS
    m = MCS(losses, size=alpha, reps=reps, block_size=block, method="R", bootstrap="stationary", seed=seed)
    m.compute()
    out = m.pvalues.rename(columns={"Pvalue": "mcs_pvalue"})
    out["in_MCS"] = out.index.isin(m.included)
    out["mean_loss"] = losses.mean().reindex(out.index)
    return out.sort_values("mean_loss")


def bootstrap_ratio_ci(loss_a, loss_b, block: int = 40, reps: int = 2000, seed: int = 0, level: float = 0.95):
    """Stationary-bootstrap CI for mean(loss_a) / mean(loss_b)."""
    from arch.bootstrap import StationaryBootstrap
    a, b = np.asarray(loss_a, float), np.asarray(loss_b, float)
    bs = StationaryBootstrap(block, a, b, seed=seed)
    draws = np.array([x[0][0].mean() / x[0][1].mean() for x in bs.bootstrap(reps)])
    lo, hi = np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(a.mean() / b.mean()), float(lo), float(hi)


def regime_difference_test(loss_diff: pd.Series, regimes: pd.Series, h: int) -> dict:
    """Joint test that the mean loss differential is equal across regimes (H2).

    Regress d_t on regime dummies (no constant) with HAC covariance (3h lags) and
    Wald-test equality of all regime means. One test instead of one per regime,
    so no multiple-testing inflation.
    """
    import statsmodels.api as sm
    df = pd.DataFrame({"d": loss_diff, "r": regimes}).dropna()
    D = pd.get_dummies(df["r"]).astype(float)
    res = sm.OLS(df["d"], D).fit(cov_type="HAC", cov_kwds={"maxlags": max(3 * h, 5)})
    names = list(D.columns)
    R = np.zeros((len(names) - 1, len(names)))
    for i in range(len(names) - 1):
        R[i, 0], R[i, i + 1] = 1, -1
    w = res.wald_test(R, scalar=True)
    return {"means": res.params.to_dict(), "wald_stat": float(w.statistic), "p_value": float(w.pvalue),
            "df": len(names) - 1}


def conformalize_band(q_lo, q_hi, y, q_lo_new, q_hi_new, alpha_tail: float = 0.10):
    """Per-tail split-conformal adjustment (CQR, Romano et al. 2019) in log space.

    Calibration scores come from out-of-sample forecasts made BEFORE the period
    being adjusted (here: validation-period quantile forecasts vs realised).
    Each tail is shifted so that its empirical miss rate on the calibration set
    equals `alpha_tail`.
    """
    lo_score = np.log(q_lo) - np.log(y)          # > 0 when y falls below the lower quantile
    hi_score = np.log(y) - np.log(q_hi)          # > 0 when y falls above the upper quantile
    n = len(y)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha_tail)) / n)
    a_lo = float(np.quantile(lo_score, level))
    a_hi = float(np.quantile(hi_score, level))
    return np.exp(np.log(q_lo_new) - a_lo), np.exp(np.log(q_hi_new) + a_hi), (a_lo, a_hi)


# --------------------------------------------------------------------------- #
# Mincer-Zarnowitz
# --------------------------------------------------------------------------- #
def mincer_zarnowitz(rv, f, h: int, log_space: bool = False) -> dict:
    """Regress realised on forecast; unbiased & efficient <=> a = 0, b = 1. HAC SEs."""
    import statsmodels.api as sm
    y, x = np.asarray(rv, float), np.asarray(f, float)
    if log_space:
        y, x = np.log(y), np.log(x)
    res = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": h})
    wald = res.wald_test("const = 0, x1 = 1", scalar=True)
    return {"a": float(res.params[0]), "b": float(res.params[1]), "b_se": float(res.bse[1]),
            "R2": float(res.rsquared), "wald_p(a=0,b=1)": float(wald.pvalue)}


# --------------------------------------------------------------------------- #
# Forecast combinations (causal)
# --------------------------------------------------------------------------- #
def realised_loss_known_at(loss: pd.DataFrame, h: int) -> pd.DataFrame:
    """Loss of the forecast made at s becomes known at the close of s+h; shift by h rows."""
    return loss.shift(h)


def combine(forecasts: pd.DataFrame, rv: pd.Series, h: int, window: int = 126,
            regimes: pd.Series | None = None, min_obs: int = 20) -> dict[str, pd.Series]:
    """Three combinations of variance forecasts, all using only information known at t.

    EQ      : equal weights
    TRAIL   : weights proportional to 1 / trailing mean QLIKE over `window` known losses
    REGIME  : weights proportional to 1 / mean QLIKE over all known past losses in today's regime
    """
    F = forecasts.sort_index()
    L = pd.DataFrame({m: qlike(rv.loc[F.index], F[m]) for m in F}, index=F.index)
    known = realised_loss_known_at(L, h)
    out = {"ENS_EQ": F.mean(axis=1)}

    trail = known.rolling(window, min_periods=min_obs).mean()
    w = (1 / trail).div((1 / trail).sum(axis=1), axis=0)
    w = w.fillna(1 / F.shape[1])
    out["ENS_TRAIL"] = (w * F).sum(axis=1)

    if regimes is not None:
        reg = regimes.reindex(F.index)
        wr = pd.DataFrame(np.nan, index=F.index, columns=F.columns)
        for r in reg.dropna().unique():
            mask = (reg == r).astype(float).replace(0, np.nan)
            # expanding mean of known losses on past days that were in regime r
            num = known.mul(mask, axis=0).fillna(0).cumsum()
            cnt = known.notna().mul(mask.notna(), axis=0).cumsum()
            mean_r = (num / cnt).where(cnt >= min_obs)
            sel = reg == r
            wr.loc[sel] = (1 / mean_r.loc[sel]).div((1 / mean_r.loc[sel]).sum(axis=1), axis=0)
        wr = wr.fillna(1 / F.shape[1])
        out["ENS_REGIME"] = (wr * F).sum(axis=1)
        out["_weights_regime"] = wr
    out["_weights_trail"] = w
    return out


# --------------------------------------------------------------------------- #
# Interval calibration
# --------------------------------------------------------------------------- #
def kupiec_pof(hits: np.ndarray, p: float) -> dict:
    """Kupiec proportion-of-failures LR test. Assumes independent hits; with
    overlapping h-day targets hits cluster, so treat the p-value as indicative."""
    hits = np.asarray(hits, bool)
    n, x = len(hits), int(hits.sum())
    phat = min(max(x / n, 1e-12), 1 - 1e-12)
    lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - phat) - x * np.log(phat))
    return {"expected": p, "observed": x / n, "n": n, "LR": float(lr), "p_value": float(stats.chi2.sf(lr, 1))}


def block_coverage_ci(hits, block: int, reps: int = 2000, seed: int = 0):
    """Stationary-bootstrap CI for a hit rate; valid under the overlap-induced clustering."""
    from arch.bootstrap import StationaryBootstrap
    bs = StationaryBootstrap(block, np.asarray(hits, float), seed=seed)
    draws = np.array([x[0][0].mean() for x in bs.bootstrap(reps)])
    return tuple(np.quantile(draws, [0.025, 0.975]))


def pinball(y, q, tau: float) -> float:
    y, q = np.asarray(y, float), np.asarray(q, float)
    u = y - q
    return float(np.mean(np.maximum(tau * u, (tau - 1) * u)))


# --------------------------------------------------------------------------- #
# Grouped SHAP for XGBoost (exact TreeSHAP via xgboost, no extra dependency)
# --------------------------------------------------------------------------- #
def xgb_grouped_shap(model, X: pd.DataFrame, groups: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Mean |SHAP| per feature and per feature group (signed contributions summed within a group first)."""
    import xgboost as xgb
    contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]   # drop bias column
    phi = pd.DataFrame(contrib, index=X.index, columns=X.columns)
    by_feature = phi.abs().mean()
    by_group = phi.T.groupby(groups.reindex(X.columns)).sum().T.abs().mean()
    return by_feature.sort_values(ascending=False), by_group.sort_values(ascending=False)
