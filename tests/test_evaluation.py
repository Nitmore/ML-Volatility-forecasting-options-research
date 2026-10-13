"""Tests for Notebook 4 evaluation tools. Run: pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src import evaluation as E
from src.metrics import qlike


def _ma_losses(n, h, shift, seed):
    """Two loss series whose differential is MA(h-1), like overlapping-target forecasts."""
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n + h)
    d = np.convolve(e, np.ones(h) / np.sqrt(h), mode="valid")[:n]
    base = rng.gamma(2, 0.2, n)
    return base + d * 0.2 + shift, base


def test_dm_size_under_overlap():
    """Under equal accuracy with MA(19) differentials, rejection rate at 5% should be near 5%."""
    rej = [E.diebold_mariano(*_ma_losses(750, 20, 0.0, s), h=20)["p_value"] < 0.05 for s in range(400)]
    assert np.mean(rej) < 0.09            # an iid t-test rejects ~68% of the time here


def test_dm_power():
    a, b = _ma_losses(750, 5, 0.06, 1)
    r = E.diebold_mariano(a, b, 5)
    assert r["p_value"] < 0.05 and r["mean_diff"] > 0


def test_mcs_keeps_best_drops_bad():
    rng = np.random.default_rng(0)
    L = pd.DataFrame(rng.gamma(2, 0.2, (800, 4)), columns=list("abcd"))
    L["d"] += 0.3
    out = E.model_confidence_set(L, alpha=0.1, block=10, reps=500)
    assert not out.loc["d", "in_MCS"]
    assert out["in_MCS"].sum() >= 1


def test_mincer_zarnowitz_on_unbiased_forecast():
    rng = np.random.default_rng(2)
    f = rng.gamma(4, 0.01, 3000)
    rv = f * rng.gamma(10, 0.1, 3000)      # E[rv | f] = f
    r = E.mincer_zarnowitz(rv, f, h=1)
    assert abs(r["b"] - 1) < 0.1 and r["wald_p(a=0,b=1)"] > 0.01


def test_combinations_are_causal():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2019-01-01", periods=600)
    F = pd.DataFrame(rng.gamma(4, 0.01, (600, 3)), index=idx, columns=["m1", "m2", "m3"])
    rv = pd.Series(rng.gamma(4, 0.01, 600), index=idx)
    reg = pd.Series(rng.choice(["low", "mid", "high"], 600), index=idx)
    base = E.combine(F, rv, h=20, regimes=reg)
    rv2 = rv.copy(); rv2.iloc[400:] *= 5           # outcomes from day 400 on (known from day 420)
    pert = E.combine(F, rv2, h=20, regimes=reg)
    for k in ("ENS_TRAIL", "ENS_REGIME"):
        pd.testing.assert_series_equal(base[k].iloc[:420], pert[k].iloc[:420])
        assert not np.allclose(base[k].iloc[421:], pert[k].iloc[421:])


def test_kupiec_correct_coverage():
    rng = np.random.default_rng(4)
    assert E.kupiec_pof(rng.random(5000) < 0.1, 0.1)["p_value"] > 0.01
    assert E.kupiec_pof(rng.random(5000) < 0.2, 0.1)["p_value"] < 1e-6


def test_pinball_minimised_at_true_quantile():
    rng = np.random.default_rng(5)
    y = rng.normal(size=20000)
    q_true = np.quantile(y, 0.9)
    assert E.pinball(y, np.full_like(y, q_true), 0.9) < E.pinball(y, np.full_like(y, q_true + 0.3), 0.9)


def test_grouped_shap_sums_correctly():
    from xgboost import XGBRegressor
    rng = np.random.default_rng(6)
    X = pd.DataFrame(rng.normal(size=(500, 4)), columns=["a1", "a2", "b1", "noise"])
    y = X.a1 + X.a2 + 0.5 * X.b1
    m = XGBRegressor(n_estimators=100, max_depth=3, verbosity=0).fit(X, y)
    feat, grp = E.xgb_grouped_shap(m, X, pd.Series({"a1": "A", "a2": "A", "b1": "B", "noise": "N"}))
    assert grp.index[0] == "A" and grp["N"] < 0.1 * grp["A"]


def test_regime_difference_test():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2020-01-01", periods=900)
    reg = pd.Series(np.repeat(["low", "mid", "high"], 300), index=idx)
    same = pd.Series(rng.normal(0, 1, 900), index=idx)
    diff = same + reg.map({"low": 0.0, "mid": 0.0, "high": 0.6})
    assert E.regime_difference_test(same, reg, 1)["p_value"] > 0.01
    assert E.regime_difference_test(diff, reg, 1)["p_value"] < 0.01


def test_conformal_band_restores_coverage():
    rng = np.random.default_rng(8)
    n = 4000
    mu = rng.normal(-4, 0.4, n)
    y = np.exp(mu + rng.normal(0, 0.5, n))
    lo, hi = np.exp(mu - 0.3), np.exp(mu + 0.3)          # far too narrow: true 10/90% is +/-0.64
    cal, new = slice(0, 2000), slice(2000, None)
    lo2, hi2, _ = E.conformalize_band(lo[cal], hi[cal], y[cal], lo[new], hi[new])
    cov_raw = np.mean((y[new] >= lo[new]) & (y[new] <= hi[new]))
    cov_adj = np.mean((y[new] >= lo2) & (y[new] <= hi2))
    assert cov_raw < 0.5 and abs(cov_adj - 0.8) < 0.03
