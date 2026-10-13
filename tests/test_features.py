"""Tests for Notebook 2 building blocks. Run: pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data import synthetic
from src.data.align import build_panel
from src import features as F


@pytest.fixture(scope="module")
def panel():
    cfg, _ = load_config()
    d = cfg["data"]
    return build_panel(synthetic.generate(d["download_start"], d["download_end"], seed=3), cfg)


def _gbm_ohlc(n_days=4000, sigma=0.20, overnight_share=0.0, steps=390, seed=1):
    """Constant-vol GBM with a fine intraday grid: true annual variance = sigma^2."""
    rng = np.random.default_rng(seed)
    dvar = sigma ** 2 / 252
    on = rng.normal(0, np.sqrt(dvar * overnight_share), n_days)
    path = np.cumsum(rng.normal(0, np.sqrt(dvar * (1 - overnight_share) / steps), (n_days, steps)), axis=1)
    path = np.concatenate([np.zeros((n_days, 1)), path], axis=1)
    c_prev = 100 * np.exp(np.concatenate([[0], np.cumsum(on + path[:, -1])[:-1]]))
    o = c_prev * np.exp(on)
    idx = pd.bdate_range("2000-01-03", periods=n_days)
    return (pd.Series(o, idx), pd.Series(o * np.exp(path.max(1)), idx),
            pd.Series(o * np.exp(path.min(1)), idx), pd.Series(o * np.exp(path[:, -1]), idx))


@pytest.mark.parametrize("est", ["parkinson", "garman_klass", "rogers_satchell", "yang_zhang"])
def test_range_estimators_unbiased_without_gaps(est):
    # A fine intraday grid matters: with a 390-step grid, range estimators run
    # ~5-8% low (the sampled high/low understate the true extremes).
    o, h, l, c = _gbm_ohlc(n_days=2000, overnight_share=0.0, steps=2000)
    fn = getattr(F, est)
    v = fn(h, l, 60) if est == "parkinson" else fn(o, h, l, c, 60)
    assert abs(v.mean() / 0.04 - 1) < 0.05


def test_yang_zhang_captures_overnight_variance():
    """With 40% overnight variance, range-only estimators miss it; Yang-Zhang does not."""
    o, h, l, c = _gbm_ohlc(n_days=2000, overnight_share=0.4, steps=2000, seed=2)
    yz = F.yang_zhang(o, h, l, c, 60).mean() / 0.04
    gk = F.garman_klass(o, h, l, c, 60).mean() / 0.04
    assert abs(yz - 1) < 0.05
    assert gk < 0.68                                    # ~ (1 - 0.4) of the true variance


def test_no_feature_leaks(panel):
    audit = F.audit_leakage(F.build_features, panel, n_dates=8, seed=1)
    assert audit["passed"].all(), audit.loc[~audit["passed"], "leaking_features"]


def test_audit_catches_a_planted_leak(panel):
    def leaky(p):
        X, cat = F.build_features(p)
        X["future_ret"] = p["ret"].shift(-1)            # uses tomorrow's return
        return X, cat
    audit = F.audit_leakage(leaky, panel, n_dates=5, seed=2)
    assert not audit["passed"].any()
    assert all("future_ret" in cols for cols in audit["leaking_features"])


def test_catalog_complete(panel):
    X, cat = F.build_features(panel)
    assert list(X.columns) == list(cat.index)
    assert set(cat["group"]) <= set(F.GROUPS)


def test_no_nan_after_warmup(panel):
    X, cat = F.build_features(panel)
    warm = int(cat["lookback_days"].max()) + 1
    assert not X.iloc[warm:].isna().any().any()
