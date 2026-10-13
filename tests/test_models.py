"""Tests for Notebook 3 building blocks. Run: pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src import models as M
from src.config import load_config
from src.data import synthetic
from src.data.align import build_panel
from src.metrics import qlike
from src.targets import build_targets
from src.validation import check_fold, make_folds


@pytest.fixture(scope="module")
def data():
    cfg, _ = load_config()
    d = cfg["data"]
    panel = build_panel(synthetic.generate(d["download_start"], d["download_end"], seed=5), cfg)
    return cfg, panel, build_targets(panel, [5, 20])


# ---------------------------------------------------------------- metrics --
def test_qlike_properties():
    assert qlike([1.0], [1.0])[0] == pytest.approx(0)
    under, over = qlike([1.0], [0.5])[0], qlike([1.0], [2.0])[0]
    assert under > over > 0                      # under-predicting variance costs more


# ---------------------------------------------------------------- folds ----
@pytest.mark.parametrize("h", [5, 20])
def test_folds_are_purged(data, h):
    cfg, panel, tg = data
    folds = make_folds(panel.index, tg[f"label_end_{h}"], cfg["data"]["sample_start"], [2019, 2020, 2021, 2022], "val")
    for f in folds:
        check_fold(f, tg[f"label_end_{h}"])
        assert f.n_purged == h


# ---------------------------------------------------------------- causality
def test_ewma_is_causal(data):
    _, panel, _ = data
    r = panel["ret"]
    base = M.ewma_forecast(r)
    t = 800
    r2 = r.copy(); r2.iloc[t + 1:] *= 3
    pd.testing.assert_series_equal(base.iloc[: t + 1], M.ewma_forecast(r2).iloc[: t + 1])
    r3 = r.copy(); r3.iloc[t] += 0.05
    assert M.ewma_forecast(r3).iloc[t] != base.iloc[t]           # forecast at t uses r_t


def test_garch_forecast_is_causal(data):
    _, panel, _ = data
    r = panel["ret"]
    dates = panel.index[(panel.index >= "2020-03-01") & (panel.index <= "2020-04-30")]
    base, _ = M.garch_forecasts(r, "GJR", dates, [5, 20], "2014-06-01")
    cut = dates[10]
    r2 = r.copy(); r2.loc[r2.index > cut] *= 3
    pert, _ = M.garch_forecasts(r2, "GJR", dates, [5, 20], "2014-06-01")
    pd.testing.assert_frame_equal(base.loc[:cut], pert.loc[:cut])
    assert not np.allclose(base.loc[dates[11]:], pert.loc[dates[11]:])


def test_gjr_recovers_simulated_persistence(data):
    from arch import arch_model
    _, panel, _ = data
    res = arch_model(panel["ret"].dropna() * 100, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
    p = res.params
    persistence = p["alpha[1]"] + p["gamma[1]"] / 2 + p["beta[1]"]
    assert abs(persistence - synthetic.PERSIST) < 0.03
    assert p["gamma[1]"] > 0.05                                    # leverage effect detected


# ---------------------------------------------------------------- smearing -
def test_smearing_gives_unbiased_mean():
    rng = np.random.default_rng(0)
    n = 3000
    idx = pd.bdate_range("2010-01-01", periods=n)
    X = pd.DataFrame({"x": rng.normal(-4, 0.5, n)}, index=idx)
    y_log = X["x"] + rng.normal(0, 0.6, n)
    label_end = pd.Series(idx, index=idx).shift(-1)
    m = M.LogVarModel(M.make_ols()).fit(X, y_log, idx[:-1], label_end)
    naive = np.exp(m.predict_log(X)).mean() / np.exp(y_log).mean()
    smeared = m.predict_var(X).mean() / np.exp(y_log).mean()
    assert naive < 0.9                                            # exp(E[log]) is biased low (~ e^{-0.18})
    assert abs(smeared - 1) < 0.05


def test_regime_thresholds_use_training_only(data):
    cfg, panel, tg = data
    f = make_folds(panel.index, tg["label_end_20"], cfg["data"]["sample_start"], [2020], "val")[0]
    s = np.log(panel["ret"].pow(2).rolling(20).mean() * 252)
    lab, (lo, hi) = M.regime_labels(s, f)
    s2 = s.copy(); s2.loc[f.evaluate] += 10                       # change evaluation data only
    _, (lo2, hi2) = M.regime_labels(s2, f)
    assert (lo, hi) == (lo2, hi2)
    assert set(lab.unique()) <= {"low", "mid", "high"}
