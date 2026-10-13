"""Forecasting models. Every forecast is annualised variance over t+1..t+h.

Level 1  naive      : historical (random walk) variance, EWMA (RiskMetrics)
Level 2  econometric: GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1), all Student-t;
                      HAR and HAR-X (log variance, OLS)
Level 3  ML         : Random Forest, XGBoost on log variance
Level 4  uncertainty: gradient-boosted quantiles (10/50/90%) of log variance

Log-target models predict E[log RV]; exp() of that is the conditional median,
which is biased low for the mean. A lognormal correction exp(var(e)/2),
with e the residuals on a held-out calibration block at the end of each
training window, converts it to a mean forecast. Notebook 1 showed log RV is
near-Gaussian (excess kurtosis ~0.1), so the parametric form is valid and has
far lower sampling error than Duan's nonparametric mean(exp(e)) on 252 rows.
Quantile models need no correction: quantiles survive exp().
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .metrics import qlike
from .validation import Fold, inner_calibration_split

ANN = 252


# --------------------------------------------------------------------------- #
# Level 1: naive benchmarks
# --------------------------------------------------------------------------- #
def hist_forecast(ret: pd.Series, h: int) -> pd.Series:
    """Random-walk forecast: next h days' variance = last h days' realised variance."""
    return ret.pow(2).rolling(h, min_periods=h).mean() * ANN


def ewma_forecast(ret: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics EWMA: s2_{t+1} = lam*s2_t + (1-lam)*r_t^2, flat across horizons."""
    r = ret.dropna()
    s2 = np.empty(len(r))
    s2[0] = np.mean(r.iloc[:20] ** 2)
    rv = r.to_numpy()
    for t in range(1, len(r)):
        s2[t] = lam * s2[t - 1] + (1 - lam) * rv[t - 1] ** 2
    # s2[t] uses returns up to t-1; the forecast made at the close of t needs r_t too
    nxt = lam * s2 + (1 - lam) * rv ** 2
    return pd.Series(nxt * ANN, index=r.index).reindex(ret.index)


# --------------------------------------------------------------------------- #
# Level 2a: GARCH family (arch package)
# --------------------------------------------------------------------------- #
GARCH_SPECS = {
    "GARCH": dict(vol="GARCH", p=1, o=0, q=1, method="analytic"),
    "GJR": dict(vol="GARCH", p=1, o=1, q=1, method="analytic"),
    "EGARCH": dict(vol="EGARCH", p=1, o=1, q=1, method="simulation"),
}


def garch_forecasts(ret: pd.Series, spec_name: str, eval_dates: pd.DatetimeIndex, horizons: list[int],
                    fit_start: str, simulations: int = 500, seed: int = 0):
    """Monthly expanding-window refit; daily causal forecasts in between.

    At the first evaluation date of each month the model is re-estimated on all
    returns strictly before that date. Within the month the parameters are held
    fixed and the conditional-variance recursion is filtered forward, so the
    forecast at t uses returns up to and including t, and nothing after.
    """
    from arch import arch_model

    spec = dict(GARCH_SPECS[spec_name])
    method = spec.pop("method")
    r = (ret.dropna() * 100).loc[fit_start:]                      # percent returns: better optimiser scaling
    eval_dates = pd.DatetimeIndex(eval_dates)
    out = pd.DataFrame(index=eval_dates, columns=horizons, dtype=float)
    params_log, prev = [], None
    for i, (_, month) in enumerate(pd.Series(eval_dates, index=eval_dates).groupby(eval_dates.to_period("M"))):
        start, end = month.iloc[0], month.iloc[-1]
        am = arch_model(r.loc[:start - pd.Timedelta(days=1)], mean="Constant", dist="t", **spec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(disp="off", starting_values=prev, options={"maxiter": 500})
        ok = res.convergence_flag == 0
        if ok or prev is None:
            prev = res.params.to_numpy()
        params_log.append({"refit_date": start, "converged": ok, **dict(zip(res.params.index, prev))})
        fixed = arch_model(r.loc[:end], mean="Constant", dist="t", **spec).fix(prev)
        kw = {}
        if method == "simulation":
            kw = dict(method="simulation", simulations=simulations, random_state=np.random.RandomState(seed + i))
        fc = fixed.forecast(horizon=max(horizons), start=start, reindex=False, **kw).variance
        fc = fc.loc[month.index]
        for h in horizons:
            out.loc[month.index, h] = fc.iloc[:, :h].sum(axis=1).to_numpy() * ANN / h / 1e4
    return out, pd.DataFrame(params_log).set_index("refit_date")


# --------------------------------------------------------------------------- #
# Level 2b / 3: log-variance regressions with smearing
# --------------------------------------------------------------------------- #
class LogVarModel:
    """Fit an estimator on log RV; predict mean variance via a lognormal correction.

    The correction is estimated out-of-sample: the estimator is first fit on the
    training window minus its last `n_calib` rows (purged); residuals e on that
    calibration block give exp(var(e)/2); the estimator is then refit
    on the full training window. Using held-out residuals matters for trees,
    whose in-sample residuals are far too small.
    """

    def __init__(self, factory, n_calib: int = 252):
        self.factory, self.n_calib = factory, n_calib

    def fit(self, X: pd.DataFrame, y_log: pd.Series, train: pd.DatetimeIndex, label_end: pd.Series):
        early, calib = inner_calibration_split(train, label_end, self.n_calib)
        m0 = self.factory().fit(X.loc[early], y_log.loc[early])
        resid = y_log.loc[calib].to_numpy() - m0.predict(X.loc[calib])
        # variance only: adding mean(e) would inject ~4% sampling noise from a 252-row block
        self.smear_ = float(np.exp(0.5 * resid.var(ddof=1)))
        self.model_ = self.factory().fit(X.loc[train], y_log.loc[train])
        return self

    def predict_log(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X)

    def predict_var(self, X: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(X)) * self.smear_


def fit_predict(factory, X, y_log, fold: Fold, label_end, n_calib: int = 252) -> pd.Series:
    m = LogVarModel(factory, n_calib).fit(X, y_log, fold.train, label_end)
    return pd.Series(m.predict_var(X.loc[fold.evaluate]), index=fold.evaluate)


def make_ols():
    from sklearn.linear_model import LinearRegression
    return lambda: LinearRegression()


def make_rf(params: dict, seed: int):
    from sklearn.ensemble import RandomForestRegressor
    return lambda: RandomForestRegressor(random_state=seed, n_jobs=-1, **params)


def make_xgb(params: dict, seed: int):
    from xgboost import XGBRegressor
    return lambda: XGBRegressor(random_state=seed, n_jobs=-1, tree_method="hist", verbosity=0, **params)


def quantile_forecasts(X, y_log, fold: Fold, quantiles=(0.1, 0.5, 0.9), seed: int = 0, params: dict | None = None):
    """Gradient-boosted conditional quantiles of variance (exp of log-variance quantiles)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    p = dict(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=40, l2_regularization=1.0)
    p.update(params or {})
    out = {}
    for q in quantiles:
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q, random_state=seed, **p)
        m.fit(X.loc[fold.train], y_log.loc[fold.train])
        out[f"q{int(q * 100):02d}"] = np.exp(m.predict(X.loc[fold.evaluate]))
    return pd.DataFrame(out, index=fold.evaluate)


# --------------------------------------------------------------------------- #
# hyper-parameter search
# --------------------------------------------------------------------------- #
def xgb_space(trial) -> dict:
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 2, 6),
        min_child_weight=trial.suggest_float("min_child_weight", 1, 30, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    )


def rf_space(trial) -> dict:
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 500, step=100),
        max_depth=trial.suggest_int("max_depth", 3, 14),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 80, log=True),
        max_features=trial.suggest_float("max_features", 0.2, 1.0),
    )


def tune(model: str, X, y_log, rv, folds: list[Fold], label_end, n_trials: int, seed: int):
    """Optuna TPE search minimising mean validation QLIKE (variance space) across folds."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space, maker = {"XGB": (xgb_space, make_xgb), "RF": (rf_space, make_rf)}[model]

    def objective(trial):
        params = space(trial)
        losses = []
        for f in folds:
            pred = fit_predict(maker(params, seed), X, y_log, f, label_end)
            losses.append(float(np.mean(qlike(rv.loc[f.evaluate], pred))))
            trial.set_user_attr(f.name, losses[-1])
        return float(np.mean(losses))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


# --------------------------------------------------------------------------- #
# regimes (thresholds from the training window only)
# --------------------------------------------------------------------------- #
def regime_labels(trailing_logvar: pd.Series, fold: Fold, quantiles=(0.33, 0.67)) -> tuple[pd.Series, tuple]:
    lo, hi = trailing_logvar.loc[fold.train].quantile(list(quantiles)).to_numpy()
    x = trailing_logvar.loc[fold.evaluate]
    lab = pd.Series(np.where(x <= lo, "low", np.where(x <= hi, "mid", "high")), index=fold.evaluate)
    return lab, (float(lo), float(hi))
