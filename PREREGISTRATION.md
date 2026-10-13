# Pre-registration

Fill this in and **commit it before running Notebook 3**. The commit timestamp
is the evidence that hypotheses, metrics and the test window were fixed before
any model saw data. Do not edit it afterwards; record any deviation in a
"Deviations" section at the bottom with a date and reason.

- **Date committed:** YYYY-MM-DD
- **Author:** Nitish Singh

## Sample and splits
- Asset: SPY daily, research sample 2015-01-01 to 2025-12-31.
- Tuning: expanding-window folds validating on 2019, 2020, 2021, 2022, with purge = h and embargo = h.
- **Sealed test: 2023-01-01 to 2025-12-31.** Evaluated once, after `frozen_config.yaml` is committed.

## Targets
- Annualised realised variance over t+1..t+h, h in {5, 20}, from daily close-to-close returns.

## Hypotheses
| ID | Statement | Test | Decision rule |
|---|---|---|---|
| H1 | XGBoost has lower out-of-sample QLIKE than GARCH(1,1), EGARCH and HAR-RV | Diebold-Mariano, HAC variance with lag h-1; Model Confidence Set | Reject equal accuracy if p < 0.05 (two-sided) |
| H2 | The ML advantage (if any) differs across volatility regimes | QLIKE by regime (train-fold 33/67% thresholds), DM per regime | Report per regime; no pooled claim if the sign flips |
| H3 | Forecast-minus-implied variance spread predicts subsequent realised-minus-implied variance | Predictive regression, Newey-West SE | Slope significant at 5% |
| H4 | The variance risk premium is predictable from forecast-model features | Out-of-sample R² vs historical-mean VRP | R²_OOS > 0 with block-bootstrap 95% CI excluding 0 |
| H5 | A forecast-driven delta-hedged straddle strategy earns positive risk-adjusted returns net of costs | Net Sharpe, Deflated Sharpe Ratio using total trial count | DSR probability > 0.95 |

## Primary metric
- QLIKE on variance. Secondary: RMSE, MAE, R²_OOS vs HAR-RV and vs EWMA, Mincer-Zarnowitz.

## Benchmarks (fixed list)
- Historical volatility, EWMA (lambda = 0.94), GARCH(1,1)-t, EGARCH(1,1)-t, GJR-GARCH(1,1)-t, HAR-RV.

## Tuning budget
- Optuna, N trials per model per horizon: ___ (fix now). Every trial is logged and counted for the DSR.

## What would count as a null result
- XGBoost inside the MCS but not significantly better than HAR-RV is reported as "no evidence ML improves on HAR", not hidden.

## Deviations
(none yet)
