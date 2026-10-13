# ML Volatility Forecasting & Options Research (SPY, 2015–2025)

Can machine-learning models improve multi-horizon realised-volatility forecasts
over GARCH-family and HAR benchmarks, and do those forecasts identify mispriced
volatility in SPY options?

## Notebooks

| # | Notebook | Answers | Key outputs |
|---|---|---|---|
| 1 | `01_data_validation_targets` | Is the data clean and the target leak-free? | validated panel, targets, data dictionary |
| 2 | `02_features_leakage` | 61 features, provably observable at t | features, leakage audit, ablation sets |
| 3 | `03_models_walkforward` | 17 models, purged walk-forward, frozen config | out-of-sample forecasts, `frozen_config.yaml` |
| 4 | `04_evaluation` | **H1, H2**: does ML beat HAR-X? does it depend on regime? | DM, MCS, regimes, stress, ablation, SHAP, intervals |
| 5 | `05_vrp_trading` | **H3, H4, H5**: VRP, information beyond IV, forecast-timed trading | VRP study, strategy test, DSR, options engine |

Logic lives in `src/` with unit tests (`pytest -q`); notebooks orchestrate and interpret.
Run them in order. Each notebook verifies the SHA256 of its inputs.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest -q                     # all tests should pass before running notebooks
jupyter lab                   # open notebooks/01_data_validation_targets.ipynb, Run All
```

Notebook 1 downloads from Yahoo Finance, FRED and CBOE on the first run and
caches raw files in `data/raw/` with a SHA256 manifest. Later runs load the cache.

**Offline / pipeline test:** set `VOLRES_DATA_MODE=synthetic` before starting
Jupyter. Everything runs on simulated GJR-GARCH data and writes to separate
`_synthetic` folders. Synthetic numbers are never research results.

## Research protocol

1. `config.yaml` is the single source of truth for dates, horizons and splits.
2. Exploratory analysis uses 2015–2022 only.
3. `PREREGISTRATION.md` is committed before Notebook 3.
4. The 2023–2025 test set is evaluated once, after `frozen_config.yaml` is committed.
5. Every Optuna trial and strategy variant is logged; the count feeds the Deflated Sharpe Ratio.
6. The trading rule is chosen on 2019–2022 and frozen (`strategy_frozen.yaml`) before the test run.

## Options data (optional)

Notebook 5, Part E, runs a delta-hedged SPY straddle backtest on historical bid/ask
quotes if a chain is saved at `data/raw/options/spy_chain.parquet` (columns:
`date, expiration, strike, type, bid, ask`, optional `iv, delta`). Without it, H5 is
tested on a variance-swap proxy (VIX² strike), clearly labelled as such.

## Data sources and licensing

Yahoo Finance (via `yfinance`), FRED (VIXCLS, VXVCLS, DGS3MO), CBOE (VIX9D).
Raw data is not committed; check each provider's terms before redistributing.
VIX is computed from SPX options, not SPY options; it is used as an implied-vol
proxy and that limitation is stated wherever it matters.
