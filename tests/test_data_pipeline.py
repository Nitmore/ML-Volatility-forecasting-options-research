"""Unit tests for Notebook 1 building blocks. Run: pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data import synthetic, validate
from src.data.align import build_panel
from src.targets import brute_force_rv, build_targets, forward_realized_variance


@pytest.fixture(scope="module")
def cfg():
    return load_config()[0]


@pytest.fixture(scope="module")
def raw(cfg):
    d = cfg["data"]
    return synthetic.generate(d["download_start"], d["download_end"], seed=7)


# ---------------------------------------------------------------- targets --
@pytest.mark.parametrize("h", [1, 5, 20])
def test_forward_rv_matches_brute_force(h):
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 300), index=pd.bdate_range("2020-01-01", periods=300))
    r.iloc[0] = np.nan
    fast, slow = forward_realized_variance(r, h), brute_force_rv(r, h)
    pd.testing.assert_series_equal(fast, slow, check_names=False)


def test_target_uses_only_future_returns():
    """Changing r_t must not change the target at t (it starts at r_{t+1})."""
    r = pd.Series(np.full(50, 0.01), index=pd.bdate_range("2021-01-01", periods=50))
    base = forward_realized_variance(r, 5)
    r2 = r.copy()
    r2.iloc[10] = 0.5
    shocked = forward_realized_variance(r2, 5)
    assert base.iloc[10] == shocked.iloc[10]                 # own return excluded
    assert shocked.iloc[5] > base.iloc[5]                    # r_10 is in window t+1..t+5 of t=5
    assert shocked.iloc[4] == base.iloc[4]                   # t=4 window is r_5..r_9


def test_label_end_dates(raw, cfg):
    panel = build_panel(raw, cfg)
    tg = build_targets(panel, [5])
    pos = 100
    assert tg["label_end_5"].iloc[pos] == panel.index[pos + 5]
    assert tg["rv_var_5"].iloc[-5:].isna().all()
    assert tg["rv_var_5"].iloc[1:-5].notna().all()


# ---------------------------------------------------------------- panel ----
def test_panel_return_decomposition(raw, cfg):
    p = build_panel(raw, cfg)
    resid = (p["ret"] - p["ret_on"] - p["ret_id"]).dropna().abs().max()
    assert resid < 1e-10


def test_ffill_never_looks_ahead(raw, cfg):
    p = build_panel(raw, cfg)
    vix = raw["vix"].iloc[:, 0]
    filled_days = p.index[p["vix_ffilled"]]
    for d in filled_days:
        prev = vix.loc[:d].dropna()
        assert p.at[d, "vix"] == prev.iloc[-1]               # value from the past only


def test_rate_in_decimal(raw, cfg):
    p = build_panel(raw, cfg)
    assert p["rate"].max() < 0.25


# ---------------------------------------------------------------- validation
def test_synthetic_passes_validation(raw, cfg):
    report = validate.run_all(raw, cfg)
    validate.assert_no_failures(report)


def test_validation_catches_corruption(raw, cfg):
    bad = {k: v.copy() for k, v in raw.items()}
    spy = bad["spy"]
    spy.iloc[50, spy.columns.get_loc("High")] = spy["Low"].iloc[50] * 0.9   # H < L
    bad["spy"] = pd.concat([spy, spy.iloc[[60]]]).sort_index()               # duplicate
    report = validate.run_all(bad, cfg)
    failed = set(report.loc[~report["passed"], "name"])
    assert "SPY: unique dates" in failed
    assert "SPY: High >= max(Open, Close)" in failed
    with pytest.raises(AssertionError):
        validate.assert_no_failures(report)


def test_validation_catches_off_event_adjustment(raw, cfg):
    bad = {k: v.copy() for k, v in raw.items()}
    bad["spy"].iloc[200, bad["spy"].columns.get_loc("AdjClose")] *= 1.01
    report = validate.run_all(bad, cfg)
    row = report.set_index("name").loc["SPY: adj-factor changes only on dividend/split dates"]
    assert not row["passed"]


# ---------------------------------------------------------------- parsers --
def test_equity_parser_handles_multiindex(monkeypatch):
    """yfinance >= 0.2.4x returns (field, ticker) MultiIndex columns."""
    import sys, types
    from src.data import acquire
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-03"], name="Date")
    fields = ["Adj Close", "Close", "Dividends", "High", "Low", "Open", "Stock Splits", "Volume"]
    df = pd.DataFrame(np.ones((2, len(fields))) * 100, index=idx,
                      columns=pd.MultiIndex.from_product([fields, ["SPY"]]))
    fake = types.SimpleNamespace(download=lambda *a, **k: df)
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    out = acquire.download_equity("SPY", "2020-01-01", "2020-01-05")
    assert list(out.columns) == ["Open", "High", "Low", "Close", "AdjClose", "Volume", "Dividends", "Splits"]
    assert out.index.name == "date"


@pytest.mark.parametrize("header", ["observation_date", "DATE"])
def test_fred_parser(monkeypatch, header):
    from src.data import acquire
    text = f"{header},VIXCLS\n2020-01-02,12.47\n2020-01-03,.\n2020-01-06,13.85\n"
    import io
    monkeypatch.setattr(acquire, "_read_csv_url", lambda url: pd.read_csv(io.StringIO(text)))
    out = acquire.download_fred("VIXCLS", "2020-01-01", "2020-01-10")
    assert out.shape == (3, 1) and np.isnan(out.iloc[1, 0]) and out.iloc[2, 0] == 13.85


def test_cboe_parser(monkeypatch):
    from src.data import acquire
    import io
    text = "DATE,OPEN,HIGH,LOW,CLOSE\n01/02/2020,11.1,11.5,10.9,11.3\n01/03/2020,12.0,13.1,11.8,12.9\n"
    monkeypatch.setattr(acquire, "_read_csv_url", lambda url: pd.read_csv(io.StringIO(text)))
    out = acquire.download_cboe_index("u", "vix9d", "2020-01-01", "2020-01-10")
    assert list(out.columns) == ["vix9d"] and out.iloc[1, 0] == 12.9


def test_acquire_falls_back_to_next_source(monkeypatch, tmp_path, cfg):
    """First source raises (like FRED's RemoteDisconnected); second succeeds and is logged."""
    from src.data import acquire
    idx = pd.bdate_range("2020-01-01", periods=5, name="date")
    ok = pd.DataFrame({"x": np.arange(5.0) + 10}, index=idx)
    calls = []

    def flaky(*a, **k):
        calls.append("cboe")
        raise ConnectionError("Remote end closed connection without response")

    monkeypatch.setattr(acquire, "download_cboe_index", flaky)
    monkeypatch.setattr(acquire, "download_fred", lambda *a, **k: ok)
    monkeypatch.setattr(acquire, "download_yahoo_close", lambda *a, **k: ok)
    monkeypatch.setattr(acquire, "download_equity", lambda *a, **k: ok)
    out = acquire.acquire_all(cfg, tmp_path)
    assert "vix" in out and calls
    assert acquire.load_manifest(tmp_path)["vix"]["source"] == "fred:VIXCLS"


def test_manual_csv_overrides_network(monkeypatch, tmp_path, cfg):
    from src.data import acquire
    (tmp_path / "manual").mkdir()
    pd.DataFrame({"observation_date": ["2020-01-02", "2020-01-03"], "DGS3MO": [1.5, 1.6]}).to_csv(
        tmp_path / "manual" / "rate_3m.csv", index=False)
    idx = pd.bdate_range("2020-01-01", periods=5, name="date")
    ok = pd.DataFrame({"x": np.arange(5.0) + 10}, index=idx)
    for fn in ("download_cboe_index", "download_fred", "download_yahoo_close", "download_equity"):
        monkeypatch.setattr(acquire, fn, lambda *a, **k: ok)
    cfg2 = {**cfg, "data": {**cfg["data"], "download_start": "2020-01-01", "download_end": "2020-01-10"}}
    out = acquire.acquire_all(cfg2, tmp_path)
    assert out["rate_3m"].iloc[:, 0].tolist() == [1.5, 1.6]
    assert acquire.load_manifest(tmp_path)["rate_3m"]["source"] == "manual:rate_3m.csv"
