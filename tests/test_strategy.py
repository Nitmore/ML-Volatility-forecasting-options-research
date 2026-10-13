"""Tests for Notebook 5: VRP tools, variance-swap proxy backtest, DSR, options engine."""
import numpy as np
import pandas as pd
import pytest

from src import options_bt as O
from src import strategy as S


# ---------------------------------------------------------------- VRP / H3 / H4
def test_encompassing_detects_extra_information():
    rng = np.random.default_rng(0)
    n = 1500
    true = rng.normal(-4, 0.5, n)
    iv = np.exp(true + 0.2 + rng.normal(0, 0.4, n))          # implied: biased + noisy view of true variance
    good_f = np.exp(true + rng.normal(0, 0.3, n))             # forecast with independent information
    rv = np.exp(true + rng.normal(0, 0.3, n))
    r = S.encompassing_regression(rv, iv, good_f, h=1)
    assert r["b_f"] > 0.3 and r["p_f"] < 0.01
    noise = np.exp(rng.normal(-4, 0.5, n))                    # forecast with no information at all
    r2 = S.encompassing_regression(rv, iv, noise, h=1)
    assert r2["p_f"] > 0.01 and abs(r2["b_f"]) < 0.1


def test_vrp_oos_r2_uses_only_known_history():
    idx = pd.bdate_range("2019-01-01", periods=400)
    vrp = pd.Series(np.r_[np.zeros(200), np.ones(200)], index=idx)
    _, bench = S.vrp_oos_r2(vrp, vrp, h=20, min_hist=10)
    assert bench.iloc[210] == 0                       # vrp jumps at 200 but is only known from 220
    assert bench.iloc[221] > 0


# ---------------------------------------------------------------- backtest
def test_trade_pnl_units():
    pnl = S.trade_pnl_vol(pd.Series([0.04]), pd.Series([0.0225]))   # IV 20, RV 15
    assert abs(pnl.iloc[0] - 0.04375) < 1e-12                        # ~ 4.4 vol points


def test_backtest_signs_costs_and_offsets():
    idx = pd.bdate_range("2023-01-02", periods=100)
    short_pnl = pd.Series(0.02, index=idx)                          # short always earns 2 vol pts
    exit_d = pd.Series(idx, index=idx).shift(-5)
    short = S.backtest(pd.Series(-1.0, index=idx), short_pnl, exit_d, h=5, cost_vol=0.005)
    long_ = S.backtest(pd.Series(1.0, index=idx), short_pnl, exit_d, h=5, cost_vol=0.005)
    flat = S.backtest(pd.Series(0.0, index=idx), short_pnl, exit_d, h=5, cost_vol=0.005)
    assert np.allclose(short.trades["net"], 0.015)
    assert np.allclose(long_.trades["net"], -0.025)
    assert flat.trades["net"].abs().sum() == 0
    assert set(short.trades.offset) == set(range(5))
    # each offset uses non-overlapping trades
    for _, g in short.trades.groupby("offset"):
        assert (g["entry"].iloc[1:].to_numpy() >= g["exit"].iloc[:-1].to_numpy()).all()


def test_positions_thresholds():
    sig = pd.Series([-0.5, -0.1, 0.05, 0.4])
    assert S.positions(sig, 0.1, -0.3).tolist() == [1.0, 0.0, 0.0, -1.0]
    assert S.positions(sig, 0.1, None).tolist() == [0.0, 0.0, 0.0, -1.0]


# ---------------------------------------------------------------- PSR / DSR
def test_psr_and_dsr_behave():
    assert S.psr(0.0, 100, 0, 3) == pytest.approx(0.5)
    assert S.psr(0.3, 100, 0, 3) > 0.99
    rng = np.random.default_rng(1)
    trials = rng.normal(0, 0.1, 50)
    d = S.dsr(0.3, 100, 0, 3, trials)
    assert d["sr_star"] > 0 and d["dsr"] < S.psr(0.3, 100, 0, 3)   # deflation lowers confidence
    # expected max of N iid N(0, s^2) Sharpes ~ simulated mean of the max
    sims = rng.normal(0, 0.1, (4000, 50)).max(axis=1).mean()
    assert abs(S.expected_max_sharpe(0.01, 50) - sims) < 0.02


# ---------------------------------------------------------------- options engine
def test_implied_vol_roundtrip():
    for cp in ("C", "P"):
        p = float(O.bs_price(500, 505, 30 / 365, 0.04, 0.013, 0.18, cp))
        assert O.implied_vol(p, 500, 505, 30 / 365, 0.04, 0.013, cp) == pytest.approx(0.18, abs=1e-5)


def _gbm(n, sigma, seed, s0=500.0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    r = rng.normal(-0.5 * sigma ** 2 / 252, sigma / np.sqrt(252), n)
    return pd.Series(s0 * np.exp(np.cumsum(r)), index=idx)


@pytest.mark.parametrize("real_vol, expect", [(0.30, "positive"), (0.10, "negative")])
def test_long_straddle_pnl_tracks_realised_vs_implied(real_vol, expect):
    """Hedged long straddle at 20% implied earns when realised > implied, loses when below."""
    pnls = []
    for seed in range(6):
        spot = _gbm(60, real_vol, seed)
        chain = O.prepare_chain(O.synthetic_chain(spot, 0.20, r=0.0, q=0.0))
        sig = pd.Series(0.0, index=spot.index); sig.iloc[0] = 1.0
        tr, _, _ = O.run_straddle_backtest(chain, spot, pd.Series(0.0, index=spot.index), sig,
                                           div_yield=0.0, hedge_cost_bps=0.0)
        pnls.append(tr["pnl"].iloc[0])
    assert (np.mean(pnls) > 0) if expect == "positive" else (np.mean(pnls) < 0)


def test_fair_implied_gives_small_pnl_and_attribution_adds_up():
    pnls, prem = [], []
    for seed in range(10):
        spot = _gbm(60, 0.20, 100 + seed)
        chain = O.prepare_chain(O.synthetic_chain(spot, 0.20, r=0.0, q=0.0))
        sig = pd.Series(0.0, index=spot.index); sig.iloc[0] = 1.0
        tr, daily, _ = O.run_straddle_backtest(chain, spot, pd.Series(0.0, index=spot.index), sig,
                                               div_yield=0.0, hedge_cost_bps=0.0)
        pnls.append(tr["pnl"].iloc[0]); prem.append(tr["entry_mid"].iloc[0] * 100)
        # option mid P&L is explained by Greeks up to a small residual
        expl = daily[["delta_pnl", "gamma_pnl", "vega_pnl", "theta_pnl"]].sum(axis=1)
        assert np.abs(daily["residual"]).sum() < 0.25 * np.abs(daily["opt_mid_pnl"]).sum() + 1e-9
        assert np.allclose(daily["opt_mid_pnl"], expl + daily["residual"])
    assert abs(np.mean(pnls)) < 0.15 * np.mean(prem)          # fair price: no systematic edge


def test_bid_ask_costs_reduce_pnl():
    spot = _gbm(60, 0.20, 7)
    sig = pd.Series(0.0, index=spot.index); sig.iloc[0] = 1.0
    res = {}
    for hs in (0.0, 0.03):
        chain = O.prepare_chain(O.synthetic_chain(spot, 0.20, r=0.0, q=0.0, half_spread_pct=hs))
        tr, _, _ = O.run_straddle_backtest(chain, spot, pd.Series(0.0, index=spot.index), sig, div_yield=0.0)
        res[hs] = tr["pnl"].iloc[0]
    assert res[0.03] < res[0.0]
