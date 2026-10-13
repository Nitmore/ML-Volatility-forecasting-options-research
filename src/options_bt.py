"""Delta-hedged ATM straddle backtest on HISTORICAL OPTION QUOTES.

Nothing here invents option prices: entries and exits fill at the quoted bid
or ask, and daily marks use the quoted mid. Black-Scholes is used only to
compute hedge ratios and Greeks for P&L attribution (from the quoted mid's
implied volatility, or the vendor's IV/delta when supplied).

Expected chain format (one row per contract per day, long format):
    date, expiration, strike, type ('C'/'P'), bid, ask
    optional: iv, delta, underlying   (underlying = SPY close used by the vendor)

Conventions
    * Option strikes reference the RAW (not dividend-adjusted) SPY price.
    * One straddle = 1 call + 1 put, 100-share multiplier.
    * Positions are non-overlapping: a new trade opens only after the last closes.
    * Hedge: shares rebalanced at each close to make the book delta-neutral.
    * Cash accrues at the daily risk-free rate (premium received/paid, hedge
      purchases/sales).
    * Early exercise is not modelled; days where a short ITM call's extrinsic
      value is below the next dividend before expiry are FLAGGED for review.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

MULT = 100


# --------------------------------------------------------------------------- #
# Black-Scholes-Merton with continuous dividend yield
# --------------------------------------------------------------------------- #
def bs_price(S, K, T, r, q, sigma, cp):
    T = np.maximum(T, 1e-8)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    put = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp(-q * T) * stats.norm.cdf(-d1)
    return np.where(np.asarray(cp) == "C", call, put)


def bs_greeks(S, K, T, r, q, sigma, cp) -> dict:
    T = np.maximum(T, 1e-8)
    sq = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sq)
    d2 = d1 - sigma * sq
    pdf = stats.norm.pdf(d1)
    is_call = np.asarray(cp) == "C"
    delta = np.where(is_call, np.exp(-q * T) * stats.norm.cdf(d1), -np.exp(-q * T) * stats.norm.cdf(-d1))
    gamma = np.exp(-q * T) * pdf / (S * sigma * sq)
    vega = S * np.exp(-q * T) * pdf * sq                                  # per 1.00 change in sigma
    theta_c = (-S * np.exp(-q * T) * pdf * sigma / (2 * sq) - r * K * np.exp(-r * T) * stats.norm.cdf(d2)
               + q * S * np.exp(-q * T) * stats.norm.cdf(d1))
    theta_p = (-S * np.exp(-q * T) * pdf * sigma / (2 * sq) + r * K * np.exp(-r * T) * stats.norm.cdf(-d2)
               - q * S * np.exp(-q * T) * stats.norm.cdf(-d1))
    theta = np.where(is_call, theta_c, theta_p)                           # per year
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def implied_vol(price, S, K, T, r, q, cp) -> float:
    intrinsic = max(0.0, (S * np.exp(-q * T) - K * np.exp(-r * T)) * (1 if cp == "C" else -1))
    if not np.isfinite(price) or price <= intrinsic + 1e-6:
        return np.nan
    f = lambda s: float(bs_price(S, K, T, r, q, s, cp)) - price
    try:
        return optimize.brentq(f, 1e-4, 5.0, xtol=1e-7)
    except ValueError:
        return np.nan


# --------------------------------------------------------------------------- #
# chain helpers
# --------------------------------------------------------------------------- #
def prepare_chain(chain: pd.DataFrame) -> dict:
    c = chain.copy()
    c["date"] = pd.to_datetime(c["date"]).dt.normalize()
    c["expiration"] = pd.to_datetime(c["expiration"]).dt.normalize()
    c["type"] = c["type"].str.upper().str[0]
    c = c[(c["bid"] > 0) & (c["ask"] >= c["bid"])]
    c["mid"] = (c["bid"] + c["ask"]) / 2
    return {d: g.set_index(["expiration", "strike", "type"]).sort_index() for d, g in c.groupby("date")}


def pick_straddle(day: pd.DataFrame, date, spot: float, dte_min: int, dte_max: int, dte_target: int):
    exps = day.index.get_level_values("expiration").unique()
    dte = pd.Series((exps - date).days, index=exps)
    ok = dte[(dte >= dte_min) & (dte <= dte_max)]
    if ok.empty:
        return None
    exp = (ok - dte_target).abs().idxmin()
    sub = day.loc[exp]
    strikes = sub.index.get_level_values("strike").unique()
    both = [k for k in strikes if (k, "C") in sub.index and (k, "P") in sub.index]
    if not both:
        return None
    k = min(both, key=lambda x: abs(x - spot))
    return exp, k


# --------------------------------------------------------------------------- #
# backtest
# --------------------------------------------------------------------------- #
def run_straddle_backtest(chain_by_date: dict, spot: pd.Series, rate: pd.Series, signal: pd.Series,
                          div_yield: float = 0.013, dte_range=(30, 45), dte_target: int = 37,
                          hold_days: int = 20, min_dte_exit: int = 5, hedge_cost_bps: float = 0.5,
                          hedge_band: float = 0.0, dividends: pd.Series | None = None):
    """Trade +1 (long) / -1 (short) straddles when the signal is non-zero.

    hedge_band: re-hedge only when |book delta| exceeds this many shares per
    straddle (0 = hedge every day).
    Returns (trades DataFrame, daily P&L DataFrame with attribution, flags list).
    """
    dates = [d for d in signal.index if d in chain_by_date and d in spot.index]
    trades, daily, flags = [], [], []
    i = 0
    while i < len(dates):
        d0 = dates[i]
        side = float(signal.loc[d0])
        if side == 0:
            i += 1
            continue
        S0 = float(spot.loc[d0])
        pick = pick_straddle(chain_by_date[d0], d0, S0, *dte_range, dte_target)
        if pick is None:
            i += 1
            continue
        exp, K = pick
        legs = [(exp, K, "C"), (exp, K, "P")]
        q0 = chain_by_date[d0].loc[legs]
        entry_px = float(q0["ask"].sum() if side > 0 else q0["bid"].sum())
        mid0 = float(q0["mid"].sum())
        cash = -side * entry_px * MULT                      # long pays, short receives
        shares, prev = 0.0, None
        spread_cost = abs(entry_px - mid0) * MULT
        hedge_cost = 0.0
        attrib_rows = []
        closed = False
        j = i
        while j < len(dates):
            d = dates[j]
            day = chain_by_date[d]
            if not all(l in day.index for l in legs):
                if prev is not None and (j - i >= hold_days or (exp - d).days <= min_dte_exit):
                    break                                   # due to close but unquoted: force-close below
                j += 1
                continue
            S = float(spot.loc[d])
            T = max((exp - d).days, 0) / 365.0
            r = float(rate.loc[d]) if d in rate.index and np.isfinite(rate.loc[d]) else 0.0
            q = day.loc[legs]
            mid = float(q["mid"].sum())
            if "iv" in q and q["iv"].notna().all():
                ivs = q["iv"].to_numpy(float)
            else:
                ivs = np.array([implied_vol(float(q.loc[l, "mid"]), S, K, T, r, div_yield, l[2]) for l in legs])
            ivs = np.where(np.isfinite(ivs), ivs, np.nanmean(ivs) if np.isfinite(ivs).any() else 0.2)
            g = bs_greeks(S, K, T, r, div_yield, ivs, np.array(["C", "P"]))
            if "delta" in q and q["delta"].notna().all():
                g["delta"] = q["delta"].to_numpy(float)
            book_delta = side * float(np.sum(g["delta"])) * MULT
            if prev is not None:
                dt_days = (d - prev["d"]).days
                cash *= np.exp(prev["r"] * dt_days / 365.0)
                dS = S - prev["S"]
                opt_pnl = side * (mid - prev["mid"]) * MULT
                hedge_pnl = shares * dS
                attrib_rows.append({
                    "date": d, "trade_entry": d0, "opt_mid_pnl": opt_pnl, "hedge_pnl": hedge_pnl,
                    "delta_pnl": side * prev["delta"] * dS * MULT,
                    "gamma_pnl": side * 0.5 * prev["gamma"] * dS ** 2 * MULT,
                    "vega_pnl": side * prev["vega_k"] @ (ivs - prev["ivs"]) * MULT,
                    "theta_pnl": side * prev["theta"] * dt_days / 365.0 * MULT,
                })
                a = attrib_rows[-1]
                a["residual"] = a["opt_mid_pnl"] - a["delta_pnl"] - a["gamma_pnl"] - a["vega_pnl"] - a["theta_pnl"]
            if prev is None:
                first_ivs = ivs
            held = j - i
            dte = (exp - d).days
            closing = held >= hold_days or dte <= min_dte_exit or j == len(dates) - 1
            if closing:
                exit_px = float(q["bid"].sum() if side > 0 else q["ask"].sum())
                spread_cost += abs(exit_px - mid) * MULT
                cash += side * exit_px * MULT
                cash += shares * S                              # unwind hedge
                hedge_cost += abs(shares) * S * hedge_cost_bps / 1e4
                shares = 0.0
                pnl = cash - hedge_cost
                closed = True
                trades.append({"entry": d0, "exit": d, "side": side, "expiry": exp, "strike": K, "S0": S0,
                               "entry_px": entry_px, "exit_px": exit_px, "entry_mid": mid0, "days": held,
                               "iv_entry": float(np.mean(first_ivs)), "iv_exit": float(np.mean(ivs)),
                               "pnl": pnl, "pnl_per_premium": pnl / (mid0 * MULT),
                               "spread_cost": spread_cost, "hedge_cost": hedge_cost})
                break
            target = -book_delta
            if prev is None or abs(target - shares) > hedge_band:
                trade_sh = target - shares
                cash -= trade_sh * S
                hedge_cost += abs(trade_sh) * S * hedge_cost_bps / 1e4
                shares = target
            if dividends is not None and side < 0:
                upcoming = dividends[(dividends.index > d) & (dividends.index <= exp) & (dividends > 0)]
                if len(upcoming) and S > K:
                    extrinsic = float(q.loc[(exp, K, "C"), "mid"]) - (S - K)
                    if extrinsic < float(upcoming.iloc[0]):
                        flags.append({"date": d, "trade_entry": d0, "issue": "short ITM call: extrinsic < next dividend"})
            prev = {"d": d, "S": S, "mid": mid, "r": r, "delta": float(np.sum(g["delta"])),
                    "gamma": float(np.sum(g["gamma"])), "vega_k": np.asarray(g["vega"], float),
                    "theta": float(np.sum(g["theta"])), "ivs": ivs}
            j += 1
        if not closed and prev is not None:
            # quotes for the held contracts disappeared: close at the last available quotes (flagged)
            last = chain_by_date[prev["d"]].loc[legs]
            exit_px = float(last["bid"].sum() if side > 0 else last["ask"].sum())
            spread_cost += abs(exit_px - prev["mid"]) * MULT
            cash += side * exit_px * MULT + shares * prev["S"]
            hedge_cost += abs(shares) * prev["S"] * hedge_cost_bps / 1e4
            trades.append({"entry": d0, "exit": prev["d"], "side": side, "expiry": exp, "strike": K, "S0": S0,
                           "entry_px": entry_px, "exit_px": exit_px, "entry_mid": mid0, "days": None,
                           "iv_entry": float(np.mean(first_ivs)), "iv_exit": float(np.mean(prev["ivs"])),
                           "pnl": cash - hedge_cost, "pnl_per_premium": (cash - hedge_cost) / (mid0 * MULT),
                           "spread_cost": spread_cost, "hedge_cost": hedge_cost})
            flags.append({"date": prev["d"], "trade_entry": d0, "issue": "forced close at last available quote"})
        daily.extend(attrib_rows)
        i = j + 1
    return pd.DataFrame(trades), pd.DataFrame(daily), flags


# --------------------------------------------------------------------------- #
# synthetic chain (for tests and pipeline checks ONLY)
# --------------------------------------------------------------------------- #
def synthetic_chain(spot: pd.Series, sigma_imp: float, r: float = 0.03, q: float = 0.013,
                    half_spread_pct: float = 0.0, strike_step: float = 1.0, width: float = 0.25) -> pd.DataFrame:
    """Black-Scholes-priced monthly-expiry chain around spot. Used only to test
    the engine's mechanics; never as a substitute for historical quotes."""
    exps = pd.date_range(spot.index[0], spot.index[-1] + pd.Timedelta(days=90), freq="WOM-3FRI")
    s0 = float(spot.iloc[0])
    strikes = np.arange(np.round(s0 * (1 - width)), np.round(s0 * (1 + width)) + strike_step, strike_step)
    g = pd.MultiIndex.from_product([spot.index, exps, strikes, ["C", "P"]],
                                   names=["date", "expiration", "strike", "type"]).to_frame(index=False)
    g["S"] = spot.reindex(g["date"]).to_numpy()
    g["T"] = (g["expiration"] - g["date"]).dt.days / 365.0
    g = g[(g["T"] >= 2 / 365) & (g["T"] <= 50 / 365) & (np.abs(g["strike"] / g["S"] - 1) < 0.12)]
    m = bs_price(g["S"].to_numpy(), g["strike"].to_numpy(), g["T"].to_numpy(), r, q, sigma_imp, g["type"].to_numpy())
    hs = half_spread_pct * m
    return pd.DataFrame({"date": g["date"].to_numpy(), "expiration": g["expiration"].to_numpy(),
                         "strike": g["strike"].to_numpy(), "type": g["type"].to_numpy(),
                         "bid": np.maximum(m - hs, 0.01), "ask": m + hs, "iv": sigma_imp})
