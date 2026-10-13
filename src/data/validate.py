"""Data validation suite.

Each check returns a `Check` record. Severity:
  FAIL -> the pipeline must stop (assert_no_failures raises)
  WARN -> plausible data issue; inspect and document
  INFO -> expected market features worth recording (e.g. crash-day returns)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Check:
    name: str
    severity: str            # FAIL | WARN | INFO
    passed: bool
    n_issues: int
    detail: str = ""
    examples: list = field(default_factory=list)


def _dates(idx, k: int = 5) -> list[str]:
    return [str(pd.Timestamp(d).date()) for d in list(idx)[:k]]


# --------------------------------------------------------------------------- #
# equity checks
# --------------------------------------------------------------------------- #
def check_index(df: pd.DataFrame, name: str) -> list[Check]:
    dup = df.index[df.index.duplicated()]
    return [
        Check(f"{name}: unique dates", "FAIL", len(dup) == 0, len(dup), "duplicate index entries", _dates(dup)),
        Check(f"{name}: sorted dates", "FAIL", bool(df.index.is_monotonic_increasing), 0 if df.index.is_monotonic_increasing else 1),
    ]


def check_ohlc(df: pd.DataFrame) -> list[Check]:
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    nonpos = df.index[(df[["Open", "High", "Low", "Close", "AdjClose"]] <= 0).any(axis=1)]
    nan = df.index[df[["Open", "High", "Low", "Close", "AdjClose"]].isna().any(axis=1)]
    tol = 1e-8 * c
    bad_hi = df.index[h < np.maximum(o, c) - tol]
    bad_lo = df.index[l > np.minimum(o, c) + tol]
    flat = df.index[(h == l)]
    return [
        Check("SPY: no missing prices", "FAIL", len(nan) == 0, len(nan), "NaN in OHLC/AdjClose", _dates(nan)),
        Check("SPY: positive prices", "FAIL", len(nonpos) == 0, len(nonpos), "non-positive price", _dates(nonpos)),
        Check("SPY: High >= max(Open, Close)", "FAIL", len(bad_hi) == 0, len(bad_hi), "", _dates(bad_hi)),
        Check("SPY: Low <= min(Open, Close)", "FAIL", len(bad_lo) == 0, len(bad_lo), "", _dates(bad_lo)),
        Check("SPY: non-degenerate range (H > L)", "WARN", len(flat) == 0, len(flat),
              "H == L breaks log-range estimators", _dates(flat)),
    ]


def check_volume(df: pd.DataFrame) -> Check:
    z = df.index[df["Volume"] <= 0]
    return Check("SPY: positive volume", "WARN", len(z) == 0, len(z), "zero/negative volume days", _dates(z))


def check_stale(df: pd.DataFrame, max_run: int) -> Check:
    same = df["Close"].diff().eq(0)
    run = same.groupby((~same).cumsum()).cumsum()
    bad = df.index[run >= max_run]
    return Check("SPY: no stale closes", "WARN", len(bad) == 0, len(bad),
                 f"runs of >= {max_run} identical closes", _dates(bad))


def check_calendar(df: pd.DataFrame, start: str, end: str) -> list[Check]:
    """Compare against the official NYSE session calendar."""
    try:
        import pandas_market_calendars as mcal
    except ImportError:
        gaps = df.index.to_series().diff().dt.days
        bad = df.index[gaps > 4]
        return [Check("SPY: calendar gaps (heuristic)", "WARN", len(bad) == 0, len(bad),
                      "install pandas_market_calendars for an exact NYSE check", _dates(bad))]
    sched = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
    sessions = pd.DatetimeIndex(sched.index).tz_localize(None).normalize()
    sessions = sessions[(sessions >= df.index.min()) & (sessions <= df.index.max())]
    missing = sessions.difference(df.index)
    extra = df.index.difference(sessions)
    return [
        Check("SPY: no missing NYSE sessions", "WARN", len(missing) == 0, len(missing), "", _dates(missing)),
        Check("SPY: no non-session rows", "WARN", len(extra) == 0, len(extra), "", _dates(extra)),
    ]


def check_dividend_adjustment(df: pd.DataFrame, tol: float = 1e-5) -> Check:
    """Adj factor (AdjClose/Close) may only change on ex-dividend or split dates.

    Yahoo applies the backward adjustment so the factor steps up on the
    ex-date. A step anywhere else means the series was revised or corrupted.
    """
    f = df["AdjClose"] / df["Close"]
    step = (f / f.shift(1) - 1).abs()
    event = (df["Dividends"] > 0) | (df["Splits"] > 0)
    bad = df.index[(step > tol) & ~event]
    bad = bad[1:] if len(bad) and bad[0] == df.index[0] else bad
    return Check("SPY: adj-factor changes only on dividend/split dates", "WARN", len(bad) == 0,
                 len(bad), f"{int(event.sum())} dividend/split events in sample", _dates(bad))


def check_extreme_returns(df: pd.DataFrame, z: float) -> Check:
    r = np.log(df["AdjClose"]).diff().dropna()
    mad = (r - r.median()).abs().median() * 1.4826
    big = r[(r - r.median()).abs() > z * mad]
    examples = [f"{d.date()}: {v:+.2%}" for d, v in big.items()][:8]
    return Check(f"SPY: returns beyond {z} robust sigma", "INFO", True, len(big),
                 "genuine market events are kept, not winsorised", examples)


# --------------------------------------------------------------------------- #
# secondary-series checks
# --------------------------------------------------------------------------- #
def check_series(s: pd.Series, name: str, bounds: tuple[float, float] | None = None) -> list[Check]:
    out = check_index(s.to_frame(), name)
    if bounds is not None:
        lo, hi = bounds
        bad = s.index[(s < lo) | (s > hi)]
        out.append(Check(f"{name}: within [{lo}, {hi}]", "FAIL", len(bad) == 0, len(bad), "", _dates(bad)))
    return out


def check_coverage(master: pd.DatetimeIndex, s: pd.Series, name: str) -> Check:
    missing = master.difference(s.dropna().index)
    return Check(f"{name}: value on every SPY session", "WARN", len(missing) == 0, len(missing),
                 "filled forward within the configured limit at alignment", _dates(missing))


def check_rate_units(s: pd.Series) -> Check:
    mx = float(s.max())
    ok = 0 <= mx < 25  # FRED DGS3MO is in percent; decimals would be < 0.25
    return Check("rate_3m: quoted in percent (to be converted)", "FAIL", ok, 0 if ok else 1, f"max = {mx:.3f}")


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def run_all(raw: dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    v = cfg["validation"]
    spy = raw["spy"]
    checks: list[Check] = []
    checks += check_index(spy, "SPY")
    checks += check_ohlc(spy)
    checks.append(check_volume(spy))
    checks.append(check_stale(spy, v["max_stale_run"]))
    checks += check_calendar(spy, cfg["data"]["download_start"], cfg["data"]["download_end"])
    checks.append(check_dividend_adjustment(spy))
    checks.append(check_extreme_returns(spy, v["extreme_return_mad_z"]))

    vix = raw["vix"].iloc[:, 0]
    checks += check_series(vix.dropna(), "VIX", tuple(v["vix_bounds"]))
    checks.append(check_coverage(spy.index, vix, "VIX"))
    for opt in ("vix9d", "vix3m"):
        if opt in raw:
            s = raw[opt].iloc[:, 0].dropna()
            checks += check_series(s, opt.upper(), tuple(v["vix_bounds"]))
            checks.append(check_coverage(spy.index, s, opt.upper()))

    rate = raw["rate_3m"].iloc[:, 0]
    checks += check_series(rate.dropna(), "rate_3m")
    checks.append(check_rate_units(rate.dropna()))
    checks.append(check_coverage(spy.index, rate, "rate_3m"))

    return pd.DataFrame([c.__dict__ for c in checks])


def repair_ohlc(spy: pd.DataFrame, max_rel_breach: float = 0.002) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clamp tiny vendor inconsistencies (e.g. High a few cents below Close).

    Only breaches smaller than `max_rel_breach` are repaired; anything larger is
    left in place so the FAIL check still stops the pipeline. Every repaired
    row is returned in a log, which the notebook saves next to the data.
    """
    df = spy.copy()
    hi_need = df[["Open", "Close"]].max(axis=1)
    lo_need = df[["Open", "Close"]].min(axis=1)
    hi_gap = (hi_need - df["High"]) / df["Close"]
    lo_gap = (df["Low"] - lo_need) / df["Close"]
    # 1e-8 matches the FAIL check's tolerance, so float noise is never "repaired"
    fix_hi = (hi_gap > 1e-8) & (hi_gap <= max_rel_breach)
    fix_lo = (lo_gap > 1e-8) & (lo_gap <= max_rel_breach)
    log = pd.DataFrame({
        "field": ["High"] * int(fix_hi.sum()) + ["Low"] * int(fix_lo.sum()),
        "old": list(df.loc[fix_hi, "High"]) + list(df.loc[fix_lo, "Low"]),
        "new": list(hi_need[fix_hi]) + list(lo_need[fix_lo]),
    }, index=list(df.index[fix_hi]) + list(df.index[fix_lo]))
    df.loc[fix_hi, "High"] = hi_need[fix_hi]
    df.loc[fix_lo, "Low"] = lo_need[fix_lo]
    log.index.name = "date"
    return df, log.sort_index()


def assert_no_failures(report: pd.DataFrame) -> None:
    fails = report[(report["severity"] == "FAIL") & ~report["passed"]]
    if len(fails):
        raise AssertionError("Validation FAILED:\n" + fails[["name", "n_issues", "examples"]].to_string())
