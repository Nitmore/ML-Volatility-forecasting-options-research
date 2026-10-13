"""Raw data acquisition with provenance.

Every raw file is written once to data/raw/ as parquet, and a manifest entry
records source, query, download time (UTC), row count, date range and a SHA256
of the file. Re-running a notebook loads the cached file unless force=True, so
results are reproducible even if a vendor later revises history.
"""
from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

MANIFEST = "manifest.json"


# --------------------------------------------------------------------------- #
# provenance helpers
# --------------------------------------------------------------------------- #
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _update_manifest(raw_dir: Path, name: str, entry: dict) -> None:
    mpath = raw_dir / MANIFEST
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    manifest[name] = entry
    mpath.write_text(json.dumps(manifest, indent=2, default=str))


def _save(df: pd.DataFrame, raw_dir: Path, name: str, source: str, query: dict) -> Path:
    path = raw_dir / f"{name}.parquet"
    df.to_parquet(path)
    _update_manifest(
        raw_dir,
        name,
        {
            "source": source,
            "query": query,
            "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rows": int(len(df)),
            "first_date": str(df.index.min().date()) if len(df) else None,
            "last_date": str(df.index.max().date()) if len(df) else None,
            "columns": list(map(str, df.columns)),
            "sha256": sha256(path),
        },
    )
    return path


def load_manifest(raw_dir: Path) -> dict:
    mpath = raw_dir / MANIFEST
    return json.loads(mpath.read_text()) if mpath.exists() else {}


def verify_manifest(raw_dir: Path) -> pd.DataFrame:
    """Re-hash every cached raw file and compare with the manifest."""
    rows = []
    for name, entry in load_manifest(raw_dir).items():
        path = raw_dir / f"{name}.parquet"
        ok = path.exists() and sha256(path) == entry["sha256"]
        rows.append({"file": name, "rows": entry["rows"], "first": entry["first_date"],
                     "last": entry["last_date"], "downloaded_utc": entry["downloaded_utc"],
                     "hash_ok": ok})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# downloaders
# --------------------------------------------------------------------------- #
def download_equity(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Daily OHLCV + Adj Close + dividends + splits from Yahoo Finance.

    Raw OHLC are split-adjusted but NOT dividend-adjusted (Yahoo convention);
    Adj Close is split- and dividend-adjusted. We keep both and derive a
    dividend adjustment factor downstream.
    """
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=False, actions=True,
                     progress=False, threads=False)
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo returned no data for {ticker}.")
    if isinstance(df.columns, pd.MultiIndex):           # yfinance >= 0.2.4x
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Adj Close": "AdjClose", "Stock Splits": "Splits"})
    for col in ("Dividends", "Splits"):
        if col not in df:
            df[col] = 0.0
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    cols = ["Open", "High", "Low", "Close", "AdjClose", "Volume", "Dividends", "Splits"]
    return df[cols].astype(float)


def _read_csv_url(url: str, retries: int = 4, backoff: float = 2.0) -> pd.DataFrame:
    """Fetch a CSV over HTTPS with retries and exponential backoff.

    FRED and CBOE occasionally drop connections (RemoteDisconnected) or rate
    limit; a browser-like User-Agent and a few spaced retries fix most cases.
    """
    import time
    import requests

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "text/csv,*/*"}
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=(10, 60), headers=headers)
            resp.raise_for_status()
            return pd.read_csv(io.StringIO(resp.text))
        except Exception as exc:                          # noqa: BLE001
            last = exc
            if attempt < retries - 1:
                time.sleep(backoff * 2 ** attempt)
    raise RuntimeError(f"{url} failed after {retries} attempts: {last}")


def download_fred(series_id: str, start: str, end: str) -> pd.DataFrame:
    """FRED series via the public CSV endpoint (no API key needed)."""
    url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
           f"&cosd={start}&coed={end}")
    df = _read_csv_url(url)
    date_col = df.columns[0]                              # 'observation_date' or 'DATE'
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.index.name = "date"
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce")     # FRED uses '.' for missing
    return s.to_frame(series_id).loc[start:end]


def download_cboe_index(url: str, name: str, start: str, end: str) -> pd.DataFrame:
    """CBOE daily index history CSV (DATE, OPEN, HIGH, LOW, CLOSE)."""
    df = _read_csv_url(url)
    df.columns = [c.strip().upper() for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE").sort_index()
    df.index.name = "date"
    col = "CLOSE" if "CLOSE" in df.columns else df.columns[-1]   # some files are DATE,<INDEX>
    out = pd.to_numeric(df[col], errors="coerce").to_frame(name)
    return out.loc[start:end]


def download_yahoo_close(symbol: str, name: str, start: str, end: str) -> pd.DataFrame:
    """Daily close of a Yahoo index symbol (e.g. ^VIX, ^VIX9D, ^VIX3M, ^IRX)."""
    import yfinance as yf

    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo returned no data for {symbol}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    out = df[["Close"]].rename(columns={"Close": name}).astype(float)
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    out.index.name = "date"
    return out


def load_manual_csv(path: Path, name: str, start: str, end: str) -> pd.DataFrame:
    """User-supplied CSV: first column = date, last column = value."""
    df = pd.read_csv(path)
    idx = pd.to_datetime(df.iloc[:, 0])
    s = pd.to_numeric(df.iloc[:, -1], errors="coerce")
    out = pd.DataFrame({name: s.to_numpy()}, index=pd.DatetimeIndex(idx, name="date")).sort_index()
    return out.loc[start:end]


def _source_fn(src: dict, name: str, start: str, end: str):
    kind = src["type"]
    if kind == "fred":
        return lambda: download_fred(src["id"], start, end), f"fred:{src['id']}"
    if kind == "cboe":
        return lambda: download_cboe_index(src["url"], name, start, end), f"cboe:{src['url']}"
    if kind == "yahoo":
        return lambda: download_yahoo_close(src["symbol"], name, start, end), f"yahoo:{src['symbol']}"
    raise ValueError(f"unknown source type {kind}")


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def acquire_all(cfg: dict, raw_dir: Path, force: bool = False) -> dict[str, pd.DataFrame]:
    """Download (or load cached) every raw dataset.

    Each secondary series has an ordered list of sources in config.yaml; the
    first that succeeds is used and recorded in the manifest. A CSV placed at
    data/raw/manual/<name>.csv overrides all network sources. Optional series
    fail soft; required ones raise with every source's error listed.
    """
    d = cfg["data"]
    start, end = d["download_start"], d["download_end"]
    manual_dir = raw_dir / "manual"
    out: dict[str, pd.DataFrame] = {}

    plans: dict[str, tuple[bool, list]] = {
        "spy": (True, [(lambda: download_equity(d["ticker"], start, end), f"yahoo:{d['ticker']}")]),
    }
    for name, spec in d["series"].items():
        plans[name] = (spec.get("required", False),
                       [_source_fn(src, name, start, end) for src in spec["sources"]])

    for name, (required, sources) in plans.items():
        path = raw_dir / f"{name}.parquet"
        if path.exists() and not force:
            out[name] = pd.read_parquet(path)
            print(f"cached     {name:<7} {len(out[name]):>5} rows  ({load_manifest(raw_dir).get(name, {}).get('source', '?')})")
            continue
        manual = manual_dir / f"{name}.csv"
        if manual.exists():
            sources = [(lambda m=manual, n=name: load_manual_csv(m, n, start, end), f"manual:{manual.name}")]
        errors = []
        for fn, label in sources:
            try:
                df = fn()
                if df.dropna().empty:
                    raise RuntimeError("empty series")
                _save(df, raw_dir, name, label, {"start": start, "end": end})
                out[name] = df
                print(f"downloaded {name:<7} {len(df):>5} rows  {df.index.min().date()} -> "
                      f"{df.index.max().date()}  [{label}]")
                break
            except Exception as exc:                      # noqa: BLE001
                errors.append(f"{label}: {exc}")
                print(f"  {name}: {label} failed, trying next source")
        else:
            msg = f"All sources failed for '{name}':\n  " + "\n  ".join(errors)
            if required:
                raise RuntimeError(msg + f"\nWorkaround: save a CSV (date, value) to {manual}")
            print(f"optional dataset '{name}' unavailable; continuing without it")
    return out


def save_synthetic(data: dict[str, pd.DataFrame], raw_dir: Path) -> None:
    """Persist synthetic data through the same manifest path (for offline runs/tests)."""
    for name, df in data.items():
        _save(df, raw_dir, name, "synthetic", {"note": "src.data.synthetic"})
