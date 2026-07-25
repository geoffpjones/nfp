"""Shared market-data readers for NFP analysis."""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
DEFAULT_MARKET_BARS_DB = WORKSPACE / "md" / "sqlite" / "market-bars-5y.db"

DEFAULT_PAIRS: Dict[str, str] = {
    "EUR/USD": "EURUSD",
    "GBP/USD": "GBPUSD",
    "USD/CAD": "USDCAD",
    "USD/JPY": "USDJPY",
}


def date_range(start_date: date, end_date: date) -> Iterable[date]:
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def cached_day_path(cache_dir: Path, pair: str, dt: date, fmt: str = "parquet") -> Path:
    return cache_dir / pair.replace("/", "_") / f"{dt.isoformat()}.{fmt}"


def _pair_symbol(pair: str) -> str:
    return DEFAULT_PAIRS.get(pair, pair.replace("/", "").replace("_", "").upper())


def _sqlite_ts(value: pd.Timestamp) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_sqlite_bars(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    frame["timestamp"] = frame["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    if "volume" not in frame.columns:
        frame["volume"] = 0
    return frame


def load_bar_window(
    market_data_path: Path,
    pair: str,
    start_ts,
    end_ts,
    *,
    columns: Iterable[str] = ("timestamp", "open", "high", "low", "close", "volume"),
) -> pd.DataFrame:
    """Load consolidated 1-minute bars from the shared SQLite store."""
    selected = list(columns)
    valid_columns = {"timestamp", "open", "high", "low", "close", "volume"}
    unknown = sorted(set(selected) - valid_columns)
    if unknown:
        raise ValueError(f"unsupported bar columns: {unknown}")

    db_path = market_data_path if market_data_path.suffix == ".db" else DEFAULT_MARKET_BARS_DB
    if not db_path.exists():
        return pd.DataFrame(columns=selected)

    sql_columns = ["ts_utc AS timestamp" if col == "timestamp" else col for col in selected]
    sql = f"""
        SELECT {", ".join(sql_columns)}
        FROM fx_bars
        WHERE symbol = ? AND ts_utc >= ? AND ts_utc < ?
        ORDER BY ts_utc ASC
    """
    params = (_pair_symbol(pair), _sqlite_ts(pd.Timestamp(start_ts)), _sqlite_ts(pd.Timestamp(end_ts)))
    with sqlite3.connect(db_path) as conn:
        frame = pd.read_sql_query(sql, conn, params=params)

    return _normalize_sqlite_bars(frame)


def load_cached_day(cache_dir: Path, pair: str, dt: date, fmt: str = "parquet") -> pd.DataFrame:
    if cache_dir.suffix == ".db":
        start = pd.Timestamp(dt)
        return load_bar_window(cache_dir, pair, start, start + pd.Timedelta(days=1))

    path = cached_day_path(cache_dir, pair, dt, fmt)
    if not path.exists():
        return pd.DataFrame()

    if fmt == "parquet":
        frame = pd.read_parquet(path)
    elif fmt == "csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"unsupported market-data cache format: {fmt}")

    if frame.empty:
        return frame

    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    if "volume" not in frame.columns:
        frame["volume"] = 0
    return frame


def latest_cached_date(cache_dir: Path, pair: str, fmt: str = "parquet") -> Optional[date]:
    if cache_dir.suffix == ".db":
        if not cache_dir.exists():
            return None
        with sqlite3.connect(cache_dir) as conn:
            row = conn.execute(
                "SELECT MAX(ts_utc) FROM fx_bars WHERE symbol = ?",
                (_pair_symbol(pair),),
            ).fetchone()
        if row is None or row[0] is None:
            return None
        return pd.Timestamp(row[0]).date()

    pair_dir = cache_dir / pair.replace("/", "_")
    if not pair_dir.exists():
        return None

    dates = []
    for path in pair_dir.glob(f"*.{fmt}"):
        try:
            dates.append(date.fromisoformat(path.stem))
        except ValueError:
            continue
    return max(dates) if dates else None


def available_date_range(market_data_path: Path, pairs: Iterable[str] = DEFAULT_PAIRS.keys()) -> Optional[tuple[date, date]]:
    db_path = market_data_path if market_data_path.suffix == ".db" else DEFAULT_MARKET_BARS_DB
    if not db_path.exists():
        return None

    symbols = [_pair_symbol(pair) for pair in pairs]
    placeholders = ",".join("?" for _ in symbols)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            f"SELECT MIN(ts_utc), MAX(ts_utc) FROM fx_bars WHERE symbol IN ({placeholders})",
            symbols,
        ).fetchone()

    if row is None or row[0] is None or row[1] is None:
        return None
    return pd.Timestamp(row[0]).date(), pd.Timestamp(row[1]).date()
