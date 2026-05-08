#!/usr/bin/env python3
"""Utilities for fetching and caching Dukascopy 1-minute candles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import lzma
from pathlib import Path
import struct
import time
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

DUKASCOPY_BASE_URL = "https://datafeed.dukascopy.com/datafeed/{symbol}/{year}/{month:02d}/{day:02d}/BID_candles_min_1.bi5"

DEFAULT_PAIRS: Dict[str, str] = {
    "EUR/USD": "EURUSD",
    "GBP/USD": "GBPUSD",
    "USD/CAD": "USDCAD",
    "USD/JPY": "USDJPY",
}

# JPY crosses are quoted with 3 decimals in Dukascopy minute candle files.
PRICE_DIVISOR_OVERRIDES: Dict[str, float] = {
    "USDJPY": 1000.0,
}


@dataclass
class FetchResult:
    pair: str
    symbol: str
    date_value: date
    url: Optional[str]
    status: str
    rows: int
    error: Optional[str] = None


def _price_divisor(symbol: str) -> float:
    normalized = symbol.upper()
    if normalized in PRICE_DIVISOR_OVERRIDES:
        return PRICE_DIVISOR_OVERRIDES[normalized]
    if normalized.endswith("JPY"):
        return 1000.0
    return 100000.0


def _dukascopy_month(dt: date) -> int:
    # Dukascopy uses zero-based month directories (00..11).
    return dt.month - 1


def _url_candidates(symbol: str, dt: date) -> List[str]:
    # Primary path is zero-based month (expected). We keep one fallback for
    # legacy scripts that used calendar month values.
    candidates: List[Tuple[int, int]] = [(_dukascopy_month(dt), dt.day), (dt.month, dt.day)]
    seen = set()
    urls: List[str] = []
    for month, day in candidates:
        if month < 0 or month > 12:
            continue
        key = (month, day)
        if key in seen:
            continue
        seen.add(key)
        urls.append(
            DUKASCOPY_BASE_URL.format(
                symbol=symbol,
                year=dt.year,
                month=month,
                day=day,
            )
        )
    return urls


def _parse_bi5_candles(payload: bytes, dt: date, symbol: str) -> pd.DataFrame:
    raw = lzma.decompress(payload)
    if len(raw) < 24:
        return pd.DataFrame()

    divisor = _price_divisor(symbol)
    midnight = datetime(dt.year, dt.month, dt.day)
    records = []

    for offset in range(0, len(raw), 24):
        chunk = raw[offset : offset + 24]
        if len(chunk) < 24:
            break
        sec_from_midnight, open_p, high_p, low_p, close_p = struct.unpack(">IIIII", chunk[:20])
        volume, = struct.unpack(">f", chunk[20:24])
        records.append(
            {
                "timestamp": midnight + timedelta(seconds=sec_from_midnight),
                "open": open_p / divisor,
                "high": high_p / divisor,
                "low": low_p / divisor,
                "close": close_p / divisor,
                "volume": volume,
            }
        )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df

    # Keep only bars that actually belong to the requested UTC day.
    day_start = datetime(dt.year, dt.month, dt.day)
    day_end = day_start + timedelta(days=1)
    df = df[(df["timestamp"] >= day_start) & (df["timestamp"] < day_end)]
    return df.reset_index(drop=True)


class DukascopyClient:
    def __init__(
        self,
        *,
        timeout_seconds: int = 20,
        max_retries: int = 4,
        backoff_seconds: float = 1.2,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            }
        )

    def fetch_minute_candles(self, symbol: str, dt: date) -> Tuple[pd.DataFrame, FetchResult]:
        last_error: Optional[str] = None

        for url in _url_candidates(symbol, dt):
            for attempt in range(self.max_retries):
                try:
                    response = self.session.get(url, timeout=self.timeout_seconds)
                    status = response.status_code

                    if status == 404:
                        break

                    if status in (429, 500, 502, 503, 504):
                        wait_s = self.backoff_seconds * (attempt + 1)
                        time.sleep(wait_s)
                        last_error = f"HTTP {status}"
                        continue

                    if status != 200:
                        last_error = f"HTTP {status}"
                        break

                    df = _parse_bi5_candles(response.content, dt, symbol)
                    if df.empty:
                        last_error = "empty payload"
                        break

                    return df, FetchResult(
                        pair="",
                        symbol=symbol,
                        date_value=dt,
                        url=url,
                        status="ok",
                        rows=len(df),
                    )
                except requests.RequestException as exc:
                    last_error = str(exc)
                    time.sleep(self.backoff_seconds * (attempt + 1))
                except lzma.LZMAError as exc:
                    last_error = f"lzma error: {exc}"
                    break
                except Exception as exc:  # noqa: BLE001 - caller receives error text
                    last_error = str(exc)
                    break

        return pd.DataFrame(), FetchResult(
            pair="",
            symbol=symbol,
            date_value=dt,
            url=None,
            status="error",
            rows=0,
            error=last_error or "no matching Dukascopy path",
        )


def _pair_directory_name(pair: str) -> str:
    return pair.replace("/", "_")


def day_cache_path(cache_dir: Path, pair: str, dt: date, fmt: str = "parquet") -> Path:
    pair_dir = cache_dir / _pair_directory_name(pair)
    pair_dir.mkdir(parents=True, exist_ok=True)
    ext = "parquet" if fmt == "parquet" else "csv"
    return pair_dir / f"{dt.isoformat()}.{ext}"


def load_cached_day(cache_dir: Path, pair: str, dt: date, fmt: str = "parquet") -> Optional[pd.DataFrame]:
    path = day_cache_path(cache_dir, pair, dt, fmt=fmt)
    if not path.exists():
        return None
    if fmt == "parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["timestamp"])


def save_cached_day(df: pd.DataFrame, cache_dir: Path, pair: str, dt: date, fmt: str = "parquet") -> Path:
    path = day_cache_path(cache_dir, pair, dt, fmt=fmt)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def fetch_or_load_day(
    client: DukascopyClient,
    pair: str,
    symbol: str,
    dt: date,
    cache_dir: Path,
    *,
    cache_format: str = "parquet",
    force_refresh: bool = False,
) -> Tuple[pd.DataFrame, FetchResult]:
    if not force_refresh:
        cached = load_cached_day(cache_dir, pair, dt, fmt=cache_format)
        if cached is not None and not cached.empty:
            return cached, FetchResult(
                pair=pair,
                symbol=symbol,
                date_value=dt,
                url=None,
                status="cached",
                rows=len(cached),
            )

    fetched, fetch_result = client.fetch_minute_candles(symbol, dt)
    fetch_result.pair = pair
    if not fetched.empty:
        save_cached_day(fetched, cache_dir, pair, dt, fmt=cache_format)
    return fetched, fetch_result


def date_range(start_date: date, end_date: date) -> Iterable[date]:
    current = start_date
    while current <= end_date:
        # Dukascopy forex candles are unavailable on most Saturdays.
        if current.weekday() != 5:
            yield current
        current += timedelta(days=1)


def latest_cached_date(cache_dir: Path, pair: str, fmt: str = "parquet") -> Optional[date]:
    pair_dir = cache_dir / _pair_directory_name(pair)
    if not pair_dir.exists():
        return None

    ext = ".parquet" if fmt == "parquet" else ".csv"
    latest: Optional[date] = None
    for path in pair_dir.glob(f"*{ext}"):
        try:
            candidate = date.fromisoformat(path.stem)
        except ValueError:
            continue
        if latest is None or candidate > latest:
            latest = candidate
    return latest
