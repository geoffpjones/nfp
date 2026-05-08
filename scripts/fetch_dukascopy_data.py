#!/usr/bin/env python3
"""Bulk fetch Dukascopy minute candles into local cache."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dukascopy_data import (  # noqa: E402
    DEFAULT_PAIRS,
    DukascopyClient,
    date_range,
    fetch_or_load_day,
)

DEFAULT_CACHE_DIR = ROOT / "data" / "dukascopy_minute"


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Dukascopy minute candles")
    parser.add_argument("--start-date", type=_parse_date, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=_parse_date, default=date.today(), help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=list(DEFAULT_PAIRS.keys()),
        help="Pairs to fetch (e.g. EUR/USD GBP/USD)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Output cache directory",
    )
    parser.add_argument(
        "--cache-format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Storage format for day files",
    )
    parser.add_argument("--force-refresh", action="store_true", help="Ignore local cache and fetch from Dukascopy")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=4, help="Retries per URL")
    args = parser.parse_args()

    end_date = args.end_date
    start_date = args.start_date or (end_date - timedelta(days=30))

    if start_date > end_date:
        raise SystemExit("start-date cannot be after end-date")

    invalid_pairs = [pair for pair in args.pairs if pair not in DEFAULT_PAIRS]
    if invalid_pairs:
        valid = ", ".join(DEFAULT_PAIRS)
        raise SystemExit(f"Unsupported pairs: {invalid_pairs}. Valid values: {valid}")

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    client = DukascopyClient(timeout_seconds=args.timeout, max_retries=args.max_retries)

    total_rows = 0
    ok = 0
    cached = 0
    failed = 0

    print(f"Fetching {len(args.pairs)} pairs from {start_date} to {end_date}")
    print(f"Cache: {args.cache_dir} ({args.cache_format})")

    for pair in args.pairs:
        symbol = DEFAULT_PAIRS[pair]
        print(f"\n[{pair}] symbol={symbol}")

        for day_value in date_range(start_date, end_date):
            frame, result = fetch_or_load_day(
                client,
                pair,
                symbol,
                day_value,
                args.cache_dir,
                cache_format=args.cache_format,
                force_refresh=args.force_refresh,
            )

            if result.status in ("ok", "cached"):
                total_rows += len(frame)
                if result.status == "cached":
                    cached += 1
                    print(f"  {day_value} cached ({len(frame):,} rows)")
                else:
                    ok += 1
                    print(f"  {day_value} fetched ({len(frame):,} rows)")
            else:
                failed += 1
                reason = result.error or "unknown error"
                print(f"  {day_value} failed ({reason})")

    print("\nSummary")
    print(f"  fetched days: {ok}")
    print(f"  cached days: {cached}")
    print(f"  failed days: {failed}")
    print(f"  total rows: {total_rows:,}")


if __name__ == "__main__":
    main()
