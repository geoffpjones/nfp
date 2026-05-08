#!/usr/bin/env python3
"""Incrementally refresh Dukascopy minute cache to current date."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
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
    latest_cached_date,
)

DEFAULT_CACHE_DIR = ROOT / "data" / "dukascopy_minute"


def main() -> None:
    parser = argparse.ArgumentParser(description="Update local Dukascopy minute cache")
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=list(DEFAULT_PAIRS.keys()),
        help="Pairs to refresh",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Cache directory",
    )
    parser.add_argument(
        "--cache-format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Cache file format",
    )
    parser.add_argument(
        "--history-days-if-empty",
        type=int,
        default=45,
        help="How far back to fetch when no cache exists",
    )
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=4, help="Retries per URL")
    args = parser.parse_args()

    invalid_pairs = [pair for pair in args.pairs if pair not in DEFAULT_PAIRS]
    if invalid_pairs:
        valid = ", ".join(DEFAULT_PAIRS)
        raise SystemExit(f"Unsupported pairs: {invalid_pairs}. Valid values: {valid}")

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    client = DukascopyClient(timeout_seconds=args.timeout, max_retries=args.max_retries)
    today = date.today()

    total_ok = 0
    total_failed = 0

    for pair in args.pairs:
        symbol = DEFAULT_PAIRS[pair]
        cached_last = latest_cached_date(args.cache_dir, pair, fmt=args.cache_format)

        if cached_last is None:
            start = today - timedelta(days=args.history_days_if_empty)
            print(f"[{pair}] no cache found, bootstrapping from {start} to {today}")
        else:
            start = cached_last + timedelta(days=1)
            print(f"[{pair}] last cached day is {cached_last}, refreshing through {today}")

        if start > today:
            print(f"[{pair}] already up to date")
            continue

        for day_value in date_range(start, today):
            frame, result = fetch_or_load_day(
                client,
                pair,
                symbol,
                day_value,
                args.cache_dir,
                cache_format=args.cache_format,
                force_refresh=True,
            )

            if result.status == "ok":
                total_ok += 1
                print(f"  {day_value} fetched ({len(frame):,} rows)")
            else:
                total_failed += 1
                reason = result.error or "unknown error"
                print(f"  {day_value} failed ({reason})")

    print("\nUpdate summary")
    print(f"  fetched days: {total_ok}")
    print(f"  failed days: {total_failed}")


if __name__ == "__main__":
    main()
