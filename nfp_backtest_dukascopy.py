#!/usr/bin/env python3
"""NFP surprise backtest using cached Dukascopy minute candles."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from dukascopy_data import DEFAULT_PAIRS, DukascopyClient, fetch_or_load_day

OUTPUT_DIR = Path("/home/gjones/work/projects/nfp/results")
DEFAULT_CACHE_DIR = Path("/home/gjones/work/projects/nfp/data/dukascopy_minute")
UTC = ZoneInfo("UTC")
NEW_YORK = ZoneInfo("America/New_York")
PAIRS = list(DEFAULT_PAIRS.keys())

# Embedded fallback dataset. Pass --events-csv to use a maintained external file.
EMBEDDED_NFP_DATA: List[Tuple[str, int, int]] = [
    ("2024-01-05", 216, 173), ("2024-02-02", 353, 180), ("2024-03-08", 275, 200),
    ("2024-04-05", 175, 200), ("2024-05-03", 275, 190), ("2024-06-07", 206, 190),
    ("2024-07-05", 206, 190), ("2024-08-02", 114, 175), ("2024-09-06", 142, 175),
    ("2024-10-04", 254, 140), ("2024-11-01", 254, 117), ("2024-12-06", 227, 200),
    ("2023-01-06", 50, 180), ("2023-02-03", 517, 185), ("2023-03-10", 311, 185),
    ("2023-04-07", 236, 180), ("2023-05-05", 339, 180), ("2023-06-02", 339, 180),
    ("2023-07-07", 306, 180), ("2023-08-04", 187, 180), ("2023-09-08", 157, 180),
    ("2023-10-06", 336, 170), ("2023-11-03", 150, 170), ("2023-12-08", 199, 180),
    ("2022-01-07", 231, 193), ("2022-02-04", 467, 200), ("2022-03-04", 431, 250),
    ("2022-04-01", 275, 250), ("2022-05-06", 397, 240), ("2022-06-03", 372, 230),
    ("2022-07-01", 372, 225), ("2022-08-05", 528, 375), ("2022-09-02", 263, 375),
    ("2022-10-07", 263, 250), ("2022-11-04", 217, 250), ("2022-12-02", 217, 200),
    ("2021-01-08", 600, 130), ("2021-02-05", 379, 130), ("2021-03-05", 649, 130),
    ("2021-04-02", 329, 130), ("2021-05-07", 266, 130), ("2021-06-04", 282, 130),
    ("2021-07-02", 339, 130), ("2021-08-06", 235, 130), ("2021-09-03", 194, 130),
    ("2021-10-08", 266, 130), ("2021-11-05", 249, 130), ("2021-12-03", 194, 130),
    ("2020-01-03", 266, 120), ("2020-02-07", 200, 120), ("2020-03-06", -700, 120),
    ("2020-04-03", -701, 120), ("2020-05-01", -2500, 120), ("2020-06-05", 2500, 120),
    ("2020-07-03", 1370, 120), ("2020-08-07", 2552, 120), ("2020-09-04", 1370, 120),
    ("2020-10-02", 666, 120), ("2020-11-06", 1370, 120), ("2020-12-04", 1400, 120),
    ("2019-01-04", 252, 180), ("2019-02-01", 256, 180), ("2019-03-08", 263, 180),
    ("2019-04-05", 196, 180), ("2019-05-03", 75, 180), ("2019-06-07", 175, 180),
    ("2019-07-05", 209, 180), ("2019-08-02", 130, 180), ("2019-09-06", 130, 180),
    ("2019-10-04", 136, 180), ("2019-11-01", 141, 180), ("2019-12-06", 145, 180),
]


def load_nfp_events(events_csv: Optional[Path], years: int) -> pd.DataFrame:
    if events_csv is not None:
        df = pd.read_csv(events_csv)
        missing_cols = {"date", "actual", "forecast"} - set(df.columns)
        if missing_cols:
            raise ValueError(f"events CSV missing columns: {sorted(missing_cols)}")
        df = df[["date", "actual", "forecast"]].copy()
    else:
        df = pd.DataFrame(EMBEDDED_NFP_DATA, columns=["date", "actual", "forecast"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "actual", "forecast"]).copy()
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    df["forecast"] = pd.to_numeric(df["forecast"], errors="coerce")
    df = df.dropna(subset=["actual", "forecast"])

    cutoff = datetime.utcnow().date() - timedelta(days=years * 365)
    df = df[df["date"].dt.date >= cutoff].sort_values("date").reset_index(drop=True)
    return df


def nfp_release_utc(day_value: date) -> datetime:
    # NFP is released at 08:30 America/New_York. Convert with timezone rules,
    # so March/November DST transitions are handled correctly.
    local_release = datetime(day_value.year, day_value.month, day_value.day, 8, 30, tzinfo=NEW_YORK)
    return local_release.astimezone(UTC).replace(tzinfo=None)


def _lookup_price(df: pd.DataFrame, target_time: datetime) -> Optional[float]:
    window = df[(df["timestamp"] >= target_time) & (df["timestamp"] <= target_time + timedelta(minutes=5))]
    if window.empty:
        return None
    return float(window.iloc[0]["close"])


def calc_event_returns(df: pd.DataFrame, event_time_utc: datetime, horizons: Iterable[int]) -> Dict[int, float]:
    returns: Dict[int, float] = {}
    base_price = _lookup_price(df, event_time_utc)
    if base_price is None:
        return returns

    for horizon in horizons:
        target_time = event_time_utc + timedelta(hours=horizon)
        target_price = _lookup_price(df, target_time)
        if target_price is None:
            continue
        returns[horizon] = ((target_price - base_price) / base_price) * 100.0

    return returns


def run_backtest(
    *,
    years: int,
    events_csv: Optional[Path],
    cache_dir: Path,
    cache_format: str,
    force_refresh: bool,
    timeout: int,
    max_retries: int,
) -> Dict[str, List[Dict[str, float]]]:
    events = load_nfp_events(events_csv, years)

    if events.empty:
        return {"events": [], "total": 0}

    client = DukascopyClient(timeout_seconds=timeout, max_retries=max_retries)
    results = []

    print(f"Running NFP backtest on {len(events)} events")
    print(f"Cache directory: {cache_dir} ({cache_format})")

    for event in events.itertuples(index=False):
        nfp_day = event.date.date()
        event_time_utc = nfp_release_utc(nfp_day)
        surprise = float(event.actual - event.forecast)
        surprise_factor = surprise / 50.0

        row = {
            "date": event.date.strftime("%Y-%m-%d"),
            "actual": float(event.actual),
            "forecast": float(event.forecast),
            "surprise": surprise,
            "surprise_factor": surprise_factor,
            "surprise_sign": "positive" if surprise > 0 else "negative",
        }

        for pair in PAIRS:
            symbol = DEFAULT_PAIRS[pair]
            frame, fetch_result = fetch_or_load_day(
                client,
                pair,
                symbol,
                nfp_day,
                cache_dir,
                cache_format=cache_format,
                force_refresh=force_refresh,
            )
            if frame.empty:
                print(f"{row['date']} {pair}: no data ({fetch_result.error or fetch_result.status})")
                continue

            returns = calc_event_returns(frame, event_time_utc, horizons=[1, 4, 6])
            for horizon, pct in returns.items():
                row[f"{pair}_returns_{horizon}h"] = pct

        results.append(row)

    return {"events": results, "total": len(results)}


def aggregate_results(results: Dict[str, List[Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    events = results["events"]
    pos_events = [event for event in events if event["surprise_sign"] == "positive"]
    neg_events = [event for event in events if event["surprise_sign"] == "negative"]

    stats: Dict[str, Dict[str, float]] = {}
    for pair in PAIRS:
        pair_stats: Dict[str, float] = {}
        for horizon in [1, 4, 6]:
            metric = f"{pair}_returns_{horizon}h"
            pos_values = [event.get(metric) for event in pos_events if pd.notna(event.get(metric))]
            neg_values = [event.get(metric) for event in neg_events if pd.notna(event.get(metric))]

            if not pos_values or not neg_values:
                continue

            key = f"{horizon}h"
            pair_stats[f"{key}_pos_avg"] = float(np.mean(pos_values))
            pair_stats[f"{key}_neg_avg"] = float(np.mean(neg_values))
            pair_stats[f"{key}_diff"] = pair_stats[f"{key}_pos_avg"] - pair_stats[f"{key}_neg_avg"]
            pair_stats[f"{key}_pos_std"] = float(np.std(pos_values))
            pair_stats[f"{key}_neg_std"] = float(np.std(neg_values))
            pair_stats[f"{key}_pos_count"] = float(len(pos_values))
            pair_stats[f"{key}_neg_count"] = float(len(neg_values))

        stats[pair] = pair_stats

    return stats


def print_results(stats: Dict[str, Dict[str, float]]) -> None:
    print("\n" + "=" * 80)
    print("NFP SURPRISE IMPACT ANALYSIS (Dukascopy)")
    print("=" * 80)

    for pair in PAIRS:
        print(f"\n{pair}")
        print("-" * 60)
        for horizon in [1, 4, 6]:
            key = f"{horizon}h"
            if f"{key}_pos_avg" not in stats.get(pair, {}):
                continue

            pos_avg = stats[pair][f"{key}_pos_avg"]
            neg_avg = stats[pair][f"{key}_neg_avg"]
            diff = stats[pair][f"{key}_diff"]
            pos_std = stats[pair][f"{key}_pos_std"]
            neg_std = stats[pair][f"{key}_neg_std"]
            pos_count = int(stats[pair][f"{key}_pos_count"])
            neg_count = int(stats[pair][f"{key}_neg_count"])

            print(f"  {key.upper()} | pos={pos_avg:+.3f}% (std {pos_std:.3f}, n={pos_count}) "
                  f"neg={neg_avg:+.3f}% (std {neg_std:.3f}, n={neg_count}) diff={diff:+.3f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NFP surprise backtest with Dukascopy minute data")
    parser.add_argument("--years", type=int, default=5, help="How many years of events to analyze")
    parser.add_argument("--events-csv", type=Path, default=None, help="Optional CSV with columns: date,actual,forecast")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "nfp_impact_dukascopy.csv", help="Aggregated output CSV")
    parser.add_argument("--events-output", type=Path, default=OUTPUT_DIR / "nfp_events_dukascopy.csv", help="Event-level output CSV")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Minute candle cache directory")
    parser.add_argument("--cache-format", choices=["parquet", "csv"], default="parquet", help="Cache file format")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cache and refetch event days")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=4, help="Retry attempts per URL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    results = run_backtest(
        years=args.years,
        events_csv=args.events_csv,
        cache_dir=args.cache_dir,
        cache_format=args.cache_format,
        force_refresh=args.force_refresh,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    stats = aggregate_results(results)
    print_results(stats)

    if results["events"]:
        pd.DataFrame(results["events"]).to_csv(args.events_output, index=False)
        print(f"\nSaved events: {args.events_output}")

    rows: List[Dict[str, float]] = []
    for pair in PAIRS:
        for horizon in [1, 4, 6]:
            key = f"{horizon}h"
            if f"{key}_pos_avg" not in stats.get(pair, {}):
                continue
            rows.append(
                {
                    "pair": pair,
                    "horizon_hours": horizon,
                    "pos_avg": stats[pair][f"{key}_pos_avg"],
                    "neg_avg": stats[pair][f"{key}_neg_avg"],
                    "diff": stats[pair][f"{key}_diff"],
                    "pos_std": stats[pair][f"{key}_pos_std"],
                    "neg_std": stats[pair][f"{key}_neg_std"],
                    "pos_count": int(stats[pair][f"{key}_pos_count"]),
                    "neg_count": int(stats[pair][f"{key}_neg_count"]),
                }
            )

    if rows:
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f"Saved aggregate stats: {args.output}")


if __name__ == "__main__":
    main()
