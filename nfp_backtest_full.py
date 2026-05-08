#!/usr/bin/env python3
"""Extended NFP analysis with scenario generation using Dukascopy data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from nfp_backtest_dukascopy import (
    DEFAULT_CACHE_DIR,
    OUTPUT_DIR,
    PAIRS,
    aggregate_results,
    print_results,
    run_backtest,
)

BASE_PRICES = {
    "EUR/USD": 1.1734,
    "GBP/USD": 1.3563,
    "USD/CAD": 1.3650,
    "USD/JPY": 156.88,
}


def build_scenarios(stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    surprise_factors = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    rows: List[Dict[str, float]] = []

    for pair in PAIRS:
        for horizon in [1, 4, 6]:
            key = f"{horizon}h"
            pos_key = f"{key}_pos_avg"
            neg_key = f"{key}_neg_avg"

            if pos_key not in stats.get(pair, {}) or neg_key not in stats.get(pair, {}):
                continue

            coeff = (stats[pair][pos_key] - stats[pair][neg_key]) / 2.0
            base_price = BASE_PRICES[pair]

            for surprise_factor in surprise_factors:
                expected_return = coeff * surprise_factor
                expected_price = base_price * (1 + expected_return / 100.0)
                rows.append(
                    {
                        "pair": pair,
                        "horizon_hours": horizon,
                        "surprise_factor": surprise_factor,
                        "expected_return_pct": expected_return,
                        "expected_price": expected_price,
                        "coefficient": coeff,
                    }
                )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NFP full analysis using Dukascopy minute data")
    parser.add_argument("--years", type=int, default=5, help="How many years of events to analyze")
    parser.add_argument("--events-csv", type=Path, default=None, help="Optional CSV with date,actual,forecast")
    parser.add_argument("--events-output", type=Path, default=OUTPUT_DIR / "nfp_events_full.csv", help="Event-level output")
    parser.add_argument("--stats-output", type=Path, default=OUTPUT_DIR / "nfp_stats_full.csv", help="Aggregate stats output")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "nfp_scenarios.csv", help="Scenario output")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Minute candle cache directory")
    parser.add_argument("--cache-format", choices=["parquet", "csv"], default="parquet", help="Cache format")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore local cache and refetch event days")
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
        print(f"Saved events: {args.events_output}")

    stats_rows = []
    for pair in PAIRS:
        for horizon in [1, 4, 6]:
            key = f"{horizon}h"
            if f"{key}_pos_avg" not in stats.get(pair, {}):
                continue
            stats_rows.append(
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

    if stats_rows:
        pd.DataFrame(stats_rows).to_csv(args.stats_output, index=False)
        print(f"Saved stats: {args.stats_output}")

    scenarios = build_scenarios(stats)
    if not scenarios.empty:
        scenarios.to_csv(args.output, index=False)
        print(f"Saved scenarios: {args.output}")


if __name__ == "__main__":
    main()
