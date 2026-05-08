#!/usr/bin/env python3
"""Generate NFP scenario charts from recent Dukascopy candles."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from dukascopy_data import DEFAULT_PAIRS, DukascopyClient, date_range, fetch_or_load_day

OUTPUT_DIR = Path("/home/gjones/work/projects/nfp/results")
DEFAULT_CACHE_DIR = Path("/home/gjones/work/projects/nfp/data/dukascopy_minute")
PAIRS = list(DEFAULT_PAIRS.keys())

FALLBACK_COEFFICIENTS = {
    "EUR/USD": {"1h": 0.0069, "4h": 0.0376, "6h": 0.0376},
    "GBP/USD": {"1h": -0.0014, "4h": 0.0462, "6h": 0.0462},
    "USD/CAD": {"1h": 0.0002, "4h": -0.0031, "6h": -0.0031},
    "USD/JPY": {"1h": -0.0215, "4h": 0.0016, "6h": 0.0154},
}

SURPRISE_FACTORS = [-2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0]


def load_coefficients(stats_path: Path) -> Dict[str, Dict[str, float]]:
    if not stats_path.exists():
        return FALLBACK_COEFFICIENTS

    df = pd.read_csv(stats_path)
    if not {"pair", "horizon_hours", "pos_avg", "neg_avg"}.issubset(df.columns):
        return FALLBACK_COEFFICIENTS

    coefficients: Dict[str, Dict[str, float]] = {pair: {} for pair in PAIRS}
    for row in df.itertuples(index=False):
        pair = row.pair
        horizon = int(row.horizon_hours)
        if pair not in coefficients:
            continue
        key = f"{horizon}h"
        coefficients[pair][key] = (float(row.pos_avg) - float(row.neg_avg)) / 2.0

    # Fill missing entries with fallback values.
    for pair in PAIRS:
        for horizon in [1, 4, 6]:
            key = f"{horizon}h"
            if key not in coefficients[pair]:
                coefficients[pair][key] = FALLBACK_COEFFICIENTS[pair][key]

    return coefficients


def load_recent_data(
    pair: str,
    days: int,
    cache_dir: Path,
    cache_format: str,
    force_refresh: bool,
    timeout: int,
    max_retries: int,
) -> pd.DataFrame:
    symbol = DEFAULT_PAIRS[pair]
    end_day = date.today()
    start_day = end_day - timedelta(days=days)

    client = DukascopyClient(timeout_seconds=timeout, max_retries=max_retries)
    frames: List[pd.DataFrame] = []

    for day_value in date_range(start_day, end_day):
        frame, _ = fetch_or_load_day(
            client,
            pair,
            symbol,
            day_value,
            cache_dir,
            cache_format=cache_format,
            force_refresh=force_refresh,
        )
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

    # Resample to hourly candles for a cleaner chart.
    hourly = (
        data.set_index("timestamp")
        .resample("1h")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["close"])
        .reset_index()
    )
    return hourly


def generate_scenarios(latest_close: float, coeffs: Dict[str, float]) -> List[Dict[str, float]]:
    scenarios = []
    for surprise in SURPRISE_FACTORS:
        row = {"surprise_factor": surprise}
        for horizon in [1, 4, 6]:
            key = f"{horizon}h"
            expected_return = coeffs.get(key, 0.0) * surprise
            row[f"{horizon}_price"] = latest_close * (1 + expected_return / 100.0)
        scenarios.append(row)
    return scenarios


def plot_pair(
    pair: str,
    recent_data: pd.DataFrame,
    coeffs: Dict[str, float],
    output_dir: Path,
) -> Path:
    latest_time = recent_data["timestamp"].iloc[-1]
    latest_close = float(recent_data["close"].iloc[-1])
    scenarios = generate_scenarios(latest_close, coeffs)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(recent_data["timestamp"], recent_data["close"], linewidth=1.8, color="#0b5fa5", label="Recent hourly close")

    colors = ["#d62828", "#f77f00", "#fcbf49", "#90be6d", "#43aa8b", "#4d908e", "#577590", "#6a4c93", "#8f2d56"]

    for idx, scenario in enumerate(scenarios):
        color = colors[idx % len(colors)]
        surprise = scenario["surprise_factor"]

        points_x = [latest_time, latest_time + timedelta(hours=1), latest_time + timedelta(hours=4), latest_time + timedelta(hours=6)]
        points_y = [latest_close, scenario["1_price"], scenario["4_price"], scenario["6_price"]]

        ax.plot(points_x, points_y, "--", linewidth=1.3, color=color, alpha=0.85, label=f"{surprise:+.1f}σ")
        ax.scatter(points_x[-1], points_y[-1], s=28, color=color, zorder=4)

    ax.set_title(
        f"{pair} NFP Scenario Paths\n"
        f"Last close: {latest_close:.5f} at {latest_time:%Y-%m-%d %H:%M UTC}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.25, linestyle=":")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.xticks(rotation=35)
    ax.legend(loc="upper left", ncol=3, fontsize=8, framealpha=0.9)
    plt.tight_layout()

    output_path = output_dir / f"nfp_chart_{pair.replace('/', '_')}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate NFP scenario charts from Dukascopy data")
    parser.add_argument("--days", type=int, default=5, help="Number of recent days to chart")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Minute candle cache directory")
    parser.add_argument("--cache-format", choices=["parquet", "csv"], default="parquet", help="Cache format")
    parser.add_argument("--force-refresh", action="store_true", help="Refetch recent days before plotting")
    parser.add_argument("--stats-file", type=Path, default=OUTPUT_DIR / "nfp_stats_full.csv", help="Stats CSV used for coefficients")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=4, help="Retry attempts per URL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    coeffs_by_pair = load_coefficients(args.stats_file)

    print("=" * 80)
    print("NFP SCENARIO CHARTS (Dukascopy-backed)")
    print("=" * 80)

    saved: List[Path] = []
    for pair in PAIRS:
        recent_data = load_recent_data(
            pair,
            days=args.days,
            cache_dir=args.cache_dir,
            cache_format=args.cache_format,
            force_refresh=args.force_refresh,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )

        if recent_data.empty:
            print(f"{pair}: skipped (no local or remote data)")
            continue

        output_path = plot_pair(pair, recent_data, coeffs_by_pair[pair], OUTPUT_DIR)
        print(f"{pair}: saved {output_path}")
        saved.append(output_path)

    print("\nGenerated files:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
