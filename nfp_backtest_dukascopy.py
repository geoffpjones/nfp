#!/usr/bin/env python3
"""NFP delta/return backtest using shared cached minute candles."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from market_data import DEFAULT_MARKET_BARS_DB, DEFAULT_PAIRS, load_cached_day

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
OUTPUT_DIR = ROOT / "results"
DEFAULT_CACHE_DIR = DEFAULT_MARKET_BARS_DB
UTC = ZoneInfo("UTC")
NEW_YORK = ZoneInfo("America/New_York")
PAIRS = list(DEFAULT_PAIRS.keys())

# Minute horizons requested by the user:
# 1, 5, 15, 30 minutes and 1, 2, 4, 6, 12 hours.
RETURN_HORIZONS_MINUTES: List[int] = [1, 5, 15, 30, 60, 120, 240, 360, 720]

# Scenario deltas for sensitivity matrix projections.
SCENARIO_DELTA_K: List[float] = [-150.0, -100.0, -75.0, -50.0, -25.0, 0.0, 25.0, 50.0, 75.0, 100.0, 150.0]

# Embedded fallback dataset. Prefer --events-csv with maintained source data.
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


def horizon_label(minutes: int) -> str:
    if minutes < 60:
        return f"{minutes}m"
    return f"{minutes // 60}h"


def horizon_labels(horizons_minutes: Iterable[int]) -> List[str]:
    return [horizon_label(minutes) for minutes in horizons_minutes]


def nfp_release_utc(day_value: date) -> datetime:
    # NFP is released at 08:30 America/New_York. Convert with timezone rules,
    # so March/November DST transitions are handled correctly.
    local_release = datetime(day_value.year, day_value.month, day_value.day, 8, 30, tzinfo=NEW_YORK)
    return local_release.astimezone(UTC).replace(tzinfo=None)


def _normalize_release_time_utc(value: pd.Series) -> pd.Series:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    # Store as naive UTC datetimes for direct comparison with candle timestamps.
    return ts.dt.tz_convert("UTC").dt.tz_localize(None)


def load_nfp_events(events_csv: Optional[Path], years: int) -> pd.DataFrame:
    if events_csv is not None:
        df = pd.read_csv(events_csv)
        missing_cols = {"date", "actual", "forecast"} - set(df.columns)
        if missing_cols:
            raise ValueError(f"events CSV missing columns: {sorted(missing_cols)}")
    else:
        df = pd.DataFrame(EMBEDDED_NFP_DATA, columns=["date", "actual", "forecast"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    df["forecast"] = pd.to_numeric(df["forecast"], errors="coerce")

    if "release_time_utc" in df.columns:
        df["release_time_utc"] = _normalize_release_time_utc(df["release_time_utc"])
    else:
        df["release_time_utc"] = df["date"].dt.date.apply(
            lambda d: nfp_release_utc(d) if pd.notna(d) else pd.NaT
        )

    df = df.dropna(subset=["date", "actual", "forecast", "release_time_utc"]).copy()

    cutoff = datetime.utcnow().date() - timedelta(days=years * 365)
    df = df[df["date"].dt.date >= cutoff].sort_values("date").reset_index(drop=True)
    if df.empty:
        return df

    df["delta"] = df["actual"] - df["forecast"]
    df["delta_k"] = df["delta"] / 1000.0

    forecast_abs = df["forecast"].abs()
    df["pct_delta"] = np.where(forecast_abs > 0, (df["delta"] / forecast_abs) * 100.0, np.nan)

    # Standard deviation of delta across entire series.
    delta_std = float(df["delta"].std(ddof=1)) if len(df) > 1 else float("nan")
    if np.isfinite(delta_std) and delta_std > 0:
        df["delta_zscore"] = df["delta"] / delta_std
    else:
        df["delta_zscore"] = np.nan

    df["delta_std_series"] = delta_std
    return df


def _lookup_price(df: pd.DataFrame, target_time: datetime) -> Optional[float]:
    window = df[(df["timestamp"] >= target_time) & (df["timestamp"] <= target_time + timedelta(minutes=5))]
    if window.empty:
        return None
    return float(window.iloc[0]["close"])


def calc_event_returns(
    df: pd.DataFrame,
    event_time_utc: datetime,
    horizons_minutes: Sequence[int],
) -> Dict[str, float]:
    returns: Dict[str, float] = {}
    base_price = _lookup_price(df, event_time_utc)
    if base_price is None:
        return returns

    for minutes in horizons_minutes:
        target_time = event_time_utc + timedelta(minutes=minutes)
        target_price = _lookup_price(df, target_time)
        if target_price is None:
            continue
        label = horizon_label(minutes)
        returns[label] = ((target_price - base_price) / base_price) * 100.0

    return returns


def _required_dates_for_event_window(event_time_utc: datetime, horizons_minutes: Sequence[int]) -> List[date]:
    max_minutes = max(horizons_minutes) if horizons_minutes else 0
    end_time = event_time_utc + timedelta(minutes=max_minutes + 5)
    start_day = event_time_utc.date()
    end_day = end_time.date()

    days: List[date] = []
    current = start_day
    while current <= end_day:
        days.append(current)
        current += timedelta(days=1)
    return days


def _load_event_window_data(
    pair: str,
    event_time_utc: datetime,
    horizons_minutes: Sequence[int],
    cache_dir: Path,
    cache_format: str,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    required_days = _required_dates_for_event_window(event_time_utc, horizons_minutes)

    for day_value in required_days:
        frame = load_cached_day(cache_dir, pair, day_value, fmt=cache_format)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return data


def run_backtest(
    *,
    years: int,
    events_csv: Optional[Path],
    cache_dir: Path,
    cache_format: str,
    horizons_minutes: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    horizons = list(horizons_minutes or RETURN_HORIZONS_MINUTES)
    events = load_nfp_events(events_csv, years)

    if events.empty:
        return {"events": [], "total": 0, "horizons_minutes": horizons, "delta_std_series": float("nan")}

    delta_std_series = float(events["delta"].std(ddof=1)) if len(events) > 1 else float("nan")
    results: List[Dict[str, object]] = []

    print(f"Running NFP backtest on {len(events)} events")
    print(f"Market-data cache directory: {cache_dir} ({cache_format})")
    print(f"Horizons: {', '.join(horizon_labels(horizons))}")

    for event in events.itertuples(index=False):
        event_time_utc: datetime = event.release_time_utc
        row: Dict[str, object] = {
            "date": event.date.strftime("%Y-%m-%d"),
            "release_time_utc": event_time_utc.strftime("%Y-%m-%d %H:%M:%S"),
            "actual": float(event.actual),
            "forecast": float(event.forecast),
            "delta": float(event.delta),
            "delta_k": float(event.delta_k),
            "pct_delta": float(event.pct_delta) if pd.notna(event.pct_delta) else np.nan,
            "delta_zscore": float(event.delta_zscore) if pd.notna(event.delta_zscore) else np.nan,
            "delta_std_series": delta_std_series,
        }

        for pair in PAIRS:
            frame = _load_event_window_data(
                pair,
                event_time_utc,
                horizons,
                cache_dir,
                cache_format,
            )
            if frame.empty:
                continue

            returns = calc_event_returns(frame, event_time_utc, horizons)
            for label, pct in returns.items():
                row[f"{pair}_returns_{label}"] = pct

        results.append(row)

    return {
        "events": results,
        "total": len(results),
        "horizons_minutes": horizons,
        "delta_std_series": delta_std_series,
    }


def build_sensitivity_stats(
    events_df: pd.DataFrame,
    horizons_minutes: Sequence[int],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    if events_df.empty:
        return pd.DataFrame(rows)

    for pair in PAIRS:
        for minutes in horizons_minutes:
            label = horizon_label(minutes)
            metric = f"{pair}_returns_{label}"
            if metric not in events_df.columns:
                continue

            subset = events_df[["delta_k", metric]].dropna()
            if len(subset) < 3:
                continue

            x = subset["delta_k"].to_numpy(dtype=float)
            y = subset[metric].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            y_hat = intercept + slope * x
            residuals = y - y_hat
            rmse = float(np.sqrt(np.mean(np.square(residuals))))

            corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else np.nan
            r2 = corr * corr if np.isfinite(corr) else np.nan

            rows.append(
                {
                    "pair": pair,
                    "horizon_label": label,
                    "horizon_minutes": minutes,
                    "n_events": int(len(subset)),
                    "slope_pct_per_1k_delta": float(slope),
                    "intercept_pct": float(intercept),
                    "corr": corr,
                    "r2": r2,
                    "rmse_pct": rmse,
                    "mean_return_pct": float(np.mean(y)),
                    "std_return_pct": float(np.std(y, ddof=1)) if len(y) > 1 else np.nan,
                    "delta_std_k_series": float(events_df["delta_k"].std(ddof=1)) if len(events_df) > 1 else np.nan,
                }
            )

    return pd.DataFrame(rows)


def build_sensitivity_matrix(stats_df: pd.DataFrame, delta_scenarios_k: Sequence[float]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if stats_df.empty:
        return pd.DataFrame(rows)

    for row in stats_df.itertuples(index=False):
        for delta_k in delta_scenarios_k:
            predicted = float(row.intercept_pct + row.slope_pct_per_1k_delta * delta_k)
            rows.append(
                {
                    "pair": row.pair,
                    "horizon_label": row.horizon_label,
                    "horizon_minutes": int(row.horizon_minutes),
                    "delta_k": float(delta_k),
                    "predicted_return_pct": predicted,
                    "slope_pct_per_1k_delta": float(row.slope_pct_per_1k_delta),
                    "intercept_pct": float(row.intercept_pct),
                    "n_events": int(row.n_events),
                }
            )

    return pd.DataFrame(rows)


def create_event_prediction_comparison(
    events_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    horizons_minutes: Sequence[int],
    *,
    target_date: Optional[date] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if events_df.empty or stats_df.empty:
        return pd.DataFrame(rows)

    date_series = pd.to_datetime(events_df["date"], errors="coerce").dt.date
    if target_date is None:
        target_date = datetime.utcnow().date() - timedelta(days=1)

    # Use latest event on or before target date. This lets users request a
    # non-release day (e.g. weekend) while still getting the most recent NFP.
    target_events = events_df[date_series <= target_date]
    source = "asof_target_date"
    if target_events.empty:
        last_idx = pd.to_datetime(events_df["date"], errors="coerce").idxmax()
        target_events = events_df.loc[[last_idx]]
        source = "latest_available"

    event_row = target_events.iloc[-1]
    event_date = str(event_row["date"])
    delta_k = float(event_row["delta_k"])
    actual = float(event_row["actual"])
    forecast = float(event_row["forecast"])

    for pair in PAIRS:
        pair_stats = stats_df[stats_df["pair"] == pair]
        if pair_stats.empty:
            continue

        for minutes in horizons_minutes:
            label = horizon_label(minutes)
            stat = pair_stats[pair_stats["horizon_minutes"] == minutes]
            if stat.empty:
                continue
            stat_row = stat.iloc[0]

            predicted = float(stat_row["intercept_pct"] + stat_row["slope_pct_per_1k_delta"] * delta_k)
            metric = f"{pair}_returns_{label}"
            actual_return = float(event_row[metric]) if metric in event_row and pd.notna(event_row[metric]) else np.nan
            prediction_error = actual_return - predicted if pd.notna(actual_return) else np.nan

            rows.append(
                {
                    "comparison_source": source,
                    "comparison_target_date": target_date.isoformat(),
                    "event_date": event_date,
                    "pair": pair,
                    "horizon_label": label,
                    "horizon_minutes": minutes,
                    "actual_nfp": actual,
                    "forecast_nfp": forecast,
                    "delta_k": delta_k,
                    "predicted_return_pct": predicted,
                    "realized_return_pct": actual_return,
                    "prediction_error_pct": prediction_error,
                    "slope_pct_per_1k_delta": float(stat_row["slope_pct_per_1k_delta"]),
                    "intercept_pct": float(stat_row["intercept_pct"]),
                }
            )

    return pd.DataFrame(rows)


def print_results(stats_df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("NFP DELTA SENSITIVITY (shared market data)")
    print("=" * 80)

    if stats_df.empty:
        print("No sensitivity statistics were produced.")
        return

    for pair in PAIRS:
        pair_stats = stats_df[stats_df["pair"] == pair].sort_values("horizon_minutes")
        if pair_stats.empty:
            continue
        print(f"\n{pair}")
        print("-" * 70)
        for row in pair_stats.itertuples(index=False):
            print(
                f"  {row.horizon_label:>3} | beta={row.slope_pct_per_1k_delta:+.6f}%/1k "
                f"alpha={row.intercept_pct:+.4f}% r2={row.r2:.3f} rmse={row.rmse_pct:.3f}% n={int(row.n_events)}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NFP delta backtest with shared minute market data")
    parser.add_argument("--years", type=int, default=5, help="How many years of events to analyze")
    parser.add_argument("--events-csv", type=Path, default=None, help="Optional CSV with columns: date,actual,forecast")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "nfp_impact_dukascopy.csv", help="Sensitivity stats output CSV")
    parser.add_argument("--events-output", type=Path, default=OUTPUT_DIR / "nfp_events_dukascopy.csv", help="Event-level output CSV")
    parser.add_argument("--matrix-output", type=Path, default=OUTPUT_DIR / "nfp_sensitivity_matrix.csv", help="Sensitivity matrix output CSV")
    parser.add_argument(
        "--yesterday-output",
        type=Path,
        default=OUTPUT_DIR / "nfp_yesterday_prediction.csv",
        help="Yesterday/latest event prediction comparison output CSV",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Minute candle source path")
    parser.add_argument("--cache-format", choices=["parquet", "csv"], default="parquet", help="Cache file format")
    parser.add_argument(
        "--comparison-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=None,
        help="Target date for predicted-vs-realized comparison (YYYY-MM-DD); uses latest event on/before date",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.cache_dir.suffix == ".db":
        args.cache_dir.parent.mkdir(parents=True, exist_ok=True)
    else:
        args.cache_dir.mkdir(parents=True, exist_ok=True)

    results = run_backtest(
        years=args.years,
        events_csv=args.events_csv,
        cache_dir=args.cache_dir,
        cache_format=args.cache_format,
        horizons_minutes=RETURN_HORIZONS_MINUTES,
    )

    events_df = pd.DataFrame(results["events"])
    if not events_df.empty:
        events_df.to_csv(args.events_output, index=False)
        print(f"\nSaved events: {args.events_output}")

    stats_df = build_sensitivity_stats(events_df, RETURN_HORIZONS_MINUTES)
    print_results(stats_df)
    if not stats_df.empty:
        stats_df.to_csv(args.output, index=False)
        print(f"Saved sensitivity stats: {args.output}")

    matrix_df = build_sensitivity_matrix(stats_df, SCENARIO_DELTA_K)
    if not matrix_df.empty:
        matrix_df.to_csv(args.matrix_output, index=False)
        print(f"Saved sensitivity matrix: {args.matrix_output}")

    comparison_df = create_event_prediction_comparison(
        events_df,
        stats_df,
        RETURN_HORIZONS_MINUTES,
        target_date=args.comparison_date,
    )
    if not comparison_df.empty:
        comparison_df.to_csv(args.yesterday_output, index=False)
        source = str(comparison_df.iloc[0]["comparison_source"])
        event_date = str(comparison_df.iloc[0]["event_date"])
        print(f"Saved {source} comparison ({event_date}): {args.yesterday_output}")


if __name__ == "__main__":
    main()
