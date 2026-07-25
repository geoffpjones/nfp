#!/usr/bin/env python3
"""Full NFP delta analysis with sensitivity matrix, scenarios, and plots."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nfp_backtest_dukascopy import (
    DEFAULT_CACHE_DIR,
    OUTPUT_DIR,
    PAIRS,
    RETURN_HORIZONS_MINUTES,
    SCENARIO_DELTA_K,
    build_sensitivity_matrix,
    build_sensitivity_stats,
    create_event_prediction_comparison,
    horizon_label,
    run_backtest,
)


def build_projection_scenarios(
    stats_df: pd.DataFrame,
    forecast_k: float,
    scenario_delta_k: List[float],
) -> pd.DataFrame:
    matrix_df = build_sensitivity_matrix(stats_df, scenario_delta_k)
    if matrix_df.empty:
        return matrix_df

    projections = matrix_df.copy()
    projections["forecast_k"] = forecast_k
    projections["scenario_actual_k"] = forecast_k + projections["delta_k"]
    projections = projections[
        [
            "pair",
            "horizon_label",
            "horizon_minutes",
            "forecast_k",
            "scenario_actual_k",
            "delta_k",
            "predicted_return_pct",
            "slope_pct_per_1k_delta",
            "intercept_pct",
            "n_events",
        ]
    ]
    return projections


def plot_returns_by_delta(
    events_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    output_dir: Path,
) -> List[Path]:
    saved: List[Path] = []
    if events_df.empty or stats_df.empty:
        return saved

    for pair in PAIRS:
        fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=False)
        axes_flat = axes.flatten()
        fig.suptitle(f"{pair} returns vs NFP delta (actual - forecast)", fontsize=14, fontweight="bold")

        for idx, minutes in enumerate(RETURN_HORIZONS_MINUTES):
            ax = axes_flat[idx]
            label = horizon_label(minutes)
            metric = f"{pair}_returns_{label}"

            if metric not in events_df.columns:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(label)
                continue

            subset = events_df[["delta_k", metric]].dropna()
            if subset.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(label)
                continue

            x = subset["delta_k"].to_numpy(dtype=float)
            y = subset[metric].to_numpy(dtype=float)
            ax.scatter(x, y, alpha=0.75, s=24, color="#155e75")

            stat = stats_df[(stats_df["pair"] == pair) & (stats_df["horizon_minutes"] == minutes)]
            if not stat.empty:
                slope = float(stat.iloc[0]["slope_pct_per_1k_delta"])
                intercept = float(stat.iloc[0]["intercept_pct"])
                r2 = float(stat.iloc[0]["r2"]) if pd.notna(stat.iloc[0]["r2"]) else np.nan

                x_line = np.linspace(np.min(x), np.max(x), 100)
                y_line = intercept + slope * x_line
                ax.plot(x_line, y_line, color="#b91c1c", linewidth=1.6)
                ax.text(
                    0.02,
                    0.96,
                    f"beta={slope:+.5f}%/1k\nr2={r2:.3f}",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
                )

            ax.axvline(0.0, color="#6b7280", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.axhline(0.0, color="#6b7280", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.set_title(label)
            ax.set_xlabel("delta (K payrolls)")
            ax.set_ylabel("return %")
            ax.grid(alpha=0.2, linestyle=":")

        plt.tight_layout()
        out = output_dir / f"nfp_delta_returns_{pair.replace('/', '_')}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved.append(out)

    return saved


def plot_yesterday_vs_predicted(comparison_df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    if comparison_df.empty:
        return None

    event_date = str(comparison_df.iloc[0]["event_date"])
    source = str(comparison_df.iloc[0]["comparison_source"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
    axes_flat = axes.flatten()
    fig.suptitle(f"NFP {source} comparison ({event_date}): predicted vs realized returns", fontsize=14, fontweight="bold")

    for idx, pair in enumerate(PAIRS):
        ax = axes_flat[idx]
        subset = comparison_df[comparison_df["pair"] == pair].sort_values("horizon_minutes")
        if subset.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(pair)
            continue

        x = np.arange(len(subset))
        predicted = subset["predicted_return_pct"].to_numpy(dtype=float)
        labels = subset["horizon_label"].astype(str).tolist()

        ax.bar(x, predicted, width=0.65, color="#0c4a6e", alpha=0.8, label="Predicted")

        realized = subset["realized_return_pct"].to_numpy(dtype=float)
        if np.isfinite(realized).any():
            ax.plot(x, realized, "o-", color="#b91c1c", linewidth=1.5, markersize=4, label="Realized")

        ax.axhline(0.0, color="#6b7280", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xticks(x, labels)
        ax.set_title(pair)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Return %")
        ax.grid(axis="y", alpha=0.2, linestyle=":")
        ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    out = output_dir / "nfp_yesterday_vs_predicted.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def parse_scenario_delta_k(value: str) -> List[float]:
    values: List[float] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("scenario delta list cannot be empty")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NFP full delta analysis using shared minute market data")
    parser.add_argument("--years", type=int, default=5, help="How many years of events to analyze")
    parser.add_argument("--events-csv", type=Path, default=None, help="Optional CSV with date,actual,forecast")
    parser.add_argument("--events-output", type=Path, default=OUTPUT_DIR / "nfp_events_full.csv", help="Event-level output")
    parser.add_argument("--stats-output", type=Path, default=OUTPUT_DIR / "nfp_stats_full.csv", help="Sensitivity stats output")
    parser.add_argument("--matrix-output", type=Path, default=OUTPUT_DIR / "nfp_sensitivity_matrix.csv", help="Sensitivity matrix output")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "nfp_fx_scenarios.csv", help="Scenario projection output")
    parser.add_argument(
        "--yesterday-output",
        type=Path,
        default=OUTPUT_DIR / "nfp_yesterday_prediction.csv",
        help="Yesterday/latest comparison output",
    )
    parser.add_argument(
        "--forecast-k",
        type=float,
        default=None,
        help="Expected NFP forecast (K jobs) used to project scenario actual values; defaults to latest forecast in events",
    )
    parser.add_argument(
        "--scenario-delta-k",
        type=parse_scenario_delta_k,
        default=SCENARIO_DELTA_K,
        help="Comma-separated delta scenarios in K jobs (e.g. -75,-50,-25,0,25,50,75)",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Minute candle source path")
    parser.add_argument("--cache-format", choices=["parquet", "csv"], default="parquet", help="Cache format")
    parser.add_argument(
        "--comparison-date",
        type=lambda s: pd.to_datetime(s, format="%Y-%m-%d").date(),
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
    if events_df.empty:
        print("No event data produced.")
        return

    events_df.to_csv(args.events_output, index=False)
    print(f"Saved events: {args.events_output}")

    stats_df = build_sensitivity_stats(events_df, RETURN_HORIZONS_MINUTES)
    if stats_df.empty:
        print("No sensitivity stats produced.")
        return
    stats_df.to_csv(args.stats_output, index=False)
    print(f"Saved sensitivity stats: {args.stats_output}")

    matrix_df = build_sensitivity_matrix(stats_df, args.scenario_delta_k)
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

    forecast_k = args.forecast_k
    if forecast_k is None:
        forecast_k = float(events_df.sort_values("date").iloc[-1]["forecast"] / 1000.0)
    projections = build_projection_scenarios(stats_df, forecast_k=forecast_k, scenario_delta_k=args.scenario_delta_k)
    if not projections.empty:
        projections.to_csv(args.output, index=False)
        print(f"Saved scenario projections: {args.output}")

    pair_plot_paths = plot_returns_by_delta(events_df, stats_df, OUTPUT_DIR)
    if pair_plot_paths:
        print("Saved return-vs-delta plots:")
        for path in pair_plot_paths:
            print(f"  {path}")

    compare_plot = plot_yesterday_vs_predicted(comparison_df, OUTPUT_DIR)
    if compare_plot is not None:
        print(f"Saved yesterday comparison plot: {compare_plot}")


if __name__ == "__main__":
    main()
