#!/usr/bin/env python3
"""Run multi-event surprise sensitivity and forward scenario analysis for FX."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cloudscraper
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market_data import DEFAULT_MARKET_BARS_DB, DEFAULT_PAIRS, load_bar_window
from nfp_backtest_dukascopy import RETURN_HORIZONS_MINUTES, horizon_label

DEFAULT_HISTORY_EVENTS = ROOT / "data" / "forexfactory_events_all.csv"
DEFAULT_CACHE_DIR = DEFAULT_MARKET_BARS_DB
DEFAULT_OUTPUT_DIR = ROOT / "results"
DEFAULT_UPCOMING_EVENTS = ROOT / "data" / "forexfactory_upcoming_events.csv"
G7_RATES_SCRIPT = Path("/media/gjones/work/projects/scrape/g7_fx_rates.py")
FF_SCRIPT = ROOT / "scripts" / "fetch_forexfactory_nfp.py"


def parse_csv_floats(value: str) -> List[float]:
    values: List[float] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("expected at least one numeric value")
    return values


def parse_csv_strings(value: str) -> List[str]:
    values = [token.strip().lower() for token in value.split(",") if token.strip()]
    if not values:
        raise ValueError("expected at least one string value")
    return values


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Identify surprise-sensitive market-moving economic events and produce "
            "daily/weekly scenario analysis using latest FX spot snapshot"
        )
    )
    parser.add_argument("--history-events-csv", type=Path, default=DEFAULT_HISTORY_EVENTS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--cache-format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--years", type=int, default=5, help="Lookback window for historical calibration")
    parser.add_argument("--min-events", type=int, default=10, help="Minimum history points per series regression")
    parser.add_argument(
        "--horizons-minutes",
        type=parse_csv_floats,
        default=RETURN_HORIZONS_MINUTES,
        help="Comma-separated return horizons in minutes",
    )
    parser.add_argument(
        "--top-series",
        type=int,
        default=40,
        help="How many top market-moving event series to carry into horizon analysis",
    )
    parser.add_argument(
        "--impact-filter",
        type=parse_csv_strings,
        default=["high", "medium"],
        help="Comma-separated impact classes used for upcoming scenario selection",
    )
    parser.add_argument(
        "--scenario-z",
        type=parse_csv_floats,
        default=[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
        help="Scenario surprise multipliers in sigma units",
    )
    parser.add_argument(
        "--scenario-raw-deltas",
        type=parse_csv_floats,
        default=None,
        help=(
            "Optional raw surprise deltas to add to scenarios (event native units, "
            "for example -25000,-5000,0,5000,25000)"
        ),
    )
    parser.add_argument(
        "--upcoming-start-date",
        type=parse_date,
        default=date.today(),
        help="Upcoming horizon start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=7,
        help="Number of days forward for event horizon",
    )
    parser.add_argument(
        "--upcoming-events-csv",
        type=Path,
        default=DEFAULT_UPCOMING_EVENTS,
        help="Persisted upcoming events CSV for the selected horizon",
    )
    parser.add_argument(
        "--skip-fetch-upcoming",
        action="store_true",
        help="Do not call ForexFactory for upcoming events; use existing upcoming-events-csv",
    )
    parser.add_argument(
        "--refresh-history",
        action="store_true",
        help="Refresh historical ForexFactory files before analysis",
    )
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout for ForexFactory week requests")
    parser.add_argument("--max-retries", type=int, default=4, help="Retries per ForexFactory week request")
    parser.add_argument("--sleep-seconds", type=float, default=0.2, help="Delay between ForexFactory week requests")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def run_cmd(cmd: List[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _to_utc_naive(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    return ts.dt.tz_convert("UTC").dt.tz_localize(None)


def load_history_events(path: Path, years: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"History events file not found: {path}")

    df = pd.read_csv(path)
    required = {
        "event_id",
        "event_base_id",
        "date",
        "release_time_utc",
        "event_name",
        "currency",
        "impact",
        "actual",
        "forecast",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"history events CSV missing columns: {missing}")

    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce")
    df["event_base_id"] = pd.to_numeric(df["event_base_id"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["release_time_utc"] = _to_utc_naive(df["release_time_utc"])
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    df["forecast"] = pd.to_numeric(df["forecast"], errors="coerce")
    df["surprise"] = df["actual"] - df["forecast"]

    df = df.dropna(subset=["event_id", "event_base_id", "date", "release_time_utc", "event_name", "currency"]).copy()
    df["event_id"] = df["event_id"].astype("int64")
    df["event_base_id"] = df["event_base_id"].astype("int64")

    cutoff = datetime.utcnow().date() - timedelta(days=years * 365)
    df = df[df["date"].dt.date >= cutoff].copy()

    df = df[df["actual"].notna() & df["forecast"].notna()].copy()
    df = df.sort_values(["release_time_utc", "event_id"]).drop_duplicates(subset=["event_id"], keep="first")
    return df.reset_index(drop=True)


def load_pair_candles(
    cache_dir: Path,
    pair: str,
    cache_format: str,
    start_day: date,
    end_day: date,
) -> pd.DataFrame:
    candles = load_bar_window(
        cache_dir,
        pair,
        pd.Timestamp(start_day),
        pd.Timestamp(end_day) + pd.Timedelta(days=1),
        columns=("timestamp", "close"),
    )
    if candles.empty:
        return candles
    candles["close"] = pd.to_numeric(candles["close"], errors="coerce")
    candles = candles.dropna(subset=["timestamp", "close"])
    return candles.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)


def _merge_forward_price(
    left_df: pd.DataFrame,
    right_candles: pd.DataFrame,
    left_on: str,
    new_col: str,
    *,
    tolerance_minutes: int = 5,
) -> pd.DataFrame:
    merge_df = pd.merge_asof(
        left_df.sort_values(left_on),
        right_candles[["timestamp", "close"]].sort_values("timestamp"),
        left_on=left_on,
        right_on="timestamp",
        direction="forward",
        tolerance=pd.Timedelta(minutes=tolerance_minutes),
    )
    merge_df = merge_df.rename(columns={"close": new_col}).drop(columns=["timestamp"])
    return merge_df


def append_event_returns(
    events_df: pd.DataFrame,
    *,
    cache_dir: Path,
    cache_format: str,
    horizons_minutes: Sequence[int],
    pairs: Sequence[str],
) -> pd.DataFrame:
    if events_df.empty:
        return events_df.copy()

    out = events_df.copy()
    start_day = out["release_time_utc"].min().date()
    max_horizon = max(int(v) for v in horizons_minutes)
    end_day = (out["release_time_utc"].max() + timedelta(minutes=max_horizon + 5)).date()

    for pair in pairs:
        print(f"Loading candles for {pair} ...")
        candles = load_pair_candles(cache_dir, pair, cache_format, start_day, end_day)
        if candles.empty:
            print(f"  No cached candles found for {pair} in {start_day} -> {end_day}")
            continue

        work = out[["event_id", "release_time_utc"]].copy().sort_values("release_time_utc")
        work = _merge_forward_price(work, candles, "release_time_utc", "base_price")

        for minutes in horizons_minutes:
            minutes_int = int(minutes)
            label = horizon_label(minutes_int)
            metric = f"{pair}_returns_{label}"
            target_col = f"target_price_{minutes_int}"

            targets = work[["event_id", "release_time_utc"]].copy()
            targets["target_time"] = targets["release_time_utc"] + pd.to_timedelta(minutes_int, unit="m")
            targets = _merge_forward_price(targets, candles, "target_time", target_col)

            merged = work[["event_id", "base_price"]].merge(
                targets[["event_id", target_col]],
                on="event_id",
                how="left",
            )
            returns = (merged[target_col] - merged["base_price"]) / merged["base_price"] * 100.0
            out = out.merge(pd.DataFrame({"event_id": merged["event_id"], metric: returns}), on="event_id", how="left")

    return out


def build_series_metadata(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for base_id, group in events_df.groupby("event_base_id"):
        impact_mode = group["impact"].dropna().astype(str)
        impact_value = impact_mode.mode().iloc[0] if not impact_mode.empty else ""
        rows.append(
            {
                "event_base_id": int(base_id),
                "event_name": str(group.iloc[0]["event_name"]),
                "currency": str(group.iloc[0]["currency"]),
                "impact": impact_value,
                "history_events": int(len(group)),
                "surprise_mean": float(group["surprise"].mean()),
                "surprise_std": float(group["surprise"].std(ddof=1)) if len(group) > 1 else np.nan,
                "surprise_median": float(group["surprise"].median()),
                "surprise_abs_mean": float(group["surprise"].abs().mean()),
            }
        )

    return pd.DataFrame(rows)


def build_sensitivity_stats(
    events_with_returns: pd.DataFrame,
    horizons_minutes: Sequence[int],
    *,
    min_events: int,
    pairs: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if events_with_returns.empty:
        return pd.DataFrame(rows)

    group_cols = ["event_base_id", "event_name", "currency", "impact"]
    for keys, group in events_with_returns.groupby(group_cols):
        event_base_id, event_name, currency, impact = keys

        for pair in pairs:
            for minutes in horizons_minutes:
                minutes_int = int(minutes)
                label = horizon_label(minutes_int)
                metric = f"{pair}_returns_{label}"
                if metric not in group.columns:
                    continue

                subset = group[["surprise", metric]].dropna()
                if len(subset) < min_events:
                    continue

                x = subset["surprise"].to_numpy(dtype=float)
                y = subset[metric].to_numpy(dtype=float)
                x_std = float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")
                if not np.isfinite(x_std) or x_std <= 0:
                    continue

                slope, intercept = np.polyfit(x, y, 1)
                y_hat = intercept + slope * x
                residual = y - y_hat
                rmse = float(np.sqrt(np.mean(np.square(residual))))

                corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else np.nan
                r2 = corr * corr if np.isfinite(corr) else np.nan

                slope_per_sigma = float(slope * x_std)
                mover_score = float(abs(slope_per_sigma) * abs(corr) * math.sqrt(len(subset))) if np.isfinite(corr) else np.nan

                rows.append(
                    {
                        "event_base_id": int(event_base_id),
                        "event_name": str(event_name),
                        "currency": str(currency),
                        "impact": str(impact),
                        "pair": pair,
                        "horizon_label": label,
                        "horizon_minutes": minutes_int,
                        "n_events": int(len(subset)),
                        "slope_pct_per_unit_surprise": float(slope),
                        "intercept_pct": float(intercept),
                        "surprise_std": x_std,
                        "slope_pct_per_1sd_surprise": slope_per_sigma,
                        "corr": corr,
                        "r2": r2,
                        "rmse_pct": rmse,
                        "mean_return_pct": float(np.mean(y)),
                        "std_return_pct": float(np.std(y, ddof=1)) if len(y) > 1 else np.nan,
                        "mean_abs_return_pct": float(np.mean(np.abs(y))),
                        "mover_score": mover_score,
                    }
                )

    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        return stats_df

    stats_df = stats_df.sort_values(["mover_score", "n_events"], ascending=[False, False]).reset_index(drop=True)
    return stats_df


def build_market_movers(stats_df: pd.DataFrame) -> pd.DataFrame:
    if stats_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for base_id, group in stats_df.groupby("event_base_id"):
        ranked = group.sort_values("mover_score", ascending=False)
        best = ranked.iloc[0]
        rows.append(
            {
                "event_base_id": int(base_id),
                "event_name": str(best["event_name"]),
                "currency": str(best["currency"]),
                "impact": str(best["impact"]),
                "max_mover_score": float(best["mover_score"]),
                "best_pair": str(best["pair"]),
                "best_horizon_label": str(best["horizon_label"]),
                "best_horizon_minutes": int(best["horizon_minutes"]),
                "best_corr": float(best["corr"]),
                "best_slope_pct_per_1sd_surprise": float(best["slope_pct_per_1sd_surprise"]),
                "best_n_events": int(best["n_events"]),
                "avg_mover_score": float(group["mover_score"].mean()),
                "median_abs_corr": float(group["corr"].abs().median()),
                "stats_rows": int(len(group)),
            }
        )

    movers = pd.DataFrame(rows)
    movers = movers.sort_values(["max_mover_score", "best_n_events"], ascending=[False, False]).reset_index(drop=True)
    movers["rank"] = np.arange(1, len(movers) + 1)
    return movers


def fetch_upcoming_events(
    start_date: date,
    end_date: date,
    *,
    timeout_seconds: int,
    max_retries: int,
    sleep_seconds: float,
) -> pd.DataFrame:
    ff = _load_module(FF_SCRIPT, "fetch_forexfactory_nfp_module")
    scraper = cloudscraper.create_scraper(browser={"browser": "chrome", "platform": "linux", "desktop": True})

    week_starts = list(ff._iter_week_starts(start_date, end_date))
    rows: List[Dict] = []

    print(f"Fetching upcoming ForexFactory events for {start_date} -> {end_date}")
    for idx, week_start in enumerate(week_starts, start=1):
        slug = ff._week_slug(week_start)
        print(f"  [{idx}/{len(week_starts)}] {slug}")
        html = ff._fetch_week_page(
            scraper,
            week_start,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        days_payload = ff._extract_days_array(html)
        rows.extend(ff._extract_all_events(days_payload, start_date, end_date))
        time.sleep(max(0.0, sleep_seconds))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["release_time_utc", "event_id"]).drop_duplicates(subset=["event_id"], keep="first")
    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce")
    df["event_base_id"] = pd.to_numeric(df["event_base_id"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["release_time_utc"] = _to_utc_naive(df["release_time_utc"])
    df["forecast"] = pd.to_numeric(df["forecast"], errors="coerce")
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    return df.reset_index(drop=True)


def load_latest_spot_map(script_path: Path, pairs: Sequence[str]) -> Dict[str, float]:
    if not script_path.exists():
        raise FileNotFoundError(f"g7 FX snapshot script not found: {script_path}")

    mod = _load_module(script_path, "g7_fx_rates_module")
    items = mod.fetch_g7_rates()
    if not items:
        raise RuntimeError("Failed to fetch live rates from g7_fx_rates.fetch_g7_rates()")

    def _extract_mid_with_fallback(prices: Dict) -> Optional[float]:
        for key in ("current", "lastClose", "previousLastClose"):
            block = prices.get(key, {}) if isinstance(prices, dict) else {}
            bid = block.get("bid")
            ask = block.get("ask")
            if bid is None or ask is None:
                continue
            try:
                return (float(bid) + float(ask)) / 2.0
            except (TypeError, ValueError):
                continue
        return None

    spot_map: Dict[str, float] = {}
    for pair in pairs:
        symbol = pair.replace("/", "")
        instrument_data = mod.find_instrument(items, symbol)
        if not instrument_data:
            continue
        prices = instrument_data.get("prices", {})
        processed = mod.process_prices(pair, prices)
        if processed and processed.get("mid") is not None:
            spot_map[pair] = float(processed["mid"])
            continue
        fallback_mid = _extract_mid_with_fallback(prices)
        if fallback_mid is not None:
            spot_map[pair] = float(fallback_mid)

    return spot_map


def build_upcoming_scenarios(
    upcoming_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    *,
    selected_series_ids: Sequence[int],
    impact_filter: Optional[Sequence[str]],
    scenario_z: Sequence[float],
    scenario_raw_deltas: Optional[Sequence[float]],
    spot_map: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if upcoming_df.empty or stats_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    selected_set = {int(v) for v in selected_series_ids}
    upcoming_filtered = upcoming_df[upcoming_df["event_base_id"].isin(selected_set)].copy()
    if impact_filter:
        impact_set = {str(v).lower() for v in impact_filter}
        upcoming_filtered = upcoming_filtered[upcoming_filtered["impact"].astype(str).str.lower().isin(impact_set)].copy()
    if upcoming_filtered.empty:
        return upcoming_filtered, pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for event in upcoming_filtered.itertuples(index=False):
        event_base_id = int(event.event_base_id)
        forecast_value = float(event.forecast) if pd.notna(event.forecast) else np.nan
        event_stats = stats_df[stats_df["event_base_id"] == event_base_id]
        if event_stats.empty or not np.isfinite(forecast_value):
            continue

        for stat in event_stats.itertuples(index=False):
            pair = str(stat.pair)
            surprise_std = float(stat.surprise_std) if pd.notna(stat.surprise_std) else np.nan
            if not np.isfinite(surprise_std) or surprise_std <= 0:
                continue

            deltas: List[Tuple[str, float]] = []
            for z in scenario_z:
                delta = float(z) * surprise_std
                deltas.append((f"z={z:+.2f}", delta))

            if scenario_raw_deltas is not None:
                for d in scenario_raw_deltas:
                    deltas.append((f"delta={float(d):+.6g}", float(d)))

            for scenario_name, delta in deltas:
                predicted_return = float(stat.intercept_pct + stat.slope_pct_per_unit_surprise * delta)
                scenario_actual = forecast_value + delta
                spot = spot_map.get(pair)
                scenario_spot = np.nan
                if spot is not None and np.isfinite(spot):
                    scenario_spot = float(spot * (1.0 + predicted_return / 100.0))

                rows.append(
                    {
                        "event_date": event.date.strftime("%Y-%m-%d") if pd.notna(event.date) else "",
                        "release_time_utc": event.release_time_utc.strftime("%Y-%m-%d %H:%M:%S")
                        if pd.notna(event.release_time_utc)
                        else "",
                        "event_id": int(event.event_id),
                        "event_base_id": event_base_id,
                        "event_name": str(event.event_name),
                        "currency": str(event.currency),
                        "impact": str(event.impact),
                        "forecast": forecast_value,
                        "pair": pair,
                        "horizon_label": str(stat.horizon_label),
                        "horizon_minutes": int(stat.horizon_minutes),
                        "scenario": scenario_name,
                        "scenario_delta": float(delta),
                        "scenario_actual": float(scenario_actual),
                        "predicted_return_pct": predicted_return,
                        "live_spot": float(spot) if spot is not None else np.nan,
                        "predicted_spot": scenario_spot,
                        "slope_pct_per_unit_surprise": float(stat.slope_pct_per_unit_surprise),
                        "intercept_pct": float(stat.intercept_pct),
                        "corr": float(stat.corr) if pd.notna(stat.corr) else np.nan,
                        "r2": float(stat.r2) if pd.notna(stat.r2) else np.nan,
                        "mover_score": float(stat.mover_score) if pd.notna(stat.mover_score) else np.nan,
                        "n_events": int(stat.n_events),
                    }
                )

    scenarios = pd.DataFrame(rows)
    if not scenarios.empty:
        scenarios = scenarios.sort_values(["release_time_utc", "event_base_id", "pair", "horizon_minutes", "scenario"]).reset_index(
            drop=True
        )

    return upcoming_filtered, scenarios


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.upcoming_events_csv.parent.mkdir(parents=True, exist_ok=True)

    horizons_minutes = [int(v) for v in args.horizons_minutes]
    pairs = list(DEFAULT_PAIRS.keys())

    if args.refresh_history:
        run_cmd(
            [
                sys.executable,
                "scripts/fetch_forexfactory_nfp.py",
                "--all-output",
                str(args.history_events_csv),
                "--nfp-output",
                str(ROOT / "data" / "nfp_events_forexfactory.csv"),
            ]
        )

    print(f"Loading historical events: {args.history_events_csv}")
    history_df = load_history_events(args.history_events_csv, years=args.years)
    if history_df.empty:
        raise SystemExit("No historical events available after filtering")

    print(f"Historical events with actual+forecast: {len(history_df)}")
    events_with_returns = append_event_returns(
        history_df,
        cache_dir=args.cache_dir,
        cache_format=args.cache_format,
        horizons_minutes=horizons_minutes,
        pairs=pairs,
    )

    events_output = args.output_dir / "event_surprise_events_with_returns.csv"
    events_with_returns.to_csv(events_output, index=False)
    print(f"Saved event-level returns: {events_output}")

    series_meta = build_series_metadata(events_with_returns)
    meta_output = args.output_dir / "event_series_metadata.csv"
    if not series_meta.empty:
        series_meta.to_csv(meta_output, index=False)
        print(f"Saved series metadata: {meta_output}")

    stats_df = build_sensitivity_stats(
        events_with_returns,
        horizons_minutes,
        min_events=args.min_events,
        pairs=pairs,
    )
    if stats_df.empty:
        raise SystemExit("No sensitivity stats produced. Try lower --min-events or check data coverage")

    stats_output = args.output_dir / "event_sensitivity_stats.csv"
    stats_df.to_csv(stats_output, index=False)
    print(f"Saved sensitivity stats: {stats_output}")

    movers_df = build_market_movers(stats_df)
    movers_output = args.output_dir / "event_market_movers.csv"
    movers_df.to_csv(movers_output, index=False)
    print(f"Saved market mover ranking: {movers_output}")

    filtered_movers = movers_df[movers_df["impact"].astype(str).str.lower().isin(set(args.impact_filter))].copy()
    filtered_movers = filtered_movers.sort_values(["max_mover_score", "best_n_events"], ascending=[False, False]).reset_index(
        drop=True
    )
    filtered_movers["filtered_rank"] = np.arange(1, len(filtered_movers) + 1)
    filtered_movers_output = args.output_dir / "event_market_movers_filtered.csv"
    filtered_movers.to_csv(filtered_movers_output, index=False)
    print(f"Saved filtered mover ranking: {filtered_movers_output}")

    selected_ids = filtered_movers.head(args.top_series)["event_base_id"].astype(int).tolist()
    print(
        f"Selected top {len(selected_ids)} event series for horizon scenarios "
        f"(impact filter: {','.join(args.impact_filter)})"
    )

    upcoming_end = args.upcoming_start_date + timedelta(days=max(0, args.horizon_days))
    if args.skip_fetch_upcoming:
        if not args.upcoming_events_csv.exists():
            raise SystemExit(
                f"--skip-fetch-upcoming was set, but file does not exist: {args.upcoming_events_csv}"
            )
        upcoming_df = pd.read_csv(args.upcoming_events_csv)
        upcoming_df["date"] = pd.to_datetime(upcoming_df["date"], errors="coerce")
        upcoming_df["release_time_utc"] = _to_utc_naive(upcoming_df["release_time_utc"])
        upcoming_df["event_id"] = pd.to_numeric(upcoming_df["event_id"], errors="coerce")
        upcoming_df["event_base_id"] = pd.to_numeric(upcoming_df["event_base_id"], errors="coerce")
        upcoming_df["forecast"] = pd.to_numeric(upcoming_df["forecast"], errors="coerce")
    else:
        upcoming_df = fetch_upcoming_events(
            args.upcoming_start_date,
            upcoming_end,
            timeout_seconds=args.timeout,
            max_retries=args.max_retries,
            sleep_seconds=args.sleep_seconds,
        )
        upcoming_df.to_csv(args.upcoming_events_csv, index=False)
        print(f"Saved upcoming events: {args.upcoming_events_csv}")

    if upcoming_df.empty:
        print("No upcoming events returned for the selected horizon")
        return

    print("Fetching live G7 spot snapshot ...")
    spot_map = load_latest_spot_map(G7_RATES_SCRIPT, pairs)
    if not spot_map:
        raise SystemExit("No live spot prices were resolved for backtest pairs")

    print("Live spot snapshot:")
    for pair in pairs:
        if pair in spot_map:
            print(f"  {pair}: {spot_map[pair]:.6f}")

    upcoming_filtered, scenarios_df = build_upcoming_scenarios(
        upcoming_df,
        stats_df,
        selected_series_ids=selected_ids,
        impact_filter=args.impact_filter,
        scenario_z=args.scenario_z,
        scenario_raw_deltas=args.scenario_raw_deltas,
        spot_map=spot_map,
    )

    upcoming_filtered_output = args.output_dir / "upcoming_market_moving_events.csv"
    upcoming_filtered.to_csv(upcoming_filtered_output, index=False)
    print(f"Saved upcoming market-moving events: {upcoming_filtered_output}")

    scenarios_output = args.output_dir / "upcoming_event_scenarios.csv"
    if scenarios_df.empty:
        print("No upcoming scenarios produced (likely no forecast values for selected events yet)")
    else:
        scenarios_df.to_csv(scenarios_output, index=False)
        print(f"Saved upcoming scenario analysis: {scenarios_output}")

    print("\nTop market movers (by surprise sensitivity):")
    cols = ["rank", "event_base_id", "currency", "event_name", "best_pair", "best_horizon_label", "max_mover_score", "best_n_events"]
    print(movers_df[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
