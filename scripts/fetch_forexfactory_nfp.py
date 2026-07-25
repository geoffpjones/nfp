#!/usr/bin/env python3
"""Fetch full Forex Factory calendar history and derive an NFP-only dataset."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sys
import time
from typing import Dict, Iterable, List, Optional, Tuple

import cloudscraper
import pandas as pd
from requests import Response

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market_data import DEFAULT_MARKET_BARS_DB, available_date_range

DEFAULT_TICK_CACHE_DIR = DEFAULT_MARKET_BARS_DB
DEFAULT_ALL_OUTPUT = ROOT / "data" / "forexfactory_events_all.csv"
DEFAULT_NFP_OUTPUT = ROOT / "data" / "nfp_events_forexfactory.csv"
BASE_URL = "https://www.forexfactory.com/calendar"

NFP_EVENT_NAMES = {
    "Non-Farm Employment Change",
    "Non-Farm Payrolls",
}


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _week_start_sunday(value: date) -> date:
    # Python weekday: Monday=0, Sunday=6.
    return value - timedelta(days=(value.weekday() + 1) % 7)


def _iter_week_starts(start_date: date, end_date: date) -> Iterable[date]:
    current = _week_start_sunday(start_date)
    while current <= end_date:
        yield current
        current += timedelta(days=7)


def _week_slug(week_start: date) -> str:
    return f"{week_start.strftime('%b').lower()}{week_start.day}.{week_start.year}"


def _extract_days_array(html: str) -> List[Dict]:
    marker = "days:"
    marker_pos = html.find(marker)
    if marker_pos < 0:
        raise ValueError("Could not find Forex Factory calendar days payload")

    start = html.find("[", marker_pos)
    if start < 0:
        raise ValueError("Could not find start of days payload")

    depth = 0
    in_string = False
    escape = False
    end: Optional[int] = None

    for idx, char in enumerate(html[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == "\"":
                in_string = False
            continue

        if char == "\"":
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break

    if end is None:
        raise ValueError("Could not find end of days payload")

    return json.loads(html[start:end])


def _parse_value(raw_value: str) -> Optional[float]:
    if raw_value is None:
        return None

    text = raw_value.strip().replace(",", "").replace("−", "-")
    if text in {"", "-", "N/A", "n/a"}:
        return None

    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)([KMBT%]?)", text)
    if not match:
        return None

    value = float(match.group(1))
    suffix = match.group(2)
    if suffix == "K":
        value *= 1_000
    elif suffix == "M":
        value *= 1_000_000
    elif suffix == "B":
        value *= 1_000_000_000
    elif suffix == "T":
        value *= 1_000_000_000_000

    return value


def _fetch_week_page(
    scraper: cloudscraper.CloudScraper,
    week_start: date,
    *,
    timeout_seconds: int,
    max_retries: int,
) -> str:
    slug = _week_slug(week_start)
    url = f"{BASE_URL}?week={slug}"

    for attempt in range(1, max_retries + 1):
        try:
            response: Response = scraper.get(url, timeout=timeout_seconds)
            if response.status_code == 200 and "days:" in response.text:
                return response.text

            wait_seconds = min(8.0, 0.8 * attempt)
            print(f"[{slug}] status={response.status_code}, retrying in {wait_seconds:.1f}s")
            time.sleep(wait_seconds)
        except Exception as exc:  # noqa: BLE001
            wait_seconds = min(8.0, 0.8 * attempt)
            print(f"[{slug}] request error: {exc}; retrying in {wait_seconds:.1f}s")
            time.sleep(wait_seconds)

    raise RuntimeError(f"Failed to fetch week {slug} after {max_retries} attempts")


def _extract_all_events(days_payload: List[Dict], start_date: date, end_date: date) -> List[Dict]:
    rows: List[Dict] = []

    for day_item in days_payload:
        for event in day_item.get("events", []):
            event_ts = event.get("dateline")
            if event_ts is None:
                continue

            event_utc = datetime.fromtimestamp(int(event_ts), tz=timezone.utc)
            event_date = event_utc.date()
            if event_date < start_date or event_date > end_date:
                continue

            event_name = event.get("name", "")
            actual_raw = (event.get("actual") or "").strip()
            forecast_raw = (event.get("forecast") or "").strip()
            previous_raw = (event.get("previous") or "").strip()

            actual = _parse_value(actual_raw)
            forecast = _parse_value(forecast_raw)
            previous = _parse_value(previous_raw)
            surprise = actual - forecast if actual is not None and forecast is not None else None

            rows.append(
                {
                    "event_id": int(event["id"]),
                    "event_base_id": int(event.get("ebaseId")) if event.get("ebaseId") is not None else None,
                    "date": event_date.isoformat(),
                    "release_time_utc": event_utc.strftime("%Y-%m-%d %H:%M:%S"),
                    "event_name": event_name,
                    "prefixed_name": event.get("prefixedName"),
                    "currency": event.get("currency"),
                    "country": event.get("country"),
                    "impact": event.get("impactName"),
                    "actual": actual,
                    "forecast": forecast,
                    "previous": previous,
                    "surprise": surprise,
                    "actual_raw": actual_raw,
                    "forecast_raw": forecast_raw,
                    "previous_raw": previous_raw,
                    "source": "forexfactory",
                    "source_url": f"{BASE_URL}?day={event_date.strftime('%b').lower()}{event_date.day}.{event_date.year}#detail={event['id']}",
                }
            )

    return rows


def _filter_nfp_from_all_events(df_all: pd.DataFrame, keep_incomplete: bool) -> pd.DataFrame:
    if df_all.empty:
        return df_all.copy()

    df = df_all[(df_all["currency"] == "USD") & (df_all["event_name"].isin(NFP_EVENT_NAMES))].copy()
    if df.empty:
        return df

    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    df["forecast"] = pd.to_numeric(df["forecast"], errors="coerce")
    df["previous"] = pd.to_numeric(df["previous"], errors="coerce")
    df["surprise"] = pd.to_numeric(df.get("surprise"), errors="coerce")

    missing_surprise = df["surprise"].isna() & df["actual"].notna() & df["forecast"].notna()
    df.loc[missing_surprise, "surprise"] = df.loc[missing_surprise, "actual"] - df.loc[missing_surprise, "forecast"]

    if not keep_incomplete:
        df = df.dropna(subset=["actual", "forecast"]).copy()

    return df


def _detect_tick_range(cache_dir: Path) -> Optional[Tuple[date, date]]:
    if cache_dir.suffix == ".db":
        return available_date_range(cache_dir)

    if not cache_dir.exists():
        return None

    all_dates: List[date] = []
    for path in cache_dir.glob("*/*"):
        if path.suffix not in {".parquet", ".csv"}:
            continue
        try:
            all_dates.append(date.fromisoformat(path.stem))
        except ValueError:
            continue

    if not all_dates:
        return None

    return min(all_dates), max(all_dates)


def _resolve_range(args: argparse.Namespace) -> Tuple[date, date, str]:
    if args.start_date and args.end_date:
        return args.start_date, args.end_date, "cli dates"

    if args.match_tick_range:
        tick_range = _detect_tick_range(args.tick_cache_dir)
        if tick_range is not None:
            start = args.start_date or tick_range[0]
            end = args.end_date or tick_range[1]
            return start, end, f"tick cache range ({args.tick_cache_dir})"

    end = args.end_date or date.today()
    start = args.start_date or (end - timedelta(days=args.years * 365))
    return start, end, "rolling years"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Forex Factory events and derive NFP actual/forecast")
    parser.add_argument("--start-date", type=_parse_date, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=_parse_date, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--years", type=int, default=5, help="Used when dates are not explicitly supplied")
    parser.add_argument(
        "--tick-cache-dir",
        type=Path,
        default=DEFAULT_TICK_CACHE_DIR,
        help="Market-data source path used to infer date range",
    )
    parser.add_argument(
        "--match-tick-range",
        dest="match_tick_range",
        action="store_true",
        default=True,
        help="Infer range from market-data source when start/end are omitted (default)",
    )
    parser.add_argument(
        "--no-match-tick-range",
        dest="match_tick_range",
        action="store_false",
        help="Do not infer range from market data; use --years/date arguments only",
    )
    parser.add_argument(
        "--all-output",
        type=Path,
        default=DEFAULT_ALL_OUTPUT,
        help="Output CSV path for all fetched events",
    )
    parser.add_argument(
        "--nfp-output",
        type=Path,
        default=DEFAULT_NFP_OUTPUT,
        help="Output CSV path for NFP-only events (filtered from all-output)",
    )
    # Backward compatibility for existing command usage.
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Deprecated alias for --nfp-output",
    )
    parser.add_argument(
        "--keep-incomplete",
        action="store_true",
        help="Keep rows where actual or forecast is missing in the NFP file",
    )
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds")
    parser.add_argument("--max-retries", type=int, default=4, help="Retries per week request")
    parser.add_argument("--sleep-seconds", type=float, default=0.2, help="Delay between week requests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output is not None:
        args.nfp_output = args.output

    start_date, end_date, range_source = _resolve_range(args)
    if start_date > end_date:
        raise SystemExit("start-date cannot be after end-date")

    week_starts = list(_iter_week_starts(start_date, end_date))
    scraper = cloudscraper.create_scraper(browser={"browser": "chrome", "platform": "linux", "desktop": True})

    rows: List[Dict] = []
    print(f"Fetching Forex Factory events for {start_date} -> {end_date} ({range_source})")
    print(f"Weeks to query: {len(week_starts)}")

    for idx, week_start in enumerate(week_starts, start=1):
        slug = _week_slug(week_start)
        print(f"[{idx}/{len(week_starts)}] week={slug}")
        html = _fetch_week_page(
            scraper,
            week_start,
            timeout_seconds=args.timeout,
            max_retries=args.max_retries,
        )
        days_payload = _extract_days_array(html)
        rows.extend(_extract_all_events(days_payload, start_date, end_date))
        time.sleep(max(0.0, args.sleep_seconds))

    if not rows:
        raise SystemExit("No Forex Factory rows found for the requested range")

    df_all = pd.DataFrame(rows)
    df_all = df_all.sort_values(["release_time_utc", "event_id"]).drop_duplicates(subset=["event_id"], keep="first")

    args.all_output.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(args.all_output, index=False)
    print(f"Saved {len(df_all)} all events: {args.all_output}")

    # Read from the persisted all-events file, then filter for NFP.
    df_from_file = pd.read_csv(args.all_output)
    df_nfp = _filter_nfp_from_all_events(df_from_file, keep_incomplete=args.keep_incomplete)

    if df_nfp.empty:
        raise SystemExit("All rows were filtered out (likely missing actual/forecast)")

    args.nfp_output.parent.mkdir(parents=True, exist_ok=True)
    df_nfp.to_csv(args.nfp_output, index=False)

    print(f"Saved {len(df_nfp)} NFP events: {args.nfp_output}")
    print(df_nfp[["date", "actual", "forecast", "surprise"]].tail(5).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
