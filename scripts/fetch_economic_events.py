#!/usr/bin/env python3
"""Fetch all upcoming economic events from Forex Factory."""

from datetime import date, datetime, timedelta
import re
import sys
import time
from pathlib import Path

import cloudscraper
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "upcoming_market_moving_events.csv"

BASE_URL = "https://www.forexfactory.com/calendar"


def _week_start_sunday(value: date) -> date:
    return value - timedelta(days=(value.weekday() + 1) % 7)


def _week_slug(week_start: date) -> str:
    return f"{week_start.strftime('%b').lower()}{week_start.day}.{week_start.year}"


def _fetch_week_page(scraper, week_start, timeout_seconds=30, max_retries=4):
    slug = _week_slug(week_start)
    url = f"{BASE_URL}?day={slug}"
    for attempt in range(max_retries):
        try:
            resp = scraper.get(url, timeout=timeout_seconds)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2)
    return None


def _extract_days_array(html):
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
    end = None
    for idx, char in enumerate(html[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == '[':
            depth += 1
        elif char == ']':
            depth -= 1
            if depth == 0:
                end = idx
                break
    if end is None:
        raise ValueError("Could not find end of days payload")
    payload_str = html[start : end + 1]
    try:
        return json.loads(payload_str)
    except json.JSONDecodeError:
        raise ValueError("Could not parse days payload as JSON")


def _extract_all_events(days_payload, start_date, end_date):
    rows = []
    for day in days_payload:
        date_str = day.get("date")
        if not date_str:
            continue
        try:
            day_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        if day_date < start_date or day_date > end_date:
            continue
        for evt in day.get("events", []):
            rows.append({
                "event_id": evt.get("id"),
                "date": date_str,
                "time_utc": evt.get("time"),
                "event_name": evt.get("name"),
                "currency": evt.get("currency", ""),
                "impact": evt.get("impact", "low"),
                "actual": evt.get("actual"),
                "forecast": evt.get("forecast"),
                "previous": evt.get("previous"),
            })
    return rows


def main():
    scraper = cloudscraper.create_scraper(browser={"browser": "chrome", "platform": "linux", "desktop": True})
    
    # Fetch next 7 days
    today = date.today()
    end_date = today + timedelta(days=7)
    
    week_start = _week_start_sunday(today)
    week_starts = []
    current = week_start
    while current <= end_date:
        week_starts.append(current)
        current += timedelta(days=7)
    
    rows = []
    print(f"Fetching economic events for {today} -> {end_date}")
    
    for week_start in week_starts:
        slug = _week_slug(week_start)
        print(f"Fetching week: {slug}")
        html = _fetch_week_page(scraper, week_start)
        if html:
            days_payload = _extract_days_array(html)
            rows.extend(_extract_all_events(days_payload, today, end_date))
        time.sleep(0.5)
    
    if not rows:
        print("No events found")
        return
    
    df = pd.DataFrame(rows)
    df = df[df['event_name'].notna()].drop_duplicates(subset=['event_id'], keep='first')
    
    # Add prefixed names and source_url
    df['prefixed_name'] = df.apply(lambda r: f"{r['currency']} {r['event_name']}" if r['currency'] else r['event_name'], axis=1)
    df['source'] = 'forexfactory'
    df['source_url'] = df.apply(lambda r: f"https://www.forexfactory.com/calendar?day={r['date'].replace('-','')}.2026#detail={r['event_id']}", axis=1)
    
    # Parse time to UTC
    df['release_time_utc'] = pd.to_datetime(df['date'] + ' ' + df['time_utc']).dt.tz_localize('UTC')
    
    # Calculate surprise
    df['surprise'] = df.apply(lambda r: r['actual'] - r['forecast'] if r['actual'] and r['forecast'] else None, axis=1)
    
    # Format raw values
    df['actual_raw'] = df['actual'].apply(lambda x: f"{x}%" if isinstance(x, float) else x)
    df['forecast_raw'] = df['forecast'].apply(lambda x: f"{x}%" if isinstance(x, float) else x)
    df['previous_raw'] = df['previous'].apply(lambda x: f"{x}%" if isinstance(x, float) else x)
    
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(df)} events: {OUTPUT}")
    print(df[['date', 'event_name', 'actual', 'forecast']].to_string(index=False))


if __name__ == "__main__":
    import json
    main()
