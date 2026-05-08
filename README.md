# NFP Surprise Impact Backtester

Backtests NFP (Non-Farm Payrolls) surprise impact on major FX pairs using Dukascopy 1-minute data.

## What Changed

The Dukascopy ingestion flow now uses:

- zero-based month URL handling (with fallback)
- symbol-aware price scaling (fixes JPY pair prices)
- retry/backoff + local day-level cache
- dedicated scripts for bulk fetch and incremental updates

## Requirements

```bash
pip install -r requirements.txt
```

## Data Consumption Scripts

### 1) Bulk fetch a date range

```bash
python scripts/fetch_dukascopy_data.py \
  --start-date 2026-01-01 \
  --end-date 2026-05-08
```

Defaults:

- pairs: `EUR/USD GBP/USD USD/CAD USD/JPY`
- cache directory: `/home/gjones/work/projects/nfp/data/dukascopy_minute`
- cache format: `parquet`

### 2) Incremental catch-up to current date

```bash
python scripts/update_dukascopy_data.py
```

If no cache exists for a pair, this bootstraps the last 45 days by default.

## Backtests

### Standard backtest

```bash
python nfp_backtest.py --years 5
```

Equivalent explicit script:

```bash
python nfp_backtest_dukascopy.py --years 5
```

Optional event CSV input (for up-to-date NFP releases):

```bash
python nfp_backtest_dukascopy.py \
  --events-csv /path/to/nfp_events.csv \
  --years 5
```

`events-csv` must include columns: `date,actual,forecast`.

### Full analysis + scenarios

```bash
python nfp_backtest_full.py --years 5
```

Outputs:

- `results/nfp_events_dukascopy.csv` (or `nfp_events_full.csv` from full script)
- `results/nfp_impact_dukascopy.csv`
- `results/nfp_stats_full.csv`
- `results/nfp_scenarios.csv`

## Charts (Real Data)

```bash
python nfp_charts.py --days 5
```

This now uses recent Dukascopy candles from cache (and can refetch with `--force-refresh`) instead of synthetic data.

## Notes

- NFP release is modeled at `08:30 America/New_York` and converted to UTC using timezone rules.
- Cached minute files are stored per pair/day to keep refreshes fast and deterministic.
