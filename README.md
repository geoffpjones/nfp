# NFP Surprise Impact Backtester

Backtests NFP (Non-Farm Payrolls) surprise impact on major FX pairs using shared 1-minute FX data.

## What Changed

FX market-data fetching has been moved out of this project. NFP reads cached
1-minute bar data from `../md/sqlite/market-bars-5y.db`; update or rebuild
that data from the shared `../md` project.

## Requirements

```bash
pip install -r requirements.txt
```

## Data Scripts

### Fetch Forex Factory NFP expected/actual history

```bash
python scripts/fetch_forexfactory_nfp.py
```

Defaults:

- infers date range from shared market-data store (`../md/sqlite/market-bars-5y.db`)
- falls back to rolling `--years 5` if no cache exists
- writes full calendar history to `data/forexfactory_events_all.csv`
- then filters that file into `data/nfp_events_forexfactory.csv` with `date,actual,forecast,surprise`

Useful overrides:

```bash
# Explicit range
python scripts/fetch_forexfactory_nfp.py --start-date 2021-06-04 --end-date 2026-05-07

# Ignore cache-derived range and use rolling years
python scripts/fetch_forexfactory_nfp.py --no-match-tick-range --years 5

# Custom output paths
python scripts/fetch_forexfactory_nfp.py \
  --all-output data/forexfactory_events_all.csv \
  --nfp-output data/nfp_events_forexfactory.csv
```

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
  --events-csv data/nfp_events_forexfactory.csv \
  --years 5
```

`events-csv` must include columns: `date,actual,forecast`.

This now computes per-event:

- `delta = actual - forecast`
- `%delta = 100 * delta / |forecast|`
- `delta_std_series` and `delta_zscore`
- returns at `1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h` for each pair

Outputs from `nfp_backtest_dukascopy.py`:

- `results/nfp_events_dukascopy.csv` (event-level deltas + all horizons)
- `results/nfp_impact_dukascopy.csv` (pair/horizon sensitivity stats)
- `results/nfp_sensitivity_matrix.csv` (predicted returns by delta scenarios)
- `results/nfp_yesterday_prediction.csv` (yesterday/latest event predicted vs realized)

### Full analysis + scenarios + plots

```bash
python nfp_backtest_full.py --years 5
```

Example with explicit scenario deltas in K jobs:

```bash
python nfp_backtest_full.py \
  --events-csv data/nfp_events_forexfactory.csv \
  --scenario-delta-k="-75,-50,-25,0,25,50,75"
```

Outputs:

- `results/nfp_events_full.csv`
- `results/nfp_stats_full.csv`
- `results/nfp_sensitivity_matrix.csv`
- `results/nfp_fx_scenarios.csv` (projection table for `forecast X`, `actual = X + delta`)
- `results/nfp_yesterday_prediction.csv`
- `results/nfp_delta_returns_*.png` (returns by delta for each pair)
- `results/nfp_yesterday_vs_predicted.png`

## Charts (Real Data)

Charts are produced by `nfp_backtest_full.py`:

- `results/nfp_delta_returns_*.png`
- `results/nfp_yesterday_vs_predicted.png`

## Cross-Event Horizon Scenario Runner

Use this to expand beyond NFP and identify event series whose surprise (`actual - forecast`) has historically moved FX.
It then fetches upcoming calendar events and creates scenario forecasts using the latest intraday G7 spot snapshot from:

- `/media/gjones/work/projects/scrape/g7_fx_rates.py`

Run a 7-day horizon analysis:

```bash
python scripts/run_event_horizon_scenarios.py --horizon-days 7
```

Useful options:

```bash
# Refresh historical ForexFactory files first
python scripts/run_event_horizon_scenarios.py \
  --refresh-history \
  --horizon-days 7

# Daily run (next 1 day) and custom scenario shocks
python scripts/run_event_horizon_scenarios.py \
  --horizon-days 1 \
  --scenario-z=-1,0,1 \
  --scenario-raw-deltas=-25000,-5000,0,5000,25000
```

Outputs:

- `results/event_surprise_events_with_returns.csv` (historical event-level surprises + return horizons)
- `results/event_series_metadata.csv` (per-series surprise distribution stats)
- `results/event_sensitivity_stats.csv` (pair/horizon surprise sensitivity coefficients)
- `results/event_market_movers.csv` (all ranked movers)
- `results/event_market_movers_filtered.csv` (impact-filtered movers used for horizon projection)
- `data/forexfactory_upcoming_events.csv` (upcoming event pull for requested date window)
- `results/upcoming_market_moving_events.csv` (upcoming events selected from filtered movers)
- `results/upcoming_event_scenarios.csv` (forecast + scenario actual values + predicted returns/spots)

## Notes

- NFP release is modeled at `08:30 America/New_York` and converted to UTC using timezone rules.
- Minute FX bars are read from the shared `../md` project.
