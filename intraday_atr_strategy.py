#!/usr/bin/env python3
"""Intraday ATR trend strategy backtest with multiple exit policies on EUR/USD - OPTIMIZED."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from market_data import DEFAULT_MARKET_BARS_DB, load_cached_day

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
OUTPUT_DIR = ROOT / "results"
DEFAULT_CACHE_DIR = DEFAULT_MARKET_BARS_DB
EURUSD_PAIR = "EUR/USD"
EURUSD_SYMBOL = "EURUSD"


def atr_band_breakout_entries(df: pd.DataFrame, atr_period: int = 14, multiplier: float = 1.0) -> pd.DataFrame:
    """Identify ATR band breakout entries."""
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    df["returns"] = df["close"].pct_change()
    df["atr"] = df["returns"].rolling(window=atr_period).std() * np.sqrt(1440) * multiplier
    
    df["session_open"] = df.groupby(df["timestamp"].dt.date)["close"].transform("first")
    df["upper_band"] = df["session_open"] + df["atr"]
    df["lower_band"] = df["session_open"] - df["atr"]
    
    df["session_max"] = df.groupby(df["timestamp"].dt.date)["high"].cummax()
    df["session_min"] = df.groupby(df["timestamp"].dt.date)["low"].cummin()
    
    df["entry_long"] = (df["close"] > df["upper_band"]) & (df["high"] > df["session_max"].shift(1))
    df["entry_short"] = (df["close"] < df["lower_band"]) & (df["low"] < df["session_min"].shift(1))
    
    return df


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate VWAP."""
    vwaps = []
    for date_val, group in df.groupby(df["timestamp"].dt.date):
        typical_price = (group["high"] + group["low"] + group["close"]) / 3
        vwap = (typical_price * group["volume"]).cumsum() / group["volume"].cumsum()
        vwaps.append(vwap.reindex(group.index).values)
    return pd.Series(np.concatenate(vwaps), index=df.index)


def calculate_psar(df: pd.DataFrame, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """Calculate Parabolic SAR."""
    psars = []
    for date_val, group in df.groupby(df["timestamp"].dt.date):
        group = group.copy()
        psar = [group["open"].iloc[0]]
        ep = [group["high"].iloc[0]]
        af = [af_start]
        
        for i in range(1, len(group)):
            prev_psar = psar[-1]
            prev_ep = ep[-1]
            prev_af = af[-1]
            current_close = group["close"].iloc[i]
            
            new_psar = prev_psar + prev_af * (prev_ep - prev_psar)
            new_psar = min(new_psar, current_close) if new_psar > current_close else new_psar
            
            if group["high"].iloc[i] > prev_ep:
                new_ep = group["high"].iloc[i]
                new_af = min(prev_af + af_step, af_max)
            else:
                new_ep = prev_ep
                new_af = prev_af
            
            psar.append(new_psar)
            ep.append(new_ep)
            af.append(new_af)
        
        psars.append(pd.Series(psar, index=group.index))
    
    return pd.concat(psars)


def calculate_returns_simple(
    entries: pd.DataFrame,
    df: pd.DataFrame,
    exit_type: str,
) -> List[Dict]:
    """Calculate returns for a given exit type - optimized."""
    trades = []
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    for idx, row in entries.iterrows():
        if not (row["entry_long"] or row["entry_short"]):
            continue
        
        entry_time = row["timestamp"]
        entry_price = row["close"]
        direction = 1 if row["entry_long"] else -1
        entry_date = entry_time.date()
        
        day_df = df[df["timestamp"].dt.date == entry_date]
        day_df = day_df[day_df["timestamp"] >= entry_time].copy()
        if len(day_df) == 0:
            continue
        
        if exit_type == "session_open":
            session_open = day_df.iloc[0]["close"]
            exit_idx = day_df[day_df["close"] <= session_open].index
            if len(exit_idx) > 0:
                exit_idx = exit_idx[0]
            else:
                exit_idx = day_df.index[-1]
            exit_price = day_df.loc[exit_idx, "close"]
            exit_time = day_df.loc[exit_idx, "timestamp"]
        
        elif exit_type == "midline":
            day_max = day_df["high"].cummax()
            day_min = day_df["low"].cummin()
            midline = (day_max + day_min) / 2
            if direction == 1:
                exit_idx = day_df[day_df["close"] <= midline].index
            else:
                exit_idx = day_df[day_df["close"] >= midline].index
            if len(exit_idx) > 0:
                exit_idx = exit_idx[0]
            else:
                exit_idx = day_df.index[-1]
            exit_price = day_df.loc[exit_idx, "close"]
            exit_time = day_df.loc[exit_idx, "timestamp"]
        
        elif exit_type == "vwap":
            typical_price = (day_df["high"] + day_df["low"] + day_df["close"]) / 3
            vwap = (typical_price * day_df["volume"]).cumsum() / day_df["volume"].cumsum()
            if direction == 1:
                exit_idx = day_df[day_df["close"] <= vwap].index
            else:
                exit_idx = day_df[day_df["close"] >= vwap].index
            if len(exit_idx) > 0:
                exit_idx = exit_idx[0]
            else:
                exit_idx = day_df.index[-1]
            exit_price = day_df.loc[exit_idx, "close"]
            exit_time = day_df.loc[exit_idx, "timestamp"]
        
        elif exit_type == "psar":
            psar = calculate_psar(day_df)
            if direction == 1:
                exit_idx = day_df[day_df["close"] <= psar].index
            else:
                exit_idx = day_df[day_df["close"] >= psar].index
            if len(exit_idx) > 0:
                exit_idx = exit_idx[0]
            else:
                exit_idx = day_df.index[-1]
            exit_price = day_df.loc[exit_idx, "close"]
            exit_time = day_df.loc[exit_idx, "timestamp"]
        
        else:
            continue
        
        hold_time = (exit_time - entry_time).total_seconds() / 60
        if hold_time <= 0 or hold_time > 480:
            continue
        
        pct_return = ((exit_price - entry_price) / entry_price) * direction * 10000
        
        trades.append({
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "direction": direction,
            "hold_minutes": hold_time,
            "pct_return": pct_return,
            "exit_type": exit_type,
        })
    
    return trades


def run_backtest(
    start_date: date,
    end_date: date,
    cache_dir: Path,
    cache_format: str = "parquet",
) -> Dict[str, pd.DataFrame]:
    """Run backtest for all exit types."""
    exit_types = ["session_open", "midline", "vwap", "psar"]
    
    all_trades = {exit_type: [] for exit_type in exit_types}
    
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:
            df = load_cached_day(cache_dir, EURUSD_PAIR, current, fmt=cache_format)
            
            if df.empty:
                current += timedelta(days=1)
                continue
            
            entries = atr_band_breakout_entries(df)
            
            for exit_type in exit_types:
                trades = calculate_returns_simple(entries, df, exit_type)
                for trade in trades:
                    trade["date"] = current
                all_trades[exit_type].extend(trades)
        
        current += timedelta(days=1)
    
    results = {}
    for exit_type in exit_types:
        if all_trades[exit_type]:
            df_trades = pd.DataFrame(all_trades[exit_type])
            results[exit_type] = df_trades
        else:
            results[exit_type] = pd.DataFrame()
    
    return results


def calculate_performance(trades_df: pd.DataFrame) -> Dict:
    """Calculate performance metrics."""
    if trades_df.empty:
        return {}
    
    returns = trades_df["pct_return"].dropna()
    if len(returns) == 0:
        return {}
    
    total_return = (1 + returns / 10000).prod() - 1
    winning_trades = returns[returns > 0]
    losing_trades = returns[returns < 0]
    
    avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
    
    sharpe = np.sqrt(252 * 8) * returns.mean() / returns.std() if len(returns) > 1 else 0
    
    cumulative = (1 + returns / 10000).cumprod()
    max_drawdown = (cumulative / cumulative.cummax() - 1).min()
    
    return {
        "exit_type": trades_df["exit_type"].iloc[0] if len(trades_df) > 0 else "unknown",
        "n_trades": len(returns),
        "win_rate": len(winning_trades) / len(returns) * 100 if len(returns) > 0 else 0,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if len(losing_trades) > 0 and avg_loss != 0 else 0,
        "total_return_pct": total_return * 10000,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_drawdown * 10000,
        "avg_return": returns.mean(),
        "std_return": returns.std(),
    }


def print_results(results: Dict[str, pd.DataFrame]) -> None:
    """Print performance comparison."""
    print("\n" + "=" * 90)
    print("INTRADAY ATR TREND STRATEGY - EXIT POLICY COMPARISON")
    print("=" * 90)
    
    metrics = []
    for exit_type, trades_df in results.items():
        perf = calculate_performance(trades_df)
        if perf:
            metrics.append(perf)
    
    if not metrics:
        print("No trades generated.")
        return
    
    metrics_df = pd.DataFrame(metrics)
    
    print("\n" + "-" * 90)
    print(f"{'Exit Type':<15} {'Trades':>8} {'Win%':>8} {'Total Ret':>12} {'Sharpe':>10} {'Max DD':>10}")
    print("-" * 90)
    
    for _, row in metrics_df.iterrows():
        print(
            f"{row['exit_type']:<15} {int(row['n_trades']):>8} "
            f"{row['win_rate']:>7.1f}% {row['total_return_pct']:>11.2f} "
            f"{row['sharpe_ratio']:>9.2f} {row['max_drawdown_pct']:>9.2f}"
        )
    
    print("-" * 90)
    
    best_sharpe = metrics_df.loc[metrics_df["sharpe_ratio"].idxmax()]
    best_return = metrics_df.loc[metrics_df["total_return_pct"].idxmax()]
    
    print(f"\nBest Sharpe: {best_sharpe['exit_type']} ({best_sharpe['sharpe_ratio']:.2f})")
    print(f"Best Return: {best_return['exit_type']} ({best_return['total_return_pct']:.2f}%)")


def create_ensemble_backtest(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create ensemble by combining all exit type trades."""
    all_trades = []
    
    for exit_type, trades_df in results.items():
        if not trades_df.empty:
            trades_df = trades_df[["entry_time", "exit_time", "entry_price", "exit_price", "direction", "pct_return"]]
            trades_df["exit_type"] = exit_type
            all_trades.append(trades_df)
    
    if not all_trades:
        return pd.DataFrame()
    
    ensemble_df = pd.concat(all_trades, ignore_index=True)
    return ensemble_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Intraday ATR trend backtest")
    parser.add_argument("--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--cache-format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "eurusd_intraday_ensemble.csv")
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.cache_dir.suffix == ".db":
        args.cache_dir.parent.mkdir(parents=True, exist_ok=True)
    else:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
    
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    
    print(f"Running backtest from {start_date} to {end_date}")
    print(f"Cache: {args.cache_dir} ({args.cache_format})")
    
    results = run_backtest(
        start_date=start_date,
        end_date=end_date,
        cache_dir=args.cache_dir,
        cache_format=args.cache_format,
    )
    
    print_results(results)
    
    ensemble_df = create_ensemble_backtest(results)
    if not ensemble_df.empty:
        ensemble_df.to_csv(args.output, index=False)
        print(f"\nSaved ensemble trades: {args.output}")
        
        ensemble_perf = calculate_performance(ensemble_df)
        print(f"\nEnsemble Performance:")
        print(f"  Total Trades: {int(ensemble_perf['n_trades'])}")
        print(f"  Win Rate: {ensemble_perf['win_rate']:.1f}%")
        print(f"  Total Return: {ensemble_perf['total_return_pct']:.2f} bps")
        print(f"  Sharpe: {ensemble_perf['sharpe_ratio']:.2f}")
        print(f"  Max DD: {ensemble_perf['max_drawdown_pct']:.2f} bps")


if __name__ == "__main__":
    main()
