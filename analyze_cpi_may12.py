#!/usr/bin/env python3
"""CPI May 12 2026 scenario analysis - final."""

import pandas as pd

# Load today's CPI data
events = pd.read_csv('/home/gjones/work/projects/nfp/results/upcoming_market_moving_events.csv')
cpi_events = events[events['date'] == '2026-05-12']

# Get forecast values
cpi_m_m = cpi_events[cpi_events['event_name'] == 'CPI m/m'].iloc[0]
forecast_cpi = float(str(cpi_m_m['forecast_raw']).replace('%', '').strip())
core_cpi = cpi_events[cpi_events['event_name'] == 'Core CPI m/m'].iloc[0]
forecast_core = float(str(core_cpi['forecast_raw']).replace('%', '').strip())

print("=" * 70)
print("US CPI MAY 12, 2026 - FORECAST & SCENARIOS")
print("=" * 70)
print(f"\nForecast: CPI m/m = {forecast_cpi:.1f}% | Core CPI = {forecast_core:.1f}%")
print("Note: Actual values pending in data feed")

print("\n" + "=" * 70)
print("FX SCENARIO ANALYSIS (Based on Historical NFP Impacts)")
print("=" * 70)

# Historical NFP impact averages from nfp_stats_full.csv
# Negative surprise (-50K): USD strengthens
# Positive surprise (+50K): USD weakens

scenarios_def = [
    ("CPI Miss (-0.3% vs 0.6%)", -50, "USD strength"),
    ("CPI Beat (+0.3% vs 0.6%)", 50, "USD weakness"),
]

# Use historical average impacts from nfp_stats_full.csv
nfp_stats = pd.read_csv('/home/gjones/work/projects/nfp/results/nfp_stats_full.csv')

for name, nfp_k, interpretation in scenarios_def:
    print(f"\n{name.upper()}")
    print(f"  → {interpretation} (similar to {nfp_k:+.0f}K NFP)")
    print("   Historical avg impact:")
    
    for h in ['1h', '4h', '6h']:
        # Get slope for each pair
        eur = nfp_stats[(nfp_stats['pair'] == 'EUR/USD') & (nfp_stats['horizon_label'] == h)].iloc[0]
        gbp = nfp_stats[(nfp_stats['pair'] == 'GBP/USD') & (nfp_stats['horizon_label'] == h)].iloc[0]
        cad = nfp_stats[(nfp_stats['pair'] == 'USD/CAD') & (nfp_stats['horizon_label'] == h)].iloc[0]
        jpy = nfp_stats[(nfp_stats['pair'] == 'USD/JPY') & (nfp_stats['horizon_label'] == h)].iloc[0]
        
        # Apply NFP surprise (in K) * slope (pct per 1K)
        eur_move = eur['slope_pct_per_1k_delta'] * nfp_k
        gbp_move = gbp['slope_pct_per_1k_delta'] * nfp_k
        cad_move = cad['slope_pct_per_1k_delta'] * nfp_k
        jpy_move = jpy['slope_pct_per_1k_delta'] * nfp_k
        
        print(f"   {h.upper()}: EUR/USD {eur_move:+.4f}% | GBP/USD {gbp_move:+.4f}% | USD/CAD {cad_move:+.4f}% | USD/JPY {jpy_move:+.4f}%")

# Market implications
print("\n" + "=" * 70)
print("MARKET IMPLICATIONS")
print("=" * 70)
print(f"\nConsensus: CPI m/m {forecast_cpi:.1f}% (vs 0.3% prior)")
print(f"Consensus: Core CPI m/m {forecast_core:.1f}% (vs 0.2% prior)")

print("\nKey Drivers:")
print("  • Headline CPI: Energy & food volatility")
print("  • Core CPI: Services inflation (sticky)")
print("  • Fed Policy: Data-dependent stance")

print("\nTrading Strategy:")
print("  • If CPI < 0.6%: USD strength, buy USD")
print("  • If CPI > 0.9%: USD weakness, sell USD")
print("  • If 0.6-0.9%: Range-bound, mean reversion")

# Save results
results = {
    'date': ['2026-05-12'],
    'event': ['US CPI m/m'],
    'forecast_cpi': [forecast_cpi],
    'forecast_core': [forecast_core],
    'status': ['Pending actual'],
}
df_results = pd.DataFrame(results)
df_results.to_csv('/home/gjones/work/projects/nfp/results/cpi_may12_2026_summary.csv', index=False)
print(f"\nResults saved to: results/cpi_may12_2026_summary.csv")
