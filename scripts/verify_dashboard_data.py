"""
Verify that dashboard will show latest data
"""
from data_loader import HealthDataLoader
import pandas as pd

print("="*60)
print("DASHBOARD DATA VERIFICATION")
print("="*60)

# Load data
loader = HealthDataLoader("files")
all_data = loader.load_all_daily_metrics()

print(f"\n[DATA LOADED]")
print(f"  Total days: {len(all_data)}")
print(f"  Date range: {all_data['date'].min().date()} to {all_data['date'].max().date()}")

# Check latest dates with actual data
latest_dates = all_data[all_data['StepCount'].notna()].tail(10)
print(f"\n[LATEST 10 DAYS WITH STEP DATA]")
for idx, row in latest_dates.iterrows():
    print(f"  {row['date'].date()}: {row['StepCount']:.0f} steps")

# Check running period
RUNNING_START = "2025-07-01"
RUNNING_END = all_data['date'].max().strftime('%Y-%m-%d')
running_data = all_data[
    (all_data['date'] >= RUNNING_START) &
    (all_data['date'] <= RUNNING_END)
]

print(f"\n[RUNNING ANALYSIS PERIOD]")
print(f"  Start: {RUNNING_START}")
print(f"  End: {RUNNING_END}")
print(f"  Total days in period: {len(running_data)}")
print(f"  Days with step data: {running_data['StepCount'].notna().sum()}")

# Check November data specifically
nov_data = all_data[all_data['date'] >= '2025-11-01']
print(f"\n[NOVEMBER 2025 DATA]")
print(f"  Total days: {len(nov_data)}")
if len(nov_data) > 0:
    print(f"  Date range: {nov_data['date'].min().date()} to {nov_data['date'].max().date()}")
    nov_with_steps = nov_data[nov_data['StepCount'].notna()]
    print(f"  Days with step data: {len(nov_with_steps)}")
    if len(nov_with_steps) > 0:
        for idx, row in nov_with_steps.iterrows():
            print(f"    {row['date'].date()}: {row['StepCount']:.0f} steps")

print(f"\n[VERIFICATION RESULT]")
if nov_data['StepCount'].notna().sum() > 0:
    print("  ✓ SUCCESS: November 2025 data is available!")
    print("  ✓ Dashboard will show latest data")
    print(f"\n  Your dashboard will now display data through {all_data['date'].max().date()}")
else:
    print("  ✗ WARNING: No November data found")
    print("  Run 'python process_bronze.py' to process latest API data")

print("="*60)
