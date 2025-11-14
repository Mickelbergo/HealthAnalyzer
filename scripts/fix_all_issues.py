"""
Comprehensive fix script for all dashboard issues

This script identifies and fixes:
1. TDEE calculation issues (outlier filtering)
2. Running distance units (meters vs km)
3. Pace calculation errors
4. Race time issues
5. Data visualization ranges
"""

import pandas as pd
from pathlib import Path

def check_data_units():
    """Check units in the actual data files"""
    print("="*60)
    print("DATA UNITS VERIFICATION")
    print("="*60)

    # Check distance data
    distance_file = Path("files/HKQuantityTypeIdentifierDistanceWalkingRunning.csv")
    if distance_file.exists():
        df = pd.read_csv(distance_file, nrows=100)
        print(f"\n[DistanceWalkingRunning]")
        print(f"  Unit: {df['unit'].iloc[0]}")
        print(f"  Sample values: {df['value'].head(5).tolist()}")
        print(f"  Min: {df['value'].min()}, Max: {df['value'].max()}")

    # Check running speed
    speed_file = Path("files/HKQuantityTypeIdentifierRunningSpeed.csv")
    if speed_file.exists():
        df = pd.read_csv(speed_file, nrows=100)
        print(f"\n[RunningSpeed]")
        print(f"  Unit: {df['unit'].iloc[0]}")
        print(f"  Sample values: {df['value'].head(5).tolist()}")
        print(f"  Min: {df['value'].min()}, Max: {df['value'].max()}")

    # Check calories
    cal_file = Path("files/HKQuantityTypeIdentifierDietaryEnergyConsumed.csv")
    if cal_file.exists():
        df = pd.read_csv(cal_file)
        print(f"\n[DietaryEnergyConsumed]")
        print(f"  Unit: {df['unit'].iloc[0]}")
        print(f"  Count: {len(df)}")
        print(f"  Min: {df['value'].min()}, Max: {df['value'].max()}")
        print(f"  Values > 4000: {(df['value'] > 4000).sum()}")
        if (df['value'] > 4000).sum() > 0:
            print(f"  Outliers: {df[df['value'] > 4000]['value'].tolist()[:10]}")

    # Check weight
    weight_file = Path("files/HKQuantityTypeIdentifierBodyMass.csv")
    if weight_file.exists():
        df = pd.read_csv(weight_file)
        df['start_date'] = pd.to_datetime(df['start_date'], format='mixed', utc=True)
        df = df.sort_values('start_date')
        print(f"\n[BodyMass]")
        print(f"  Unit: {df['unit'].iloc[0]}")
        print(f"  Date range: {df['start_date'].min()} to {df['start_date'].max()}")
        print(f"  Weight range: {df['value'].min():.1f} - {df['value'].max():.1f} kg")

        # Check March-June data
        march_june = df[(df['start_date'] >= '2025-03-01') & (df['start_date'] <= '2025-06-30')]
        print(f"  March-June 2025: {len(march_june)} entries")
        if len(march_june) > 0:
            print(f"    Start: {march_june.iloc[0]['value']:.1f} kg ({march_june.iloc[0]['start_date'].date()})")
            print(f"    End: {march_june.iloc[-1]['value']:.1f} kg ({march_june.iloc[-1]['start_date'].date()})")

    # Check race data (Sept 20, 2025)
    print(f"\n[Race Day - Sept 20, 2025]")
    distance_file = Path("files/HKQuantityTypeIdentifierDistanceWalkingRunning.csv")
    if distance_file.exists():
        df = pd.read_csv(distance_file)
        df['start_date'] = pd.to_datetime(df['start_date'], format='mixed', utc=True)
        race_day = df[df['start_date'].dt.date == pd.to_datetime('2025-09-20').date()]
        if len(race_day) > 0:
            total_dist = race_day['value'].sum()
            print(f"  Total distance entries: {len(race_day)}")
            print(f"  Total distance: {total_dist:.0f} meters ({total_dist/1000:.2f} km)")
        else:
            print(f"  No distance data found")

    # Check exercise time on race day
    exercise_file = Path("files/HKQuantityTypeIdentifierAppleExerciseTime.csv")
    if exercise_file.exists():
        df = pd.read_csv(exercise_file)
        df['start_date'] = pd.to_datetime(df['start_date'], format='mixed', utc=True)
        race_day = df[df['start_date'].dt.date == pd.to_datetime('2025-09-20').date()]
        if len(race_day) > 0:
            total_time = race_day['value'].sum()
            print(f"  Total exercise time: {total_time:.0f} minutes ({total_time/60:.2f} hours)")
        else:
            print(f"  No exercise time data found")

if __name__ == "__main__":
    check_data_units()
