"""
Calculate actual running distance from workouts and distance data
"""
import pandas as pd
from pathlib import Path
from typing import Tuple


def get_running_only_distance(data_dir: str, start_date: str = None) -> pd.DataFrame:
    """
    Calculate daily running distance by matching workout times with distance data

    Args:
        data_dir: Directory containing CSV files
        start_date: Optional start date filter

    Returns:
        DataFrame with date and running_distance_km
    """
    data_path = Path(data_dir)

    # Load workouts
    workouts = pd.read_csv(data_path / 'workouts.csv')
    workouts['startDate'] = pd.to_datetime(workouts['startDate'], format='mixed', utc=True)
    workouts['endDate'] = pd.to_datetime(workouts['endDate'], format='mixed', utc=True)

    # Filter for running workouts only
    running_workouts = workouts[
        workouts['workoutActivityType'] == 'HKWorkoutActivityTypeRunning'
    ].copy()

    print(f"Found {len(running_workouts)} running workouts")

    # Load distance data
    distance_df = pd.read_csv(data_path / 'HKQuantityTypeIdentifierDistanceWalkingRunning.csv')
    distance_df['start_date'] = pd.to_datetime(distance_df['start_date'], format='mixed', utc=True)
    distance_df['end_date'] = pd.to_datetime(distance_df['end_date'], format='mixed', utc=True)

    # Remove duplicates
    distance_df = distance_df.drop_duplicates(subset=['start_date', 'end_date', 'value'], keep='first')

    # For each running workout, find distance entries that fall within the workout time
    running_distances = []

    for _, workout in running_workouts.iterrows():
        workout_start = workout['startDate']
        workout_end = workout['endDate']
        workout_date = workout_start.date()
        workout_duration_minutes = workout['duration']

        # Find distance entries during this workout
        # Use strict time matching - only distance recorded during the actual workout
        mask = (
            (distance_df['start_date'] >= workout_start) &
            (distance_df['end_date'] <= workout_end)
        )

        workout_distances = distance_df[mask]
        total_distance = workout_distances['value'].sum()

        # Sanity check: running pace should be reasonable
        # Typical running pace: 4-8 min/km (12-15 km/h or 7.5-3.75 min/km)
        # Max reasonable pace for amateur: 3 min/km (20 km/h)
        # Min reasonable pace: 8 min/km (7.5 km/h)
        if total_distance > 0 and workout_duration_minutes > 0:
            pace_min_per_km = workout_duration_minutes / total_distance

            # Only include if pace is reasonable for running (3-8 min/km)
            # This filters out cases where we captured too much walking distance
            if 3 <= pace_min_per_km <= 8:
                running_distances.append({
                    'date': workout_date,
                    'distance_km': total_distance,
                    'duration_minutes': workout_duration_minutes,
                    'pace_min_per_km': pace_min_per_km
                })
            else:
                print(f"  Skipped workout on {workout_date}: distance={total_distance:.1f}km, pace={pace_min_per_km:.1f} min/km (unreasonable)")

    # Convert to DataFrame and aggregate by date (in case multiple runs per day)
    if running_distances:
        running_df = pd.DataFrame(running_distances)
        daily_running = running_df.groupby('date')['distance_km'].sum().reset_index()
        daily_running['date'] = pd.to_datetime(daily_running['date'])

        # Filter by start_date if provided
        if start_date:
            daily_running = daily_running[daily_running['date'] >= start_date]

        return daily_running
    else:
        return pd.DataFrame(columns=['date', 'distance_km'])


def calculate_weekly_running_distance(running_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate weekly running distance totals

    Args:
        running_df: DataFrame with date and distance_km columns

    Returns:
        DataFrame with week and distance_km columns
    """
    if running_df.empty:
        return pd.DataFrame(columns=['week', 'distance_km'])

    running_df = running_df.copy()
    running_df['week'] = running_df['date'].dt.to_period('W')

    weekly = running_df.groupby('week').agg({
        'distance_km': 'sum'
    }).reset_index()

    weekly['week'] = weekly['week'].dt.to_timestamp()

    return weekly
