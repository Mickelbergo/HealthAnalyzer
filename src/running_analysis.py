"""
Running and Cardiovascular Analysis Module

Analyzes running performance, heart rate data, and training progress
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')


class RunningAnalyzer:
    """Analyze running workouts and performance metrics"""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with daily health data

        Args:
            data: DataFrame with running and heart rate metrics
        """
        self.data = data.copy()
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data = self.data.sort_values('date').reset_index(drop=True)

    def identify_runs(self, min_distance: float = 1.0) -> pd.DataFrame:
        """
        Identify running days based on distance threshold

        Args:
            min_distance: Minimum distance in km to consider as a run

        Returns:
            DataFrame with only running days
        """
        if 'DistanceWalkingRunning' not in self.data.columns:
            return pd.DataFrame()

        # Distance is already in km from Apple Health export
        runs = self.data[self.data['DistanceWalkingRunning'] >= min_distance].copy()

        # Calculate pace (min/km) if we have distance and time
        if 'AppleExerciseTime' in runs.columns:
            # Exercise time is in minutes, distance already in km
            runs['pace_min_per_km'] = runs['AppleExerciseTime'] / runs['DistanceWalkingRunning']

        return runs.reset_index(drop=True)

    def calculate_running_metrics(self) -> pd.DataFrame:
        """
        Calculate comprehensive running performance metrics

        Returns:
            DataFrame with calculated metrics
        """
        df = self.data.copy()

        # Distance is already in km from Apple Health
        if 'DistanceWalkingRunning' in df.columns:
            df['distance_km'] = df['DistanceWalkingRunning']

        # Calculate pace if running speed is available
        if 'RunningSpeed' in df.columns:
            # RunningSpeed is in km/hr, convert to min/km
            # pace (min/km) = 60 / speed (km/hr)
            df['pace_min_per_km'] = 60 / df['RunningSpeed']

        # Calculate weekly totals
        df['week'] = df['date'].dt.to_period('W')

        weekly_stats = df.groupby('week').agg({
            'distance_km': 'sum',
            'RunningSpeed': 'mean',
            'HeartRate': 'mean',
            'RunningPower': 'mean',
            'RunningStrideLength': 'mean',
        }).reset_index()

        weekly_stats['week'] = weekly_stats['week'].dt.to_timestamp()

        return df, weekly_stats

    def analyze_training_progression(
        self,
        target_date: Optional[str] = None
    ) -> Dict:
        """
        Analyze training progression leading up to a target event

        Args:
            target_date: Date of target event (e.g., race day)

        Returns:
            Dictionary with progression analysis
        """
        df = self.data.copy()

        if target_date:
            target = pd.to_datetime(target_date)
            df = df[df['date'] <= target]

        # Convert distance to km
        if 'DistanceWalkingRunning' in df.columns:
            df['distance_km'] = df['DistanceWalkingRunning'] / 1000
        else:
            return {}

        # Weekly aggregations
        df['week'] = df['date'].dt.to_period('W')
        weekly = df.groupby('week').agg({
            'distance_km': 'sum',
            'HeartRate': 'mean',
            'RestingHeartRate': 'mean',
        }).reset_index()

        weekly['week'] = weekly['week'].dt.to_timestamp()

        # Calculate trends
        analysis = {
            'total_distance_km': df['distance_km'].sum(),
            'avg_weekly_distance': weekly['distance_km'].mean(),
            'peak_weekly_distance': weekly['distance_km'].max(),
            'training_weeks': len(weekly),
            'avg_heart_rate': df['HeartRate'].mean() if 'HeartRate' in df.columns else None,
            'avg_resting_hr': df['RestingHeartRate'].mean() if 'RestingHeartRate' in df.columns else None,
        }

        # Calculate weekly distance trend (linear regression)
        if len(weekly) >= 3:
            from scipy import stats
            x = np.arange(len(weekly))
            y = weekly['distance_km'].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            analysis['weekly_distance_trend'] = slope
            analysis['trend_r_squared'] = r_value ** 2

        return analysis

    def detect_peak_performance_days(self, metric: str = 'RunningSpeed') -> pd.DataFrame:
        """
        Detect days with peak performance

        Args:
            metric: Metric to use for detecting peaks

        Returns:
            DataFrame with top performance days
        """
        if metric not in self.data.columns:
            return pd.DataFrame()

        df = self.data.dropna(subset=[metric]).copy()
        df = df.sort_values(metric, ascending=False)

        return df.head(10)


class HeartRateAnalyzer:
    """Analyze heart rate patterns and cardiovascular fitness"""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with daily health data

        Args:
            data: DataFrame with heart rate metrics
        """
        self.data = data.copy()
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data = self.data.sort_values('date').reset_index(drop=True)

    def calculate_hr_zones(
        self,
        max_hr: Optional[int] = None,
        age: Optional[int] = None
    ) -> Dict[str, Tuple[int, int]]:
        """
        Calculate heart rate training zones

        Args:
            max_hr: Maximum heart rate (if None, estimate from age)
            age: Age in years (used if max_hr not provided)

        Returns:
            Dictionary with HR zones
        """
        # Estimate max HR if not provided
        if max_hr is None:
            if age is None:
                age = 25  # default assumption
            max_hr = 220 - age

        zones = {
            'Zone 1 (Recovery)': (int(max_hr * 0.50), int(max_hr * 0.60)),
            'Zone 2 (Aerobic)': (int(max_hr * 0.60), int(max_hr * 0.70)),
            'Zone 3 (Tempo)': (int(max_hr * 0.70), int(max_hr * 0.80)),
            'Zone 4 (Threshold)': (int(max_hr * 0.80), int(max_hr * 0.90)),
            'Zone 5 (Max)': (int(max_hr * 0.90), max_hr),
        }

        return zones

    def analyze_hr_trends(self) -> Dict:
        """
        Analyze heart rate trends over time

        Returns:
            Dictionary with trend analysis
        """
        df = self.data.copy()

        # Ensure we have heart rate data
        hr_cols = ['HeartRate', 'RestingHeartRate', 'HeartRateVariabilitySDNN']
        available_cols = [col for col in hr_cols if col in df.columns]

        if not available_cols:
            return {}

        analysis = {}

        # Resting Heart Rate trend (lower is better)
        if 'RestingHeartRate' in df.columns:
            rhr_data = df.dropna(subset=['RestingHeartRate'])
            if len(rhr_data) >= 7:
                analysis['avg_resting_hr'] = rhr_data['RestingHeartRate'].mean()
                analysis['min_resting_hr'] = rhr_data['RestingHeartRate'].min()
                analysis['resting_hr_trend'] = self._calculate_trend(
                    rhr_data['date'],
                    rhr_data['RestingHeartRate']
                )

        # HRV trend (higher is better)
        if 'HeartRateVariabilitySDNN' in df.columns:
            hrv_data = df.dropna(subset=['HeartRateVariabilitySDNN'])
            if len(hrv_data) >= 7:
                analysis['avg_hrv'] = hrv_data['HeartRateVariabilitySDNN'].mean()
                analysis['max_hrv'] = hrv_data['HeartRateVariabilitySDNN'].max()
                analysis['hrv_trend'] = self._calculate_trend(
                    hrv_data['date'],
                    hrv_data['HeartRateVariabilitySDNN']
                )

        # Average exercise HR
        if 'HeartRate' in df.columns:
            hr_data = df.dropna(subset=['HeartRate'])
            if len(hr_data) >= 7:
                analysis['avg_exercise_hr'] = hr_data['HeartRate'].mean()

        return analysis

    def _calculate_trend(self, dates: pd.Series, values: pd.Series) -> float:
        """
        Calculate linear trend

        Returns:
            Slope of the trend line (change per day)
        """
        from scipy import stats

        # Convert dates to days since start
        days = (dates - dates.min()).dt.total_seconds() / 86400
        slope, intercept, r_value, p_value, std_err = stats.linregress(days, values)

        return slope

    def calculate_cardiovascular_fitness_score(self) -> float:
        """
        Calculate a composite cardiovascular fitness score

        Score is based on:
        - Resting heart rate (lower is better)
        - HRV (higher is better)
        - VO2 Max (higher is better)

        Returns:
            Fitness score (0-100)
        """
        df = self.data.copy()
        score = 50  # baseline

        # Resting HR component (ideal: 40-50 bpm)
        if 'RestingHeartRate' in df.columns:
            avg_rhr = df['RestingHeartRate'].mean()
            if pd.notna(avg_rhr):
                # Score: 100 at 40 bpm, 0 at 80 bpm
                rhr_score = max(0, min(100, (80 - avg_rhr) * 2.5))
                score = score * 0.7 + rhr_score * 0.3

        # HRV component (ideal: >50 ms)
        if 'HeartRateVariabilitySDNN' in df.columns:
            avg_hrv = df['HeartRateVariabilitySDNN'].mean()
            if pd.notna(avg_hrv):
                # Score: higher is better, cap at 100
                hrv_score = min(100, avg_hrv * 1.5)
                score = score * 0.7 + hrv_score * 0.3

        # VO2 Max component
        if 'VO2Max' in df.columns:
            avg_vo2 = df['VO2Max'].mean()
            if pd.notna(avg_vo2):
                # Score: 100 at 60, 0 at 30
                vo2_score = max(0, min(100, (avg_vo2 - 30) * 3.33))
                score = score * 0.7 + vo2_score * 0.3

        return score


def analyze_race_performance(
    data: pd.DataFrame,
    race_date: str,
    race_distance_km: float = 21.0975,
    window_days: int = 30
) -> Dict:
    """
    Analyze performance around a specific race

    Args:
        data: DataFrame with health metrics
        race_date: Date of the race
        race_distance_km: Distance of the race (default: half marathon)
        window_days: Days before/after race to analyze

    Returns:
        Dictionary with race analysis
    """
    race_dt = pd.to_datetime(race_date)
    start_date = race_dt - timedelta(days=window_days)
    end_date = race_dt + timedelta(days=window_days)

    # Filter to race window
    df = data[(data['date'] >= start_date) & (data['date'] <= end_date)].copy()

    # Get race day data
    race_day = df[df['date'].dt.date == race_dt.date()]

    analysis = {
        'race_date': race_date,
        'race_distance_km': race_distance_km,
    }

    if not race_day.empty:
        race_data = race_day.iloc[0]

        # Race performance metrics
        if 'DistanceWalkingRunning' in race_data:
            actual_distance = race_data['DistanceWalkingRunning'] / 1000
            analysis['actual_distance_km'] = actual_distance

        if 'AppleExerciseTime' in race_data:
            race_time_min = race_data['AppleExerciseTime']
            analysis['race_time_minutes'] = race_time_min
            analysis['average_pace_min_per_km'] = race_time_min / race_distance_km

        if 'HeartRate' in race_data:
            analysis['average_race_hr'] = race_data['HeartRate']

        if 'RunningPower' in race_data:
            analysis['average_power_watts'] = race_data['RunningPower']

    # Training before race
    pre_race = df[df['date'] < race_dt]
    if not pre_race.empty and 'DistanceWalkingRunning' in pre_race.columns:
        analysis['pre_race_total_km'] = (pre_race['DistanceWalkingRunning'] / 1000).sum()
        analysis['pre_race_avg_weekly_km'] = analysis['pre_race_total_km'] / (window_days / 7)

    # Recovery after race
    post_race = df[df['date'] > race_dt]
    if not post_race.empty:
        if 'RestingHeartRate' in post_race.columns:
            analysis['post_race_avg_rhr'] = post_race['RestingHeartRate'].mean()

    return analysis
