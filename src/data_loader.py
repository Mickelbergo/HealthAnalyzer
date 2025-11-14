"""
Health Data Loader and Processor

This module provides utilities for loading and processing Apple Health data
from CSV files exported from Apple Health.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class HealthDataLoader:
    """Load and process Apple Health CSV data"""

    def __init__(self, data_dir: str = "files"):
        """
        Initialize the data loader

        Args:
            data_dir: Directory containing CSV files from Apple Health export
        """
        self.data_dir = Path(data_dir)
        self.data_cache: Dict[str, pd.DataFrame] = {}

    def load_metric(self, metric_name: str) -> pd.DataFrame:
        """
        Load a specific health metric from CSV

        Args:
            metric_name: Name of the metric (e.g., 'BodyMass', 'HeartRate')

        Returns:
            DataFrame with the metric data
        """
        # Check cache first
        if metric_name in self.data_cache:
            return self.data_cache[metric_name].copy()

        # Try different file naming patterns
        possible_names = [
            f"HKQuantityTypeIdentifier{metric_name}.csv",
            f"HKCategoryTypeIdentifier{metric_name}.csv",
            f"{metric_name}.csv",
        ]

        for filename in possible_names:
            filepath = self.data_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                # Parse dates - handle multiple formats
                if 'start_date' in df.columns:
                    df['start_date'] = pd.to_datetime(df['start_date'], format='mixed', utc=True)
                if 'end_date' in df.columns:
                    df['end_date'] = pd.to_datetime(df['end_date'], format='mixed', utc=True, errors='coerce')
                if 'creation_date' in df.columns:
                    df['creation_date'] = pd.to_datetime(df['creation_date'], format='mixed', utc=True, errors='coerce')

                # Remove duplicate entries (same start_date, end_date, and value)
                # This fixes the issue where data is imported multiple times
                before_count = len(df)
                df = df.drop_duplicates(subset=['start_date', 'end_date', 'value'], keep='first')
                after_count = len(df)
                if before_count != after_count:
                    print(f"Info: Removed {before_count - after_count} duplicate entries from {metric_name}")

                # Cache the data
                self.data_cache[metric_name] = df.copy()
                return df

        # If not found, return empty DataFrame
        print(f"Warning: Metric '{metric_name}' not found in {self.data_dir}")
        return pd.DataFrame()

    def get_daily_aggregates(
        self,
        metric_name: str,
        agg_func: str = 'mean',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get daily aggregates for a metric

        Args:
            metric_name: Name of the metric
            agg_func: Aggregation function ('mean', 'sum', 'max', 'min', 'last')
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)

        Returns:
            DataFrame with date index and aggregated values
        """
        df = self.load_metric(metric_name)

        if df.empty:
            return pd.DataFrame()

        # Filter by date range if specified
        if start_date:
            df = df[df['start_date'] >= start_date]
        if end_date:
            df = df[df['start_date'] <= end_date]

        # Extract date and aggregate
        df['date'] = df['start_date'].dt.date

        if agg_func == 'last':
            # Take the last value of each day
            daily = df.groupby('date')['value'].last()
        else:
            daily = df.groupby('date')['value'].agg(agg_func)

        return daily.to_frame(metric_name).reset_index()

    def load_all_daily_metrics(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load all metrics and aggregate by day

        Args:
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)

        Returns:
            DataFrame with all daily metrics
        """
        # Define metrics and their aggregation functions
        metrics_config = {
            # Body metrics
            'BodyMass': 'last',
            'Height': 'last',

            # Energy metrics
            'ActiveEnergyBurned': 'sum',
            'BasalEnergyBurned': 'sum',
            'DietaryEnergyConsumed': 'sum',

            # Activity metrics
            'StepCount': 'sum',
            'DistanceWalkingRunning': 'sum',
            'FlightsClimbed': 'sum',
            'AppleExerciseTime': 'sum',
            'AppleStandTime': 'sum',

            # Heart metrics
            'HeartRate': 'mean',
            'RestingHeartRate': 'mean',
            'HeartRateVariabilitySDNN': 'mean',
            'WalkingHeartRateAverage': 'mean',
            'OxygenSaturation': 'mean',

            # Running metrics
            'RunningSpeed': 'mean',
            'RunningPower': 'mean',
            'RunningStrideLength': 'mean',
            'RunningGroundContactTime': 'mean',
            'RunningVerticalOscillation': 'mean',

            # Other metrics
            'VO2Max': 'mean',
            'RespiratoryRate': 'mean',
            'PhysicalEffort': 'mean',
        }

        # Load each metric
        daily_dfs = []
        for metric, agg_func in metrics_config.items():
            df = self.get_daily_aggregates(metric, agg_func, start_date, end_date)
            if not df.empty:
                daily_dfs.append(df.set_index('date'))

        # Merge all metrics
        if daily_dfs:
            result = pd.concat(daily_dfs, axis=1)
            result.index = pd.to_datetime(result.index)
            result = result.sort_index()
            return result.reset_index()

        return pd.DataFrame()

    def get_sleep_duration(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate daily sleep duration from SleepAnalysis data

        Returns:
            DataFrame with date and sleep_hours
        """
        df = self.load_metric('SleepAnalysis')

        if df.empty:
            return pd.DataFrame()

        # Filter by date range
        if start_date:
            df = df[df['start_date'] >= start_date]
        if end_date:
            df = df[df['start_date'] <= end_date]

        # Only count actual sleep states (exclude Awake and InBed)
        # Valid sleep states: AsleepCore, AsleepDeep, AsleepREM, AsleepUnspecified
        sleep_states = [
            'HKCategoryValueSleepAnalysisAsleepCore',
            'HKCategoryValueSleepAnalysisAsleepDeep',
            'HKCategoryValueSleepAnalysisAsleepREM',
            'HKCategoryValueSleepAnalysisAsleepUnspecified'
        ]
        df = df[df['value'].isin(sleep_states)]

        if df.empty:
            return pd.DataFrame()

        # Calculate sleep duration in hours
        df['duration_hours'] = (df['end_date'] - df['start_date']).dt.total_seconds() / 3600

        # Group by date (use end_date as the morning of that day)
        df['date'] = df['end_date'].dt.date

        # Sum sleep durations for each day
        sleep_daily = df.groupby('date')['duration_hours'].sum()

        return sleep_daily.to_frame('sleep_hours').reset_index()


class TDEEAnalyzer:
    """
    Analyze Total Daily Energy Expenditure (TDEE) and estimate maintenance calories
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with daily health data

        Args:
            data: DataFrame with columns: date, BodyMass, DietaryEnergyConsumed
        """
        self.data = data.copy()
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data = self.data.sort_values('date').reset_index(drop=True)

    def calculate_tdee(
        self,
        window: int = 7,
        min_days: int = 5
    ) -> pd.DataFrame:
        """
        Calculate TDEE using the energy balance equation

        Formula: TDEE = Calories_In - (Weight_Change_kg * 7700 kcal/kg) / Days

        Args:
            window: Rolling window size in days
            min_days: Minimum number of days with data required

        Returns:
            DataFrame with TDEE estimates
        """
        df = self.data.copy()

        # Ensure we have required columns
        required = ['date', 'BodyMass', 'DietaryEnergyConsumed']
        if not all(col in df.columns for col in required):
            raise ValueError(f"Data must contain columns: {required}")

        # Remove rows with missing data
        df = df.dropna(subset=['BodyMass', 'DietaryEnergyConsumed'])

        if len(df) < min_days:
            print(f"Warning: Not enough data ({len(df)} days, need {min_days})")
            return pd.DataFrame()

        # Calculate weight change over rolling window
        df['weight_change_kg'] = df['BodyMass'].diff(window)
        df['avg_calories'] = df['DietaryEnergyConsumed'].rolling(window).mean()

        # Energy balance: 1 kg fat = ~7700 kcal
        # TDEE = avg_calories - (weight_change_kg * 7700 / window_days)
        kcal_per_kg = 7700
        df['tdee_estimate'] = df['avg_calories'] - (df['weight_change_kg'] * kcal_per_kg / window)

        # Use IQR method to detect and cap extreme outliers in TDEE estimates
        # (but keep the values, just mark outliers as less reliable)
        Q1 = df['tdee_estimate'].quantile(0.25)
        Q3 = df['tdee_estimate'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR  # Use 3*IQR for very conservative outlier detection
        upper_bound = Q3 + 3 * IQR

        # Mark extreme outliers as NaN (these are likely calculation artifacts from data gaps)
        df.loc[(df['tdee_estimate'] < lower_bound) | (df['tdee_estimate'] > upper_bound), 'tdee_estimate'] = pd.NA

        # Calculate confidence based on weight change magnitude
        # Smaller weight changes are more reliable for TDEE estimation
        df['weight_change_abs'] = df['weight_change_kg'].abs()
        df['tdee_confidence'] = 1.0 / (1.0 + df['weight_change_abs'])

        # Add rolling average TDEE for smoothing
        df['tdee_smooth'] = df['tdee_estimate'].rolling(window, min_periods=3).mean()

        return df

    def get_tdee_summary(self, window: int = 7) -> Dict:
        """
        Get summary statistics for TDEE analysis

        Args:
            window: Rolling window size for TDEE calculation

        Returns:
            Dictionary with summary statistics
        """
        df = self.calculate_tdee(window=window)

        if df.empty:
            return {}

        # Filter to rows with valid TDEE estimates
        valid = df.dropna(subset=['tdee_estimate'])

        if valid.empty:
            return {}

        summary = {
            'avg_tdee': valid['tdee_estimate'].mean(),
            'median_tdee': valid['tdee_estimate'].median(),
            'std_tdee': valid['tdee_estimate'].std(),
            'avg_calories_in': valid['DietaryEnergyConsumed'].mean(),
            'start_weight': valid['BodyMass'].iloc[0],
            'end_weight': valid['BodyMass'].iloc[-1],
            'total_weight_change': valid['BodyMass'].iloc[-1] - valid['BodyMass'].iloc[0],
            'days_analyzed': len(valid),
            'avg_daily_deficit': valid['DietaryEnergyConsumed'].mean() - valid['tdee_estimate'].mean(),
        }

        return summary


def calculate_bmr(weight_kg: float, age: int, sex: str, height_cm: float = 175) -> float:
    """
    Calculate Basal Metabolic Rate using Oxford-Henry equations

    Args:
        weight_kg: Body weight in kg
        age: Age in years
        sex: 'male' or 'female'
        height_cm: Height in cm

    Returns:
        Estimated BMR in kcal/day
    """
    if sex.lower() in ['m', 'male']:
        if age < 30:
            return 15.057 * weight_kg + 692.2
        elif age < 60:
            return 11.472 * weight_kg + 873.1
        else:
            return 11.711 * weight_kg + 587.7
    else:  # female
        if age < 30:
            return 14.818 * weight_kg + 486.6
        elif age < 60:
            return 8.126 * weight_kg + 845.6
        else:
            return 9.082 * weight_kg + 658.5
