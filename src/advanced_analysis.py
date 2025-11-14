"""
Advanced Health Analysis Module

Additional fitness insights using sleep, activity, nutrition, and recovery data
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class SleepAnalyzer:
    """Analyze sleep patterns and their impact on performance"""

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.data['date'] = pd.to_datetime(self.data['date'])

    def analyze_sleep_quality(self) -> Dict:
        """
        Analyze sleep duration and its relationship with performance

        Returns:
            Dictionary with sleep analysis
        """
        if 'sleep_hours' not in self.data.columns:
            return {}

        df = self.data.dropna(subset=['sleep_hours']).copy()

        if len(df) < 7:
            return {}

        analysis = {
            'avg_sleep_hours': df['sleep_hours'].mean(),
            'median_sleep_hours': df['sleep_hours'].median(),
            'std_sleep_hours': df['sleep_hours'].std(),
            'min_sleep_hours': df['sleep_hours'].min(),
            'max_sleep_hours': df['sleep_hours'].max(),
            'nights_tracked': len(df),
        }

        # Sleep consistency score (lower std = more consistent)
        consistency_score = max(0, 100 - (df['sleep_hours'].std() * 20))
        analysis['consistency_score'] = consistency_score

        # Days with insufficient sleep (<6 hours)
        insufficient = (df['sleep_hours'] < 6).sum()
        analysis['nights_insufficient_sleep'] = insufficient
        analysis['pct_insufficient_sleep'] = (insufficient / len(df) * 100)

        # Optimal sleep (7-9 hours)
        optimal = ((df['sleep_hours'] >= 7) & (df['sleep_hours'] <= 9)).sum()
        analysis['nights_optimal_sleep'] = optimal
        analysis['pct_optimal_sleep'] = (optimal / len(df) * 100)

        # Weekly pattern
        df['day_of_week'] = df['date'].dt.day_name()
        weekly_avg = df.groupby('day_of_week')['sleep_hours'].mean()
        analysis['sleep_by_day'] = weekly_avg.to_dict()

        return analysis

    def sleep_performance_correlation(self) -> Dict:
        """Correlate sleep with next-day performance metrics"""

        if 'sleep_hours' not in self.data.columns:
            return {}

        df = self.data.copy()
        correlations = {}

        # Shift sleep to align with next day performance
        df['prev_night_sleep'] = df['sleep_hours'].shift(1)

        metrics_to_correlate = [
            'ActiveEnergyBurned',
            'StepCount',
            'DistanceWalkingRunning',
            'HeartRate',
            'RestingHeartRate',
            'HeartRateVariabilitySDNN',
        ]

        for metric in metrics_to_correlate:
            if metric in df.columns:
                valid_data = df.dropna(subset=['prev_night_sleep', metric])
                if len(valid_data) >= 10:
                    corr, p_value = stats.pearsonr(
                        valid_data['prev_night_sleep'],
                        valid_data[metric]
                    )
                    correlations[metric] = {
                        'correlation': corr,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }

        return correlations


class NutritionAnalyzer:
    """Analyze nutrition patterns and macronutrient balance"""

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.data['date'] = pd.to_datetime(self.data['date'])

    def analyze_macros(self) -> Dict:
        """Analyze macronutrient intake and balance"""

        macro_cols = ['DietaryProtein', 'DietaryCarbohydrates', 'DietaryFatTotal']
        available = [col for col in macro_cols if col in self.data.columns]

        if not available:
            return {}

        df = self.data.dropna(subset=available).copy()

        if df.empty:
            return {}

        analysis = {}

        # Average macros
        for col in available:
            metric_name = col.replace('Dietary', '').replace('Total', '')
            analysis[f'avg_{metric_name.lower()}_g'] = df[col].mean()

        # Macro ratios (if all three available)
        if len(available) == 3:
            df['total_macros'] = df[available].sum(axis=1)

            analysis['avg_protein_pct'] = (df['DietaryProtein'] / df['total_macros'] * 100).mean()
            analysis['avg_carbs_pct'] = (df['DietaryCarbohydrates'] / df['total_macros'] * 100).mean()
            analysis['avg_fat_pct'] = (df['DietaryFatTotal'] / df['total_macros'] * 100).mean()

            # Protein per kg body weight (if weight available)
            if 'BodyMass' in df.columns:
                df_with_weight = df.dropna(subset=['BodyMass'])
                if not df_with_weight.empty:
                    analysis['avg_protein_per_kg'] = (
                        df_with_weight['DietaryProtein'] / df_with_weight['BodyMass']
                    ).mean()

        # Micronutrient tracking
        micros = {
            'DietaryCalcium': 'calcium_mg',
            'DietaryIron': 'iron_mg',
            'DietaryMagnesium': 'magnesium_mg',
            'DietaryPotassium': 'potassium_mg',
            'DietaryVitaminC': 'vitamin_c_mg',
        }

        for col, name in micros.items():
            if col in self.data.columns:
                micro_data = self.data.dropna(subset=[col])
                if not micro_data.empty:
                    analysis[f'avg_{name}'] = micro_data[col].mean()

        return analysis


class RecoveryAnalyzer:
    """Analyze recovery patterns and stress indicators"""

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.data['date'] = pd.to_datetime(self.data['date'])

    def calculate_recovery_score(self) -> pd.DataFrame:
        """
        Calculate daily recovery score based on HRV, RHR, and sleep

        Returns:
            DataFrame with recovery scores
        """
        df = self.data.copy()

        # Initialize score
        df['recovery_score'] = 50.0

        # HRV component (higher is better)
        if 'HeartRateVariabilitySDNN' in df.columns:
            hrv_data = df['HeartRateVariabilitySDNN'].dropna()
            if not hrv_data.empty:
                # Normalize HRV (50 = average, each 10ms adds/subtracts 10 points)
                df['hrv_score'] = 50 + (df['HeartRateVariabilitySDNN'] - 50)
                df['hrv_score'] = df['hrv_score'].clip(0, 100)
                df['recovery_score'] = df['recovery_score'] * 0.6 + df['hrv_score'] * 0.4

        # RHR component (lower is better)
        if 'RestingHeartRate' in df.columns:
            rhr_data = df['RestingHeartRate'].dropna()
            if not rhr_data.empty:
                # Normalize RHR (60 = average, each bpm above/below adjusts score)
                df['rhr_score'] = 50 + (60 - df['RestingHeartRate']) * 2
                df['rhr_score'] = df['rhr_score'].clip(0, 100)
                df['recovery_score'] = df['recovery_score'] * 0.7 + df['rhr_score'] * 0.3

        # Sleep component
        if 'sleep_hours' in df.columns:
            df['sleep_score'] = ((df['sleep_hours'] / 8) * 100).clip(0, 100)
            df['recovery_score'] = df['recovery_score'] * 0.8 + df['sleep_score'] * 0.2

        return df[['date', 'recovery_score']].dropna()

    def detect_overtraining_risk(self) -> Dict:
        """
        Detect potential overtraining based on recovery trends

        Returns:
            Dictionary with overtraining risk analysis
        """
        recovery_df = self.calculate_recovery_score()

        if len(recovery_df) < 14:
            return {}

        # Calculate 7-day rolling average
        recovery_df['recovery_7d'] = recovery_df['recovery_score'].rolling(7).mean()

        # Trend over last 14 days
        recent = recovery_df.tail(14)
        x = np.arange(len(recent))
        y = recent['recovery_score'].values

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        analysis = {
            'current_recovery_score': recovery_df['recovery_score'].iloc[-1],
            'avg_recovery_score': recovery_df['recovery_score'].mean(),
            '7d_avg_recovery': recovery_df['recovery_7d'].iloc[-1],
            'recovery_trend': slope,  # Points per day
            'trend_direction': 'improving' if slope > 0 else 'declining',
        }

        # Overtraining indicators
        current_score = recovery_df['recovery_score'].iloc[-1]
        avg_score = recovery_df['recovery_score'].mean()

        if current_score < 40 and slope < -1:
            analysis['overtraining_risk'] = 'HIGH'
        elif current_score < 50 and slope < -0.5:
            analysis['overtraining_risk'] = 'MODERATE'
        else:
            analysis['overtraining_risk'] = 'LOW'

        return analysis


class ActivityPatternAnalyzer:
    """Analyze activity patterns and consistency"""

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.data['date'] = pd.to_datetime(self.data['date'])

    def analyze_consistency(self) -> Dict:
        """
        Analyze activity consistency and patterns

        Returns:
            Dictionary with consistency metrics
        """
        df = self.data.copy()

        analysis = {}

        # Step consistency
        if 'StepCount' in df.columns:
            steps = df.dropna(subset=['StepCount'])
            if not steps.empty:
                analysis['avg_daily_steps'] = steps['StepCount'].mean()
                analysis['step_consistency_cv'] = (steps['StepCount'].std() / steps['StepCount'].mean())

                # Active days (>5000 steps)
                active_days = (steps['StepCount'] >= 5000).sum()
                analysis['pct_active_days'] = (active_days / len(steps) * 100)

                # Very active days (>10000 steps)
                very_active = (steps['StepCount'] >= 10000).sum()
                analysis['pct_very_active_days'] = (very_active / len(steps) * 100)

        # Exercise time consistency
        if 'AppleExerciseTime' in df.columns:
            exercise = df.dropna(subset=['AppleExerciseTime'])
            if not exercise.empty:
                analysis['avg_exercise_minutes'] = exercise['AppleExerciseTime'].mean()

                # Days with >30 min exercise
                active_exercise = (exercise['AppleExerciseTime'] >= 30).sum()
                analysis['pct_days_30min_exercise'] = (active_exercise / len(exercise) * 100)

        # Weekly patterns
        df['day_of_week'] = df['date'].dt.day_name()
        df['is_weekend'] = df['date'].dt.dayofweek >= 5

        if 'StepCount' in df.columns:
            weekday_steps = df[~df['is_weekend']]['StepCount'].mean()
            weekend_steps = df[df['is_weekend']]['StepCount'].mean()
            if pd.notna(weekday_steps) and pd.notna(weekend_steps):
                analysis['weekday_avg_steps'] = weekday_steps
                analysis['weekend_avg_steps'] = weekend_steps
                analysis['weekend_vs_weekday_ratio'] = weekend_steps / weekday_steps

        return analysis

    def find_peak_performance_windows(self) -> Dict:
        """Identify time periods with best performance"""

        df = self.data.copy()

        if 'ActiveEnergyBurned' not in df.columns:
            return {}

        # Calculate 7-day rolling performance score
        df['performance_7d'] = df['ActiveEnergyBurned'].rolling(7).mean()

        # Find top 3 performance windows
        top_windows = df.nlargest(3, 'performance_7d')[['date', 'performance_7d']]

        analysis = {
            'peak_performance_dates': [
                {
                    'date': str(row['date'].date()),
                    'avg_calories_7d': row['performance_7d']
                }
                for _, row in top_windows.iterrows()
            ]
        }

        return analysis


class WeightLossAnalyzer:
    """Extended weight loss analysis"""

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.data['date'] = pd.to_datetime(self.data['date'])

    def analyze_weight_loss_phases(self) -> Dict:
        """Identify different phases of weight loss"""

        if 'BodyMass' not in self.data.columns:
            return {}

        df = self.data.dropna(subset=['BodyMass']).copy()

        if len(df) < 14:
            return {}

        # Calculate weekly rate of change
        df['weight_change_7d'] = df['BodyMass'].diff(7)
        df['rate_kg_per_week'] = -df['weight_change_7d']  # Negative for loss

        # Categorize phases
        df['phase'] = 'maintenance'
        df.loc[df['rate_kg_per_week'] > 0.5, 'phase'] = 'rapid_loss'
        df.loc[(df['rate_kg_per_week'] > 0.2) & (df['rate_kg_per_week'] <= 0.5), 'phase'] = 'steady_loss'
        df.loc[(df['rate_kg_per_week'] > -0.2) & (df['rate_kg_per_week'] <= 0.2), 'phase'] = 'maintenance'
        df.loc[df['rate_kg_per_week'] < -0.2, 'phase'] = 'gaining'

        phase_counts = df['phase'].value_counts()

        analysis = {
            'total_weight_change': df['BodyMass'].iloc[-1] - df['BodyMass'].iloc[0],
            'avg_weekly_rate': df['rate_kg_per_week'].mean(),
            'max_weekly_loss': df['rate_kg_per_week'].max(),
            'days_tracked': len(df),
        }

        for phase in ['rapid_loss', 'steady_loss', 'maintenance', 'gaining']:
            if phase in phase_counts:
                analysis[f'days_in_{phase}'] = int(phase_counts[phase])

        return analysis

    def predict_goal_date(self, goal_weight: float) -> Dict:
        """Predict when goal weight will be reached"""

        if 'BodyMass' not in self.data.columns:
            return {}

        df = self.data.dropna(subset=['BodyMass']).copy()

        if len(df) < 14:
            return {}

        current_weight = df['BodyMass'].iloc[-1]

        if current_weight <= goal_weight:
            return {'message': 'Goal already achieved!', 'current_weight': current_weight}

        # Calculate recent trend (last 30 days)
        recent = df.tail(30)
        x = np.arange(len(recent))
        y = recent['BodyMass'].values

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Predict days to goal
        if slope >= 0:
            return {'message': 'Weight not decreasing', 'current_trend': slope}

        days_to_goal = (goal_weight - current_weight) / slope
        predicted_date = df['date'].iloc[-1] + pd.Timedelta(days=days_to_goal)

        return {
            'current_weight': current_weight,
            'goal_weight': goal_weight,
            'weight_to_lose': current_weight - goal_weight,
            'current_rate_kg_per_day': slope,
            'estimated_days_to_goal': int(days_to_goal),
            'predicted_goal_date': str(predicted_date.date()),
            'confidence_r_squared': r_value ** 2,
        }
