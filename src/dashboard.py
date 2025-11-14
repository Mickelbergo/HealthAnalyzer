"""
Health Analytics Dashboard - Enhanced Edition

Interactive dashboard with advanced fitness analytics and beautiful visualizations
"""

import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data_loader import HealthDataLoader, TDEEAnalyzer, calculate_bmr
from running_analysis import RunningAnalyzer, HeartRateAnalyzer
from advanced_analysis import (
    SleepAnalyzer, NutritionAnalyzer, RecoveryAnalyzer,
    ActivityPatternAnalyzer, WeightLossAnalyzer
)

# Color scheme
COLORS = {
    'primary': '#2E86AB',
    'success': '#06A77D',
    'warning': '#F77F00',
    'danger': '#D62828',
    'info': '#16A085',
    'purple': '#9B59B6',
    'teal': '#1ABC9C',
    'dark': '#2C3E50',
}

# Initialize the Dash app with modern theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.LUX],  # Modern, clean theme
    suppress_callback_exceptions=True
)
app.title = "Health Analytics Dashboard"

# Load data
print("Loading health data...")
# Find data directory relative to this file
from pathlib import Path
script_dir = Path(__file__).parent
project_root = script_dir.parent
data_dir = project_root / "data" / "processed" / "files"

# Fallback to old location if new structure doesn't exist
if not data_dir.exists():
    data_dir = project_root / "files"

# IMPORTANT: Only use data from March 24, 2025 onwards
DATA_START_DATE = "2025-03-24"

loader = HealthDataLoader(str(data_dir))
all_data = loader.load_all_daily_metrics(start_date=DATA_START_DATE)
sleep_data = loader.get_sleep_duration(start_date=DATA_START_DATE)

# Merge sleep data
if not sleep_data.empty:
    sleep_data['date'] = pd.to_datetime(sleep_data['date'])
    all_data = all_data.merge(sleep_data, on='date', how='left')

print(f"Loaded {len(all_data)} days of health data")
print(f"Date range: {all_data['date'].min()} to {all_data['date'].max()}")

# Define date ranges dynamically based on available data
# TDEE period: where we have both weight and calorie data
tdee_data_check = all_data[all_data['BodyMass'].notna() & all_data['DietaryEnergyConsumed'].notna()]
if len(tdee_data_check) > 0:
    TDEE_START = tdee_data_check['date'].min().strftime('%Y-%m-%d')
    TDEE_END = tdee_data_check['date'].max().strftime('%Y-%m-%d')
else:
    # Fallback to fixed dates if no data
    TDEE_START = "2025-03-01"
    TDEE_END = "2025-06-30"

# Running period: where we have distance data
running_data_check = all_data[all_data['DistanceWalkingRunning'].notna()]
if len(running_data_check) > 0:
    RUNNING_START = running_data_check['date'].min().strftime('%Y-%m-%d')
    RUNNING_END = running_data_check['date'].max().strftime('%Y-%m-%d')
else:
    # Fallback
    RUNNING_START = "2025-07-01"
    RUNNING_END = all_data['date'].max().strftime('%Y-%m-%d')

print(f"Analysis periods:")
print(f"  TDEE: {TDEE_START} to {TDEE_END} ({len(tdee_data_check)} days with data)")
print(f"  Running: {RUNNING_START} to {RUNNING_END} ({len(running_data_check)} days with data)")


def create_metric_card(title, value, subtitle=None, icon=None, color='primary'):
    """Create a modern metric card"""
    card_color = COLORS.get(color, COLORS['primary'])

    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.H6(title, className="text-muted mb-2"),
                html.H2(value, className="mb-0", style={'color': card_color, 'fontWeight': 'bold'}),
                html.P(subtitle if subtitle else "", className="text-muted small mb-0")
            ])
        ])
    ], className="mb-3 shadow-sm")


def create_tdee_analysis_section():
    """Create enhanced TDEE analysis with beautiful visualizations"""

    tdee_data = all_data[
        (all_data['date'] >= TDEE_START) &
        (all_data['date'] <= TDEE_END)
    ].copy()

    # Exclude days with zero or missing calorie data for more accurate TDEE
    # (these are days where food tracking wasn't done)
    tdee_data = tdee_data[
        (tdee_data['DietaryEnergyConsumed'].notna()) &
        (tdee_data['DietaryEnergyConsumed'] > 0)
    ]

    if tdee_data.empty:
        return html.Div("No data available for TDEE analysis period")

    analyzer = TDEEAnalyzer(tdee_data)
    tdee_results = analyzer.calculate_tdee(window=7)
    summary = analyzer.get_tdee_summary(window=7)

    # Metric cards
    cards = dbc.Row([
        dbc.Col(create_metric_card(
            "Weight Lost",
            f"{abs(summary.get('total_weight_change', 0)):.1f} kg",
            f"{summary.get('start_weight', 0):.1f} → {summary.get('end_weight', 0):.1f} kg",
            color='success'
        ), width=3),
        dbc.Col(create_metric_card(
            "Average TDEE",
            f"{summary.get('avg_tdee', 0):.0f}",
            "kcal/day",
            color='primary'
        ), width=3),
        dbc.Col(create_metric_card(
            "Daily Deficit",
            f"{abs(summary.get('avg_daily_deficit', 0)):.0f}",
            "kcal/day",
            color='warning'
        ), width=3),
        dbc.Col(create_metric_card(
            "Loss Rate",
            f"{abs(summary.get('total_weight_change', 0) / summary.get('days_analyzed', 1) * 7):.2f}",
            "kg/week",
            color='info'
        ), width=3),
    ], className="mb-4")

    # Create main visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Weight Progression',
            'Calorie Balance',
            'TDEE Estimation',
            'Weekly Summary'
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": True}, {"type": "bar"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Plot 1: Weight with trend
    fig.add_trace(
        go.Scatter(
            x=tdee_data['date'],
            y=tdee_data['BodyMass'],
            mode='lines+markers',
            name='Weight',
            line=dict(color=COLORS['primary'], width=3),
            marker=dict(size=6)
        ),
        row=1, col=1
    )

    # Plot 2: Daily calorie comparison
    if not tdee_results.empty:
        fig.add_trace(
            go.Scatter(
                x=tdee_results['date'],
                y=tdee_results['DietaryEnergyConsumed'],
                mode='markers',
                name='Consumed',
                marker=dict(color=COLORS['warning'], size=8, opacity=0.7)
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=tdee_results['date'],
                y=tdee_results['tdee_smooth'],
                mode='lines',
                name='TDEE',
                line=dict(color=COLORS['success'], width=3)
            ),
            row=1, col=2
        )

        # Plot 3: TDEE trend with confidence
        fig.add_trace(
            go.Scatter(
                x=tdee_results['date'],
                y=tdee_results['tdee_estimate'],
                mode='markers',
                name='Daily TDEE',
                marker=dict(color=COLORS['info'], size=4, opacity=0.5)
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=tdee_results['date'],
                y=tdee_results['tdee_smooth'],
                mode='lines',
                name='7-day Average',
                line=dict(color=COLORS['primary'], width=3)
            ),
            row=2, col=1
        )

        # Plot 4: Weekly energy balance
        tdee_results['week'] = tdee_results['date'].dt.to_period('W')
        weekly = tdee_results.groupby('week').agg({
            'DietaryEnergyConsumed': 'mean',
            'tdee_smooth': 'mean'
        }).reset_index()
        weekly['balance'] = weekly['DietaryEnergyConsumed'] - weekly['tdee_smooth']
        weekly['week'] = weekly['week'].dt.to_timestamp()

        colors_bar = [COLORS['danger'] if x < 0 else COLORS['success'] for x in weekly['balance']]

        fig.add_trace(
            go.Bar(
                x=weekly['week'],
                y=weekly['balance'],
                name='Weekly Balance',
                marker=dict(color=colors_bar)
            ),
            row=2, col=2
        )

    # Update layout
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Week", row=2, col=2)

    # Set Y-axis range for weight chart (don't start at 0)
    weight_min = tdee_data['BodyMass'].min()
    weight_max = tdee_data['BodyMass'].max()
    y_margin = max(3, (weight_max - weight_min) * 0.1)  # At least 3kg margin or 10% of range
    fig.update_yaxes(
        title_text="Weight (kg)",
        range=[weight_min - y_margin, weight_max + y_margin],
        row=1, col=1
    )

    fig.update_yaxes(title_text="Calories (kcal)", row=1, col=2)
    fig.update_yaxes(title_text="TDEE (kcal)", row=2, col=1)
    fig.update_yaxes(title_text="Balance (kcal)", row=2, col=2)

    fig.update_layout(
        height=800,
        showlegend=True,
        hovermode='closest',
        template='plotly_white',
        font=dict(size=11)
    )

    return html.Div([
        html.H3("TDEE Analysis: March - June 2025", className="mb-4"),
        cards,
        dcc.Graph(figure=fig, config={'displayModeBar': False}),
    ])


def create_sleep_recovery_section():
    """NEW: Sleep and recovery analysis"""

    sleep_analyzer = SleepAnalyzer(all_data)
    recovery_analyzer = RecoveryAnalyzer(all_data)

    sleep_quality = sleep_analyzer.analyze_sleep_quality()
    sleep_corr = sleep_analyzer.sleep_performance_correlation()
    recovery = recovery_analyzer.detect_overtraining_risk()
    recovery_scores = recovery_analyzer.calculate_recovery_score()

    if not sleep_quality:
        return html.Div("No sleep data available")

    # Metric cards
    cards = dbc.Row([
        dbc.Col(create_metric_card(
            "Avg Sleep",
            f"{sleep_quality.get('avg_sleep_hours', 0):.1f}h",
            f"{sleep_quality.get('nights_tracked', 0)} nights tracked",
            color='primary'
        ), width=3),
        dbc.Col(create_metric_card(
            "Sleep Score",
            f"{sleep_quality.get('consistency_score', 0):.0f}/100",
            "Consistency",
            color='success'
        ), width=3),
        dbc.Col(create_metric_card(
            "Recovery",
            f"{recovery.get('current_recovery_score', 0):.0f}/100",
            recovery.get('trend_direction', 'unknown'),
            color='info'
        ), width=3),
        dbc.Col(create_metric_card(
            "Overtraining Risk",
            recovery.get('overtraining_risk', 'UNKNOWN'),
            f"7d avg: {recovery.get('7d_avg_recovery', 0):.0f}",
            color='warning' if recovery.get('overtraining_risk') == 'MODERATE' else 'success'
        ), width=3),
    ], className="mb-4")

    # Visualizations
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Sleep Duration Over Time',
            'Sleep by Day of Week',
            'Recovery Score Trend',
            'Sleep vs Performance Correlations'
        ),
        specs=[
            [{"secondary_y": False}, {"type": "bar"}],
            [{"secondary_y": False}, {"type": "bar"}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    # Plot 1: Sleep over time
    sleep_df = all_data.dropna(subset=['sleep_hours'])
    fig.add_trace(
        go.Scatter(
            x=sleep_df['date'],
            y=sleep_df['sleep_hours'],
            mode='markers',
            name='Sleep Hours',
            marker=dict(color=COLORS['purple'], size=4, opacity=0.6)
        ),
        row=1, col=1
    )

    # Add 7-day average
    sleep_df['sleep_7d'] = sleep_df['sleep_hours'].rolling(7).mean()
    fig.add_trace(
        go.Scatter(
            x=sleep_df['date'],
            y=sleep_df['sleep_7d'],
            mode='lines',
            name='7-day Avg',
            line=dict(color=COLORS['primary'], width=3)
        ),
        row=1, col=1
    )

    # Reference lines
    fig.add_hline(y=7, line_dash="dash", line_color="green", opacity=0.5, row=1, col=1)
    fig.add_hline(y=9, line_dash="dash", line_color="green", opacity=0.5, row=1, col=1)

    # Plot 2: Sleep by day of week
    if 'sleep_by_day' in sleep_quality:
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        sleep_by_day = sleep_quality['sleep_by_day']
        day_values = [sleep_by_day.get(day, 0) for day in days_order]

        colors_days = [COLORS['danger'] if v < 6 else COLORS['success'] if v >= 7 and v <= 9 else COLORS['warning'] for v in day_values]

        fig.add_trace(
            go.Bar(
                x=days_order,
                y=day_values,
                name='Avg Sleep',
                marker=dict(color=colors_days),
                text=[f"{v:.1f}h" for v in day_values],
                textposition='outside'
            ),
            row=1, col=2
        )

    # Plot 3: Recovery score over time
    if not recovery_scores.empty:
        fig.add_trace(
            go.Scatter(
                x=recovery_scores['date'],
                y=recovery_scores['recovery_score'],
                mode='lines',
                name='Recovery Score',
                line=dict(color=COLORS['teal'], width=2),
                fill='tozeroy',
                fillcolor=f"rgba(26, 188, 156, 0.1)"
            ),
            row=2, col=1
        )

    # Plot 4: Sleep correlations
    if sleep_corr:
        sig_corr = {k: v for k, v in sleep_corr.items() if v['significant']}
        if sig_corr:
            metrics = list(sig_corr.keys())
            corr_values = [sig_corr[m]['correlation'] for m in metrics]

            colors_corr = [COLORS['success'] if v > 0 else COLORS['danger'] for v in corr_values]

            fig.add_trace(
                go.Bar(
                    y=metrics,
                    x=corr_values,
                    orientation='h',
                    name='Correlation',
                    marker=dict(color=colors_corr),
                    text=[f"{v:.3f}" for v in corr_values],
                    textposition='outside'
                ),
                row=2, col=2
            )

    # Update layout
    fig.update_yaxes(title_text="Hours", row=1, col=1)
    fig.update_yaxes(title_text="Hours", row=1, col=2)
    fig.update_yaxes(title_text="Score", row=2, col=1)
    fig.update_xaxes(title_text="Correlation Coefficient", row=2, col=2)

    fig.update_layout(
        height=800,
        showlegend=True,
        hovermode='closest',
        template='plotly_white',
        font=dict(size=11)
    )

    return html.Div([
        html.H3("Sleep & Recovery Analysis", className="mb-4"),
        cards,
        dcc.Graph(figure=fig, config={'displayModeBar': False}),
        dbc.Alert([
            html.H5("Key Insights", className="alert-heading"),
            html.Hr(),
            html.P(f"• Optimal sleep (7-9h): {sleep_quality.get('pct_optimal_sleep', 0):.0f}% of nights"),
            html.P(f"• Insufficient sleep (<6h): {sleep_quality.get('pct_insufficient_sleep', 0):.0f}% of nights"),
            html.P(f"• Best sleep day: {max(sleep_quality.get('sleep_by_day', {}).items(), key=lambda x: x[1])[0] if sleep_quality.get('sleep_by_day') else 'N/A'}"),
        ], color="info", className="mt-4")
    ])


def create_activity_patterns_section():
    """NEW: Activity patterns and consistency analysis"""

    activity_analyzer = ActivityPatternAnalyzer(all_data)
    activity = activity_analyzer.analyze_consistency()

    if not activity:
        return html.Div("No activity data available")

    # Metric cards
    cards = dbc.Row([
        dbc.Col(create_metric_card(
            "Avg Steps",
            f"{activity.get('avg_daily_steps', 0):.0f}",
            "per day",
            color='primary'
        ), width=3),
        dbc.Col(create_metric_card(
            "Active Days",
            f"{activity.get('pct_active_days', 0):.0f}%",
            ">5,000 steps",
            color='success'
        ), width=3),
        dbc.Col(create_metric_card(
            "Very Active",
            f"{activity.get('pct_very_active_days', 0):.0f}%",
            ">10,000 steps",
            color='warning'
        ), width=3),
        dbc.Col(create_metric_card(
            "Exercise",
            f"{activity.get('avg_exercise_minutes', 0):.0f} min",
            "per day",
            color='info'
        ), width=3),
    ], className="mb-4")

    # Create visualizations
    df = all_data.copy()
    df['day_of_week'] = df['date'].dt.day_name()
    df['is_weekend'] = df['date'].dt.dayofweek >= 5

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Steps Over Time',
            'Weekday vs Weekend',
            'Exercise Minutes Distribution',
            'Activity Consistency'
        ),
        specs=[
            [{"secondary_y": False}, {"type": "box"}],
            [{"type": "histogram"}, {"secondary_y": False}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    # Plot 1: Steps over time with 7-day average
    steps_df = df.dropna(subset=['StepCount'])
    fig.add_trace(
        go.Scatter(
            x=steps_df['date'],
            y=steps_df['StepCount'],
            mode='markers',
            name='Daily Steps',
            marker=dict(color=COLORS['info'], size=3, opacity=0.5)
        ),
        row=1, col=1
    )

    steps_df['steps_7d'] = steps_df['StepCount'].rolling(7).mean()
    fig.add_trace(
        go.Scatter(
            x=steps_df['date'],
            y=steps_df['steps_7d'],
            mode='lines',
            name='7-day Avg',
            line=dict(color=COLORS['primary'], width=3)
        ),
        row=1, col=1
    )

    # Reference lines
    fig.add_hline(y=5000, line_dash="dash", line_color="orange", opacity=0.5, row=1, col=1)
    fig.add_hline(y=10000, line_dash="dash", line_color="green", opacity=0.5, row=1, col=1)

    # Plot 2: Weekday vs Weekend box plot
    if 'StepCount' in df.columns:
        fig.add_trace(
            go.Box(
                y=df[~df['is_weekend']]['StepCount'],
                name='Weekday',
                marker=dict(color=COLORS['primary'])
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Box(
                y=df[df['is_weekend']]['StepCount'],
                name='Weekend',
                marker=dict(color=COLORS['success'])
            ),
            row=1, col=2
        )

    # Plot 3: Exercise time histogram
    if 'AppleExerciseTime' in df.columns:
        exercise_df = df.dropna(subset=['AppleExerciseTime'])
        fig.add_trace(
            go.Histogram(
                x=exercise_df['AppleExerciseTime'],
                name='Exercise Minutes',
                marker=dict(color=COLORS['warning']),
                nbinsx=30
            ),
            row=2, col=1
        )

    # Plot 4: Consistency metric (CV over time)
    if 'StepCount' in df.columns:
        df_monthly = df.set_index('date').resample('M')['StepCount'].agg(['mean', 'std'])
        df_monthly['cv'] = (df_monthly['std'] / df_monthly['mean']) * 100

        fig.add_trace(
            go.Scatter(
                x=df_monthly.index,
                y=df_monthly['cv'],
                mode='lines+markers',
                name='Variability',
                line=dict(color=COLORS['purple'], width=2),
                marker=dict(size=8)
            ),
            row=2, col=2
        )

    # Update layout
    fig.update_yaxes(title_text="Steps", row=1, col=1)
    fig.update_yaxes(title_text="Steps", row=1, col=2)
    fig.update_xaxes(title_text="Minutes", row=2, col=1)
    fig.update_yaxes(title_text="CV (%)", row=2, col=2)

    fig.update_layout(
        height=800,
        showlegend=True,
        hovermode='closest',
        template='plotly_white',
        font=dict(size=11)
    )

    return html.Div([
        html.H3("Activity Patterns & Consistency", className="mb-4"),
        cards,
        dcc.Graph(figure=fig, config={'displayModeBar': False}),
        dbc.Alert([
            html.H5("Consistency Analysis", className="alert-heading"),
            html.Hr(),
            html.P(f"• Weekend activity: {activity.get('weekend_vs_weekday_ratio', 0):.2f}x weekday level"),
            html.P(f"• Days with 30+ min exercise: {activity.get('pct_days_30min_exercise', 0):.0f}%"),
        ], color="success", className="mt-4")
    ])


def create_running_analysis_section():
    """Enhanced running analysis"""

    running_data = all_data[
        (all_data['date'] >= RUNNING_START) &
        (all_data['date'] <= RUNNING_END)
    ].copy()

    if running_data.empty:
        return html.Div("No data available for running analysis period")

    run_analyzer = RunningAnalyzer(running_data)
    hr_analyzer = HeartRateAnalyzer(running_data)

    df_runs, weekly_runs = run_analyzer.calculate_running_metrics()
    hr_trends = hr_analyzer.analyze_hr_trends()
    fitness_score = hr_analyzer.calculate_cardiovascular_fitness_score()

    # Metric cards
    cards = dbc.Row([
        dbc.Col(create_metric_card(
            "Resting HR",
            f"{hr_trends.get('avg_resting_hr', 0):.0f}",
            "bpm",
            color='primary'
        ), width=3),
        dbc.Col(create_metric_card(
            "HRV",
            f"{hr_trends.get('avg_hrv', 0):.0f}",
            "ms",
            color='success'
        ), width=3),
        dbc.Col(create_metric_card(
            "Fitness Score",
            f"{fitness_score:.0f}/100",
            "cardiovascular",
            color='warning'
        ), width=3),
        dbc.Col(create_metric_card(
            "Total Distance",
            f"{(df_runs['distance_km'].sum() if 'distance_km' in df_runs.columns else 0):.0f}",
            "km",
            color='info'
        ), width=3),
    ], className="mb-4")

    # Create visualization
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Weekly Distance',
            'Resting HR Trend',
            'HRV Trend',
            'Running Power',
            'VO2 Max',
            'Pace Distribution'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.08
    )

    # Plot 1: Weekly distance
    if not weekly_runs.empty and 'distance_km' in weekly_runs.columns:
        fig.add_trace(
            go.Bar(
                x=weekly_runs['week'],
                y=weekly_runs['distance_km'],
                name='Distance',
                marker=dict(color=COLORS['primary'])
            ),
            row=1, col=1
        )

    # Plot 2: Resting HR
    if 'RestingHeartRate' in df_runs.columns:
        rhr_data = df_runs.dropna(subset=['RestingHeartRate'])
        fig.add_trace(
            go.Scatter(
                x=rhr_data['date'],
                y=rhr_data['RestingHeartRate'],
                mode='lines+markers',
                name='RHR',
                line=dict(color=COLORS['danger'], width=2),
                marker=dict(size=4)
            ),
            row=1, col=2
        )

    # Plot 3: HRV
    if 'HeartRateVariabilitySDNN' in df_runs.columns:
        hrv_data = df_runs.dropna(subset=['HeartRateVariabilitySDNN'])
        fig.add_trace(
            go.Scatter(
                x=hrv_data['date'],
                y=hrv_data['HeartRateVariabilitySDNN'],
                mode='lines+markers',
                name='HRV',
                line=dict(color=COLORS['purple'], width=2),
                marker=dict(size=4)
            ),
            row=1, col=3
        )

    # Plot 4: Running power
    if 'RunningPower' in df_runs.columns:
        power_data = df_runs.dropna(subset=['RunningPower'])
        fig.add_trace(
            go.Scatter(
                x=power_data['date'],
                y=power_data['RunningPower'],
                mode='markers',
                name='Power',
                marker=dict(color=COLORS['warning'], size=6)
            ),
            row=2, col=1
        )

    # Plot 5: VO2 Max
    if 'VO2Max' in df_runs.columns:
        vo2_data = df_runs.dropna(subset=['VO2Max'])
        fig.add_trace(
            go.Scatter(
                x=vo2_data['date'],
                y=vo2_data['VO2Max'],
                mode='lines+markers',
                name='VO2 Max',
                line=dict(color=COLORS['teal'], width=2),
                marker=dict(size=6)
            ),
            row=2, col=2
        )

    # Plot 6: Pace distribution
    if 'pace_min_per_km' in df_runs.columns:
        pace_data = df_runs.dropna(subset=['pace_min_per_km'])
        pace_data = pace_data[pace_data['pace_min_per_km'] < 15]  # Remove outliers
        fig.add_trace(
            go.Histogram(
                x=pace_data['pace_min_per_km'],
                name='Pace',
                marker=dict(color=COLORS['success']),
                nbinsx=20
            ),
            row=2, col=3
        )

    # Update layout
    fig.update_yaxes(title_text="km", row=1, col=1)
    fig.update_yaxes(title_text="bpm", row=1, col=2)
    fig.update_yaxes(title_text="ms", row=1, col=3)
    fig.update_yaxes(title_text="watts", row=2, col=1)
    fig.update_xaxes(title_text="min/km", row=2, col=3)

    fig.update_layout(
        height=700,
        showlegend=False,
        template='plotly_white',
        font=dict(size=10)
    )

    return html.Div([
        html.H3("Running & Cardiovascular Analysis", className="mb-4"),
        cards,
        dcc.Graph(figure=fig, config={'displayModeBar': False}),
    ])


def create_overview_section():
    """Enhanced overview with new metrics"""

    # Additional analyzers
    weight_analyzer = WeightLossAnalyzer(all_data)
    phases = weight_analyzer.analyze_weight_loss_phases()
    prediction = weight_analyzer.predict_goal_date(70.0)

    return html.Div([
        html.H3("Health Metrics Overview", className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("📊 Data Summary")),
                    dbc.CardBody([
                        html.P(f"Total days tracked: {len(all_data)}", className="mb-2"),
                        html.P(f"Date range: {all_data['date'].min().strftime('%Y-%m-%d')} to {all_data['date'].max().strftime('%Y-%m-%d')}", className="mb-2"),
                        html.Hr(),
                        html.H6("Available Metrics:", className="mb-2"),
                        html.Div([
                            dbc.Badge(f"{col}: {all_data[col].notna().sum()} days", color="primary", className="me-2 mb-2")
                            for col in all_data.columns[:10]
                            if col != 'date' and all_data[col].notna().sum() > 0
                        ])
                    ])
                ], className="shadow-sm")
            ], width=6),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("🎯 Weight Loss Progress")),
                    dbc.CardBody([
                        html.H6("Phase Breakdown:"),
                        dbc.Progress([
                            dbc.Progress(
                                value=phases.get('days_in_rapid_loss', 0) if phases else 0,
                                color="danger",
                                bar=True,
                                label="Rapid"
                            ),
                            dbc.Progress(
                                value=phases.get('days_in_steady_loss', 0) if phases else 0,
                                color="warning",
                                bar=True,
                                label="Steady"
                            ),
                            dbc.Progress(
                                value=phases.get('days_in_maintenance', 0) if phases else 0,
                                color="success",
                                bar=True,
                                label="Maintenance"
                            ),
                        ], className="mb-3"),
                        html.Hr(),
                        html.P(f"Total weight change: {phases.get('total_weight_change', 0):.1f} kg" if phases else "No data"),
                        html.P(f"Average rate: {phases.get('avg_weekly_rate', 0):.2f} kg/week" if phases else ""),
                        html.Hr(),
                        html.H6("Goal Prediction (70 kg):") if prediction and 'predicted_goal_date' in prediction else html.Div(),
                        html.P(f"🎯 {prediction.get('predicted_goal_date', 'N/A')}" if prediction and 'predicted_goal_date' in prediction else "", className="text-success fw-bold"),
                    ])
                ], className="shadow-sm")
            ], width=6),
        ], className="mb-4")
    ])


# App layout with modern styling
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("💪 Health Analytics Dashboard", className="text-center mb-2 mt-4"),
                html.P("Comprehensive analysis of your Apple Watch health data", className="text-center text-muted mb-4")
            ])
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="📊 Overview", tab_id="overview"),
                dbc.Tab(label="🎯 TDEE Analysis", tab_id="tdee"),
                dbc.Tab(label="🏃 Running", tab_id="running"),
                dbc.Tab(label="😴 Sleep & Recovery", tab_id="sleep"),
                dbc.Tab(label="📈 Activity Patterns", tab_id="activity"),
            ], id="tabs", active_tab="overview", className="mb-4")
        ])
    ]),

    dbc.Row([
        dbc.Col([
            html.Div(id="tab-content")
        ])
    ]),

    html.Footer([
        html.Hr(),
        html.P("Health Analytics Dashboard • Built with Plotly Dash", className="text-center text-muted small")
    ], className="mt-5 mb-3")
], fluid=True, style={'backgroundColor': '#f8f9fa'})


@callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab")
)
def render_tab_content(active_tab):
    """Render content based on selected tab"""

    if active_tab == "tdee":
        return create_tdee_analysis_section()
    elif active_tab == "running":
        return create_running_analysis_section()
    elif active_tab == "sleep":
        return create_sleep_recovery_section()
    elif active_tab == "activity":
        return create_activity_patterns_section()
    else:  # overview
        return create_overview_section()


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting Enhanced Health Analytics Dashboard...")
    print("="*60)
    print("\n📊 Dashboard Features:")
    print("  • TDEE Analysis with visual insights")
    print("  • Sleep & Recovery tracking")
    print("  • Activity patterns & consistency")
    print("  • Running performance metrics")
    print("  • Weight loss phase analysis")
    print("\n🌐 Open your browser and navigate to: http://127.0.0.1:8050/")
    print("\n⌨️  Press Ctrl+C to stop the server\n")
    print("="*60 + "\n")

    app.run_server(debug=True, host='0.0.0.0', port=8050)
