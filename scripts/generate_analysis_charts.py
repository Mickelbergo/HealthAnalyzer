"""
Generate analysis charts for GitHub README
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import HealthDataLoader, TDEEAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# Matplotlib settings
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Load data
data_dir = Path(__file__).parent.parent / 'data' / 'processed' / 'files'
loader = HealthDataLoader(str(data_dir))
START_DATE = '2025-03-24'

print("Loading health data...")
all_data = loader.load_all_daily_metrics(start_date=START_DATE)
sleep_data = loader.get_sleep_duration(start_date=START_DATE)

# Merge sleep data
if not sleep_data.empty:
    sleep_data['date'] = pd.to_datetime(sleep_data['date'])
    all_data = all_data.merge(sleep_data, on='date', how='left')

# Create docs directory
docs_dir = Path(__file__).parent.parent / 'docs'
docs_dir.mkdir(exist_ok=True)

print(f"Loaded {len(all_data)} days of data from {all_data['date'].min()} to {all_data['date'].max()}")

# 1. Weight Loss Progress
print("\n1. Generating weight loss chart...")
weight_data = all_data[all_data['BodyMass'].notna()].copy()
if not weight_data.empty:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(weight_data['date'], weight_data['BodyMass'],
            color='#2E86AB', linewidth=2.5, marker='o', markersize=4)

    # Add trend line
    z = np.polyfit(mdates.date2num(weight_data['date']), weight_data['BodyMass'], 1)
    p = np.poly1d(z)
    ax.plot(weight_data['date'], p(mdates.date2num(weight_data['date'])),
            "r--", alpha=0.8, linewidth=2, label=f'Trend: {-z[0]*7:.2f} kg/week')

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weight (kg)', fontsize=12, fontweight='bold')
    ax.set_title('Weight Loss Progress', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(docs_dir / 'weight_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Start: {weight_data['BodyMass'].iloc[0]:.1f} kg")
    print(f"   Current: {weight_data['BodyMass'].iloc[-1]:.1f} kg")
    print(f"   Lost: {weight_data['BodyMass'].iloc[0] - weight_data['BodyMass'].iloc[-1]:.1f} kg")

# 2. Running Pace Improvement
print("\n2. Generating running pace improvement chart...")
running_data = all_data[
    (all_data['RunningSpeed'].notna()) &
    (all_data['DistanceWalkingRunning'] > 1.0)
].copy()

if not running_data.empty:
    running_data['pace_min_per_km'] = 60 / running_data['RunningSpeed']
    running_data = running_data[running_data['pace_min_per_km'] < 10]
    running_data = running_data.sort_values('date')
    running_data['pace_7d'] = running_data['pace_min_per_km'].rolling(7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(running_data['date'], running_data['pace_min_per_km'],
               alpha=0.3, s=30, color='#F77F00', label='Individual Runs')
    ax.plot(running_data['date'], running_data['pace_7d'],
            color='#D62828', linewidth=2.5, label='7-Day Average')

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pace (min/km)', fontsize=12, fontweight='bold')
    ax.set_title('Running Pace Improvement Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(docs_dir / 'pace_improvement.png', dpi=150, bbox_inches='tight')
    plt.close()

    initial_pace = running_data['pace_min_per_km'].head(7).mean()
    recent_pace = running_data['pace_min_per_km'].tail(7).mean()
    print(f"   Initial avg: {initial_pace:.2f} min/km")
    print(f"   Recent avg: {recent_pace:.2f} min/km")
    print(f"   Improvement: {initial_pace - recent_pace:.2f} min/km ({(initial_pace - recent_pace)*60:.0f}s per km faster)")

# 3. TDEE and Calorie Deficit
print("\n3. Generating TDEE analysis chart...")
tdee_data = all_data[
    (all_data['BodyMass'].notna()) &
    (all_data['DietaryEnergyConsumed'].notna()) &
    (all_data['DietaryEnergyConsumed'] > 0)
].copy()

if len(tdee_data) > 7:
    analyzer = TDEEAnalyzer(tdee_data)
    tdee_results = analyzer.calculate_tdee(window=7)

    if not tdee_results.empty and 'tdee_smooth' in tdee_results.columns:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Top: Calories In vs TDEE
        ax1.scatter(tdee_results['date'], tdee_results['DietaryEnergyConsumed'],
                   alpha=0.4, s=30, color='#06A77D', label='Calories Consumed')
        ax1.plot(tdee_results['date'], tdee_results['tdee_smooth'],
                color='#2E86AB', linewidth=2.5, label='TDEE (7-day avg)')
        ax1.set_ylabel('Calories (kcal)', fontsize=12, fontweight='bold')
        ax1.set_title('Daily Energy Balance: Consumption vs Expenditure', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

        # Bottom: Daily Deficit
        tdee_results['deficit'] = tdee_results['tdee_smooth'] - tdee_results['DietaryEnergyConsumed']
        colors = ['#D62828' if x > 0 else '#06A77D' for x in tdee_results['deficit']]
        ax2.bar(tdee_results['date'], tdee_results['deficit'], color=colors, alpha=0.6)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Daily Deficit (kcal)', fontsize=12, fontweight='bold')
        ax2.set_title('Daily Calorie Deficit (Red = Deficit, Green = Surplus)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(docs_dir / 'tdee_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

        summary = analyzer.get_tdee_summary(window=7)
        print(f"   Average TDEE: {summary.get('avg_tdee', 0):.0f} kcal/day")
        print(f"   Average intake: {summary.get('avg_calories_in', 0):.0f} kcal/day")
        print(f"   Average deficit: {abs(summary.get('avg_daily_deficit', 0)):.0f} kcal/day")

# 4. Sleep Patterns
print("\n4. Generating sleep analysis chart...")
if not sleep_data.empty:
    sleep_plot = sleep_data[sleep_data['sleep_hours'] > 0].copy()
    sleep_plot['sleep_7d'] = sleep_plot['sleep_hours'].rolling(7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(sleep_plot['date'], sleep_plot['sleep_hours'],
              alpha=0.3, s=30, color='#9B59B6', label='Daily Sleep')
    ax.plot(sleep_plot['date'], sleep_plot['sleep_7d'],
           color='#2C3E50', linewidth=2.5, label='7-Day Average')

    # Add reference lines for optimal sleep
    ax.axhspan(7, 9, alpha=0.1, color='green', label='Optimal Range (7-9h)')

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sleep (hours)', fontsize=12, fontweight='bold')
    ax.set_title('Sleep Duration Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(docs_dir / 'sleep_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   Average: {sleep_plot['sleep_hours'].mean():.1f} hours/night")
    print(f"   Optimal nights (7-9h): {((sleep_plot['sleep_hours'] >= 7) & (sleep_plot['sleep_hours'] <= 9)).sum() / len(sleep_plot) * 100:.0f}%")

# 5. Activity Overview
print("\n5. Generating activity overview...")
activity_data = all_data[all_data['StepCount'].notna()].copy()
if not activity_data.empty:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Steps
    activity_data['steps_7d'] = activity_data['StepCount'].rolling(7, min_periods=1).mean()
    ax1.bar(activity_data['date'], activity_data['StepCount'],
           alpha=0.3, color='#16A085', label='Daily Steps')
    ax1.plot(activity_data['date'], activity_data['steps_7d'],
            color='#2C3E50', linewidth=2.5, label='7-Day Average')
    ax1.axhline(y=10000, color='r', linestyle='--', alpha=0.5, label='10k Goal')
    ax1.set_ylabel('Steps', fontsize=12, fontweight='bold')
    ax1.set_title('Daily Step Count', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    # Distance
    distance_data = activity_data[activity_data['DistanceWalkingRunning'].notna()].copy()
    if not distance_data.empty:
        distance_data['distance_7d'] = distance_data['DistanceWalkingRunning'].rolling(7, min_periods=1).mean()
        ax2.bar(distance_data['date'], distance_data['DistanceWalkingRunning'],
               alpha=0.3, color='#F77F00', label='Daily Distance')
        ax2.plot(distance_data['date'], distance_data['distance_7d'],
                color='#D62828', linewidth=2.5, label='7-Day Average')
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Distance (km)', fontsize=12, fontweight='bold')
        ax2.set_title('Walking/Running Distance', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(docs_dir / 'activity_overview.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   Average steps: {activity_data['StepCount'].mean():.0f}/day")
    print(f"   Days >10k steps: {(activity_data['StepCount'] >= 10000).sum() / len(activity_data) * 100:.0f}%")

print("\n" + "="*60)
print("All charts generated successfully in docs/ folder!")
print("="*60)
