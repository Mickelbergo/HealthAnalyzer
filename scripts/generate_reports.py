"""
Generate Health Analytics Reports

This script generates comprehensive analysis reports and exports them as JSON/CSV
"""

import json
from pathlib import Path
import pandas as pd
from datetime import datetime

from data_loader import HealthDataLoader, TDEEAnalyzer, calculate_bmr
from running_analysis import RunningAnalyzer, HeartRateAnalyzer, analyze_race_performance
from advanced_analysis import (
    SleepAnalyzer, NutritionAnalyzer, RecoveryAnalyzer,
    ActivityPatternAnalyzer, WeightLossAnalyzer
)


def generate_all_reports():
    """Generate comprehensive health analysis reports"""

    print("="*60)
    print("HEALTH ANALYTICS REPORT GENERATOR")
    print("="*60)
    print()

    # Load data
    print("Loading health data...")
    loader = HealthDataLoader("files")
    all_data = loader.load_all_daily_metrics()
    sleep_data = loader.get_sleep_duration()

    # Merge sleep data
    if not sleep_data.empty:
        sleep_data['date'] = pd.to_datetime(sleep_data['date'])
        all_data = all_data.merge(sleep_data, on='date', how='left')

    print(f"[OK] Loaded {len(all_data)} days of health data")
    print(f"  Date range: {all_data['date'].min().date()} to {all_data['date'].max().date()}")
    print()

    # Create reports directory
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # ===== TDEE ANALYSIS (March - June 2025) =====
    print("-" * 60)
    print("TDEE ANALYSIS: March - June 2025")
    print("-" * 60)

    tdee_data = all_data[
        (all_data['date'] >= "2025-03-01") &
        (all_data['date'] <= "2025-06-30")
    ].copy()

    if not tdee_data.empty:
        analyzer = TDEEAnalyzer(tdee_data)
        tdee_results = analyzer.calculate_tdee(window=7)
        summary = analyzer.get_tdee_summary(window=7)

        print(f"\n[TDEE SUMMARY]")
        print(f"  - Days analyzed: {summary.get('days_analyzed', 0)}")
        print(f"  - Starting weight: {summary.get('start_weight', 0):.1f} kg")
        print(f"  - Ending weight: {summary.get('end_weight', 0):.1f} kg")
        print(f"  - Total weight change: {summary.get('total_weight_change', 0):.2f} kg")
        print(f"  - Weight loss rate: {abs(summary.get('total_weight_change', 0) / summary.get('days_analyzed', 1) * 7):.2f} kg/week")
        print(f"  - Average calorie intake: {summary.get('avg_calories_in', 0):.0f} kcal/day")
        print(f"  - Estimated TDEE: {summary.get('avg_tdee', 0):.0f} +/- {summary.get('std_tdee', 0):.0f} kcal/day")
        print(f"  - Average daily deficit: {summary.get('avg_daily_deficit', 0):.0f} kcal")

        # Save detailed results
        tdee_results.to_csv(reports_dir / "tdee_analysis_detailed.csv", index=False)
        with open(reports_dir / "tdee_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n[OK] Saved TDEE analysis to {reports_dir}/tdee_*.csv and .json")
    else:
        print("[WARNING] No data available for TDEE analysis period")

    print()

    # ===== RUNNING ANALYSIS (July - September 2025) =====
    print("-" * 60)
    print("RUNNING & CARDIOVASCULAR ANALYSIS: July - September 2025")
    print("-" * 60)

    running_data = all_data[
        (all_data['date'] >= "2025-07-01") &
        (all_data['date'] <= "2025-09-30")
    ].copy()

    if not running_data.empty:
        run_analyzer = RunningAnalyzer(running_data)
        hr_analyzer = HeartRateAnalyzer(running_data)

        # Running metrics
        df_runs, weekly_runs = run_analyzer.calculate_running_metrics()

        if 'distance_km' in df_runs.columns:
            total_distance = df_runs['distance_km'].sum()
            print(f"\n[RUNNING SUMMARY]")
            print(f"  - Total distance: {total_distance:.1f} km")
            print(f"  - Average weekly distance: {weekly_runs['distance_km'].mean():.1f} km")
            print(f"  - Peak weekly distance: {weekly_runs['distance_km'].max():.1f} km")
            print(f"  - Training weeks: {len(weekly_runs)}")

        # Heart rate analysis
        hr_trends = hr_analyzer.analyze_hr_trends()
        fitness_score = hr_analyzer.calculate_cardiovascular_fitness_score()

        print(f"\n[CARDIOVASCULAR METRICS]")
        print(f"  - Average resting HR: {hr_trends.get('avg_resting_hr', 0):.0f} bpm")
        print(f"  - Min resting HR: {hr_trends.get('min_resting_hr', 0):.0f} bpm")
        print(f"  - Average HRV: {hr_trends.get('avg_hrv', 0):.0f} ms")
        print(f"  - Max HRV: {hr_trends.get('max_hrv', 0):.0f} ms")
        print(f"  - Cardiovascular fitness score: {fitness_score:.0f}/100")

        # Race analysis
        race_analysis = analyze_race_performance(
            all_data,
            "2025-09-20",
            race_distance_km=21.0975,
            window_days=30
        )

        if 'race_time_minutes' in race_analysis:
            print(f"\n[HALF MARATHON PERFORMANCE - Sept 20, 2025]")
            print(f"  - Race time: {race_analysis['race_time_minutes']:.0f} minutes ({race_analysis['race_time_minutes']//60:.0f}h {race_analysis['race_time_minutes']%60:.0f}min)")
            print(f"  - Average pace: {race_analysis['average_pace_min_per_km']:.2f} min/km")
            print(f"  - Average HR: {race_analysis.get('average_race_hr', 0):.0f} bpm")
            print(f"  - Distance: {race_analysis.get('actual_distance_km', 21.0975):.2f} km")

        # Save results
        weekly_runs.to_csv(reports_dir / "running_weekly_summary.csv", index=False)
        with open(reports_dir / "running_summary.json", 'w') as f:
            json.dump({
                'total_distance_km': total_distance if 'distance_km' in df_runs.columns else 0,
                'hr_trends': hr_trends,
                'fitness_score': fitness_score,
                'race_performance': race_analysis,
            }, f, indent=2, default=str)

        print(f"\n[OK] Saved running analysis to {reports_dir}/running_*.csv and .json")
    else:
        print("[WARNING] No data available for running analysis period")

    print()

    # ===== OVERALL HEALTH SUMMARY =====
    print("-" * 60)
    print("OVERALL HEALTH SUMMARY")
    print("-" * 60)

    # Calculate BMR estimate (assuming user parameters)
    latest_weight = all_data['BodyMass'].dropna().iloc[-1] if 'BodyMass' in all_data.columns else 75
    bmr_estimate = calculate_bmr(weight_kg=latest_weight, age=25, sex='male', height_cm=175)

    print(f"\n[CURRENT STATUS]")
    print(f"  - Latest weight: {latest_weight:.1f} kg")
    print(f"  - Estimated BMR: {bmr_estimate:.0f} kcal/day")
    print(f"  - Total days tracked: {len(all_data)}")

    # Data completeness
    completeness = {}
    for col in all_data.columns:
        if col != 'date':
            completeness[col] = {
                'total_days': all_data[col].notna().sum(),
                'percentage': (all_data[col].notna().sum() / len(all_data) * 100)
            }

    print(f"\n[DATA COMPLETENESS]")
    for metric, stats in sorted(completeness.items(), key=lambda x: x[1]['total_days'], reverse=True)[:10]:
        print(f"  - {metric}: {stats['total_days']} days ({stats['percentage']:.1f}%)")

    # Save overall summary
    with open(reports_dir / "overall_summary.json", 'w') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'total_days': len(all_data),
            'date_range': {
                'start': str(all_data['date'].min().date()),
                'end': str(all_data['date'].max().date()),
            },
            'current_weight_kg': latest_weight,
            'estimated_bmr': bmr_estimate,
            'data_completeness': completeness,
        }, f, indent=2, default=str)

    print(f"\n[OK] Saved overall summary to {reports_dir}/overall_summary.json")

    print()

    # ===== ADVANCED ANALYSES =====
    print("-" * 60)
    print("ADVANCED FITNESS ANALYSES")
    print("-" * 60)

    advanced_results = {}

    # Sleep Analysis
    print("\n[SLEEP ANALYSIS]")
    sleep_analyzer = SleepAnalyzer(all_data)
    sleep_quality = sleep_analyzer.analyze_sleep_quality()
    sleep_corr = sleep_analyzer.sleep_performance_correlation()

    if sleep_quality:
        print(f"  - Average sleep: {sleep_quality.get('avg_sleep_hours', 0):.1f} hours/night")
        print(f"  - Sleep consistency score: {sleep_quality.get('consistency_score', 0):.0f}/100")
        print(f"  - Optimal sleep nights (7-9h): {sleep_quality.get('pct_optimal_sleep', 0):.0f}%")
        print(f"  - Insufficient sleep (<6h): {sleep_quality.get('pct_insufficient_sleep', 0):.0f}%")
        print(f"  - Nights tracked: {sleep_quality.get('nights_tracked', 0)}")

        if sleep_corr:
            print(f"\n  Sleep-Performance Correlations:")
            for metric, corr_data in sleep_corr.items():
                if corr_data['significant']:
                    direction = "positive" if corr_data['correlation'] > 0 else "negative"
                    print(f"    - {metric}: {corr_data['correlation']:.3f} ({direction})")

        advanced_results['sleep_quality'] = sleep_quality
        advanced_results['sleep_correlations'] = sleep_corr

    # Nutrition Analysis
    print("\n[NUTRITION ANALYSIS]")
    nutrition_analyzer = NutritionAnalyzer(all_data)
    nutrition = nutrition_analyzer.analyze_macros()

    if nutrition:
        if 'avg_protein_g' in nutrition:
            print(f"  - Average protein: {nutrition['avg_protein_g']:.0f}g/day")
        if 'avg_carbohydrates_g' in nutrition:
            print(f"  - Average carbs: {nutrition['avg_carbohydrates_g']:.0f}g/day")
        if 'avg_fattotal_g' in nutrition:
            print(f"  - Average fat: {nutrition['avg_fattotal_g']:.0f}g/day")

        if 'avg_protein_pct' in nutrition:
            print(f"\n  Macro Distribution:")
            print(f"    - Protein: {nutrition['avg_protein_pct']:.1f}%")
            print(f"    - Carbs: {nutrition['avg_carbs_pct']:.1f}%")
            print(f"    - Fat: {nutrition['avg_fat_pct']:.1f}%")

        if 'avg_protein_per_kg' in nutrition:
            print(f"\n  - Protein per kg body weight: {nutrition['avg_protein_per_kg']:.2f}g/kg")

        advanced_results['nutrition'] = nutrition

    # Recovery Analysis
    print("\n[RECOVERY & OVERTRAINING RISK]")
    recovery_analyzer = RecoveryAnalyzer(all_data)
    recovery = recovery_analyzer.detect_overtraining_risk()

    if recovery:
        print(f"  - Current recovery score: {recovery.get('current_recovery_score', 0):.0f}/100")
        print(f"  - 7-day average: {recovery.get('7d_avg_recovery', 0):.0f}/100")
        print(f"  - Recovery trend: {recovery.get('trend_direction', 'unknown')}")
        print(f"  - Overtraining risk: {recovery.get('overtraining_risk', 'UNKNOWN')}")

        advanced_results['recovery'] = recovery

    # Activity Pattern Analysis
    print("\n[ACTIVITY PATTERNS]")
    activity_analyzer = ActivityPatternAnalyzer(all_data)
    activity = activity_analyzer.analyze_consistency()

    if activity:
        if 'avg_daily_steps' in activity:
            print(f"  - Average daily steps: {activity['avg_daily_steps']:.0f}")
        if 'pct_active_days' in activity:
            print(f"  - Active days (>5000 steps): {activity['pct_active_days']:.0f}%")
        if 'pct_very_active_days' in activity:
            print(f"  - Very active days (>10000 steps): {activity['pct_very_active_days']:.0f}%")

        if 'weekday_avg_steps' in activity:
            print(f"\n  Weekday vs Weekend:")
            print(f"    - Weekday avg: {activity['weekday_avg_steps']:.0f} steps")
            print(f"    - Weekend avg: {activity['weekend_avg_steps']:.0f} steps")
            print(f"    - Ratio: {activity['weekend_vs_weekday_ratio']:.2f}x")

        if 'avg_exercise_minutes' in activity:
            print(f"\n  - Average exercise: {activity['avg_exercise_minutes']:.0f} min/day")
            print(f"  - Days with 30+ min exercise: {activity.get('pct_days_30min_exercise', 0):.0f}%")

        advanced_results['activity_patterns'] = activity

    # Weight Loss Phase Analysis
    print("\n[WEIGHT LOSS PHASES]")
    weight_analyzer = WeightLossAnalyzer(all_data)
    phases = weight_analyzer.analyze_weight_loss_phases()

    if phases:
        print(f"  - Total weight change: {phases.get('total_weight_change', 0):.2f} kg")
        print(f"  - Average rate: {phases.get('avg_weekly_rate', 0):.2f} kg/week")
        print(f"  - Maximum weekly loss: {phases.get('max_weekly_loss', 0):.2f} kg")

        print(f"\n  Time in different phases:")
        if 'days_in_rapid_loss' in phases:
            print(f"    - Rapid loss (>0.5 kg/week): {phases['days_in_rapid_loss']} days")
        if 'days_in_steady_loss' in phases:
            print(f"    - Steady loss (0.2-0.5 kg/week): {phases['days_in_steady_loss']} days")
        if 'days_in_maintenance' in phases:
            print(f"    - Maintenance: {phases['days_in_maintenance']} days")
        if 'days_in_gaining' in phases:
            print(f"    - Gaining: {phases['days_in_gaining']} days")

        advanced_results['weight_loss_phases'] = phases

    # Goal Weight Prediction
    goal_weight = 70.0  # Example goal
    prediction = weight_analyzer.predict_goal_date(goal_weight)

    if 'predicted_goal_date' in prediction:
        print(f"\n[GOAL PREDICTION - Target: {goal_weight} kg]")
        print(f"  - Current weight: {prediction['current_weight']:.1f} kg")
        print(f"  - Weight to lose: {prediction['weight_to_lose']:.1f} kg")
        print(f"  - Current rate: {prediction['current_rate_kg_per_day']*7:.2f} kg/week")
        print(f"  - Estimated days to goal: {prediction['estimated_days_to_goal']}")
        print(f"  - Predicted goal date: {prediction['predicted_goal_date']}")
        print(f"  - Prediction confidence: {prediction['confidence_r_squared']:.2%}")

        advanced_results['goal_prediction'] = prediction

    # Save all advanced analyses
    with open(reports_dir / "advanced_analysis.json", 'w') as f:
        json.dump(advanced_results, f, indent=2, default=str)

    print(f"\n[OK] Saved advanced analysis to {reports_dir}/advanced_analysis.json")

    print()
    print("="*60)
    print("REPORT GENERATION COMPLETE")
    print(f"All reports saved to: {reports_dir.absolute()}")
    print("="*60)


if __name__ == "__main__":
    generate_all_reports()
