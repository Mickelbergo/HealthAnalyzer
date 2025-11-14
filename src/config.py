"""
Configuration File for HealthAnalyzer

Customize this file with your personal parameters and analysis preferences.
"""

# ========================================
# PERSONAL INFORMATION
# ========================================

# Used for BMR and heart rate zone calculations
PERSONAL_INFO = {
    'age': 25,              # Your age in years
    'sex': 'male',          # 'male' or 'female'
    'height_cm': 175,       # Your height in centimeters
    'max_hr': None,         # Your max heart rate (or None to estimate from age)
}

# ========================================
# ANALYSIS PERIODS
# ========================================

# TDEE Analysis Period (Weight loss/gain tracking)
TDEE_PERIOD = {
    'start': "2025-03-01",  # Start date (YYYY-MM-DD)
    'end': "2025-06-30",    # End date (YYYY-MM-DD)
    'window_days': 7,       # Rolling window for TDEE calculation
}

# Running Analysis Period (Training tracking)
RUNNING_PERIOD = {
    'start': "2025-07-01",  # Start date (YYYY-MM-DD)
    'end': "2025-09-30",    # End date (YYYY-MM-DD)
    'race_date': "2025-09-20",  # Race date (if applicable)
    'race_distance_km': 21.0975,  # Race distance (21.0975 = half marathon)
}

# ========================================
# ANALYSIS PARAMETERS
# ========================================

# TDEE Calculation
TDEE_CONFIG = {
    'kcal_per_kg_fat': 7700,    # Energy content of 1 kg body fat
    'min_days': 5,               # Minimum days required for analysis
    'tef_fraction': 0.10,        # Thermic Effect of Food (default: 10%)
}

# Running Analysis
RUNNING_CONFIG = {
    'min_run_distance_km': 1.0,  # Minimum distance to count as a run
    'recovery_hr_threshold': 100, # HR threshold for recovery runs
}

# Heart Rate Zones (will be calculated if max_hr is None)
# Based on % of max heart rate
HR_ZONES = {
    'Zone 1 (Recovery)': (0.50, 0.60),
    'Zone 2 (Aerobic)': (0.60, 0.70),
    'Zone 3 (Tempo)': (0.70, 0.80),
    'Zone 4 (Threshold)': (0.80, 0.90),
    'Zone 5 (Max)': (0.90, 1.00),
}

# ========================================
# DASHBOARD SETTINGS
# ========================================

DASHBOARD_CONFIG = {
    'host': '0.0.0.0',      # Host address (0.0.0.0 = all interfaces)
    'port': 8050,           # Port for dashboard
    'debug': True,          # Debug mode
    'theme': 'BOOTSTRAP',   # Bootstrap theme
}

# ========================================
# API SETTINGS
# ========================================

API_CONFIG = {
    'host': '0.0.0.0',      # Host address for API
    'port': 8000,           # Port for API
    'data_dir': 'bronze',   # Directory for raw ingested data
}

# ========================================
# DATA DIRECTORIES
# ========================================

DATA_PATHS = {
    'csv_files': 'files',               # CSV files from Apple Health
    'bronze_layer': 'bronze',           # Raw JSON data from API
    'reports': 'reports',               # Generated reports
    'apple_export': 'apple_health_export',  # Manual Apple Health export
}

# ========================================
# METRICS TO TRACK
# ========================================

# Metrics and their aggregation functions
# 'sum' = total per day, 'mean' = average per day, 'last' = last value of day
METRICS_CONFIG = {
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

# ========================================
# VISUALIZATION SETTINGS
# ========================================

VIZ_CONFIG = {
    'color_scheme': {
        'primary': '#2E86AB',      # Blue
        'success': '#06A77D',      # Green
        'warning': '#F77F00',      # Orange
        'danger': '#D62828',       # Red
        'info': '#16A085',         # Teal
        'purple': '#9B59B6',       # Purple
    },
    'plot_height': 900,
    'plot_template': 'plotly_white',
}
