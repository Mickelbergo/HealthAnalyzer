# Create a notebook-friendly version of the pipeline with a simple function API (no argparse).
# Saves to /mnt/data/health_bmr_pipeline_notebook.py and provides example usage.

from pathlib import Path


"""
Notebook-friendly Apple Health pipeline:
- Import functions and call run_pipeline(...) directly (no argparse).
- Aggregates daily features from apple-health-parser CSVs.
- Computes Oxford Henry BMR and "Apple-like" adjusted BMR.
- Estimates maintenance calories.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# -----------------------------
# Utility helpers
# -----------------------------

def load_csv_safely(path: Path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"[WARN] Could not read {path}: {e}")
            return None
    else:
        print(f"[WARN] Missing expected file: {path}")
        return None

def ensure_datetime(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan

# -----------------------------
# Oxford Henry BMR (adults only, simplified bands)
# -----------------------------

def bmr_oxford_henry(weight_kg: float, age: int, sex: str) -> float:
    """
    Returns BMR (kcal/day) using Oxford Henry adult equations.
    Henry CJ. Public Health Nutr. 2005.
    """
    if pd.isna(weight_kg) or weight_kg <= 0:
        return np.nan
    sex = sex.lower()
    if age < 18:
        return np.nan

    if sex == "male":
        if 18 <= age < 30:
            return 15.057 * weight_kg + 692.2
        elif 30 <= age <= 60:
            return 11.472 * weight_kg + 873.1
        else:
            return 11.711 * weight_kg + 587.7
    elif sex == "female":
        if 18 <= age < 30:
            return 14.818 * weight_kg + 486.6
        elif 30 <= age <= 60:
            return 8.126 * weight_kg + 845.6
        else:
            return 9.082 * weight_kg + 658.5
    else:
        return np.nan

def adjust_bmr(bmr_base: float,
               resting_hr: float | None = None,
               hrv_ms: float | None = None,
               sleep_hours: float | None = None,
               vo2max: float | None = None,
               sleep_min_threshold: float = 6.0) -> float:
    """Heuristic 'Apple-like' personalization of BMR."""
    if pd.isna(bmr_base):
        return np.nan
    bmr = bmr_base

    if resting_hr is not None and not pd.isna(resting_hr):
        bmr *= (1.0 + (60.0 - resting_hr) / 600.0)

    if hrv_ms is not None and not pd.isna(hrv_ms):
        if hrv_ms < 30:
            bmr *= 0.97
        elif hrv_ms > 60:
            bmr *= 1.02

    if sleep_hours is not None and not pd.isna(sleep_hours):
        if sleep_hours < sleep_min_threshold:
            bmr *= 0.95

    if vo2max is not None and not pd.isna(vo2max):
        if vo2max > 50:
            bmr *= 1.03
        elif vo2max < 30:
            bmr *= 0.97

    return bmr

# -----------------------------
# Feature engineering
# -----------------------------

HK = dict(
    BODY_MASS="HKQuantityTypeIdentifierBodyMass",
    HEIGHT="HKQuantityTypeIdentifierHeight",
    HR="HKQuantityTypeIdentifierHeartRate",
    RHR="HKQuantityTypeIdentifierRestingHeartRate",
    HRV="HKQuantityTypeIdentifierHeartRateVariabilitySDNN",
    VO2="HKQuantityTypeIdentifierVO2Max",
    STEPS="HKQuantityTypeIdentifierStepCount",
    DIST_RW="HKQuantityTypeIdentifierDistanceWalkingRunning",
    FLIGHTS="HKQuantityTypeIdentifierFlightsClimbed",
    EX_MIN="HKQuantityTypeIdentifierAppleExerciseTime",
    STAND_MIN="HKQuantityTypeIdentifierAppleStandTime",
    AE="HKQuantityTypeIdentifierActiveEnergyBurned",
    BE="HKQuantityTypeIdentifierBasalEnergyBurned",
    DIET_EN="HKQuantityTypeIdentifierDietaryEnergyConsumed",
    SLEEP="HKCategoryTypeIdentifierSleepAnalysis",
    WALK_HR_AVG="HKQuantityTypeIdentifierWalkingHeartRateAverage",
)

SLEEP_VALUES_ASLEEP = {"Asleep", "InBed", "asleep", "asleepCore", "asleepREM", "asleepDeep"}

def aggregate_records_daily(records: pd.DataFrame) -> pd.DataFrame:
    if records is None or records.empty:
        return pd.DataFrame()

    ensure_datetime(records, ["startDate", "endDate", "creationDate"])
    records["date"] = records["startDate"].dt.date
    records["value_float"] = records["value"].apply(safe_float)

    energy = records[records["type"].isin([HK["AE"], HK["BE"], HK["DIET_EN"]])].copy()
    energy_pivot = (energy
                    .pivot_table(index="date", columns="type", values="value_float", aggfunc="sum")
                    .rename_axis(None, axis=1))

    qty_sum_types = [HK["STEPS"], HK["DIST_RW"], HK["FLIGHTS"], HK["EX_MIN"], HK["STAND_MIN"]]
    qty_sum = (records[records["type"].isin(qty_sum_types)]
               .pivot_table(index="date", columns="type", values="value_float", aggfunc="sum")
               .rename_axis(None, axis=1))

    hr_mean_types = [HK["HR"], HK["RHR"], HK["HRV"], HK["VO2"], HK["WALK_HR_AVG"]]
    hr_mean = (records[records["type"].isin(hr_mean_types)]
               .pivot_table(index="date", columns="type", values="value_float", aggfunc="mean")
               .rename_axis(None, axis=1))

    bm = records[records["type"] == HK["BODY_MASS"]][["date", "startDate", "value_float"]].copy()
    bm = bm.sort_values(["date", "startDate"]).groupby("date").tail(1).set_index("date")
    bm = bm.rename(columns={"value_float": HK["BODY_MASS"]})[[HK["BODY_MASS"]]]

    sleep = records[records["type"] == HK["SLEEP"]].copy()
    sleep["dur_hours"] = (sleep["endDate"] - sleep["startDate"]).dt.total_seconds() / 3600.0

    if "value" in sleep.columns:
        sleep_asleep = sleep[sleep["value"].isin(SLEEP_VALUES_ASLEEP)]
        if not sleep_asleep.empty:
            sleep_daily = sleep_asleep.groupby("date")["dur_hours"].sum().to_frame("sleep_hours")
        else:
            sleep_daily = sleep.groupby("date")["dur_hours"].sum().to_frame("sleep_hours")
    else:
        sleep_daily = sleep.groupby("date")["dur_hours"].sum().to_frame("sleep_hours")

    dfs = [energy_pivot, qty_sum, hr_mean, bm, sleep_daily]
    daily = None
    for d in dfs:
        if d is not None and not d.empty:
            daily = d if daily is None else daily.join(d, how="outer")

    if daily is None:
        return pd.DataFrame()

    rename_map = {
        HK["AE"]: "active_kcal",
        HK["BE"]: "basal_kcal",
        HK["DIET_EN"]: "diet_kcal",
        HK["STEPS"]: "steps",
        HK["DIST_RW"]: "distance_m",
        HK["FLIGHTS"]: "flights",
        HK["EX_MIN"]: "exercise_min",
        HK["STAND_MIN"]: "stand_min",
        HK["HR"]: "hr_mean",
        HK["RHR"]: "resting_hr_mean",
        HK["HRV"]: "hrv_ms_mean",
        HK["VO2"]: "vo2max_ml_kg_min",
        HK["WALK_HR_AVG"]: "walking_hr_avg",
        HK["BODY_MASS"]: "weight_kg",
    }
    daily = daily.rename(columns=rename_map)

    if "distance_m" in daily.columns:
        med = daily["distance_m"].median(skipna=True)
        if not pd.isna(med) and med < 200:
            daily["distance_m"] = daily["distance_m"] * 1000.0

    return daily.sort_index()

def derive_extra_features(daily: pd.DataFrame, height_cm: float) -> pd.DataFrame:
    df = daily.copy()
    if "weight_kg" in df.columns and height_cm and height_cm > 0:
        h_m = height_cm / 100.0
        df["bmi"] = df["weight_kg"] / (h_m * h_m)
    else:
        df["bmi"] = np.nan

    for col in ["resting_hr_mean", "hrv_ms_mean", "vo2max_ml_kg_min", "sleep_hours", "weight_kg"]:
        if col in df.columns:
            df[f"{col}_7d"] = df[col].rolling(window=7, min_periods=3).mean()

    return df

# -----------------------------
# Public API
# -----------------------------

def run_pipeline(data_dir: str | Path,
                 sex: str,
                 age: int,
                 height_cm: float,
                 tef_frac: float = 0.10,
                 sleep_min_threshold: float = 6.0,
                 out_suffix: str = ""):
    """
    Run the full pipeline from notebook:
        daily_features.csv
        daily_energy_estimates.csv
    Returns the two DataFrames (features_df, energy_df).
    """
    data_dir = Path(data_dir)
    records = load_csv_safely(data_dir / "records.csv")
    if records is None or records.empty:
        raise FileNotFoundError(f"No records.csv found or file is empty in {data_dir}")

    daily = aggregate_records_daily(records)
    daily = derive_extra_features(daily, height_cm)

    # Weight ffill & BMRs
    daily["weight_ffill"] = daily.get("weight_kg", pd.Series(index=daily.index)).ffill()
    daily["bmr_oxford"] = daily["weight_ffill"].apply(lambda w: bmr_oxford_henry(w, age, sex))

    rhr = daily.get("resting_hr_mean_7d", daily.get("resting_hr_mean"))
    hrv = daily.get("hrv_ms_mean_7d", daily.get("hrv_ms_mean"))
    slp = daily.get("sleep_hours_7d", daily.get("sleep_hours"))
    v02 = daily.get("vo2max_ml_kg_min_7d", daily.get("vo2max_ml_kg_min"))

    daily["bmr_adjusted"] = [
        adjust_bmr(b, rh, hv, sh, v02_i, sleep_min_threshold=sleep_min_threshold)
        for b, rh, hv, sh, v02_i in zip(daily["bmr_oxford"], rhr, hrv, slp, v02)
    ]

    ae = daily.get("active_kcal", pd.Series(index=daily.index, dtype=float))
    diet = daily.get("diet_kcal", pd.Series(index=daily.index, dtype=float))
    daily["tef_kcal"] = diet * tef_frac
    daily["maintenance_kcal"] = daily["bmr_adjusted"] + ae + daily["tef_kcal"]

    keep_cols = [
        "weight_kg", "bmi",
        "steps", "distance_m", "flights", "exercise_min", "stand_min",
        "hr_mean", "resting_hr_mean", "hrv_ms_mean", "vo2max_ml_kg_min", "walking_hr_avg",
        "sleep_hours", "active_kcal", "diet_kcal", "basal_kcal",
        "bmr_oxford", "bmr_adjusted", "tef_kcal", "maintenance_kcal"
    ]
    present = [c for c in keep_cols if c in daily.columns]
    energy_df = daily[present].copy()

    # Save next to data_dir
    features_path = data_dir / f"daily_features{out_suffix}.csv"
    energy_path = data_dir / f"daily_energy_estimates{out_suffix}.csv"
    daily.to_csv(features_path, index=True)
    energy_df.to_csv(energy_path, index=True)

    print(f"[OK] Wrote {features_path}")
    print(f"[OK] Wrote {energy_path}")
    return daily, energy_df
