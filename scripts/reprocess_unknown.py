"""
Reprocess unknown type data from bronze layer to correct HK types
"""
import json
from pathlib import Path
import hashlib
from datetime import datetime

# Map metric names to HK types
METRIC_NAME_TO_HK_TYPE = {
    "apple_stand_hour": "HKCategoryTypeIdentifierAppleStandHour",
    "apple_exercise_time": "HKQuantityTypeIdentifierAppleExerciseTime",
    "apple_stand_time": "HKQuantityTypeIdentifierAppleStandTime",
    "walking_asymmetry_percentage": "HKQuantityTypeIdentifierWalkingAsymmetryPercentage",
    "stair_speed_down": "HKQuantityTypeIdentifierStairDescentSpeed",
    "stair_speed_up": "HKQuantityTypeIdentifierStairAscentSpeed",
    "walking_double_support_percentage": "HKQuantityTypeIdentifierWalkingDoubleSupportPercentage",
    "step_count": "HKQuantityTypeIdentifierStepCount",
    "distance_walking_running": "HKQuantityTypeIdentifierDistanceWalkingRunning",
    "active_energy_burned": "HKQuantityTypeIdentifierActiveEnergyBurned",
    "basal_energy_burned": "HKQuantityTypeIdentifierBasalEnergyBurned",
    "heart_rate": "HKQuantityTypeIdentifierHeartRate",
    "resting_heart_rate": "HKQuantityTypeIdentifierRestingHeartRate",
    "heart_rate_variability": "HKQuantityTypeIdentifierHeartRateVariabilitySDNN",
    "body_mass": "HKQuantityTypeIdentifierBodyMass",
    "height": "HKQuantityTypeIdentifierHeight",
    "vo2_max": "HKQuantityTypeIdentifierVO2Max",
    "sleep_analysis": "HKCategoryTypeIdentifierSleepAnalysis",
}

def normalize_datestr(s: str) -> str:
    """Return YYYY-MM-DD from an ISO datetime."""
    try:
        if isinstance(s, str):
            # Handle formats like "2025-11-13 07:00:00 +0100"
            s = s.replace(" +", "+")
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt.date().isoformat()
    except Exception:
        pass
    return (s or "")[:10]

def safe_uuid(sample: dict) -> str:
    """Generate UUID from sample content."""
    h = hashlib.sha1(json.dumps(sample, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return h

def save_sample(hk_type: str, sample: dict, bronze_root: Path) -> str:
    """Write one JSON file per sample."""
    start = sample.get("start") or sample.get("dateTime") or datetime.now().isoformat()
    day = normalize_datestr(start)
    uid = safe_uuid(sample)

    folder = bronze_root / f"type={hk_type}" / f"date={day}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{uid}.json"

    # Skip if already exists
    if path.exists():
        return str(path)

    with path.open("w", encoding="utf-8") as f:
        json.dump({
            "type": hk_type,
            "received_ts": datetime.now().isoformat(),
            **sample
        }, f, ensure_ascii=False)

    return str(path)

def reprocess_unknown_files():
    """Reprocess all unknown type files."""
    bronze_root = Path("bronze")
    unknown_dir = bronze_root / "type=unknown"

    if not unknown_dir.exists():
        print("No unknown directory found")
        return

    print("Reprocessing unknown files...")
    total_samples = 0
    files_processed = 0

    # Find all JSON files in unknown directory
    for json_file in unknown_dir.rglob("*.json"):
        files_processed += 1
        print(f"\nProcessing {json_file.name}...")

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check if it has metrics format
        if "data" in data and isinstance(data["data"], dict):
            payload_data = data["data"]
        else:
            payload_data = data

        if "metrics" in payload_data:
            metrics = payload_data.get("metrics") or []

            for metric in metrics:
                metric_name = metric.get("name", "unknown")
                hk_type = METRIC_NAME_TO_HK_TYPE.get(metric_name, f"HKQuantityTypeIdentifier{metric_name}")

                # Each data point becomes a separate sample
                data_points = metric.get("data") or []
                if not isinstance(data_points, list):
                    continue

                for data_point in data_points:
                    # Convert iOS format to standard format
                    sample = {
                        "start": data_point.get("date") or data_point.get("dateTime"),
                        "value": data_point.get("qty") or data_point.get("value"),
                        "unit": metric.get("units", ""),
                        "source": data_point.get("source", "iOS App")
                    }
                    save_sample(hk_type, sample, bronze_root)
                    total_samples += 1

            print(f"  Extracted {len(metrics)} metrics from file")

    print(f"\n[OK] Reprocessed {files_processed} files")
    print(f"[OK] Created {total_samples} individual sample files")
    print(f"\nNow run: python process_bronze.py")

if __name__ == "__main__":
    reprocess_unknown_files()
