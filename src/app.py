# app.py
# A tiny REST API to receive Apple Health JSON and save it locally.
# Run with: uvicorn app:app --reload --port 8000

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# ---------- settings ----------
load_dotenv()
SECRET = os.getenv("INGEST_TOKEN")  # the same value you'll put in your exporter app
# Find project root (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent
SAVE_ROOT = PROJECT_ROOT / "data" / "raw" / "bronze"
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

# ---------- app ----------
app = FastAPI(title="Local Health Ingest")

def extract_key(request: Request) -> Optional[str]:
    """Accept key from: Authorization: Bearer <token> OR X-API-Key header OR ?key=..."""
    headers = {k.lower(): v for k, v in (request.headers or {}).items()}
    auth = headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth.split(" ", 1)[1]
    x_api_key = headers.get("x-api-key")
    if x_api_key:
        return x_api_key
    # also allow query param ?key=...
    key = request.query_params.get("key")
    if key:
        return key
    return None

def ensure_auth(request: Request):
    token = extract_key(request)
    if not SECRET:
        raise HTTPException(status_code=500, detail="Server misconfigured: no INGEST_TOKEN set.")
    if token != SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized (bad or missing key).")

def normalize_datestr(s: str) -> str:
    """Return YYYY-MM-DD from an ISO datetime like 2025-06-01T07:10:00+02:00."""
    try:
        # Try strict parse; fall back to naive slice.
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.date().isoformat()
    except Exception:
        return (s or "")[:10]

def safe_uuid(sample: Dict[str, Any]) -> str:
    """Use given uuid if present; else hash the sample for idempotent storage."""
    if "uuid" in sample and sample["uuid"]:
        return str(sample["uuid"])
    h = hashlib.sha1(json.dumps(sample, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return h

def save_sample(hk_type: str, sample: Dict[str, Any]) -> str:
    """Write one JSON file per sample into bronze/type=.../date=YYYY-MM-DD/<uuid>.json"""
    start = sample.get("start") or sample.get("dateTime") or datetime.now(timezone.utc).isoformat()
    day = normalize_datestr(start)
    uid = safe_uuid(sample)

    folder = SAVE_ROOT / f"type={hk_type}" / f"date={day}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{uid}.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump({
            "type": hk_type,
            "received_ts": datetime.now(timezone.utc).isoformat(),
            **sample
        }, f, ensure_ascii=False)

    return str(path)

@app.get("/")
def index():
    return {"ok": True, "message": "Health ingest is running. POST to /ingest/health with your key."}

@app.post("/ingest/health")
async def ingest(request: Request):
    # 1) Auth
    ensure_auth(request)

    # 2) Read JSON body (the exporter app will send it)
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Body must be JSON.")

    # 3) We accept three common shapes:
    #   A) {"type": "...", "samples": [ {...}, {...} ]}
    #   B) {"records": [ {"type":"...", ...}, ... ] }
    #   C) {"metrics": [ {"name": "...", "data": [...], ...}, ... ] } (iOS app format)
    written = 0
    saved_paths: List[str] = []

    if isinstance(payload, dict) and "samples" in payload and "type" in payload:
        hk_type = str(payload.get("type"))
        samples = payload.get("samples") or []
        if not isinstance(samples, list):
            raise HTTPException(status_code=400, detail="'samples' must be a list.")
        for s in samples:
            saved_paths.append(save_sample(hk_type, s))
            written += 1

    elif isinstance(payload, dict) and "records" in payload:
        records = payload.get("records") or []
        if not isinstance(records, list):
            raise HTTPException(status_code=400, detail="'records' must be a list.")
        for r in records:
            hk_type = str(r.get("type") or "unknown")
            saved_paths.append(save_sample(hk_type, r))
            written += 1

    elif isinstance(payload, dict) and "metrics" in payload:
        # iOS app format: {"metrics": [{"name": "...", "data": [...]}]}
        metrics = payload.get("metrics") or []
        if not isinstance(metrics, list):
            raise HTTPException(status_code=400, detail="'metrics' must be a list.")

        # Map metric names to HK types
        metric_name_to_hk_type = {
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

        for metric in metrics:
            metric_name = metric.get("name", "unknown")
            hk_type = metric_name_to_hk_type.get(metric_name, f"HKQuantityTypeIdentifier{metric_name}")

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
                saved_paths.append(save_sample(hk_type, sample))
                written += 1

    else:
        # As a fallback, treat the whole thing as one record with an inferred type
        hk_type = str(payload.get("type") or "unknown")
        saved_paths.append(save_sample(hk_type, payload))
        written = 1

    return JSONResponse({"ok": True, "written": written, "saved": saved_paths[:5]})
