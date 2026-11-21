"""
FastAPI Backend for the Physiology-Informed Cardiometabolic Digital Twin (V4).
Run locally: uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.ingest_physionet import load_cohort
from engine.digital_twin import CardiometabolicDigitalTwin, WhatIfScenario, compute_mage

app = FastAPI(
    title="Cardiometabolic Digital Twin API",
    description="Patient-Centric EKF + Physiology-Informed ODE Digital Twin",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.environ.get("PHYSIONET_DATA_DIR", str(PROJECT_ROOT / "data" / "physionet_raw"))
COHORT = load_cohort(DATA_DIR if os.path.isdir(DATA_DIR) else None)
TWINS: Dict[str, CardiometabolicDigitalTwin] = {
    pid: CardiometabolicDigitalTwin(df) for pid, df in COHORT.items()
}


class SimulateRequest(BaseModel):
    start_idx: int = Field(default=432, ge=12, le=1800)
    post_meal_walk_min: float = Field(default=25.0, ge=0.0, le=60.0)
    carb_reduction_pct: float = Field(default=15.0, ge=-50.0, le=50.0)
    stress_reduction_pct: float = Field(default=25.0, ge=-50.0, le=50.0)
    hrv_recovery_boost: float = Field(default=7.5, ge=-15.0, le=15.0)
    horizon_hours: int = Field(default=12, ge=4, le=24)


class LogEventRequest(BaseModel):
    idx: int = Field(default=432, ge=12, le=1800)
    event_type: Literal["meal", "walk", "workout", "break", "sleep", "work"]
    carbs_g: float = Field(default=50.0, ge=0.0, le=150.0)
    duration_min: float = Field(default=20.0, ge=5.0, le=600.0)
    label: str = ""


def _build_daily_schedule(win_df: pd.DataFrame, logged_events: List[Dict]) -> List[Dict]:
    events: List[Dict] = []
    if win_df.empty:
        return events

    ts_series = pd.to_datetime(win_df["timestamp"])
    hours = ts_series.dt.hour
    sleep_mask = ((hours >= 22) | (hours < 7)) & (win_df["acc_mag"] < 0.08)
    if sleep_mask.sum() >= 12:
        sleep_ts = ts_series[sleep_mask]
        events.append({
            "time": sleep_ts.iloc[0].strftime("%I:%M %p").lstrip("0"),
            "end_time": sleep_ts.iloc[-1].strftime("%I:%M %p").lstrip("0"),
            "type": "sleep",
            "title": "Sleep & Recovery",
            "detail": f"~{round(sleep_mask.sum() * 5 / 60, 1)} hrs rest",
        })

    meal_rows = win_df[win_df["carbs_g"] >= 15.0]
    for _, r in meal_rows.iterrows():
        t = pd.to_datetime(r["timestamp"])
        h = t.hour
        m_name = "Breakfast" if 5 <= h < 11 else "Lunch" if 11 <= h < 16 else "Dinner" if 16 <= h < 22 else "Late Snack"
        events.append({
            "time": t.strftime("%I:%M %p").lstrip("0"),
            "type": "meal",
            "title": m_name,
            "detail": f"~{int(round(r['carbs_g']))}g carbs",
        })

    walk_rows = win_df[win_df["acc_mag"] >= 0.18]
    last_walk_t = None
    for _, r in walk_rows.iterrows():
        t = pd.to_datetime(r["timestamp"])
        if last_walk_t is None or (t - last_walk_t).total_seconds() > 3600:
            events.append({
                "time": t.strftime("%I:%M %p").lstrip("0"),
                "type": "walk",
                "title": "Active Movement",
                "detail": "Elevated activity detected",
            })
            last_walk_t = t

    for le in logged_events:
        t = pd.to_datetime(le["timestamp"])
        events.append({
            "time": t.strftime("%I:%M %p").lstrip("0"),
            "type": "logged",
            "title": le["detail"],
            "detail": "Logged by you",
        })

    return events


@app.get("/")
def serve_frontend():
    index_path = PROJECT_ROOT / "frontend" / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(str(index_path))


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "patients_loaded": len(TWINS),
        "data_source": "PhysioNet BIG IDEAs Lab v1.1.3" if os.path.isdir(DATA_DIR) else "PhysioNet-Schema 72h Cohort",
    }


@app.get("/api/patients")
def list_patients():
    items = []
    for pid, twin in TWINS.items():
        df = twin.df
        items.append({
            "patient_id": pid,
            "label": str(df["label"].iloc[0]),
            "hba1c": float(df["hba1c"].iloc[0]),
            "age": int(df["age"].iloc[0]) if "age" in df.columns else None,
            "gender": str(df["gender"].iloc[0]),
            "total_steps": len(df),
        })
    return {"patients": items}


@app.get("/api/patients/{patient_id}/timeline")
def get_timeline(patient_id: str, end_idx: int = Query(default=432, ge=12, le=2000)):
    if patient_id not in TWINS:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    twin = TWINS[patient_id]
    clamped_end = min(end_idx, len(twin.df) - 1)
    states = twin.run_until(clamped_end)
    sub_df = twin.df.iloc[: clamped_end + 1]
    latest = states[-1]
    row = sub_df.iloc[-1]

    win_24h = sub_df.iloc[max(0, len(sub_df) - 288) :]
    g_24h = win_24h["glucose_mg_dl"].values
    tir_24h = round(float(np.mean((g_24h >= 70.0) & (g_24h <= 140.0)) * 100.0), 1)
    hours_in_range = round(tir_24h / 100.0 * (len(win_24h) * 5.0 / 60.0), 1)
    window_hours = round(len(win_24h) * 5.0 / 60.0, 1)
    mage_24h = compute_mage(g_24h)

    future_meals = twin.df.iloc[clamped_end + 1 :]
    future_meals = future_meals[future_meals["carbs_g"] >= 15.0]
    next_meal_idx = int(future_meals.index[0]) if not future_meals.empty else min(len(twin.df) - 1, clamped_end + 24)

    personal_baseline = {
        "mean_glucose": round(float(sub_df["glucose_mg_dl"].mean()), 1),
        "mean_hr": round(float(sub_df["hr_bpm"].mean()), 1),
        "mean_hrv": round(float(sub_df["hrv_rmssd_ms"].mean()), 1),
        "mean_eda": round(float(sub_df["eda_us"].mean()), 2),
        "current_hr_delta": round(float(row["hr_bpm"]) - float(sub_df["hr_bpm"].mean()), 1),
        "current_glucose_delta": round(latest.glucose_est - float(sub_df["glucose_mg_dl"].mean()), 1),
    }

    # Multi-parameter drivers for Health Coach
    carb_driver = round(min(40.0, latest.gut_carb_load * 0.45), 1)
    stress_driver = round(max(0.0, (latest.autonomic_stress - 0.45) * 18.0), 1)
    si_driver = round(max(0.0, (0.95 - latest.insulin_sensitivity) * 35.0), 1)
    act_offset = round(-min(20.0, max(0.0, latest.activity_load - 0.06) * 45.0), 1)

    hr_act_driver = round(min(40.0, max(0.0, latest.activity_load - 0.05) * 80.0), 1)
    hr_stress_driver = round(min(30.0, max(0.0, latest.autonomic_stress - 0.4) * 20.0), 1)
    hr_recovery = round(-min(25.0, max(0.0, float(row["hrv_rmssd_ms"]) - 35.0) * 0.6), 1)

    drivers_by_metric = {
        "glucose": [
            {"name": "Recent Meal Carbs Still Digesting", "impact": carb_driver},
            {"name": "Insulin Response (Lower S_I)", "impact": si_driver},
            {"name": "Stress Hormones (EDA)", "impact": stress_driver},
            {"name": "Physical Activity Burn (ACC)", "impact": act_offset},
        ],
        "heart": [
            {"name": "Physical Movement / Exercise", "impact": hr_act_driver},
            {"name": "Autonomic Stress Load", "impact": hr_stress_driver},
            {"name": "Vagal Recovery (HRV Cushion)", "impact": hr_recovery},
        ],
        "stress": [
            {"name": "Sympathetic Skin Response (EDA)", "impact": round(min(40.0, latest.autonomic_stress * 22.0), 1)},
            {"name": "Low HRV / Fatigue Penalty", "impact": round(max(0.0, (48.0 - float(row["hrv_rmssd_ms"])) * 0.8), 1)},
            {"name": "Rest / Movement Relief", "impact": round(-min(25.0, max(0.0, float(row["hrv_rmssd_ms"]) - 40.0) * 0.7), 1)},
        ],
    }

    early_warning = twin.get_early_warning(clamped_end)
    daily_schedule = _build_daily_schedule(win_24h, twin.logged_events)

    # Human-readable simulated day & time
    start_ts = pd.to_datetime(twin.df["timestamp"].iloc[0])
    cur_ts = pd.to_datetime(latest.timestamp)
    day_num = (cur_ts.normalize() - start_ts.normalize()).days + 1
    clock_label = f"Day {day_num} · {cur_ts.strftime('%I:%M %p').lstrip('0')}"

    return {
        "patient_id": patient_id,
        "end_idx": clamped_end,
        "max_idx": len(twin.df) - 1,
        "clock_label": clock_label,
        "next_meal_idx": next_meal_idx,
        "tir_24h_pct": tir_24h,
        "hours_in_range_24h": hours_in_range,
        "window_hours_24h": window_hours,
        "mage_24h_mg_dl": mage_24h,
        "personal_baseline": personal_baseline,
        "early_warning": early_warning,
        "daily_schedule": daily_schedule,
        "logged_events": twin.logged_events,
        "latest_state": {
            "timestamp": latest.timestamp,
            "glucose_est": latest.glucose_est,
            "glucose_std": latest.glucose_std,
            "ci_lower_90": latest.ci_lower_90,
            "ci_upper_90": latest.ci_upper_90,
            "insulin_sensitivity": latest.insulin_sensitivity,
            "autonomic_stress": latest.autonomic_stress,
            "activity_load": latest.activity_load,
            "cardiometabolic_risk_score": latest.cardiometabolic_risk_score,
            "risk_band": latest.risk_band,
            "hr_bpm": float(row["hr_bpm"]),
            "hrv_rmssd_ms": float(row["hrv_rmssd_ms"]),
        },
        "drivers": drivers_by_metric["glucose"],
        "drivers_by_metric": drivers_by_metric,
        "series": {
            "timestamps": [s.timestamp for s in states],
            "glucose_obs": [round(float(v), 2) for v in sub_df["glucose_mg_dl"].tolist()],
            "glucose_est": [s.glucose_est for s in states],
            "ci_lower_90": [s.ci_lower_90 for s in states],
            "ci_upper_90": [s.ci_upper_90 for s in states],
            "risk_score": [s.cardiometabolic_risk_score for s in states],
            "insulin_sensitivity": [s.insulin_sensitivity for s in states],
            "hr_bpm": [round(float(v), 1) for v in sub_df["hr_bpm"].tolist()],
            "hrv_rmssd_ms": [round(float(v), 1) for v in sub_df["hrv_rmssd_ms"].tolist()],
            "eda_us": [round(float(v), 3) for v in sub_df["eda_us"].tolist()],
            "acc_mag": [round(float(v), 3) for v in sub_df["acc_mag"].tolist()],
            "carbs_g": [round(float(v), 1) for v in sub_df["carbs_g"].tolist()],
        },
    }


@app.post("/api/patients/{patient_id}/log_event")
def log_event(patient_id: str, req: LogEventRequest):
    if patient_id not in TWINS:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    evt = TWINS[patient_id].inject_event(
        idx=req.idx,
        event_type=req.event_type,
        carbs_g=req.carbs_g,
        duration_min=req.duration_min,
        label=req.label,
    )
    return {"status": "logged", "event": evt}


@app.post("/api/patients/{patient_id}/reset_events")
def reset_events(patient_id: str):
    if patient_id not in TWINS:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    TWINS[patient_id].reset_events()
    return {"status": "reset"}


@app.post("/api/patients/{patient_id}/simulate")
def simulate_what_if(patient_id: str, req: SimulateRequest):
    if patient_id not in TWINS:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    twin = TWINS[patient_id]
    scenario = WhatIfScenario(
        post_meal_walk_min=req.post_meal_walk_min,
        carb_reduction_pct=req.carb_reduction_pct,
        stress_reduction_pct=req.stress_reduction_pct,
        hrv_recovery_boost=req.hrv_recovery_boost,
        horizon_hours=req.horizon_hours,
    )
    return twin.simulate_what_if(req.start_idx, scenario)


@app.get("/api/patients/{patient_id}/ablation")
def get_ablation(patient_id: str):
    if patient_id not in TWINS:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return TWINS[patient_id].evaluate_ablation()
