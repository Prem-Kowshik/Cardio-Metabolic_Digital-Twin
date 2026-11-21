"""
Automated pytest suite for the Cardiometabolic Digital Twin (V5).
Run: pytest -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app, TWINS
from engine.digital_twin import WhatIfScenario

client = TestClient(app)


def test_health_and_patients():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["patients_loaded"] >= 4

    pts = client.get("/api/patients").json()["patients"]
    assert len(pts) >= 4
    assert all("patient_id" in p and "hba1c" in p for p in pts)


def test_timeline_structure_and_max_idx():
    pid = "PN-001"
    r = client.get(f"/api/patients/{pid}/timeline?end_idx=432")
    assert r.status_code == 200
    data = r.json()
    assert data["max_idx"] == 863
    assert "clock_label" in data
    assert "daily_schedule" in data
    assert "alert_items" in data["early_warning"]
    assert set(data["drivers_by_metric"].keys()) == {"glucose", "heart", "stress"}


def test_high_heart_rate_triggers_alert_even_during_workout():
    twin = TWINS["PN-001"]
    twin.reset_events()
    # Force HR to 154 bpm during an active workout (acc_mag = 0.55)
    twin.df.loc[432, "hr_bpm"] = 154.0
    twin.df.loc[432, "acc_mag"] = 0.55
    twin.run_until(432)
    ew = twin.get_early_warning(432)
    assert ew["critical_popup"] is True
    assert any(item["vital"] == "Heart Rate" for item in ew["alert_items"])
    twin.reset_events()


def test_all_six_activity_types_and_reset():
    pid = "PN-001"
    activities = [
        {"event_type": "meal", "carbs_g": 80, "duration_min": 20},
        {"event_type": "walk", "carbs_g": 0, "duration_min": 20},
        {"event_type": "workout", "carbs_g": 0, "duration_min": 30},
        {"event_type": "break", "carbs_g": 0, "duration_min": 10},
        {"event_type": "sleep", "carbs_g": 0, "duration_min": 480},
        {"event_type": "work", "carbs_g": 0, "duration_min": 60},
    ]
    for act in activities:
        res = client.post(f"/api/patients/{pid}/log_event", json={"idx": 432, **act})
        assert res.status_code == 200
        assert res.json()["status"] == "logged"

    rs = client.post(f"/api/patients/{pid}/reset_events")
    assert rs.status_code == 200
    assert len(TWINS[pid].logged_events) == 0


def test_what_if_bidirectional_monotonicity():
    twin = TWINS["PN-001"]
    twin.reset_events()
    healthy = twin.simulate_what_if(
        432, WhatIfScenario(post_meal_walk_min=30, carb_reduction_pct=30, stress_reduction_pct=25, hrv_recovery_boost=10)
    )
    cheat = twin.simulate_what_if(
        432, WhatIfScenario(post_meal_walk_min=0, carb_reduction_pct=-45, stress_reduction_pct=-40, hrv_recovery_boost=-15)
    )
    assert healthy["verdict"]["status"] == "positive"
    assert healthy["summary"]["glucose_delta"] < 0
    assert cheat["verdict"]["status"] == "negative"
    assert cheat["summary"]["glucose_delta"] > 0


def test_ablation_beats_static_baseline():
    r = client.get("/api/patients/PN-001/ablation")
    assert r.status_code == 200
    ab = r.json()
    assert ab["dynamic_digital_twin"]["rmse_mg_dl"] < ab["static_baseline"]["rmse_mg_dl"]
    assert ab["dynamic_digital_twin"]["uncertainty_calibration_90"] >= 80.0
    assert ab["clarke_summary"]["A"] + ab["clarke_summary"]["B"] >= 95.0
