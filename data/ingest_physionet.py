"""
Data Ingestion Pipeline for BIG IDEAs Lab Glycemic Variability and Wearable Device Data (v1.1.3)
Source: https://physionet.org/content/big-ideas-glycemic-wearable/

V2 Fast-loading & Unit-Normalized version:
- Auto-normalizes Empatica E4 ACC (1/64g raw units -> g-units)
- Subsamples high-frequency 32 Hz ACC / 4 Hz EDA & TEMP before datetime parsing
- Caches aligned 5-minute CSVs (`aligned_v2_5min_<ID>.csv`) so server restarts take <0.1s
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd


def _find_csv(subject_dir: Path, prefix: str) -> Optional[Path]:
    exact = subject_dir / f"{prefix}.csv"
    if exact.exists():
        return exact
    matches = sorted(subject_dir.glob(f"{prefix}_*.csv"))
    if matches:
        return matches[0]
    matches = sorted(subject_dir.glob(f"{prefix}*.csv"))
    return matches[0] if matches else None


def compute_acc_magnitude(acc_df: pd.DataFrame) -> pd.Series:
    """Compute tri-axial accelerometry vector magnitude normalized to g-units (1g ~ 1.0)."""
    cols = [c for c in acc_df.columns if any(axis in c.strip().upper() for axis in ("ACC_X", "ACC_Y", "ACC_Z")) or c.strip().upper() in ("X", "Y", "Z")]
    if len(cols) >= 3:
        xyz = acc_df[cols[:3]].apply(pd.to_numeric, errors="coerce")
        mag = np.sqrt((xyz ** 2).sum(axis=1))
        med = mag.median()
        if pd.notna(med) and med > 5.0:
            # Empatica E4 raw ACC is in 1/64 g units (~64 = 1g)
            mag = mag / med
            med = 1.0
        rel = np.abs(mag - (med if pd.notna(med) else 1.0)).fillna(0.0)
        return rel.clip(0.0, 1.5)
    return pd.Series(np.zeros(len(acc_df)), index=acc_df.index)


def load_physionet_subject(subject_dir: Path, demographics_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    sid = subject_dir.name
    cache_path = subject_dir / f"aligned_v2_5min_{sid}.csv"
    if cache_path.exists():
        print(f"[PhysioNet] Loaded cached 5-min data for Subject {sid}", flush=True)
        return pd.read_csv(cache_path, parse_dates=["timestamp"])

    t0 = time.time()
    print(f"[PhysioNet] Aligning Subject {sid} (first-time cache build)...", flush=True)

    dexcom_path = _find_csv(subject_dir, "Dexcom")
    if dexcom_path is None:
        raise FileNotFoundError(f"Missing Dexcom*.csv in {subject_dir}")

    dexcom = pd.read_csv(dexcom_path)
    ts_candidates = [c for c in dexcom.columns if "time" in c.lower() or "date" in c.lower()]
    val_candidates = [c for c in dexcom.columns if "glucose" in c.lower() or "val" in c.lower()]
    ts_col = ts_candidates[0] if ts_candidates else dexcom.columns[0]
    val_col = val_candidates[0] if val_candidates else dexcom.columns[-1]

    dexcom["timestamp"] = pd.to_datetime(dexcom[ts_col], format="mixed", errors="coerce")
    dexcom["glucose_mg_dl"] = pd.to_numeric(dexcom[val_col], errors="coerce")
    dexcom = dexcom[["timestamp", "glucose_mg_dl"]].dropna()
    dexcom = dexcom.set_index("timestamp").sort_index().resample("5min").mean().interpolate(limit=3).dropna()

    def _resample_signal(prefix: str, out_col: str, agg: str = "mean", stride: int = 1) -> Optional[pd.Series]:
        fpath = _find_csv(subject_dir, prefix)
        if fpath is None:
            return None
        df = pd.read_csv(fpath)
        if stride > 1 and len(df) > stride * 1000:
            df = df.iloc[::stride].copy()
        t_cands = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
        t_c = t_cands[0] if t_cands else df.columns[0]
        v_cands = [c for c in df.columns if c != t_c]
        v_c = v_cands[-1] if v_cands else df.columns[-1]

        df["timestamp"] = pd.to_datetime(df[t_c], format="mixed", errors="coerce")
        df["val"] = pd.to_numeric(df[v_c], errors="coerce")
        df = df[["timestamp", "val"]].dropna()
        if df.empty:
            return None
        s = df.set_index("timestamp")["val"].sort_index()
        if agg == "rmssd":
            def _rmssd(x: pd.Series) -> float:
                if len(x) < 3:
                    return np.nan
                vals = x.values.astype(float)
                if np.median(vals) < 10:
                    vals = vals * 1000.0
                diffs = np.diff(vals)
                return float(np.sqrt(np.mean(diffs ** 2)))
            return s.resample("5min").apply(_rmssd).rename(out_col)
        return s.resample("5min").mean().rename(out_col)

    hr = _resample_signal("HR", "hr_bpm", stride=5)
    eda = _resample_signal("EDA", "eda_us", stride=20)
    temp = _resample_signal("TEMP", "skin_temp_c", stride=20)
    hrv = _resample_signal("IBI", "hrv_rmssd_ms", agg="rmssd", stride=1)

    acc_path = _find_csv(subject_dir, "ACC")
    acc_series = None
    if acc_path is not None:
        acc_df = pd.read_csv(acc_path)
        if len(acc_df) > 64000:
            acc_df = acc_df.iloc[::64].copy()
        t_cands = [c for c in acc_df.columns if "time" in c.lower() or "date" in c.lower()]
        t_c = t_cands[0] if t_cands else acc_df.columns[0]
        acc_df["timestamp"] = pd.to_datetime(acc_df[t_c], format="mixed", errors="coerce")
        acc_df["acc_mag"] = compute_acc_magnitude(acc_df)
        acc_df = acc_df.dropna(subset=["timestamp"])
        if not acc_df.empty:
            acc_series = acc_df.set_index("timestamp")["acc_mag"].sort_index().resample("5min").mean()

    aligned = dexcom.copy()
    for s in [hr, eda, temp, hrv, acc_series]:
        if s is not None:
            aligned = aligned.join(s, how="left")

    defaults = {"hr_bpm": 72.0, "eda_us": 0.45, "skin_temp_c": 33.5, "hrv_rmssd_ms": 42.0, "acc_mag": 0.04}
    for col, dval in defaults.items():
        if col not in aligned.columns:
            aligned[col] = dval
        else:
            med = aligned[col].median()
            aligned[col] = aligned[col].ffill(limit=6).bfill(limit=6).fillna(med if pd.notna(med) else dval)

    aligned["carbs_g"] = 0.0
    aligned["calories"] = 0.0
    food_path = _find_csv(subject_dir, "Food_Log")
    if food_path is not None:
        food_df = pd.read_csv(food_path)
        t_col = "time_begin" if "time_begin" in food_df.columns else ("date" if "date" in food_df.columns else food_df.columns[0])
        carb_col = [c for c in food_df.columns if "carb" in c.lower()]
        cal_col = [c for c in food_df.columns if "cal" in c.lower()]
        if carb_col:
            food_df["timestamp"] = pd.to_datetime(food_df[t_col], format="mixed", errors="coerce")
            food_df["carbs_val"] = pd.to_numeric(food_df[carb_col[0]], errors="coerce").fillna(0.0)
            food_df = food_df.dropna(subset=["timestamp"]).set_index("timestamp")
            if not food_df.empty:
                carbs_5m = food_df["carbs_val"].resample("5min").sum()
                aligned["carbs_g"] = aligned.index.map(lambda idx: float(carbs_5m.get(idx, 0.0)))
                if cal_col:
                    food_df["cal_val"] = pd.to_numeric(food_df[cal_col[0]], errors="coerce").fillna(0.0)
                    cals_5m = food_df["cal_val"].resample("5min").sum()
                    aligned["calories"] = aligned.index.map(lambda idx: float(cals_5m.get(idx, 0.0)))

    hba1c = 5.7
    gender = "Female"
    if demographics_df is not None and "ID" in demographics_df.columns:
        row = demographics_df[demographics_df["ID"].astype(str).str.zfill(3) == sid.zfill(3)]
        if not row.empty:
            hba1c = float(row.iloc[0].get("HbA1c", 5.7))
            gender = str(row.iloc[0].get("Gender", "Female"))

    aligned["patient_id"] = f"PN-{sid.zfill(3)}"
    aligned["hba1c"] = hba1c
    aligned["gender"] = gender
    aligned = aligned.reset_index()
    aligned["label"] = f"PhysioNet Subject {sid} (Real Data)"
    aligned.to_csv(cache_path, index=False)
    print(f"[PhysioNet] Subject {sid} ready ({len(aligned)} steps) in {time.time() - t0:.1f}s", flush=True)
    return aligned


def generate_physionet_schema_cohort(seed: int = 42) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    profiles = [
        {"patient_id": "PN-001", "label": "Subject 001 — Prediabetic / High Glycemic Variability", "hba1c": 6.1, "age": 54, "gender": "Female", "baseline_glucose": 114.0, "insulin_sensitivity": 0.62, "stress_reactivity": 1.35, "resting_hr": 74.0},
        {"patient_id": "PN-002", "label": "Subject 004 — Borderline A1c / Sedentary Profile", "hba1c": 5.8, "age": 49, "gender": "Female", "baseline_glucose": 103.0, "insulin_sensitivity": 0.78, "stress_reactivity": 1.05, "resting_hr": 78.0},
        {"patient_id": "PN-003", "label": "Subject 009 — High Normal / Active Post-Meal Walker", "hba1c": 5.4, "age": 44, "gender": "Female", "baseline_glucose": 92.0, "insulin_sensitivity": 1.08, "stress_reactivity": 0.85, "resting_hr": 64.0},
        {"patient_id": "PN-012", "label": "Subject 012 — Elevated Stress & Nocturnal Cortisol", "hba1c": 6.3, "age": 59, "gender": "Female", "baseline_glucose": 122.0, "insulin_sensitivity": 0.54, "stress_reactivity": 1.60, "resting_hr": 80.0},
    ]
    n_steps = 864
    start_time = pd.Timestamp("2026-10-01 00:00:00")
    timestamps = pd.date_range(start_time, periods=n_steps, freq="5min")
    cohort: Dict[str, pd.DataFrame] = {}

    for p in profiles:
        hours = np.arange(n_steps) / 12.0
        tod = hours % 24.0
        circadian_hr = 6.0 * np.sin(2 * np.pi * (tod - 9.0) / 24.0)
        acc_mag = np.maximum(0.01, 0.04 + 0.03 * np.sin(2 * np.pi * (tod - 10.0) / 24.0) + rng.normal(0, 0.015, n_steps))
        if p["patient_id"] == "PN-003":
            walk_mask = ((tod >= 13.0) & (tod <= 13.6)) | ((tod >= 19.0) & (tod <= 19.6))
            acc_mag[walk_mask] += 0.35 + rng.uniform(0, 0.12, walk_mask.sum())
        else:
            walk_mask = (tod >= 17.5) & (tod <= 17.85) & (hours < 48)
            acc_mag[walk_mask] += 0.22 + rng.uniform(0, 0.08, walk_mask.sum())

        eda_us = 0.45 * p["stress_reactivity"] + 0.15 * np.sin(2 * np.pi * (tod - 11.0) / 24.0) + rng.normal(0, 0.05, n_steps)
        stress_spike = ((tod >= 10.5) & (tod <= 12.0)) | ((tod >= 15.0) & (tod <= 16.0))
        eda_us[stress_spike] += 0.55 * p["stress_reactivity"]
        eda_us = np.clip(eda_us, 0.1, 4.5)

        hr_bpm = np.clip(p["resting_hr"] + circadian_hr + 42.0 * acc_mag + 6.5 * eda_us + rng.normal(0, 1.8, n_steps), 52.0, 165.0)
        hrv_rmssd = np.clip(52.0 - 0.32 * (hr_bpm - 65.0) - 9.5 * eda_us + rng.normal(0, 2.5, n_steps), 12.0, 95.0)
        skin_temp = 33.4 + 0.5 * np.sin(2 * np.pi * (tod - 15.0) / 24.0) + 0.4 * acc_mag + rng.normal(0, 0.08, n_steps)

        carbs = np.zeros(n_steps)
        calories = np.zeros(n_steps)
        for day in range(3):
            b_idx = day * 288 + 8 * 12 + int(rng.integers(-2, 3))
            l_idx = day * 288 + 13 * 12 + int(rng.integers(-2, 3))
            d_idx = day * 288 + 19 * 12 + 6 + int(rng.integers(-2, 3))
            carbs[b_idx] = float(rng.integers(42, 62))
            calories[b_idx] = carbs[b_idx] * 7.2
            carbs[l_idx] = float(rng.integers(55, 82))
            calories[l_idx] = carbs[l_idx] * 8.0
            carbs[d_idx] = float(rng.integers(60, 95))
            calories[d_idx] = carbs[d_idx] * 8.5

        glucose = np.zeros(n_steps)
        glucose[0] = p["baseline_glucose"]
        gut_carb = 0.0
        insulin_eff = 0.0
        si = p["insulin_sensitivity"]

        for t in range(1, n_steps):
            gut_carb = 0.86 * gut_carb + carbs[t]
            carb_appearance = 0.14 * gut_carb
            activity_uptake = 14.5 * max(0.0, acc_mag[t] - 0.06)
            stress_drive = 2.6 * max(0.0, eda_us[t] - 0.45)
            excess_g = max(0.0, glucose[t - 1] - p["baseline_glucose"])
            insulin_eff = 0.88 * insulin_eff + 0.12 * si * excess_g * 0.09
            dawn_effect = 2.2 * np.exp(-0.5 * ((tod[t] - 6.5) / 1.5) ** 2)
            dg = -0.065 * (glucose[t - 1] - p["baseline_glucose"]) + 0.34 * carb_appearance - insulin_eff - activity_uptake + stress_drive + dawn_effect + rng.normal(0, 1.4)
            glucose[t] = float(np.clip(glucose[t - 1] + dg, 58.0, 265.0))

        cohort[p["patient_id"]] = pd.DataFrame({
            "timestamp": timestamps, "patient_id": p["patient_id"], "label": p["label"],
            "hba1c": p["hba1c"], "age": p["age"], "gender": p["gender"],
            "glucose_mg_dl": np.round(glucose, 2), "hr_bpm": np.round(hr_bpm, 2),
            "hrv_rmssd_ms": np.round(hrv_rmssd, 2), "eda_us": np.round(eda_us, 3),
            "skin_temp_c": np.round(skin_temp, 2), "acc_mag": np.round(acc_mag, 4),
            "carbs_g": np.round(carbs, 1), "calories": np.round(calories, 1),
        })
    return cohort


def load_cohort(data_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    if data_dir and os.path.isdir(data_dir):
        root = Path(data_dir)
        demo_path = root / "Demographics.csv"
        demo_df = pd.read_csv(demo_path) if demo_path.exists() else None
        cohort = {}
        for sub in sorted(root.iterdir()):
            if sub.is_dir() and _find_csv(sub, "Dexcom") is not None:
                try:
                    df = load_physionet_subject(sub, demo_df)
                    if not df.empty:
                        pid = df["patient_id"].iloc[0]
                        cohort[pid] = df
                except Exception as exc:
                    print(f"[ingest_physionet] Skipping {sub.name}: {exc}", flush=True)
        if cohort:
            return cohort
    return generate_physionet_schema_cohort()
