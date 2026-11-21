"""
Physiology-Informed Cardiometabolic Digital Twin Engine (V5)

Fixes & Features:
1. Coupled Glucose-Gut-Insulin-Wearable State-Space Model (Bergman Minimal Model inspired)
2. Extended Kalman Filter (EKF) with tuned S_I adaptability (Q[2,2]=0.006) & 90% CI
3. Structured Multi-Vital Alerts (`alert_items`) — HR >= 145 triggers regardless of activity,
   resting HR >= 115 triggers when still, plus Glucose, Stress, S_I, and Past-Bedtime checks
4. Interactive Multi-Activity Logger (Meal, Walk, Workout, Break, Sleep, Desk Work)
5. Bidirectional Counterfactual "What-If" ODE Simulator (supports healthy & "Cheat Day" scenarios)
6. Static Baseline vs. Dynamic Digital Twin Ablation + MAGE + Clarke Error Grid Evaluation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


@dataclass
class TwinState:
    step_idx: int
    timestamp: str
    glucose_est: float
    glucose_std: float
    gut_carb_load: float
    insulin_sensitivity: float
    autonomic_stress: float
    activity_load: float
    cardiometabolic_risk_score: float
    risk_band: str
    ci_lower_90: float
    ci_upper_90: float


@dataclass
class WhatIfScenario:
    post_meal_walk_min: float = 0.0
    carb_reduction_pct: float = 0.0
    stress_reduction_pct: float = 0.0
    hrv_recovery_boost: float = 0.0
    horizon_hours: int = 12


def compute_mage(glucose: np.ndarray) -> float:
    if len(glucose) < 6:
        return 0.0
    sd = float(np.std(glucose))
    if sd < 1e-3:
        return 0.0
    diffs = np.diff(glucose)
    extrema_idx = [0]
    for i in range(1, len(diffs)):
        if (diffs[i - 1] > 0 and diffs[i] <= 0) or (diffs[i - 1] < 0 and diffs[i] >= 0):
            extrema_idx.append(i)
    extrema_idx.append(len(glucose) - 1)
    extrema_vals = glucose[extrema_idx]
    excursions = np.abs(np.diff(extrema_vals))
    sig_excursions = excursions[excursions > sd]
    if len(sig_excursions) == 0:
        return round(float(np.max(glucose) - np.min(glucose)), 1)
    return round(float(np.mean(sig_excursions)), 1)


def clarke_zone(ref: float, pred: float) -> str:
    if (ref <= 70 and pred <= 70) or abs(pred - ref) <= 0.20 * ref:
        return "A"
    if (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
        return "E"
    if ((ref >= 70 and ref <= 290) and pred >= ref + 110) or ((ref >= 130 and ref <= 180) and pred <= (7.0 / 5.0) * ref - 182):
        return "C"
    if (ref >= 240 and (pred >= 70 and pred <= 180)) or (ref <= 175 / 3 and (pred >= 70 and pred <= 180)) or ((ref >= 175 / 3 and ref <= 70) and pred >= (6.0 / 5.0) * ref):
        return "D"
    return "B"


class CardiometabolicDigitalTwin:
    def __init__(self, patient_df: pd.DataFrame):
        self.raw_df = patient_df.reset_index(drop=True).copy()
        self.df = self.raw_df.copy()
        self.patient_id = str(self.df["patient_id"].iloc[0])
        self.hba1c = float(self.df["hba1c"].iloc[0])
        self.baseline_g = float(np.median(self.df["glucose_mg_dl"].iloc[:min(144, len(self.df))]))
        self.x = np.array([float(self.df["glucose_mg_dl"].iloc[0]), 0.0, 0.75, float(self.df["eda_us"].iloc[0])], dtype=float)
        self.P = np.diag([25.0, 4.0, 0.08, 0.05])
        # Tuned Q[2,2]=0.006 so S_I adapts visibly to sustained walks, sleep, or heavy meals
        self.Q = np.diag([4.5, 1.2, 0.006, 0.02])
        self.R = np.array([[9.0]])
        self.history: List[TwinState] = []
        self.logged_events: List[Dict] = []

    def reset_events(self) -> None:
        self.df = self.raw_df.copy()
        self.logged_events = []

    def inject_event(
        self,
        idx: int,
        event_type: str,
        carbs_g: float = 0.0,
        duration_min: float = 20.0,
        label: str = "",
    ) -> Dict:
        idx = int(np.clip(idx, 12, len(self.df) - 2))
        steps = max(1, min(len(self.df) - idx - 1, int(round(duration_min / 5.0))))
        advance_steps = 0

        if event_type == "meal" and carbs_g > 0:
            self.df.loc[idx, "carbs_g"] = float(self.df.loc[idx, "carbs_g"]) + carbs_g
            gut = carbs_g
            for k in range(1, min(25, len(self.df) - idx)):
                gut = 0.86 * gut
                self.df.loc[idx + k, "glucose_mg_dl"] = float(
                    np.clip(self.df.loc[idx + k, "glucose_mg_dl"] + 0.30 * gut * 0.14 * 4.5, 55.0, 285.0)
                )
            detail = label or f"Meal (+{carbs_g:.0f}g carbs)"

        elif event_type == "walk":
            for k in range(steps):
                self.df.loc[idx + k, "acc_mag"] = max(float(self.df.loc[idx + k, "acc_mag"]), 0.35)
                self.df.loc[idx + k, "hr_bpm"] = min(155.0, float(self.df.loc[idx + k, "hr_bpm"]) + 18.0)
                self.df.loc[idx + k, "glucose_mg_dl"] = float(
                    np.clip(self.df.loc[idx + k, "glucose_mg_dl"] - 3.2 * (k + 1) * 0.35, 62.0, 285.0)
                )
            detail = label or f"Brisk Walk ({duration_min:.0f} min)"

        elif event_type == "workout":
            for k in range(steps):
                self.df.loc[idx + k, "acc_mag"] = max(float(self.df.loc[idx + k, "acc_mag"]), 0.55)
                self.df.loc[idx + k, "hr_bpm"] = min(176.0, float(self.df.loc[idx + k, "hr_bpm"]) + 48.0)
                self.df.loc[idx + k, "glucose_mg_dl"] = float(
                    np.clip(self.df.loc[idx + k, "glucose_mg_dl"] - 4.8 * (k + 1) * 0.35, 60.0, 285.0)
                )
            detail = label or f"Workout / Jog ({duration_min:.0f} min)"

        elif event_type == "break":
            for k in range(steps):
                self.df.loc[idx + k, "acc_mag"] = 0.02
                self.df.loc[idx + k, "eda_us"] = max(0.15, float(self.df.loc[idx + k, "eda_us"]) * 0.65)
                self.df.loc[idx + k, "hr_bpm"] = max(55.0, float(self.df.loc[idx + k, "hr_bpm"]) - 6.0)
                self.df.loc[idx + k, "hrv_rmssd_ms"] = min(95.0, float(self.df.loc[idx + k, "hrv_rmssd_ms"]) + 8.0)
            detail = label or f"Rest Break ({duration_min:.0f} min)"

        elif event_type == "sleep":
            for k in range(steps):
                self.df.loc[idx + k, "acc_mag"] = 0.015
                self.df.loc[idx + k, "eda_us"] = max(0.15, float(self.df.loc[idx + k, "eda_us"]) * 0.60)
                self.df.loc[idx + k, "hr_bpm"] = max(52.0, float(self.df.loc[idx + k, "hr_bpm"]) - 10.0)
                self.df.loc[idx + k, "hrv_rmssd_ms"] = min(95.0, float(self.df.loc[idx + k, "hrv_rmssd_ms"]) + 12.0)
                self.df.loc[idx + k, "glucose_mg_dl"] = float(
                    np.clip(self.df.loc[idx + k, "glucose_mg_dl"] - 0.25 * min(k, 16), 65.0, 285.0)
                )
            advance_steps = min(steps, 24)
            hrs = round(duration_min / 60.0, 1)
            detail = label or f"Sleep / Nap ({hrs} hr)"

        elif event_type == "work":
            for k in range(steps):
                self.df.loc[idx + k, "acc_mag"] = 0.02
                self.df.loc[idx + k, "eda_us"] = min(4.2, float(self.df.loc[idx + k, "eda_us"]) + 0.35)
                self.df.loc[idx + k, "hr_bpm"] = min(145.0, float(self.df.loc[idx + k, "hr_bpm"]) + 5.0)
                self.df.loc[idx + k, "hrv_rmssd_ms"] = max(14.0, float(self.df.loc[idx + k, "hrv_rmssd_ms"]) - 4.0)
            advance_steps = steps
            hrs = round(duration_min / 60.0, 1)
            detail = label or f"Desk Work ({hrs} hr)"

        else:
            return {"idx": idx, "type": "noop", "detail": "No changes applied", "advance_steps": 0}

        evt = {
            "idx": idx,
            "timestamp": str(self.df.loc[idx, "timestamp"]),
            "type": event_type,
            "detail": detail,
            "advance_steps": advance_steps,
        }
        self.logged_events.append(evt)
        return evt

    def _compute_risk_score(self, glucose: float, si: float, stress: float, hrv: float) -> Tuple[float, str]:
        glycemic_component = np.clip((glucose - 85.0) / 115.0, 0.0, 1.0) * 45.0
        si_component = np.clip((1.0 - si) / 0.6, 0.0, 1.0) * 25.0
        stress_component = np.clip((stress - 0.35) / 1.4, 0.0, 1.0) * 15.0
        hrv_component = np.clip((45.0 - hrv) / 30.0, 0.0, 1.0) * 15.0
        score = float(np.clip(glycemic_component + si_component + stress_component + hrv_component, 2.0, 99.0))
        if score >= 65.0:
            band = "High"
        elif score >= 38.0:
            band = "Moderate"
        else:
            band = "Low"
        return round(score, 1), band

    def step_assimilate(self, idx: int) -> TwinState:
        row = self.df.iloc[idx]
        g_prev, gut_prev, si_prev, stress_prev = self.x

        carbs_in = float(row["carbs_g"])
        acc = float(row["acc_mag"])
        eda = float(row["eda_us"])
        hrv = float(self.df["hrv_rmssd_ms"].iloc[idx])

        gut_next = 0.86 * gut_prev + carbs_in
        carb_appearance = 0.14 * gut_next
        activity_uptake = min(6.0, 14.5 * max(0.0, acc - 0.06))
        stress_next = 0.70 * stress_prev + 0.30 * eda
        stress_drive = 2.6 * max(0.0, stress_next - 0.45)

        excess_g = max(0.0, g_prev - self.baseline_g)
        insulin_action = si_prev * excess_g * 0.09

        g_pred = (
            g_prev
            - 0.065 * (g_prev - self.baseline_g)
            + 0.34 * carb_appearance
            - insulin_action
            - activity_uptake
            + stress_drive
        )
        g_pred = float(np.clip(g_pred, 50.0, 300.0))
        si_pred = float(np.clip(si_prev, 0.25, 1.45))
        x_pred = np.array([g_pred, gut_next, si_pred, stress_next])

        F = np.eye(4)
        F[0, 0] = 1.0 - 0.065 - (0.09 * si_prev if g_prev > self.baseline_g else 0.0)
        F[0, 1] = 0.34 * 0.14 * 0.86
        F[0, 2] = -0.09 * excess_g
        F[0, 3] = 2.6 * 0.70 if stress_next > 0.45 else 0.0
        F[1, 1] = 0.86
        F[3, 3] = 0.70

        P_pred = F @ self.P @ F.T + self.Q

        H = np.array([[1.0, 0.0, 0.0, 0.0]])
        y_obs = float(row["glucose_mg_dl"])
        innov = y_obs - x_pred[0]
        S = H @ P_pred @ H.T + self.R
        K = (P_pred @ H.T) / S[0, 0]

        self.x = x_pred + K.flatten() * innov
        self.x[0] = float(np.clip(self.x[0], 50.0, 300.0))
        self.x[1] = float(max(0.0, self.x[1]))
        self.x[2] = float(np.clip(self.x[2], 0.25, 1.45))
        self.x[3] = float(np.clip(self.x[3], 0.1, 4.5))
        self.P = (np.eye(4) - K @ H) @ P_pred

        g_std = float(np.sqrt(max(1.0, self.P[0, 0])))
        risk_score, risk_band = self._compute_risk_score(self.x[0], self.x[2], self.x[3], hrv)

        state = TwinState(
            step_idx=idx,
            timestamp=str(row["timestamp"]),
            glucose_est=round(float(self.x[0]), 2),
            glucose_std=round(g_std, 2),
            gut_carb_load=round(float(self.x[1]), 2),
            insulin_sensitivity=round(float(self.x[2]), 3),
            autonomic_stress=round(float(self.x[3]), 3),
            activity_load=round(acc, 4),
            cardiometabolic_risk_score=risk_score,
            risk_band=risk_band,
            ci_lower_90=round(float(self.x[0] - 1.645 * g_std), 1),
            ci_upper_90=round(float(self.x[0] + 1.645 * g_std), 1),
        )
        self.history.append(state)
        return state

    def run_until(self, end_idx: int) -> List[TwinState]:
        self.history = []
        self.x = np.array([float(self.df["glucose_mg_dl"].iloc[0]), 0.0, 0.75, float(self.df["eda_us"].iloc[0])], dtype=float)
        self.P = np.diag([25.0, 4.0, 0.08, 0.05])
        limit = min(end_idx, len(self.df) - 1)
        for i in range(limit + 1):
            self.step_assimilate(i)
        return self.history

    def get_early_warning(self, start_idx: int) -> Dict:
        """
        Multi-parameter 30-minute forecast + independent vital checks.
        Every vital produces its own structured item in `alert_items` so no vital suppresses another.
        """
        g, gut, si, stress = self.x.copy()
        n_steps = min(6, len(self.df) - 1 - start_idx)
        for k in range(1, n_steps + 1):
            row_f = self.df.iloc[start_idx + k]
            gut = 0.86 * gut + float(row_f["carbs_g"])
            act_up = min(6.0, 14.5 * max(0.0, float(row_f["acc_mag"]) - 0.06))
            stress = 0.70 * stress + 0.30 * float(row_f["eda_us"])
            st_drv = 2.6 * max(0.0, stress - 0.45)
            ins_act = si * max(0.0, g - self.baseline_g) * 0.09
            g = float(np.clip(g - 0.065 * (g - self.baseline_g) + 0.34 * 0.14 * gut - ins_act - act_up + st_drv, 50.0, 290.0))

        row = self.df.iloc[start_idx]
        hr_now = float(row["hr_bpm"])
        acc_now = float(row["acc_mag"])
        eda_now = float(self.x[3])
        si_now = float(self.x[2])
        latest_state = self.history[-1] if self.history else None
        risk_now = latest_state.cardiometabolic_risk_score if latest_state else 35.0
        ts = pd.to_datetime(row["timestamp"])

        alert_items: List[Dict] = []
        critical_popup = False
        proj = round(g, 1)
        delta = round(g - float(self.x[0]), 1)

        # 1. Heart Rate check (independent of glucose; >=145 always triggers!)
        if hr_now >= 145.0:
            critical_popup = True
            alert_items.append({
                "vital": "Heart Rate",
                "severity": "critical",
                "value": f"{hr_now:.0f} bpm",
                "text": f"Heart rate is very high ({hr_now:.0f} bpm)—slow down, hydrate, and catch your breath.",
            })
        elif hr_now >= 115.0 and acc_now < 0.15:
            critical_popup = True
            alert_items.append({
                "vital": "Resting Heart Rate",
                "severity": "critical",
                "value": f"{hr_now:.0f} bpm",
                "text": f"Resting heart rate is elevated ({hr_now:.0f} bpm without heavy movement). Pause for slow breathing.",
            })
        elif hr_now >= 102.0 and acc_now < 0.15:
            alert_items.append({
                "vital": "Heart Rate",
                "severity": "warning",
                "value": f"{hr_now:.0f} bpm",
                "text": f"Heart rate is slightly elevated at rest ({hr_now:.0f} bpm).",
            })

        # 2. Blood Sugar 30-min forecast check
        if proj >= 170.0:
            critical_popup = True
            alert_items.append({
                "vital": "Blood Sugar (30m)",
                "severity": "critical",
                "value": f"{proj} mg/dL",
                "text": f"Blood sugar is projected to spike to ~{proj} mg/dL in 30 min. A 15–20 min walk can blunt this peak.",
            })
        elif proj >= 145.0:
            alert_items.append({
                "vital": "Blood Sugar (30m)",
                "severity": "warning",
                "value": f"{proj} mg/dL",
                "text": f"Blood sugar is trending upward (~{proj} mg/dL in 30 min). Consider a short walk.",
            })
        elif proj <= 72.0:
            critical_popup = True
            alert_items.append({
                "vital": "Blood Sugar (30m)",
                "severity": "critical",
                "value": f"{proj} mg/dL",
                "text": f"Blood sugar may dip to ~{proj} mg/dL in 30 min. Keep a light snack nearby.",
            })

        # 3. Stress (EDA) check
        if eda_now >= 1.60:
            critical_popup = True
            alert_items.append({
                "vital": "Stress (EDA)",
                "severity": "critical",
                "value": f"{eda_now:.2f} uS",
                "text": f"Stress response is very high ({eda_now:.2f} uS). Take a 10-minute quiet break.",
            })
        elif eda_now >= 1.25:
            alert_items.append({
                "vital": "Stress (EDA)",
                "severity": "warning",
                "value": f"{eda_now:.2f} uS",
                "text": f"Stress levels are elevated ({eda_now:.2f} uS), which can nudge blood sugar up.",
            })

        # 4. Insulin Response check
        if si_now < 0.40:
            alert_items.append({
                "vital": "Insulin Response",
                "severity": "warning",
                "value": f"S_I {si_now:.2f}",
                "text": "Insulin response is reduced—lighter carbs and post-meal walks will help most today.",
            })

        # 5. Overall Cardiometabolic Strain check
        if risk_now >= 68.0:
            critical_popup = True

        # 6. Past-bedtime sleep check
        sleep_logged = any(e["type"] == "sleep" and abs(e["idx"] - start_idx) <= 48 for e in self.logged_events)
        if ((ts.hour == 23 and ts.minute >= 30) or ts.hour <= 3) and acc_now >= 0.045 and not sleep_logged:
            alert_items.append({
                "vital": "Bedtime",
                "severity": "warning",
                "value": ts.strftime("%I:%M %p").lstrip("0"),
                "text": "It's past bedtime and no sleep is logged yet—short sleep reduces tomorrow's insulin response.",
            })

        alerts = [a["text"] for a in alert_items]
        if critical_popup:
            level = "critical"
            title = "High-Priority Health Alert"
            message = " ".join(alerts)
        elif alert_items:
            level = "warning"
            title = "Gentle Heads-Up"
            message = " ".join(alerts)
        else:
            level = "steady"
            title = "All Vitals Steady"
            message = f"Blood sugar (~{proj} mg/dL in 30m), heart rate ({hr_now:.0f} bpm), and stress are all within comfortable ranges."

        return {
            "level": level,
            "critical_popup": critical_popup,
            "title": title,
            "projected_30m": proj,
            "delta_30m": delta,
            "alert_items": alert_items,
            "alerts": alerts,
            "message": message,
        }

    def simulate_what_if(self, start_idx: int, scenario: WhatIfScenario) -> Dict:
        self.run_until(start_idx)
        n_horizon = min(scenario.horizon_hours * 12, len(self.df) - 1 - start_idx)
        if n_horizon < 6:
            n_horizon = min(144, len(self.df) - 1 - start_idx)

        def _rollout(apply_intervention: bool) -> List[Dict]:
            g, gut, si, stress = self.x.copy()
            var_g = float(self.P[0, 0])
            if apply_intervention:
                si_mult = 1.0 + 0.0022 * scenario.post_meal_walk_min + 0.009 * scenario.hrv_recovery_boost
                si = float(np.clip(si * si_mult, 0.25, 1.45))

            traj = []
            walk_steps = int(round(scenario.post_meal_walk_min / 5.0))
            walk_countdown = 0

            for k in range(1, n_horizon + 1):
                row = self.df.iloc[start_idx + k]
                carbs = float(row["carbs_g"])
                acc = float(row["acc_mag"])
                eda = float(row["eda_us"])
                hrv = float(row["hrv_rmssd_ms"])

                if apply_intervention:
                    carbs = max(0.0, carbs * (1.0 - scenario.carb_reduction_pct / 100.0))
                    eda = float(np.clip(eda * (1.0 - scenario.stress_reduction_pct / 100.0), 0.1, 4.5))
                    hrv = float(np.clip(hrv + scenario.hrv_recovery_boost, 12.0, 95.0))
                    if row["carbs_g"] >= 20.0 and walk_steps > 0:
                        walk_countdown = walk_steps
                    if walk_countdown > 0:
                        acc = max(acc, 0.34)
                        walk_countdown -= 1

                gut = 0.86 * gut + carbs
                carb_app = 0.14 * gut
                act_uptake = min(6.0, 14.5 * max(0.0, acc - 0.06))
                stress = 0.70 * stress + 0.30 * eda
                stress_drive = 2.6 * max(0.0, stress - 0.45)
                excess_g = max(0.0, g - self.baseline_g)
                ins_act = si * excess_g * 0.09

                g = float(np.clip(
                    g - 0.065 * (g - self.baseline_g) + 0.34 * carb_app - ins_act - act_uptake + stress_drive,
                    55.0, 295.0
                ))
                var_g = min(420.0, 0.92 * var_g + 5.2)
                std_g = float(np.sqrt(var_g))
                risk_score, _ = self._compute_risk_score(g, si, stress, hrv)

                traj.append({
                    "timestamp": str(row["timestamp"]),
                    "step": k,
                    "glucose": round(g, 2),
                    "ci_lower_90": round(g - 1.645 * std_g, 1),
                    "ci_upper_90": round(g + 1.645 * std_g, 1),
                    "risk_score": risk_score,
                    "observed_glucose": round(float(row["glucose_mg_dl"]), 2),
                })
            return traj

        baseline_traj = _rollout(apply_intervention=False)
        whatif_traj = _rollout(apply_intervention=True)

        base_g = np.array([p["glucose"] for p in baseline_traj])
        whatif_g = np.array([p["glucose"] for p in whatif_traj])
        base_r = np.array([p["risk_score"] for p in baseline_traj])
        whatif_r = np.array([p["risk_score"] for p in whatif_traj])

        base_tir = float(np.mean((base_g >= 70.0) & (base_g <= 140.0)) * 100.0)
        whatif_tir = float(np.mean((whatif_g >= 70.0) & (whatif_g <= 140.0)) * 100.0)
        g_delta = round(float(np.mean(whatif_g) - np.mean(base_g)), 1)
        tir_gain = round(whatif_tir - base_tir, 1)
        risk_red = round(float(np.mean(base_r) - np.mean(whatif_r)), 1)

        if g_delta >= 3.0 or risk_red <= -2.0 or tir_gain <= -2.0:
            verdict = "negative"
            verdict_title = "Negative Health Impact (Higher Strain)"
            verdict_text = f"This scenario is projected to raise average blood sugar by +{abs(g_delta)} mg/dL and increase cardiometabolic strain by +{abs(risk_red)} points."
        elif g_delta <= -3.0 or tir_gain >= 2.0 or risk_red >= 3.0:
            verdict = "positive"
            verdict_title = "Positive Health Impact"
            verdict_text = f"These changes are projected to lower average blood sugar by {abs(g_delta)} mg/dL, improve Time-in-Range by +{tir_gain}%, and reduce strain by {risk_red} points."
        else:
            verdict = "neutral"
            verdict_title = "Minimal Change"
            verdict_text = "Your sliders are close to your usual routine. Try testing a post-meal walk or a Cheat Day scenario to see the contrast."

        return {
            "patient_id": self.patient_id,
            "start_idx": start_idx,
            "horizon_hours": scenario.horizon_hours,
            "baseline_trajectory": baseline_traj,
            "whatif_trajectory": whatif_traj,
            "verdict": {
                "status": verdict,
                "title": verdict_title,
                "message": verdict_text,
            },
            "summary": {
                "baseline_mean_glucose": round(float(np.mean(base_g)), 1),
                "whatif_mean_glucose": round(float(np.mean(whatif_g)), 1),
                "glucose_delta": g_delta,
                "baseline_peak_glucose": round(float(np.max(base_g)), 1),
                "whatif_peak_glucose": round(float(np.max(whatif_g)), 1),
                "baseline_tir_pct": round(base_tir, 1),
                "whatif_tir_pct": round(whatif_tir, 1),
                "tir_gain_pct": tir_gain,
                "baseline_mean_risk": round(float(np.mean(base_r)), 1),
                "whatif_mean_risk": round(float(np.mean(whatif_r)), 1),
                "risk_reduction": risk_red,
            },
        }

    def evaluate_ablation(self) -> Dict:
        n = len(self.df)
        split = n // 2
        horizon = 6

        feat_cols = ["glucose_mg_dl", "hr_bpm", "hrv_rmssd_ms", "eda_us", "acc_mag", "carbs_g"]
        X_train = self.df[feat_cols].iloc[: split - horizon].values
        y_train = self.df["glucose_mg_dl"].iloc[horizon:split].values

        mu = X_train.mean(axis=0)
        sd = X_train.std(axis=0) + 1e-6
        X_tr_n = np.column_stack([np.ones(len(X_train)), (X_train - mu) / sd])
        reg = 5.0 * np.eye(X_tr_n.shape[1])
        reg[0, 0] = 0.0
        w = np.linalg.solve(X_tr_n.T @ X_tr_n + reg, X_tr_n.T @ y_train)

        X_test = self.df[feat_cols].iloc[split : n - horizon].values
        y_test = self.df["glucose_mg_dl"].iloc[split + horizon : n].values
        X_te_n = np.column_stack([np.ones(len(X_test)), (X_test - mu) / sd])
        static_preds = X_te_n @ w

        self.run_until(split - 1)
        twin_preds = []
        twin_low = []
        twin_high = []

        for idx in range(split, n - horizon):
            self.step_assimilate(idx)
            g, gut, si, stress = self.x.copy()
            var_g = float(self.P[0, 0])
            for k in range(1, horizon + 1):
                r_f = self.df.iloc[idx + k]
                gut = 0.86 * gut + float(r_f["carbs_g"])
                act_up = min(6.0, 14.5 * max(0.0, float(r_f["acc_mag"]) - 0.06))
                stress = 0.70 * stress + 0.30 * float(r_f["eda_us"])
                st_drv = 2.6 * max(0.0, stress - 0.45)
                ins_act = si * max(0.0, g - self.baseline_g) * 0.09
                g = float(np.clip(g - 0.065 * (g - self.baseline_g) + 0.34 * 0.14 * gut - ins_act - act_up + st_drv, 50.0, 300.0))
                var_g = 0.92 * var_g + 5.2
            std_g = float(np.sqrt(var_g))
            twin_preds.append(g)
            twin_low.append(g - 1.645 * std_g)
            twin_high.append(g + 1.645 * std_g)

        twin_preds_arr = np.array(twin_preds)
        static_rmse = float(np.sqrt(np.mean((static_preds - y_test) ** 2)))
        static_mae = float(np.mean(np.abs(static_preds - y_test)))
        twin_rmse = float(np.sqrt(np.mean((twin_preds_arr - y_test) ** 2)))
        twin_mae = float(np.mean(np.abs(twin_preds_arr - y_test)))

        def _f1(preds: np.ndarray, truth: np.ndarray) -> float:
            p_pos = preds >= 140.0
            t_pos = truth >= 140.0
            tp = np.sum(p_pos & t_pos)
            fp = np.sum(p_pos & ~t_pos)
            fn = np.sum(~p_pos & t_pos)
            if tp == 0:
                return 0.0
            return float(2 * tp / (2 * tp + fp + fn))

        coverage_90 = float(np.mean((y_test >= np.array(twin_low)) & (y_test <= np.array(twin_high))) * 100.0)
        zones = [clarke_zone(float(r), float(p)) for r, p in zip(y_test, twin_preds_arr)]
        total_z = max(1, len(zones))
        clarke_summary = {z: round(zones.count(z) / total_z * 100.0, 1) for z in ["A", "B", "C", "D", "E"]}

        step_sample = max(1, len(y_test) // 150)
        clarke_points = {
            "reference": [round(float(v), 1) for v in y_test[::step_sample]],
            "predicted": [round(float(v), 1) for v in twin_preds_arr[::step_sample]],
            "zone": zones[::step_sample],
        }

        return {
            "patient_id": self.patient_id,
            "horizon_minutes": 30,
            "held_out_samples": int(len(y_test)),
            "mage_mg_dl": compute_mage(y_test),
            "clarke_summary": clarke_summary,
            "clarke_points": clarke_points,
            "static_baseline": {
                "model": "Static Ridge Regression (No Latent State)",
                "rmse_mg_dl": round(static_rmse, 2),
                "mae_mg_dl": round(static_mae, 2),
                "hyperglycemia_f1": round(_f1(static_preds, y_test), 3),
                "uncertainty_calibration_90": None,
            },
            "dynamic_digital_twin": {
                "model": "EKF + Physiology-Informed ODE Twin",
                "rmse_mg_dl": round(twin_rmse, 2),
                "mae_mg_dl": round(twin_mae, 2),
                "hyperglycemia_f1": round(_f1(twin_preds_arr, y_test), 3),
                "uncertainty_calibration_90": round(coverage_90, 1),
            },
            "rmse_improvement_pct": round((static_rmse - twin_rmse) / max(1e-6, static_rmse) * 100.0, 1),
        }