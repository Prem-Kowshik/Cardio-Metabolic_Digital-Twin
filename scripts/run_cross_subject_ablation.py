"""
Cross-Subject Ablation Benchmark Script for CS F434 Report & Presentation.
Evaluates Static Ridge Regression vs. Dynamic EKF Digital Twin across all subjects
in `data/physionet_raw` (if downloaded) or the built-in schema-matched cohort.

Usage:
    python3 scripts/run_cross_subject_ablation.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.ingest_physionet import load_cohort
from engine.digital_twin import CardiometabolicDigitalTwin


def main() -> None:
    data_dir = os.environ.get("PHYSIONET_DATA_DIR", str(PROJECT_ROOT / "data" / "physionet_raw"))
    cohort = load_cohort(data_dir if os.path.isdir(data_dir) else None)

    rows = []
    for pid, df in cohort.items():
        twin = CardiometabolicDigitalTwin(df)
        res = twin.evaluate_ablation()
        rows.append({
            "Patient": pid,
            "HbA1c (%)": float(df["hba1c"].iloc[0]),
            "Static RMSE": res["static_baseline"]["rmse_mg_dl"],
            "Twin RMSE": res["dynamic_digital_twin"]["rmse_mg_dl"],
            "RMSE Improvement (%)": res["rmse_improvement_pct"],
            "Static MAE": res["static_baseline"]["mae_mg_dl"],
            "Twin MAE": res["dynamic_digital_twin"]["mae_mg_dl"],
            "90% CI Coverage (%)": res["dynamic_digital_twin"]["uncertainty_calibration_90"],
            "MAGE (mg/dL)": res["mage_mg_dl"],
            "Clarke Zone A+B (%)": round(res["clarke_summary"]["A"] + res["clarke_summary"]["B"], 1),
        })

    out_df = pd.DataFrame(rows)
    out_csv = PROJECT_ROOT / "cross_subject_ablation_results.csv"
    out_df.to_csv(out_csv, index=False)

    print("\n=== Cross-Subject Ablation: Static Baseline vs. Dynamic Digital Twin (30-min Ahead) ===")
    print(out_df.to_string(index=False))
    print(f"\nMean RMSE Improvement: {out_df['RMSE Improvement (%)'].mean():.1f}%")
    print(f"Mean 90% CI Coverage:  {out_df['90% CI Coverage (%)'].mean():.1f}%")
    print(f"Saved results to: {out_csv}\n")


if __name__ == "__main__":
    main()
