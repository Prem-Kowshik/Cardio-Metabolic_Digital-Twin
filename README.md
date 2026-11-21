# Cardiometabolic Digital Twin (Patient-Centric + EKF/ODE Engine)

An upgraded full-stack longitudinal digital twin built for **CS F434**, combining a **patient-centric 4-tab UI** with **Extended Kalman Filter (EKF) state assimilation**, **counterfactual ODE simulations**, and **held-out clinical validation**.

## Dataset: BIG IDEAs Lab Glycemic Variability & Wearable Device Data (v1.1.3)
- **Source**: [PhysioNet Open-Access Dataset](https://physionet.org/content/big-ideas-glycemic-wearable/) (no credentialing required)
- **Download 2 sample subjects quickly into `data/physionet_raw`**:
  ```bash
  mkdir -p data/physionet_raw
  wget -q -P data/physionet_raw https://physionet.org/files/big-ideas-glycemic-wearable/1.1.3/Demographics.csv
  wget -r -N -c -np -nH --cut-dirs=4 -P data/physionet_raw https://physionet.org/files/big-ideas-glycemic-wearable/1.1.3/001/
  wget -r -N -c -np -nH --cut-dirs=4 -P data/physionet_raw https://physionet.org/files/big-ideas-glycemic-wearable/1.1.3/002/
  ```
  *(If `data/physionet_raw` is omitted, the backend automatically loads a schema-identical 72-hour 4-patient cohort.)*

## Quickstart
```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```
Open **http://localhost:8000** in your browser.

## Cross-Subject Ablation (For Report / Presentation)
```bash
python3 scripts/run_cross_subject_ablation.py
```
