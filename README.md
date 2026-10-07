# KAN-Based Sustainable Fishing Prediction and Decision Support System for Kerala Fisheries

An explainable, data-driven machine learning platform for sustainable fisheries management along the Kerala coast (8.0°N–13.0°N, 74.5°E–77.5°E). Uses **Kolmogorov-Arnold Networks (KAN)** with learnable B-spline edge activations and **Constrained Effort Optimization** to forecast fish abundance ({t+1}$), prevent recruitment overfishing, and guide artisanal trolling operations.

---

## Key Features

1. **Real Data Ingestion & Availability Gate:** Real-time data acquisition from **NASA POWER API** (Atmospheric Climate), **Copernicus Marine / NOAA OISST v2.1** (SST & Wave Height), and **ICAR-CMFRI** (Marine Fish Landings Technical Reports, 2012–2024) across 9 maritime districts and 4 key species (*Oil Sardine, Indian Mackerel, Seer Fish, Tuna*).
2. **Epistemological Source Ledger:** Every variable is tagged as observed, derived_proxy, or unavailable in data/raw/source_ledger.json.
3. **Multi-Model Chronological Benchmark:** Strict out-of-sample holdout test (Train: 2012–2021, Val: 2022–2023, Test: 2024–2025) comparing Linear Regression, Ridge, Random Forest, Gradient Boosting, XGBoost, MLP, and KAN.
4. **Intrinsic KAN Interpretability:** Mathematical extraction of 1D univariate B-spline response curves $\\phi_i(x)$ for continuous ecological explainability without post-hoc black-box surrogates.
5. **Sustainable Decision Support Engine:** Constrained effort grid optimization, Dynamic Fishing Window evaluation, Overfishing Risk scoring, and Pelagic Trolling Suitability analysis.

---

## Project Structure

`	ext
DA-microproject/
├── data/
│   ├── raw/                      # Real acquired climate, marine, and CMFRI data
│   │   ├── source_ledger.json    # Machine-readable variable source ledger
│   │   └── kerala_marine_fisheries_raw.csv
│   ├── processed/                # Species timeseries & data quality report
│   └── data_metadata.json        # Comprehensive dataset schema & licensing metadata
├── notebooks/
│   └── KAN_Sustainable_Fisheries_Kerala_Master.ipynb # End-to-end executed master notebook
├── reports/
│   ├── DATA_AVAILABILITY_REPORT.md # Per-source audit & proxy documentation
│   ├── MODEL_COMPARISON.md       # Chronological benchmark results & analysis
│   └── RESEARCH_REPORT.md        # Technical research report & policy guidelines
├── models/
│   └── experiment_tracking.json  # Complete experiment tracking logs
├── src/
│   ├── data/                     # Data acquisition, validation, & fallback modules
│   │   ├── fetch_real_data.py
│   │   ├── validate_data.py
│   │   └── demonstration_fallback.py
│   ├── features/                 # Lag engineering & chronological splits
│   │   └── engineer_features.py
│   ├── models/                   # KAN PyTorch architecture & baselines
│   │   ├── kan_model.py
│   │   ├── baselines.py
│   │   └── evaluate.py
│   ├── optimization/             # Sustainability & effort optimization
│   │   ├── sustainability.py
│   │   └── fishing_window.py
│   └── prediction/               # Inference & decision support predictor
│       └── predictor.py
├── requirements.txt
└── README.md
`

---

## Quickstart Guide

### 1. Installation
Ensure Python 3.10+ is installed, then install dependencies:
`ash
pip install -r requirements.txt
`

### 2. Fetch Real Data & Validate
Run the data acquisition and validation pipeline:
`ash
python src/data/fetch_real_data.py
python src/data/validate_data.py
`

### 3. Run Benchmark & Experiment Tracking
Train all models and generate the benchmark report:
`ash
python src/models/evaluate.py
`

### 4. Run the Master Research Notebook
Launch the end-to-end Master Notebook:
`ash
jupyter notebook notebooks/KAN_Sustainable_Fisheries_Kerala_Master.ipynb
`
Or execute headlessly:
`ash
jupyter nbconvert --to notebook --execute notebooks/KAN_Sustainable_Fisheries_Kerala_Master.ipynb --output KAN_Sustainable_Fisheries_Kerala_Master.ipynb
`

---

## Research Questions & Hypotheses

- **RQ1 (Forecasting):** Real environmental lags and autoregressive terms forecast next-period CPUE (^2 \\approx 0.80$ with tree ensembles).
- **RQ2 (KAN Benchmark):** KAN outperforms linear and MLP models on validation data (^2 = 0.4253$ vs .0756$ linear, $-0.1196$ MLP) and offers lower test RMSE (.72$ vs .18$ linear).
- **RQ3 (Interpretability):** Learned B-splines reveal non-monotonic SST thermal optima (27.5–29.0°C) and fishing effort saturation thresholds.
- **RQ4 (Optimization):** Constrained optimization constrains Overfishing Risk to $<35\\%$ while preserving fleet economic returns.
- **RQ5 (Trolling Advisory):** Composite index provides actionable daily trolling recommendations for artisanal fishers.

---

## Data Licenses & Attribution
- **NASA POWER:** Public Domain / CC0 Open Data.
- **Copernicus Marine / Open-Meteo:** CC BY 4.0.
- **ICAR-CMFRI:** Open Access Scientific Technical Publications, Govt. of India.
- **Kerala Fisheries:** Open Government Data License (OGDL).
