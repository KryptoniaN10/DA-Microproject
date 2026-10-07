# KAN-Based Sustainable Fishing Prediction and Decision Support System for Kerala Fisheries 🐟📊

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-informational.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning and Data Analytics decision-support platform for marine fisheries along the **Kerala coast (Southeastern Arabian Sea)**.

The system uses **Kolmogorov-Arnold Networks (KAN)** with learnable 1D B-spline activation functions on network edges to model complex non-linear interactions between oceanographic drivers, fishing effort, historical Catch Per Unit Effort (CPUE), and multi-species ecological proxies.

---

## 📌 Key Objectives & Scientific Distinctions

1. **Abundance vs. Catch Disentanglement**:
   Catch is NOT abundance. High catch can occur purely due to heavy fishing pressure. We normalize landing data by fishing effort (standardized boat trips) to derive true **Catch Per Unit Effort (CPUE)** as an abundance proxy:
   $$\text{CPUE} = \frac{\text{Catch (kg)}}{\text{Standardized Fishing Effort (Trips)}}$$

2. **Sustainability-First Objective**:
   The goal is NOT to blindly maximize catch, but to:
   > *"Find fishing periods that provide a high expected yield while keeping fish abundance indicators above sustainable biological thresholds and reducing overfishing risk."*

3. **Kolmogorov-Arnold Networks (KAN) with Explainability**:
   Unlike black-box Multi-Layer Perceptrons (MLPs), KAN places learnable univariate B-splines $\phi(x)$ on network edges:
   $$f(x) = \sum_{q=1}^{2n+1} \Phi_q \left( \sum_{p=1}^n \phi_{q,p}(x_p) \right)$$
   This enables direct visual extraction of learned physical curves (e.g. thermal response envelopes, upwelling benefits, and overfishing penalties).

4. **Zero-Leakage Chronological Evaluation**:
   Strict forward temporal splitting:
   - **Training Set**: 2012 – 2021 (10 years)
   - **Validation Set**: 2022 – 2023 (2 years)
   - **Out-of-Sample Test Set**: 2024 – 2025 (2 years)

---

## 🗂️ Project Repository Structure

```
e:\DA microproject\
├── config\
│   └── config.yaml                          # Master parameters, species baselines, thresholds
├── data\
│   ├── raw\                                 # Sourced CMFRI, Kerala Fisheries & NOAA datasets
│   │   ├── cmfri_kerala_species_landings.csv
│   │   ├── kerala_fishing_effort_boats.csv
│   │   ├── imd_copernicus_kerala_coastal_env.csv
│   │   └── data_metadata.json
│   └── processed\                           # Cleaned, validated & feature-engineered datasets
│       ├── integrated_kerala_fisheries_env.csv
│       ├── oil_sardine_timeseries.csv
│       ├── indian_mackerel_timeseries.csv
│       └── data_quality_report.json
├── models\                                  # Serialized model weights & scalers
│   ├── kan_oil_sardine.pt
│   ├── baseline_models.joblib
│   ├── scaler_X.joblib
│   └── scaler_y.joblib
├── notebooks\                               # Complete interactive Jupyter Notebook suite
│   ├── 01_Data_Exploration_and_Preprocessing.ipynb
│   ├── 02_KAN_Sustainable_Fisheries_Modeling.ipynb
│   ├── 03_Decision_Support_and_Optimization.ipynb
│   └── Master_KAN_Kerala_Fisheries_System.ipynb # Unified end-to-end Master Notebook
├── reports\                                 # Research and data audit documents
│   ├── DATA_AVAILABILITY_REPORT.md
│   ├── RESEARCH_REPORT.md
│   └── MODEL_COMPARISON.md
├── src\                                     # Modular Python source library
│   ├── data\
│   │   ├── fetch_and_clean.py
│   │   └── validate_data.py
│   ├── features\
│   │   └── engineer_features.py
│   ├── models\
│   │   ├── baselines.py
│   │   ├── kan_model.py
│   │   └── evaluate.py
│   ├── prediction\
│   │   └── predictor.py
│   └── optimization\
│       ├── sustainability.py
│       └── fishing_window.py
├── run_full_pipeline.py                     # One-click execution script
├── generate_notebooks.py                    # Notebook compilation script
├── requirements.txt                         # Python dependencies
└── README.md
```

---

## 📊 Model Performance Comparison (Test Set: 2024–2025)

| Model Architecture | Test MAE (kg/trip) | Test RMSE (kg/trip) | Test $R^2$ Score | Test MAPE (%) | Interpretability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | 791.85 | 1192.37 | 0.7740 | 71.66% | High (Linear only) |
| **Ridge Regression** | 794.69 | 1199.93 | 0.7712 | 71.69% | High (Linear only) |
| **Multi-Layer Perceptron (MLP)** | 757.31 | 1175.37 | 0.7804 | 58.62% | Low (Black-box) |
| **Gradient Boosting** | 398.03 | 693.70 | 0.9235 | 18.96% | Medium (Trees) |
| **XGBoost Regressor** | 384.08 | 683.38 | 0.9258 | 19.37% | Medium (Trees) |
| **Random Forest Regressor** | 369.17 | 657.26 | 0.9313 | 18.62% | Medium (Trees) |
| **Kolmogorov-Arnold Network (KAN)** | **508.92** | **799.21** | **0.8985** | **30.78%** | **High (Univariate 1D Splines)** |

---

## 🚀 Quickstart Guide

### 1. Clone & Install Dependencies
```bash
cd "e:/DA microproject"
pip install -r requirements.txt
```

### 2. Execute Full End-to-End Pipeline
```bash
python run_full_pipeline.py
```

### 3. Launch the Interactive Jupyter Notebooks
```bash
jupyter notebook notebooks/Master_KAN_Kerala_Fisheries_System.ipynb
```
Or explore individual modular notebooks:
- [01_Data_Exploration_and_Preprocessing.ipynb](file:///e:/DA%20microproject/notebooks/01_Data_Exploration_and_Preprocessing.ipynb)
- [02_KAN_Sustainable_Fisheries_Modeling.ipynb](file:///e:/DA%20microproject/notebooks/02_KAN_Sustainable_Fisheries_Modeling.ipynb)
- [03_Decision_Support_and_Optimization.ipynb](file:///e:/DA%20microproject/notebooks/03_Decision_Support_and_Optimization.ipynb)

---

## 🔬 Core Decision-Support Indicators

- **Fishing Suitability Score ($0 - 100$)**:
  $$\text{Suitability} = 0.35 \cdot S_{\text{abundance}} + 0.25 \cdot S_{\text{env}} + 0.20 \cdot S_{\text{yield}} - 0.20 \cdot R_{\text{overfishing}}$$
- **Overfishing Risk Score ($0 - 100$ & Low/Medium/High)**:
  $$\text{Risk} \propto \text{Fishing Pressure} \times \text{Abundance Deficit} \times \text{Declining CPUE Momentum}$$
- **Constrained Safe Effort Optimization**:
  $$\max_{\text{Effort}} \text{Expected Catch} \quad \text{s.t.} \quad \text{Risk} \le 45.0, \; \hat{A}_t \ge A_{\text{thresh}}$$
- **Trolling Suitability Engine**:
  Evaluates wave height ($< 1.8\text{m}$), wind velocity ($3.0 - 6.0\text{ m/s}$), and surface thermal window for apex gamefish (Seer Fish & Tuna).
