"""
Feature Engineering Module for Kerala Fisheries CPUE Time-Series.
Calculates lagged indicators, rolling CPUE metrics, climate anomalies,
ecological predator-prey proxies, and chronological train/val/test splits.
"""

import os
import numpy as np
import pandas as pd

def engineer_fisheries_features(df, target_species="Oil Sardine"):
    """
    Performs feature engineering specifically tailored for time-series forecasting
    of relative fish abundance / CPUE for a selected target species.
    """
    # Standardize column naming
    df = df.copy()
    if "cpue_kg_per_boat_day" in df.columns and "cpue_kg_trip" not in df.columns:
        df["cpue_kg_trip"] = df["cpue_kg_per_boat_day"]
    if "fishing_effort_boat_days" in df.columns and "fishing_effort_trips" not in df.columns:
        df["fishing_effort_trips"] = df["fishing_effort_boat_days"]
    if "catch_tonnes" in df.columns and "catch_kg" not in df.columns:
        df["catch_kg"] = df["catch_tonnes"] * 1000.0
        
    df["date"] = pd.to_datetime(df["date"])
    
    # Compute cross-species ecological indexes (prey, predator, trophic ratios)
    # Oil Sardine (prey), Indian Mackerel (intermediate), Seer Fish & Tuna (apex predators)
    pivoted_cpue = df.pivot_table(index=["district", "date"], columns="species", values="cpue_kg_trip", aggfunc="mean").reset_index()
    pivoted_cpue["prey_index"] = pivoted_cpue.get("Oil Sardine", 100.0)
    pivoted_cpue["predator_index"] = (pivoted_cpue.get("Seer Fish", 20.0) + pivoted_cpue.get("Tuna", 30.0)) / 2.0
    pivoted_cpue["trophic_ratio"] = pivoted_cpue["prey_index"] / (pivoted_cpue["predator_index"] + 1e-4)
    pivoted_cpue["lotka_volterra_proxy"] = pivoted_cpue["prey_index"] * pivoted_cpue["predator_index"] / 1000.0
    
    # Filter dataset for target species
    df_sp = df[df["species"] == target_species].copy()
    df_sp = pd.merge(df_sp, pivoted_cpue[["district", "date", "prey_index", "predator_index", "trophic_ratio", "lotka_volterra_proxy"]], on=["district", "date"], how="left")
    
    # Monsoon ban indicator (June and July: 52-day mechanized trawl ban)
    df_sp["is_monsoon_ban"] = df_sp["month"].isin([6, 7]).astype(float)
    
    # Sort chronologically within each district
    df_sp = df_sp.sort_values(by=["district", "date"]).reset_index(drop=True)
    
    engineered_dfs = []
    for district, group in df_sp.groupby("district"):
        g = group.copy()
        
        # 1. Temporal / Seasonal cyclic encodings
        g["month_sin"] = np.sin(2 * np.pi * g["month"] / 12.0)
        g["month_cos"] = np.cos(2 * np.pi * g["month"] / 12.0)
        
        # 2. Lagged CPUE features (t-1, t-2, t-3, t-12 seasonal lag)
        g["cpue_lag_1"] = g["cpue_kg_trip"].shift(1)
        g["cpue_lag_2"] = g["cpue_kg_trip"].shift(2)
        g["cpue_lag_3"] = g["cpue_kg_trip"].shift(3)
        g["cpue_lag_12"] = g["cpue_kg_trip"].shift(12) # same month last year
        
        # 3. Rolling CPUE statistics (3-month and 6-month trend)
        g["cpue_rolling_mean_3"] = g["cpue_kg_trip"].shift(1).rolling(window=3, min_periods=1).mean()
        g["cpue_rolling_std_3"] = g["cpue_kg_trip"].shift(1).rolling(window=3, min_periods=1).std().fillna(0)
        g["cpue_rolling_mean_6"] = g["cpue_kg_trip"].shift(1).rolling(window=6, min_periods=1).mean()
        
        # 4. Momentum / CPUE velocity (growth rate)
        g["cpue_change_1m"] = (g["cpue_lag_1"] - g["cpue_lag_2"]) / (g["cpue_lag_2"] + 1e-5)
        
        # 5. Environmental anomalies (deviation from district-month historical climatology)
        monthly_clim_sst = g.groupby("month")["sea_surface_temperature_c"].transform("mean")
        monthly_clim_rain = g.groupby("month")["rainfall_mm"].transform("mean")
        
        g["sst_anomaly"] = g["sea_surface_temperature_c"] - monthly_clim_sst
        g["rainfall_anomaly"] = g["rainfall_mm"] - monthly_clim_rain
        
        # 6. Lagged Environmental features (environmental forcing on recruitment)
        g["sst_lag_1"] = g["sea_surface_temperature_c"].shift(1)
        g["upwelling_lag_1"] = g["upwelling_index"].shift(1)
        g["chl_a_lag_1"] = g["chlorophyll_a_mg_m3"].shift(1)
        
        # 7. Fishing effort dynamics
        g["effort_lag_1"] = g["fishing_effort_trips"].shift(1)
        g["effort_change_1m"] = (g["fishing_effort_trips"] - g["effort_lag_1"]) / (g["effort_lag_1"] + 1e-5)
        
        # 8. Target variables for forecasting
        # Primary Target: Future CPUE at t+1 (Next period abundance indicator)
        g["target_cpue_next"] = g["cpue_kg_trip"].shift(-1)
        
        # Secondary Target: Next period expected total catch (kg)
        g["target_catch_next"] = g["catch_kg"].shift(-1)
        
        engineered_dfs.append(g)
        
    df_feat = pd.concat(engineered_dfs, ignore_index=True)
    
    # Drop rows with NaN from lags and target
    df_clean = df_feat.dropna(subset=["cpue_lag_1", "cpue_lag_2", "cpue_lag_3", "target_cpue_next"]).copy()
    
    # Fill seasonal lag for earliest records
    df_clean["cpue_lag_12"] = df_clean["cpue_lag_12"].fillna(df_clean["cpue_rolling_mean_3"])
    
    return df_clean

def get_feature_columns():
    """Returns standard feature list used across ML and KAN models."""
    return [
        # Environmental / Oceanographic Features
        "sea_surface_temperature_c",
        "sst_anomaly",
        "rainfall_mm",
        "wind_speed_ms",
        "wave_height_m",
        "upwelling_index",
        "chlorophyll_a_mg_m3",
        # Fisheries Effort & Pressure Features
        "fishing_effort_trips",
        "fishing_pressure",
        "is_monsoon_ban",
        # Historical & Autoregressive Abundance Features
        "cpue_lag_1",
        "cpue_lag_2",
        "cpue_lag_3",
        "cpue_lag_12",
        "cpue_rolling_mean_3",
        "cpue_change_1m",
        # Ecological & Predator-Prey Indices
        "prey_index",
        "predator_index",
        "trophic_ratio",
        "lotka_volterra_proxy",
        # Cyclical Seasonality
        "month_sin",
        "month_cos"
    ]

def split_chronological(df, train_end_year=2021, val_end_year=2023):
    """
    Performs chronological train/validation/test split with zero future lookahead.
    Train: <= train_end_year (e.g. 2012-2021)
    Validation: train_end_year+1 to val_end_year (e.g. 2022-2023)
    Test: > val_end_year (e.g. 2024-2025)
    """
    df_train = df[df["year"] <= train_end_year].copy()
    df_val = df[(df["year"] > train_end_year) & (df["year"] <= val_end_year)].copy()
    df_test = df[df["year"] > val_end_year].copy()
    
    print(f"[SPLIT] Chronological Split summary:")
    print(f"  - Train ({df_train['year'].min()}-{train_end_year}): {len(df_train)} samples")
    print(f"  - Val   ({train_end_year+1}-{val_end_year}): {len(df_val)} samples")
    print(f"  - Test  ({val_end_year+1}-{df_test['year'].max()}): {len(df_test)} samples")
    
    return df_train, df_val, df_test

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    df = pd.read_csv("data/raw/kerala_marine_fisheries_raw.csv")
    df_feat = engineer_fisheries_features(df, "Oil Sardine")
    df_tr, df_v, df_te = split_chronological(df_feat)
    print(f"[SUCCESS] Features engineered: {df_feat.shape}, Features count: {len(get_feature_columns())}")
