"""
Feature engineering module for Kerala fisheries CPUE time-series.
Calculates lagged indicators, rolling CPUE metrics, climate anomalies, and ecological proxies.
"""

import os
import numpy as np
import pandas as pd

def engineer_fisheries_features(df, target_species="Oil Sardine"):
    """
    Performs feature engineering specifically tailored for time-series forecasting
    of relative fish abundance / CPUE for a selected target species.
    """
    # Filter dataset for target species
    df_sp = df[df["species"] == target_species].copy()
    
    # Ensure chronological sort within each district
    df_sp["date"] = pd.to_datetime(df_sp["date"])
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
        # Primary Target: Future CPUE at t+1 (Next month abundance indicator)
        g["target_cpue_next"] = g["cpue_kg_trip"].shift(-1)
        
        # Secondary Target: Next month expected total catch (kg)
        g["target_catch_next"] = g["catch_kg"].shift(-1)
        
        engineered_dfs.append(g)
        
    df_feat = pd.concat(engineered_dfs, ignore_index=True)
    
    # Drop rows with NaN from lags (first 12 months for 12-month lag, and last month for target)
    df_clean = df_feat.dropna(subset=["cpue_lag_1", "cpue_lag_2", "cpue_lag_3", "target_cpue_next"]).copy()
    
    # Backfill or fillna for the 12-month lag in earliest records if any
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

if __name__ == "__main__":
    from src.data.fetch_and_clean import generate_fisheries_environmental_dataset
    df = generate_fisheries_environmental_dataset()
    df_feat = engineer_fisheries_features(df, "Oil Sardine")
    print(f"[SUCCESS] Engineered features for Oil Sardine: {df_feat.shape[0]} samples, {len(get_feature_columns())} features")
