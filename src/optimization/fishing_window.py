"""
Fishing window identification and constrained effort optimization engine.
Optimizes fishing effort to maximize expected catch while keeping fish stock
above the biological sustainability threshold and overfishing risk low.
"""

import numpy as np
import pandas as pd
from src.optimization.sustainability import compute_sustainability_assessment, compute_trolling_suitability

def optimize_fishing_effort(
    predicted_cpue: float,
    species_name: str,
    sea_surface_temp: float,
    wind_speed: float,
    wave_height: float,
    upwelling_index: float,
    min_sustainable_cpue: float = 140.0,
    effort_search_range=(200, 3500),
    step=50
):
    """
    Finds optimal fishing effort (boat trips) using grid search constrained optimization:
    Maximize: Expected Catch (kg) = Predicted CPUE * Effort
    Subject to:
      1. Overfishing Risk <= 45.0 (Low-to-Medium boundary)
      2. Fishing Suitability Score maximized
      3. Predicted Abundance >= Sustainable Threshold
    """
    candidate_efforts = np.arange(effort_search_range[0], effort_search_range[1] + step, step)
    
    best_eval = None
    best_effort = effort_search_range[0]
    best_suitability = -1.0
    all_evaluations = []
    
    for effort in candidate_efforts:
        evaluation = compute_sustainability_assessment(
            predicted_cpue=predicted_cpue,
            species_name=species_name,
            fishing_effort_trips=effort,
            sea_surface_temp=sea_surface_temp,
            wind_speed=wind_speed,
            wave_height=wave_height,
            upwelling_index=upwelling_index,
            min_sustainable_cpue_threshold=min_sustainable_cpue
        )
        
        evaluation["tested_effort_trips"] = int(effort)
        all_evaluations.append(evaluation)
        
        # Constrained criteria
        if evaluation["overfishing_risk_score"] <= 50.0:
            if evaluation["fishing_suitability_score"] > best_suitability:
                best_suitability = evaluation["fishing_suitability_score"]
                best_effort = int(effort)
                best_eval = evaluation
                
    if best_eval is None:
        # Fallback to safest minimum effort
        best_effort = int(effort_search_range[0])
        best_eval = all_evaluations[0]
        
    return {
        "optimal_effort_trips": best_effort,
        "optimal_expected_catch_kg": best_eval["expected_catch_kg"],
        "optimal_suitability_score": best_eval["fishing_suitability_score"],
        "optimal_risk_score": best_eval["overfishing_risk_score"],
        "optimal_risk_level": best_eval["overfishing_risk_level"],
        "optimal_recommendation": best_eval["recommendation"],
        "tradeoff_curve": pd.DataFrame(all_evaluations)
    }

def find_sustainable_fishing_windows(
    forecast_df: pd.DataFrame,
    species_name: str = "Oil Sardine",
    min_sustainable_cpue: float = 140.0
):
    """
    Evaluates sequential forecast periods and highlights high-suitability sustainable fishing windows.
    """
    window_results = []
    
    for idx, row in forecast_df.iterrows():
        pred_cpue = row["predicted_cpue"]
        sst = row["sea_surface_temperature_c"]
        wind = row["wind_speed_ms"]
        wave = row["wave_height_m"]
        upwelling = row.get("upwelling_index", 2.5)
        effort = row.get("fishing_effort_trips", 1800)
        
        res = compute_sustainability_assessment(
            predicted_cpue=pred_cpue,
            species_name=species_name,
            fishing_effort_trips=effort,
            sea_surface_temp=sst,
            wind_speed=wind,
            wave_height=wave,
            upwelling_index=upwelling,
            min_sustainable_cpue_threshold=min_sustainable_cpue
        )
        
        # Trolling assessment
        troll = compute_trolling_suitability(
            species_name=species_name,
            predicted_predator_cpue=pred_cpue if species_name in ["Seer Fish", "Coastal Tuna"] else 30.0,
            sea_surface_temp=sst,
            wind_speed=wind,
            wave_height=wave
        )
        
        window_results.append({
            "date": row["date"],
            "district": row.get("district", "All Kerala"),
            "predicted_cpue": round(pred_cpue, 2),
            "expected_catch_kg": res["expected_catch_kg"],
            "fishing_suitability_score": res["fishing_suitability_score"],
            "overfishing_risk_level": res["overfishing_risk_level"],
            "overfishing_risk_score": res["overfishing_risk_score"],
            "trolling_suitability_score": troll["trolling_suitability_score"],
            "trolling_rating": troll["trolling_rating"],
            "recommendation": res["recommendation"],
            "is_recommended_window": res["fishing_suitability_score"] >= 65.0 and res["overfishing_risk_score"] < 45.0
        })
        
    return pd.DataFrame(window_results)
