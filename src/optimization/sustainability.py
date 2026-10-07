"""
Sustainability assessment, scoring, and overfishing risk computation engine.
Converts ML abundance/CPUE predictions and environmental factors into actionable decision indices.
"""

import numpy as np

def compute_sustainability_assessment(
    predicted_cpue: float,
    species_name: str,
    fishing_effort_trips: float,
    sea_surface_temp: float,
    wind_speed: float,
    wave_height: float,
    upwelling_index: float,
    cpue_trend_growth: float = 0.0,
    historical_district_baseline_cpue: float = 180.0,
    min_sustainable_cpue_threshold: float = 140.0,
    weights: dict = None
):
    """
    Computes:
    1. Abundance Suitability (0-100)
    2. Environmental Suitability (0-100)
    3. Yield Potential (0-100)
    4. Overfishing Risk Score (0-100) & Categorical Risk Level (Low/Medium/High)
    5. Fishing Suitability Score (0-100)
    6. Management Recommendation & Decision Explanation
    """
    if weights is None:
        weights = {
            "abundance": 0.35,
            "environmental": 0.25,
            "yield": 0.20,
            "overfishing_penalty": 0.20
        }
        
    # 1. Abundance Suitability (0 - 100)
    # Ratio of predicted CPUE relative to minimum sustainable threshold
    abundance_ratio = predicted_cpue / max(1.0, min_sustainable_cpue_threshold)
    abundance_score = np.clip(abundance_ratio * 70.0, 0.0, 100.0)
    
    # 2. Environmental Suitability (0 - 100)
    # Optimal SST window for tropical pelagics: 27.5 - 29.5°C
    if 27.5 <= sea_surface_temp <= 29.5:
        temp_score = 100.0 - abs(sea_surface_temp - 28.5) * 15.0
    elif sea_surface_temp < 27.5:
        temp_score = max(20.0, 100.0 - (27.5 - sea_surface_temp) * 35.0)
    else: # Thermal stress > 29.5°C
        temp_score = max(10.0, 100.0 - (sea_surface_temp - 29.5) * 45.0)
        
    # Ocean sea state safety & operational factor
    if wave_height <= 1.5:
        wave_score = 100.0
    elif wave_height <= 2.5:
        wave_score = 75.0 - (wave_height - 1.5) * 35.0
    else:
        wave_score = max(10.0, 40.0 - (wave_height - 2.5) * 20.0)
        
    # Upwelling nutrient enhancement
    upwelling_score = min(100.0, 40.0 + upwelling_index * 12.0)
    
    env_score = 0.45 * temp_score + 0.35 * wave_score + 0.20 * upwelling_score
    env_score = float(np.clip(env_score, 0.0, 100.0))
    
    # 3. Expected Yield (Catch = CPUE * Effort)
    expected_catch_kg = round(predicted_cpue * fishing_effort_trips, 1)
    
    # Yield potential normalized
    ref_expected_catch = min_sustainable_cpue_threshold * 1500.0
    yield_score = float(np.clip((expected_catch_kg / max(1.0, ref_expected_catch)) * 80.0, 0.0, 100.0))
    
    # 4. Overfishing Risk Score (0 - 100)
    # Increases when:
    # - Predicted CPUE is below baseline threshold (abundance deficit)
    # - Fishing effort / pressure is high
    # - CPUE momentum is declining (negative trend)
    abundance_deficit = max(0.0, (min_sustainable_cpue_threshold - predicted_cpue) / max(1.0, min_sustainable_cpue_threshold))
    effort_pressure_norm = np.clip(fishing_effort_trips / 2200.0, 0.2, 2.5)
    declining_penalty = 0.3 if cpue_trend_growth < -0.05 else (0.0 if cpue_trend_growth >= 0 else 0.15)
    
    # Risk equation
    raw_risk = (abundance_deficit * 55.0) + (effort_pressure_norm * 28.0) + (declining_penalty * 25.0)
    
    # Extreme wave height or hostile monsoon safety adds risk
    if wave_height > 3.0:
        raw_risk += 20.0
        
    overfishing_risk_score = float(np.clip(raw_risk, 5.0, 100.0))
    
    if overfishing_risk_score < 35.0:
        risk_level = "Low"
    elif overfishing_risk_score < 68.0:
        risk_level = "Medium"
    else:
        risk_level = "High"
        
    # 5. Master Fishing Suitability Score (0 - 100)
    raw_suitability = (
        weights["abundance"] * abundance_score +
        weights["environmental"] * env_score +
        weights["yield"] * yield_score -
        weights["overfishing_penalty"] * (overfishing_risk_score * 0.8)
    )
    fishing_suitability_score = float(np.clip(raw_suitability, 0.0, 100.0))
    
    # 6. Recommendation and Actionable Insights
    reasons = []
    if predicted_cpue >= min_sustainable_cpue_threshold * 1.15:
        reasons.append("Predicted fish abundance is strong and exceeds sustainability baseline.")
    elif predicted_cpue >= min_sustainable_cpue_threshold:
        reasons.append("Predicted fish abundance is moderate, within safe biological limits.")
    else:
        reasons.append("Predicted abundance is below threshold; stock conservation precaution needed.")
        
    if env_score > 75.0:
        reasons.append("Sea surface temperature and upwelling conditions are highly favorable.")
    elif env_score < 45.0:
        reasons.append("Environmental conditions (elevated SST / rough sea state) are unfavorable.")
        
    if wave_height > 2.8:
        reasons.append("Rough sea state: high wave height (> 2.8m) limits small-craft artisanal safety.")
        
    if risk_level == "High":
        recommendation = "STRICT PRECAUTION: Reduce fishing effort or implement localized gear restriction."
        action_code = "RESTRICT_EFFORT"
    elif risk_level == "Medium":
        recommendation = "CONTROLLED HARVEST: Suitable for moderate, regulated fishing effort."
        action_code = "CONTROLLED_FISHING"
    else:
        recommendation = "OPTIMAL FISHING: Highly favorable window for sustainable commercial operations."
        action_code = "FAVORABLE_WINDOW"
        
    return {
        "species": species_name,
        "predicted_cpue_kg_trip": round(predicted_cpue, 2),
        "expected_catch_kg": expected_catch_kg,
        "fishing_suitability_score": round(fishing_suitability_score, 1),
        "overfishing_risk_score": round(overfishing_risk_score, 1),
        "overfishing_risk_level": risk_level,
        "abundance_score": round(abundance_score, 1),
        "environmental_suitability_score": round(env_score, 1),
        "yield_score": round(yield_score, 1),
        "recommendation": recommendation,
        "action_code": action_code,
        "reasons": reasons
    }

def compute_trolling_suitability(
    species_name: str,
    predicted_predator_cpue: float,
    sea_surface_temp: float,
    wind_speed: float,
    wave_height: float
):
    """
    Computes dedicated trolling suitability for apex pelagic target species (Seerfish, Tuna).
    Trolling requires clear surface water, active pelagic hunting temperatures, and manageable chop.
    """
    # Thermal hunting window for Seerfish & Tuna
    temp_fit = np.exp(-0.5 * ((sea_surface_temp - 29.0) / 1.5) ** 2)
    
    # Wind and wave operational trolling conditions (smooth trolling requires wind 3-6 m/s, waves < 1.8m)
    if wave_height <= 1.2:
        sea_factor = 1.0
    elif wave_height <= 2.0:
        sea_factor = 0.75
    else:
        sea_factor = max(0.1, 1.0 - (wave_height - 1.2) * 0.45)
        
    if 2.5 <= wind_speed <= 6.0:
        wind_factor = 1.0
    else:
        wind_factor = max(0.3, 1.0 - abs(wind_speed - 4.5) * 0.15)
        
    # Abundance factor
    abund_factor = min(1.0, predicted_predator_cpue / 30.0)
    
    trolling_score = (
        0.35 * (temp_fit * 100.0) +
        0.30 * (sea_factor * 100.0) +
        0.15 * (wind_factor * 100.0) +
        0.20 * (abund_factor * 100.0)
    )
    trolling_score = float(np.clip(trolling_score, 0.0, 100.0))
    
    if trolling_score >= 75.0:
        trolling_rating = "Highly Favorable for Pelagic Trolling"
    elif trolling_score >= 50.0:
        trolling_rating = "Moderate Trolling Conditions"
    else:
        trolling_rating = "Low Trolling Suitability (Rough Seas / Suboptimal Temperature)"
        
    return {
        "species": species_name,
        "trolling_suitability_score": round(trolling_score, 1),
        "trolling_rating": trolling_rating,
        "sea_state_operational": "Safe" if wave_height < 2.0 else "Caution / Rough",
        "thermal_condition": "Optimal" if 28.0 <= sea_surface_temp <= 29.8 else "Suboptimal"
    }
