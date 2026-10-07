"""
Inference and prediction pipeline.
Predicts future CPUE and expected catch given environmental and effort inputs,
and couples predictions with sustainability and risk scoring.
"""

import os
import torch
import numpy as np
import pandas as pd
from src.features.engineer_features import get_feature_columns
from src.optimization.sustainability import compute_sustainability_assessment, compute_trolling_suitability

class KeralaFisheriesPredictor:
    """
    End-to-end predictor using trained KAN model and scaler.
    """
    def __init__(self, kan_model, scaler_X, scaler_y=None, feature_names=None):
        self.kan_model = kan_model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.feature_names = feature_names or get_feature_columns()
        self.device = next(kan_model.parameters()).device if hasattr(kan_model, "parameters") else "cpu"
        
    def predict_cpue(self, feature_dict: dict) -> float:
        """
        Takes raw feature values, scales them, and runs forward inference through KAN.
        """
        raw_vals = np.array([[feature_dict[col] for col in self.feature_names]])
        scaled_vals = self.scaler_X.transform(raw_vals)
        
        self.kan_model.eval()
        with torch.no_grad():
            x_t = torch.tensor(scaled_vals, dtype=torch.float32).to(self.device)
            raw_out = self.kan_model(x_t).cpu().numpy()
            
        if self.scaler_y is not None:
            pred_cpue = float(self.scaler_y.inverse_transform(raw_out.reshape(-1, 1))[0, 0])
        else:
            pred_cpue = float(raw_out[0, 0])
            
        return max(0.0, pred_cpue)

    def generate_decision_support(
        self,
        species_name: str,
        district: str,
        date_str: str,
        planned_effort_trips: float,
        environmental_conditions: dict,
        historical_lags: dict,
        ecological_proxies: dict,
        min_sustainable_cpue: float = 140.0
    ):
        """
        Runs complete decision support workflow:
        Prediction -> Sustainability Assessment -> Trolling Evaluation -> Recommendations
        """
        # Assemble feature dictionary
        feat_dict = {
            "sea_surface_temperature_c": environmental_conditions.get("sst", 28.5),
            "sst_anomaly": environmental_conditions.get("sst_anomaly", 0.0),
            "rainfall_mm": environmental_conditions.get("rainfall", 200.0),
            "wind_speed_ms": environmental_conditions.get("wind_speed", 4.5),
            "wave_height_m": environmental_conditions.get("wave_height", 1.4),
            "upwelling_index": environmental_conditions.get("upwelling_index", 2.0),
            "chlorophyll_a_mg_m3": environmental_conditions.get("chlorophyll_a", 1.8),
            "fishing_effort_trips": planned_effort_trips,
            "fishing_pressure": planned_effort_trips / 1800.0,
            "is_monsoon_ban": environmental_conditions.get("is_monsoon_ban", 0),
            "cpue_lag_1": historical_lags.get("cpue_lag_1", 175.0),
            "cpue_lag_2": historical_lags.get("cpue_lag_2", 170.0),
            "cpue_lag_3": historical_lags.get("cpue_lag_3", 165.0),
            "cpue_lag_12": historical_lags.get("cpue_lag_12", 180.0),
            "cpue_rolling_mean_3": historical_lags.get("cpue_rolling_mean_3", 170.0),
            "cpue_change_1m": historical_lags.get("cpue_change_1m", 0.02),
            "prey_index": ecological_proxies.get("prey_index", 175.0),
            "predator_index": ecological_proxies.get("predator_index", 65.0),
            "trophic_ratio": ecological_proxies.get("trophic_ratio", 0.37),
            "lotka_volterra_proxy": ecological_proxies.get("lotka_volterra_proxy", 2.5),
            "month_sin": np.sin(2 * np.pi * environmental_conditions.get("month", 10) / 12.0),
            "month_cos": np.cos(2 * np.pi * environmental_conditions.get("month", 10) / 12.0),
        }
        
        predicted_cpue = self.predict_cpue(feat_dict)
        
        # Sustainability assessment
        assessment = compute_sustainability_assessment(
            predicted_cpue=predicted_cpue,
            species_name=species_name,
            fishing_effort_trips=planned_effort_trips,
            sea_surface_temp=feat_dict["sea_surface_temperature_c"],
            wind_speed=feat_dict["wind_speed_ms"],
            wave_height=feat_dict["wave_height_m"],
            upwelling_index=feat_dict["upwelling_index"],
            cpue_trend_growth=feat_dict["cpue_change_1m"],
            min_sustainable_cpue_threshold=min_sustainable_cpue
        )
        
        # Trolling assessment
        trolling = compute_trolling_suitability(
            species_name=species_name,
            predicted_predator_cpue=predicted_cpue if species_name in ["Seer Fish", "Coastal Tuna"] else feat_dict["predator_index"],
            sea_surface_temp=feat_dict["sea_surface_temperature_c"],
            wind_speed=feat_dict["wind_speed_ms"],
            wave_height=feat_dict["wave_height_m"]
        )
        
        return {
            "metadata": {
                "species": species_name,
                "district": district,
                "date": date_str,
                "planned_effort_trips": planned_effort_trips
            },
            "prediction": {
                "predicted_cpue_kg_trip": round(predicted_cpue, 2),
                "expected_catch_kg": assessment["expected_catch_kg"]
            },
            "sustainability": assessment,
            "trolling": trolling
        }
