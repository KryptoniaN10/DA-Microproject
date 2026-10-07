"""
Demonstration Fallback Dataset Generator.
Used ONLY when live external network fetching fails or in offline demonstration mode.
Strictly watermarks all records, metadata, and columns with:
'DEMONSTRATION DATA — NOT REAL FISHERY DATA'
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

KERALA_DISTRICTS = [
    "Thiruvananthapuram", "Kollam", "Alappuzha", "Ernakulam", 
    "Thrissur", "Malappuram", "Kozhikode", "Kannur", "Kasaragod"
]

SPECIES_LIST = ["Oil Sardine", "Indian Mackerel", "Seer Fish", "Tuna"]

WATERMARK_STRING = "DEMONSTRATION DATA — NOT REAL FISHERY DATA"


def generate_demonstration_dataset(start_year=2012, end_year=2025, seed=42):
    """
    Generates an offline demonstration fallback dataset explicitly watermarked
    to guarantee zero misrepresentation as real observed fisheries records.
    """
    np.random.seed(seed)
    records = []
    
    district_weights = {
        "Kozhikode": 0.19, "Ernakulam": 0.17, "Alappuzha": 0.15,
        "Kollam": 0.14, "Kannur": 0.11, "Thiruvananthapuram": 0.09,
        "Malappuram": 0.07, "Thrissur": 0.05, "Kasaragod": 0.03
    }
    
    for yr in range(start_year, end_year + 1):
        for dist, d_wt in district_weights.items():
            base_effort = int(d_wt * 26000)
            for mo in range(1, 13):
                q = (mo - 1) // 3 + 1
                
                # Synthetic seasonal environmental curves
                sst = 28.5 + 1.2 * np.cos(2 * np.pi * (mo - 4) / 12) + np.random.normal(0, 0.2)
                upw = float(np.clip(np.sin(np.pi * (mo - 3) / 6) if 4 <= mo <= 9 else 0.08, 0.05, 1.0))
                rain = float(max(5.0, 350.0 * np.exp(-((mo - 7)**2) / 2.5) + np.random.normal(0, 25)))
                wind = float(max(1.5, 4.0 + 3.0 * np.exp(-((mo - 7)**2) / 2.0) + np.random.normal(0, 0.4)))
                salinity = float(35.0 - 2.8 * (rain / 500.0))
                chl_a = float(0.6 + 3.0 * upw + 0.4 * (rain / 400.0))
                
                effort_fac = 0.65 if mo in [6, 7] else (1.35 if mo in [10, 11, 12] else 1.0)
                effort = float(base_fleet_effort := base_effort * effort_fac * np.random.uniform(0.9, 1.1))
                
                for sp in SPECIES_LIST:
                    sp_base = {"Oil Sardine": 800.0, "Indian Mackerel": 450.0, "Tuna": 280.0, "Seer Fish": 160.0}[sp]
                    catch = max(2.0, sp_base * d_wt * 5.0 * (0.8 + 0.5 * upw) * np.random.uniform(0.85, 1.15))
                    cpue = (catch * 1000.0) / max(1.0, effort)
                    pressure = float(np.clip(effort / (base_effort * 1.6), 0.05, 0.95))
                    overfishing_risk = float(np.clip(pressure * 1.2 - cpue / 400.0, 0.0, 1.0))
                    
                    records.append({
                        "date": f"{yr}-{mo:02d}-01",
                        "year": yr,
                        "month": mo,
                        "quarter": q,
                        "district": dist,
                        "species": sp,
                        "catch_tonnes": round(catch, 2),
                        "fishing_effort_boat_days": round(effort, 1),
                        "cpue_kg_per_boat_day": round(cpue, 2),
                        "sea_surface_temperature_c": round(sst, 2),
                        "salinity_psu": round(salinity, 2),
                        "chlorophyll_a_mg_m3": round(chl_a, 2),
                        "upwelling_index": round(upw, 2),
                        "rainfall_mm": round(rain, 1),
                        "wind_speed_ms": round(wind, 2),
                        "wave_height_m": round(1.2 + 1.6 * upw, 2),
                        "air_temperature_c": round(sst - 0.5, 2),
                        "fishing_pressure": round(pressure, 3),
                        "overfishing_risk_label": round(overfishing_risk, 3),
                        "data_authenticity_watermark": WATERMARK_STRING,
                        "data_type": "demonstration_synthetic"
                    })
                    
    df = pd.DataFrame(records)
    return df


def save_demonstration_fallback(out_dir="data/raw"):
    """Saves watermarked demonstration fallback CSV and metadata."""
    os.makedirs(out_dir, exist_ok=True)
    df = generate_demonstration_dataset()
    out_path = os.path.join(out_dir, "demonstration_fallback_fisheries.csv")
    df.to_csv(out_path, index=False)
    
    meta = {
        "dataset_type": WATERMARK_STRING,
        "warning": "THIS IS SYNTHETIC DEMONSTRATION DATA ONLY. NOT REAL FISHERY DATA.",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "species": SPECIES_LIST,
        "districts": KERALA_DISTRICTS
    }
    with open(os.path.join(out_dir, "demonstration_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Fallback] Saved watermarked demonstration data to {out_path}")
    return df


if __name__ == "__main__":
    save_demonstration_fallback()
