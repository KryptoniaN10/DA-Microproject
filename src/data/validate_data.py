"""
Data validation and quality assurance module for Kerala Marine Fisheries.
Audits missing records, physical plausibility bounds, unit consistency, and anomaly checks.
"""

import os
import json
import numpy as np
import pandas as pd

def validate_fisheries_dataset(df):
    """
    Executes automated data quality validation on the raw/integrated dataset.
    Returns validation report dictionary and cleaned DataFrame.
    """
    report = {
        "total_records": int(len(df)),
        "total_columns": int(len(df.columns)),
        "columns_list": list(df.columns),
        "missing_values_per_column": {col: int(df[col].isnull().sum()) for col in df.columns},
        "duplicate_rows": int(df.duplicated().sum()),
        "integrity_checks": {},
        "summary_statistics": {},
        "data_quality_score": 100.0
    }
    
    # Check 1: Non-negative catch and effort
    neg_catch = int((df["catch_kg"] < 0).sum())
    neg_effort = int((df["fishing_effort_trips"] <= 0).sum())
    report["integrity_checks"]["negative_catch_count"] = neg_catch
    report["integrity_checks"]["zero_or_negative_effort_count"] = neg_effort
    
    # Check 2: Physical bounds for environmental variables
    sst_outliers = int(((df["sea_surface_temperature_c"] < 20.0) | (df["sea_surface_temperature_c"] > 35.0)).sum())
    rainfall_negative = int((df["rainfall_mm"] < 0).sum())
    wave_negative = int((df["wave_height_m"] < 0).sum())
    wind_negative = int((df["wind_speed_ms"] < 0).sum())
    
    report["integrity_checks"]["sst_out_of_bounds"] = sst_outliers
    report["integrity_checks"]["rainfall_negative"] = rainfall_negative
    report["integrity_checks"]["wave_height_negative"] = wave_negative
    report["integrity_checks"]["wind_speed_negative"] = wind_negative
    
    # Check 3: CPUE consistency (CPUE = catch_kg / fishing_effort_trips)
    computed_cpue = df["catch_kg"] / df["fishing_effort_trips"]
    cpue_diff = np.abs(computed_cpue - df["cpue_kg_trip"])
    cpue_inconsistent = int((cpue_diff > 0.05).sum())
    report["integrity_checks"]["cpue_inconsistency_count"] = cpue_inconsistent
    
    # Check 4: Date parsing verification
    try:
        pd.to_datetime(df["date"])
        report["integrity_checks"]["date_parsing_valid"] = True
    except Exception as e:
        report["integrity_checks"]["date_parsing_valid"] = False
        
    # Check 5: Species and District coverage
    report["summary_statistics"]["unique_districts"] = int(df["district"].nunique())
    report["summary_statistics"]["districts_list"] = sorted(df["district"].unique().tolist())
    report["summary_statistics"]["unique_species"] = int(df["species"].nunique())
    report["summary_statistics"]["species_list"] = sorted(df["species"].unique().tolist())
    report["summary_statistics"]["temporal_range"] = {
        "start_date": str(df["date"].min()),
        "end_date": str(df["date"].max()),
        "years": sorted(df["year"].unique().tolist())
    }
    
    # Deduct quality points for any defects found
    penalties = (
        neg_catch * 5 + 
        neg_effort * 5 + 
        sst_outliers * 2 + 
        rainfall_negative * 2 + 
        cpue_inconsistent * 5 +
        sum(report["missing_values_per_column"].values()) * 0.5
    )
    report["data_quality_score"] = max(0.0, 100.0 - penalties)
    
    return report

def run_validation_pipeline(base_dir="e:/DA microproject"):
    """Reads raw dataset, validates it, and writes quality report."""
    raw_path = os.path.join(base_dir, "data", "raw", "cmfri_kerala_species_landings.csv")
    from src.data.fetch_and_clean import generate_fisheries_environmental_dataset
    
    df = generate_fisheries_environmental_dataset(2012, 2025)
    report = validate_fisheries_dataset(df)
    
    processed_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    with open(os.path.join(processed_dir, "data_quality_report.json"), "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"[VALIDATION SUCCESS] Data Quality Score: {report['data_quality_score']:.1f}/100")
    print(f"Total Records: {report['total_records']}, Districts: {report['summary_statistics']['unique_districts']}, Species: {report['summary_statistics']['unique_species']}")
    return report

if __name__ == "__main__":
    run_validation_pipeline()
