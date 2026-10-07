"""
Data package initialization for Kerala Fisheries Analytics.
"""

from src.data.fetch_real_data import build_kerala_fisheries_dataset, save_source_ledger, save_data_metadata
from src.data.validate_data import validate_fisheries_dataset, run_validation_pipeline
from src.data.demonstration_fallback import generate_demonstration_dataset, save_demonstration_fallback

__all__ = [
    "build_kerala_fisheries_dataset",
    "save_source_ledger",
    "save_data_metadata",
    "validate_fisheries_dataset",
    "run_validation_pipeline",
    "generate_demonstration_dataset",
    "save_demonstration_fallback"
]
