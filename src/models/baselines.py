"""
Baseline Machine Learning models for comparative benchmarking against KAN.
Includes Linear Regression, Ridge, Random Forest, Gradient Boosting / XGBoost, and MLP.
"""

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

def get_baseline_models():
    """Returns a dictionary of configured baseline ML models."""
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, 
            max_depth=12, 
            min_samples_split=4, 
            random_state=42, 
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, 
            learning_rate=0.08, 
            max_depth=5, 
            random_state=42
        ),
        "Multi-Layer Perceptron (MLP)": MLPRegressor(
            hidden_layer_sizes=(64, 32), 
            activation="relu", 
            max_iter=300, 
            random_state=42
        )
    }
    
    # Try importing XGBoost if available
    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=100, 
            learning_rate=0.08, 
            max_depth=5, 
            random_state=42,
            n_jobs=-1
        )
    except Exception:
        pass
        
    return models
