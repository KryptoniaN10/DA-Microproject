"""
Model training, cross-validation, and evaluation framework.
Implements chronological split evaluation:
- Training: 2012-2021
- Validation: 2022-2023
- Testing: 2024-2025
Computes MAE, RMSE, R², MAPE, and directional accuracy.
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.models.baselines import get_baseline_models
from src.models.kan_model import KolmogorovArnoldNetwork
from src.features.engineer_features import get_feature_columns

def compute_metrics(y_true, y_pred):
    """Calculates all key regression evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error (avoiding zero division)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
    
    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 4),
        "MAPE": round(mape, 2)
    }

def train_and_evaluate_all(df_feat, target_col="target_cpue_next", epochs=150, lr=0.005):
    """
    Executes full chronological training, validation, and test evaluation.
    """
    feature_cols = get_feature_columns()
    
    # Chronological Split
    train_mask = df_feat["year"] <= 2021
    val_mask = (df_feat["year"] >= 2022) & (df_feat["year"] <= 2023)
    test_mask = df_feat["year"] >= 2024
    
    X_train_raw = df_feat.loc[train_mask, feature_cols].values
    y_train = df_feat.loc[train_mask, target_col].values
    
    X_val_raw = df_feat.loc[val_mask, feature_cols].values
    y_val = df_feat.loc[val_mask, target_col].values
    
    X_test_raw = df_feat.loc[test_mask, feature_cols].values
    y_test = df_feat.loc[test_mask, target_col].values
    
    # Scalers fitted ONLY on training data to prevent leakage
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train_raw)
    X_val = scaler_X.transform(X_val_raw)
    X_test = scaler_X.transform(X_test_raw)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    
    results = []
    trained_models = {}
    
    # 1. Train and Evaluate Baseline Models
    baselines = get_baseline_models()
    for name, model in baselines.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        val_metrics = compute_metrics(y_val, y_val_pred)
        test_metrics = compute_metrics(y_test, y_test_pred)
        
        results.append({
            "Model": name,
            "Train_Time_s": round(train_time, 3),
            "Val_MAE": val_metrics["MAE"],
            "Val_RMSE": val_metrics["RMSE"],
            "Val_R2": val_metrics["R2"],
            "Val_MAPE_%": val_metrics["MAPE"],
            "Test_MAE": test_metrics["MAE"],
            "Test_RMSE": test_metrics["RMSE"],
            "Test_R2": test_metrics["R2"],
            "Test_MAPE_%": test_metrics["MAPE"],
            "Test_Predictions": y_test_pred
        })
        trained_models[name] = model
        
    # 2. Train and Evaluate Kolmogorov-Arnold Network (KAN)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_features = len(feature_cols)
    kan = KolmogorovArnoldNetwork(
        layers_hidden=[in_features, 16, 8, 1],
        grid_size=5,
        spline_order=3,
        grid_range=(-2.5, 2.5)
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(kan.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=12)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(-1).to(device)
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val_scaled, dtype=torch.float32).unsqueeze(-1).to(device)
    
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    t0 = time.time()
    best_val_loss = float("inf")
    best_kan_state = None
    
    for epoch in range(epochs):
        kan.train()
        optimizer.zero_grad()
        preds = kan(X_train_t)
        loss = criterion(preds, y_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(kan.parameters(), max_norm=1.0)
        optimizer.step()
        
        kan.eval()
        with torch.no_grad():
            val_preds = kan(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_kan_state = {k: v.cpu() for k, v in kan.state_dict().items()}
                
    train_time = time.time() - t0
    
    # Load best weights
    kan.load_state_dict({k: v.to(device) for k, v in best_kan_state.items()})
    kan.eval()
    
    with torch.no_grad():
        y_val_pred_scaled = kan(X_val_t).cpu().numpy()
        y_test_pred_scaled = kan(X_test_t).cpu().numpy()
        
    y_val_pred_kan = scaler_y.inverse_transform(y_val_pred_scaled).flatten()
    y_test_pred_kan = scaler_y.inverse_transform(y_test_pred_scaled).flatten()
        
    val_metrics_kan = compute_metrics(y_val, y_val_pred_kan)
    test_metrics_kan = compute_metrics(y_test, y_test_pred_kan)
    
    results.append({
        "Model": "Kolmogorov-Arnold Network (KAN)",
        "Train_Time_s": round(train_time, 3),
        "Val_MAE": val_metrics_kan["MAE"],
        "Val_RMSE": val_metrics_kan["RMSE"],
        "Val_R2": val_metrics_kan["R2"],
        "Val_MAPE_%": val_metrics_kan["MAPE"],
        "Test_MAE": test_metrics_kan["MAE"],
        "Test_RMSE": test_metrics_kan["RMSE"],
        "Test_R2": test_metrics_kan["R2"],
        "Test_MAPE_%": test_metrics_kan["MAPE"],
        "Test_Predictions": y_test_pred_kan
    })
    trained_models["KAN"] = kan
    
    df_results = pd.DataFrame(results)
    
    return {
        "results_df": df_results,
        "trained_models": trained_models,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "test_ground_truth": y_test,
        "test_metadata": df_feat.loc[test_mask, ["date", "year", "month", "district", "species"]].reset_index(drop=True)
    }

if __name__ == "__main__":
    from src.data.fetch_and_clean import generate_fisheries_environmental_dataset
    from src.features.engineer_features import engineer_fisheries_features
    df = generate_fisheries_environmental_dataset()
    df_feat = engineer_fisheries_features(df, "Oil Sardine")
    out = train_and_evaluate_all(df_feat)
    print(out["results_df"][["Model", "Val_R2", "Test_R2", "Test_MAE", "Test_RMSE"]])
