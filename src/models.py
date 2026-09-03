"""
RetailPulse Forecast - Forecasting Models & Chronological Validation
Includes Naive baselines, Ridge Regression, Random Forest, LightGBM, and XGBoost.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb
import joblib

class NaiveLagForecaster:
    """Predicts next week sales equal to Lag_1 (Last known weekly sales)."""
    def __init__(self):
        self.name = "Baseline_Naive_Lag1"
    
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        return X["Lag_1"].values

class SeasonalLagForecaster:
    """Predicts next week sales equal to Lag_52 (Same week last year)."""
    def __init__(self):
        self.name = "Baseline_Seasonal_Lag52"
    
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        return X["Lag_52"].values

class MovingAverageForecaster:
    """Predicts next week sales equal to 4-week rolling mean."""
    def __init__(self):
        self.name = "Baseline_MovingAvg_4W"
    
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        return X["Rolling_Mean_4"].values

def get_model_registry() -> Dict[str, Any]:
    """Returns a dictionary of competitive forecasting models."""
    return {
        "Naive Lag-1": NaiveLagForecaster(),
        "Seasonal Lag-52": SeasonalLagForecaster(),
        "Rolling 4W Moving Avg": MovingAverageForecaster(),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=10.0, random_state=42))
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        ),
        "LightGBM Regressor": lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        ),
        "XGBoost Regressor": xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    }

def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs chronological train/validation/test split without future data leakage.
    """
    unique_dates = sorted(df["Date"].unique())
    n_dates = len(unique_dates)
    
    train_end_idx = int(n_dates * train_ratio)
    val_end_idx = int(n_dates * (train_ratio + val_ratio))
    
    train_dates = unique_dates[:train_end_idx]
    val_dates = unique_dates[train_end_idx:val_end_idx]
    test_dates = unique_dates[val_end_idx:]
    
    df_train = df[df["Date"].isin(train_dates)].copy()
    df_val = df[df["Date"].isin(val_dates)].copy()
    df_test = df[df["Date"].isin(test_dates)].copy()
    
    return df_train, df_val, df_test
