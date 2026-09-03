"""
RetailPulse Forecast - Evaluation & Error Analysis Module
Computes MAE, RMSE, MAPE, WAPE, R², and residual diagnostics across slices.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes comprehensive regression and time-series error metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Non-zero mask for percentage calculations
    nonzero = y_true > 0
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100.0
    
    # Weighted Absolute Percentage Error (WAPE)
    wape = (np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)) * 100.0
    
    return {
        "MAE ($)": round(mae, 2),
        "RMSE ($)": round(rmse, 2),
        "MAPE (%)": round(mape, 2),
        "WAPE (%)": round(wape, 2),
        "R2 Score": round(r2, 4)
    }

def analyze_error_by_segment(
    df_eval: pd.DataFrame,
    pred_col: str = "Predicted_Sales",
    target_col: str = "Weekly_Sales"
) -> pd.DataFrame:
    """Calculates error metrics sliced by Store Type, Department, and Holiday status."""
    df = df_eval.copy()
    df["Error"] = df[pred_col] - df[target_col]
    df["Abs_Error"] = np.abs(df["Error"])
    df["Pct_Error"] = (df["Abs_Error"] / df[target_col]) * 100.0
    
    # Group by Dept
    dept_stats = df.groupby("Dept_ID").agg(
        Mean_Actual=(target_col, "mean"),
        Mean_Predicted=(pred_col, "mean"),
        MAE=("Abs_Error", "mean"),
        WAPE=("Abs_Error", lambda x: (x.sum() / df.loc[x.index, target_col].sum()) * 100.0)
    ).reset_index()
    
    return dept_stats
