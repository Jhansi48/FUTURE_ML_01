"""
RetailPulse Forecast - Evaluation & Error Analysis Module
Computes MAE, RMSE, MAPE, WAPE, R², and residual diagnostics across slices.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes comprehensive regression and time-series error metrics:
    MAE, RMSE, MAPE, WAPE, and R² Score.
    
    Robustly handles zero denominators, single-sample or zero-variance targets,
    and non-finite edge cases.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            "MAE ($)": 0.0,
            "RMSE ($)": 0.0,
            "MAPE (%)": 0.0,
            "WAPE (%)": 0.0,
            "R2 Score": 0.0
        }
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"Shape mismatch: y_true has length {len(y_true)} but y_pred has length {len(y_pred)}")
        
    # Replace non-finite values if present
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.all(valid_mask):
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        if len(y_true) == 0:
            return {
                "MAE ($)": 0.0,
                "RMSE ($)": 0.0,
                "MAPE (%)": 0.0,
                "WAPE (%)": 0.0,
                "R2 Score": 0.0
            }

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    
    # R2 Score with protection against zero variance or < 2 samples
    if len(y_true) < 2 or np.var(y_true) < 1e-12:
        r2 = 1.0 if np.allclose(y_true, y_pred, atol=1e-5) else 0.0
    else:
        r2 = float(r2_score(y_true, y_pred))
    
    # Non-zero mask for percentage calculations (MAPE)
    nonzero = np.abs(y_true) > 1e-8
    if np.any(nonzero):
        mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100.0)
    else:
        mape = 0.0 if np.allclose(y_pred, 0.0, atol=1e-5) else 100.0
    
    # Weighted Absolute Percentage Error (WAPE) with zero denominator handling
    sum_true = float(np.sum(np.abs(y_true)))
    sum_abs_err = float(np.sum(np.abs(y_true - y_pred)))
    if sum_true > 1e-8:
        wape = float((sum_abs_err / sum_true) * 100.0)
    else:
        wape = 0.0 if sum_abs_err < 1e-8 else 100.0
    
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
    """Calculates error metrics sliced by Department on the evaluation set."""
    df = df_eval.copy()
    df["Error"] = df[pred_col] - df[target_col]
    df["Abs_Error"] = np.abs(df["Error"])
    df["Sq_Error"] = df["Error"] ** 2
    
    dept_names_map = {1: "Electronics", 2: "Apparel", 3: "Grocery", 4: "Home & Garden", 5: "Toys", 6: "Health & Beauty"}
    if "Dept_Name" not in df.columns and "Dept_ID" in df.columns:
        df["Dept_Name"] = df["Dept_ID"].map(dept_names_map)
        
    group_cols = ["Dept_ID", "Dept_Name"] if "Dept_Name" in df.columns else ["Dept_ID"]
    
    def safe_dept_wape(sub_df):
        total_act = sub_df[target_col].sum()
        total_err = sub_df["Abs_Error"].sum()
        return (total_err / total_act * 100.0) if total_act > 1e-8 else 0.0

    dept_stats = df.groupby(group_cols).agg(
        Mean_Actual=(target_col, "mean"),
        Mean_Predicted=(pred_col, "mean"),
        MAE=("Abs_Error", "mean"),
        RMSE=("Sq_Error", lambda x: np.sqrt(x.mean()))
    ).reset_index()
    
    # Add WAPE using safe helper
    wape_values = []
    for _, row in dept_stats.iterrows():
        sub_df = df[(df["Dept_ID"] == row["Dept_ID"])]
        wape_values.append(safe_dept_wape(sub_df))
    dept_stats["WAPE"] = wape_values
    
    return dept_stats
