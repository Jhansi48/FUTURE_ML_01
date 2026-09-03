"""
RetailPulse Forecast - Time-Based Feature Engineering Module
Creates chronological lag features, rolling window aggregates, holiday proximity,
and cyclical calendar encodings without look-ahead bias.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts calendar and cyclical temporal attributes."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Calendar features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfMonth"] = df["Date"].dt.day
    df["Quarter"] = df["Date"].dt.quarter
    df["Is_Month_End"] = df["Date"].dt.is_month_end.astype(int)
    df["Is_Quarter_End"] = df["Date"].dt.is_quarter_end.astype(int)
    
    # Cyclical trigonometric encodings
    df["Sin_Week"] = np.sin(2 * np.pi * df["WeekOfYear"] / 52.0)
    df["Cos_Week"] = np.cos(2 * np.pi * df["WeekOfYear"] / 52.0)
    df["Sin_Month"] = np.sin(2 * np.pi * df["Month"] / 12.0)
    df["Cos_Month"] = np.cos(2 * np.pi * df["Month"] / 12.0)
    
    # Fill Holiday Name
    if "Holiday_Name" in df.columns:
        df["Holiday_Name"] = df["Holiday_Name"].fillna("None")
    
    return df

def create_lag_and_rolling_features(
    df: pd.DataFrame,
    group_cols: List[str] = ["Store_ID", "Dept_ID"],
    target_col: str = "Weekly_Sales",
    lags: List[int] = [1, 2, 4, 8, 12, 52],
    windows: List[int] = [4, 8, 12, 26, 52]
) -> pd.DataFrame:
    """
    Creates strictly historical lag and rolling window features grouped by store and department.
    Guarantees zero future leakage by shifting the target before computing rolling statistics.
    """
    df = df.sort_values(by=group_cols + ["Date"]).reset_index(drop=True)
    
    # Generate Lag Features
    for lag in lags:
        df[f"Lag_{lag}"] = df.groupby(group_cols)[target_col].shift(lag)
        
    # Generate Rolling Statistics (grouped by store and dept, on shifted series)
    for window in windows:
        shifted_grouped = df.groupby(group_cols)[target_col].shift(1)
        # Apply rolling per group
        df[f"Rolling_Mean_{window}"] = df.groupby(group_cols)[target_col].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=2).mean()
        )
        df[f"Rolling_Std_{window}"] = df.groupby(group_cols)[target_col].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=2).std()
        )
        df[f"Rolling_Min_{window}"] = df.groupby(group_cols)[target_col].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=2).min()
        )
        df[f"Rolling_Max_{window}"] = df.groupby(group_cols)[target_col].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=2).max()
        )

    # Exponential Moving Averages per group
    df["EMA_4"] = df.groupby(group_cols)[target_col].transform(
        lambda s: s.shift(1).ewm(span=4, adjust=False).mean()
    )
    df["EMA_12"] = df.groupby(group_cols)[target_col].transform(
        lambda s: s.shift(1).ewm(span=12, adjust=False).mean()
    )

    # Fill rolling std nulls with 0
    for window in windows:
        df[f"Rolling_Std_{window}"] = df[f"Rolling_Std_{window}"].fillna(0.0)

    # Sales Growth Ratio (Lag 1 vs Lag 4)
    df["Sales_Growth_Rate_1v4"] = (df["Lag_1"] - df["Lag_4"]) / (df["Lag_4"] + 1e-5)
    
    return df

def prepare_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Full feature engineering pipeline and column selection."""
    df_feat = create_time_features(df)
    df_feat = create_lag_and_rolling_features(df_feat)
    
    # Categorical encoding (Store Type and Dept Name)
    df_feat = pd.get_dummies(df_feat, columns=["Store_Type", "Dept_Name"], drop_first=True)
    
    # Drop initial rows where annual lags (52 weeks) are NaN to ensure robust training data
    df_clean = df_feat.dropna().reset_index(drop=True)
    
    # Exclude non-feature columns
    excluded = ["Date", "Weekly_Sales", "Holiday_Name"]
    feature_cols = [c for c in df_clean.columns if c not in excluded]
    
    return df_clean, feature_cols
