"""
RetailPulse Forecast - Visualization & Reporting Module
Generates high-resolution publication-quality figures for historical trends,
forecast vs actual with statistically derived prediction intervals, residual diagnostics,
and inventory safety stock planning.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set visual style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

def plot_historical_trends(df: pd.DataFrame, output_path: str):
    """Plots multi-year aggregate weekly sales trend and department breakdown."""
    plt.figure(figsize=(14, 6))
    
    weekly_agg = df.groupby("Date")["Weekly_Sales"].sum().reset_index()
    
    plt.plot(weekly_agg["Date"], weekly_agg["Weekly_Sales"] / 1e3, label="Total Weekly Sales ($k)", color="#1f77b4", linewidth=2.2)
    
    holidays = df[df["IsHoliday"] == 1]["Date"].unique()
    for h in holidays:
        plt.axvline(pd.to_datetime(h), color="#ff7f0e", linestyle="--", alpha=0.35, linewidth=1.0)
    
    plt.title("RetailPulse Historical Sales Overview (2021 - 2024)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Total Weekly Sales ($ in Thousands)", fontsize=11)
    plt.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close("all")

def plot_seasonality_and_decomposition(df: pd.DataFrame, output_path: str):
    """Visualizes monthly sales distributions and holiday lift across departments."""
    df_plot = df.copy()
    if "Month" not in df_plot.columns:
        df_plot["Month"] = pd.to_datetime(df_plot["Date"]).dt.month
        
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.boxplot(data=df_plot, x="Month", y="Weekly_Sales", ax=axes[0], palette="Blues_r", hue="Month", legend=False)
    axes[0].set_title("Monthly Sales Distribution (Seasonal Cycle)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Month of Year", fontsize=10)
    axes[0].set_ylabel("Weekly Sales ($)", fontsize=10)
    
    sns.barplot(data=df_plot, x="Dept_Name", y="Weekly_Sales", hue="IsHoliday", ax=axes[1], palette=["#4575b4", "#d73027"])
    axes[1].set_title("Holiday Lift Impact by Department", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Department", fontsize=10)
    axes[1].set_ylabel("Average Weekly Sales ($)", fontsize=10)
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend(title="Holiday Week", labels=["Regular Week", "Holiday Week"])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close("all")

def plot_forecast_vs_actual(df_test: pd.DataFrame, output_path: str, model_name: str = "Champion Model", val_rmse: float = 2500.0):
    """
    Plots actual vs predicted sales on out-of-time test horizon with a statistically
    derived empirical 95% prediction interval (±1.96 * sigma_residual).
    """
    plt.figure(figsize=(14, 6))
    
    test_agg = df_test.groupby("Date").agg({
        "Weekly_Sales": "sum",
        "Predicted_Sales": "sum"
    }).reset_index()
    
    n_series = df_test["Dept_ID"].nunique() * df_test["Store_ID"].nunique()
    # Aggregate error standard deviation scales with sqrt(N_series)
    agg_std = val_rmse * np.sqrt(n_series)
    
    lower_bound = np.maximum(0, test_agg["Predicted_Sales"] - 1.96 * agg_std) / 1e3
    upper_bound = (test_agg["Predicted_Sales"] + 1.96 * agg_std) / 1e3
    
    plt.plot(test_agg["Date"], test_agg["Weekly_Sales"] / 1e3, label="Actual Out-of-Time Sales ($k)", color="#2b5c8f", marker="o", markersize=4, linewidth=2)
    plt.plot(test_agg["Date"], test_agg["Predicted_Sales"] / 1e3, label=f"Unbiased Test Forecast ({model_name}) ($k)", color="#e6550d", linestyle="--", marker="s", markersize=4, linewidth=2)
    
    plt.fill_between(
        test_agg["Date"],
        lower_bound,
        upper_bound,
        color="#e6550d",
        alpha=0.15,
        label="Statistical 95% Prediction Interval (±1.96 σ_val)"
    )
    
    plt.title(f"RetailPulse Out-of-Time Forecast vs Actuals — Champion: {model_name}", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Weekly Aggregate Sales ($ in Thousands)", fontsize=11)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close("all")

def plot_residual_diagnostics(df_test: pd.DataFrame, output_path: str):
    """Generates residual distribution and prediction vs error scatter."""
    residuals = df_test["Weekly_Sales"] - df_test["Predicted_Sales"]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    sns.histplot(residuals, kde=True, ax=axes[0], color="#2b8cbe", bins=30)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_title("Test Residual Error Distribution (Actual - Predicted)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Residual Error ($)", fontsize=10)
    axes[0].set_ylabel("Count", fontsize=10)
    
    axes[1].scatter(df_test["Predicted_Sales"], residuals, alpha=0.4, color="#31a354", edgecolors="none")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_title("Residuals vs Fitted Values (Homoscedasticity Check)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Fitted Values ($)", fontsize=10)
    axes[1].set_ylabel("Residuals ($)", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close("all")

def plot_feature_importance(importances: pd.DataFrame, output_path: str, top_n: int = 15):
    """Bar chart of top predictive features."""
    plt.figure(figsize=(10, 7))
    top_df = importances.head(top_n).sort_values(by="Importance", ascending=True)
    
    plt.barh(top_df["Feature"], top_df["Importance"], color="#3182bd", edgecolor="#08519c")
    plt.title(f"Top {top_n} Predictive Features for Sales Forecasting", fontsize=13, fontweight="bold", pad=10)
    plt.xlabel("Feature Importance Score (Gain)", fontsize=11)
    plt.ylabel("Engineered Feature", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close("all")

def plot_inventory_recommendations(df_test: pd.DataFrame, output_path: str, lead_time_weeks: int = 2):
    """
    Visualizes recommended safety stock and reorder point buffer using statistical
    forecast error standard deviation: Safety Stock = Z_0.95 * sigma_error * sqrt(lead_time).
    """
    plt.figure(figsize=(12, 6))
    
    df_eval = df_test.copy()
    df_eval["Residual"] = df_eval["Weekly_Sales"] - df_eval["Predicted_Sales"]
    
    dept_agg = df_eval.groupby("Dept_ID").agg(
        Mean_Actual=("Weekly_Sales", "mean"),
        Mean_Forecast=("Predicted_Sales", "mean"),
        RMSE_Residual=("Residual", lambda r: np.sqrt(np.mean(r**2)))
    ).reset_index()
    
    z_95 = 1.645 # 95% service level normal quantile
    dept_agg["Safety_Stock"] = z_95 * dept_agg["RMSE_Residual"] * np.sqrt(lead_time_weeks)
    dept_agg["Reorder_Point"] = (dept_agg["Mean_Forecast"] * lead_time_weeks) + dept_agg["Safety_Stock"]
    
    x = np.arange(len(dept_agg))
    width = 0.25
    
    plt.bar(x - width, dept_agg["Mean_Forecast"] * lead_time_weeks, width, label=f"Cycle Demand ({lead_time_weeks}W Lead Time)", color="#41b6c4")
    plt.bar(x, dept_agg["Safety_Stock"], width, label="Statistical Safety Stock (95% SLA)", color="#fd8d3c")
    plt.bar(x + width, dept_agg["Reorder_Point"], width, label="Total Reorder Point (ROP)", color="#225ea8")
    
    plt.xticks(x, [f"Dept {d}" for d in dept_agg["Dept_ID"]])
    plt.title("Statistical Inventory Reorder Policy by Department (95% Service Level)", fontsize=13, fontweight="bold", pad=10)
    plt.xlabel("Department", fontsize=11)
    plt.ylabel("Inventory Value ($ USD)", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close("all")
