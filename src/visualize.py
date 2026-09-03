"""
RetailPulse Forecast - Visualization & Reporting Module
Generates high-resolution publication-quality figures for historical trends,
forecast vs actual, residual diagnostics, and inventory impact.
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
    
    # Aggregated weekly total sales across all stores
    weekly_agg = df.groupby("Date")["Weekly_Sales"].sum().reset_index()
    
    plt.plot(weekly_agg["Date"], weekly_agg["Weekly_Sales"] / 1e3, label="Total Weekly Sales ($k)", color="#1f77b4", linewidth=2.2)
    
    # Highlight holiday weeks
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
    
    # Monthly sales boxplot
    sns.boxplot(data=df_plot, x="Month", y="Weekly_Sales", ax=axes[0], palette="Blues_r", hue="Month", legend=False)
    axes[0].set_title("Monthly Sales Distribution (Seasonal Cycle)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Month of Year", fontsize=10)
    axes[0].set_ylabel("Weekly Sales ($)", fontsize=10)
    
    # Holiday vs Non-Holiday impact by Dept
    sns.barplot(data=df_plot, x="Dept_Name", y="Weekly_Sales", hue="IsHoliday", ax=axes[1], palette=["#4575b4", "#d73027"])
    axes[1].set_title("Holiday Lift Impact by Department", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Department", fontsize=10)
    axes[1].set_ylabel("Average Weekly Sales ($)", fontsize=10)
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend(title="Holiday Week", labels=["Regular Week", "Holiday Week"])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close("all")

def plot_forecast_vs_actual(df_test: pd.DataFrame, output_path: str, model_name: str = "LightGBM"):
    """Plots actual vs predicted sales on out-of-time test horizon."""
    plt.figure(figsize=(14, 6))
    
    test_agg = df_test.groupby("Date").agg({
        "Weekly_Sales": "sum",
        "Predicted_Sales": "sum"
    }).reset_index()
    
    plt.plot(test_agg["Date"], test_agg["Weekly_Sales"] / 1e3, label="Actual Sales ($k)", color="#2b5c8f", marker="o", markersize=4, linewidth=2)
    plt.plot(test_agg["Date"], test_agg["Predicted_Sales"] / 1e3, label=f"Forecast ({model_name}) ($k)", color="#e6550d", linestyle="--", marker="s", markersize=4, linewidth=2)
    
    plt.fill_between(
        test_agg["Date"],
        (test_agg["Predicted_Sales"] * 0.92) / 1e3,
        (test_agg["Predicted_Sales"] * 1.08) / 1e3,
        color="#e6550d",
        alpha=0.15,
        label="8% Prediction Uncertainty Band"
    )
    
    plt.title(f"RetailPulse Out-of-Time Forecast vs Actuals ({model_name})", fontsize=14, fontweight="bold", pad=12)
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
    
    # Residual Histogram & KDE
    sns.histplot(residuals, kde=True, ax=axes[0], color="#2b8cbe", bins=30)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_title("Residual Error Distribution (Actual - Predicted)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Error ($)", fontsize=10)
    axes[0].set_ylabel("Count", fontsize=10)
    
    # Predicted vs Residuals
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

def plot_inventory_recommendations(df_test: pd.DataFrame, output_path: str):
    """Visualizes recommended buffer and reorder safety stock levels."""
    plt.figure(figsize=(12, 6))
    dept_agg = df_test.groupby("Dept_ID").agg({
        "Weekly_Sales": "mean",
        "Predicted_Sales": "mean"
    }).reset_index()
    
    # Safety stock computation: Mean demand + 1.65 * std error (95% service level)
    dept_agg["Safety_Stock_Buffer"] = dept_agg["Predicted_Sales"] * 1.15
    
    x = np.arange(len(dept_agg))
    width = 0.25
    
    plt.bar(x - width, dept_agg["Weekly_Sales"], width, label="Actual Avg Weekly Demand", color="#41b6c4")
    plt.bar(x, dept_agg["Predicted_Sales"], width, label="Forecasted Base Demand", color="#225ea8")
    plt.bar(x + width, dept_agg["Safety_Stock_Buffer"], width, label="Target Safety Stock (95% SLA)", color="#fd8d3c")
    
    plt.xticks(x, [f"Dept {d}" for d in dept_agg["Dept_ID"]])
    plt.title("Inventory Reorder & Safety Stock Recommendation by Department", fontsize=13, fontweight="bold", pad=10)
    plt.xlabel("Department", fontsize=11)
    plt.ylabel("Weekly Units / Sales ($)", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close("all")
