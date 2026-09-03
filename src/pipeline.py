"""
RetailPulse Forecast - End-to-End Training & Rigorous Evaluation Pipeline
Implements rigorous chronological split methodology:
- 70% Train: Model training only
- 15% Validation: Candidate model comparison & champion selection
- 15% Test: Unbiased out-of-time evaluation of the selected champion
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import save_and_load_dataset
from src.features import prepare_feature_matrix
from src.models import get_model_registry, chronological_split
from src.evaluate import compute_metrics, analyze_error_by_segment
from src.visualize import (
    plot_historical_trends,
    plot_seasonality_and_decomposition,
    plot_forecast_vs_actual,
    plot_residual_diagnostics,
    plot_feature_importance,
    plot_inventory_recommendations
)

def run_pipeline():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(base_dir, "data", "retail_store_sales.csv")
    models_dir = os.path.join(base_dir, "models")
    figures_dir = os.path.join(base_dir, "outputs", "figures")
    metrics_dir = os.path.join(base_dir, "outputs", "metrics")
    reports_dir = os.path.join(base_dir, "reports")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    print("=" * 75)
    print("STEP 1: Ingesting & Verifying Enterprise Retail Sales Dataset")
    print("=" * 75)
    df_raw = save_and_load_dataset(data_path)
    print(f"Data range: {df_raw['Date'].min()} to {df_raw['Date'].max()}")
    print(f"Total rows: {len(df_raw)}, Stores: {df_raw['Store_ID'].nunique()}, Depts: {df_raw['Dept_ID'].nunique()}")
    
    plot_historical_trends(df_raw, os.path.join(figures_dir, "historical_sales_trends.png"))
    plot_seasonality_and_decomposition(df_raw, os.path.join(figures_dir, "seasonality_and_decomposition.png"))
    
    print("\n" + "=" * 75)
    print("STEP 2: Time-Based Feature Engineering (Strict Zero Look-Ahead Leakage)")
    print("=" * 75)
    df_feat, feature_cols = prepare_feature_matrix(df_raw)
    print(f"Engineered features ({len(feature_cols)} total): {feature_cols[:8]} ...")
    print(f"Dataset shape after lag alignment: {df_feat.shape}")
    
    print("\n" + "=" * 75)
    print("STEP 3: Chronological Out-Of-Time Splits (Train 70% / Val 15% / Test 15%)")
    print("=" * 75)
    df_train, df_val, df_test = chronological_split(df_feat, train_ratio=0.70, val_ratio=0.15)
    
    X_train = df_train[feature_cols]
    y_train = df_train["Weekly_Sales"]
    X_val = df_val[feature_cols]
    y_val = df_val["Weekly_Sales"]
    X_test = df_test[feature_cols]
    y_test = df_test["Weekly_Sales"]
    
    print(f"Train Period (70%):      {df_train['Date'].min().date()} to {df_train['Date'].max().date()} ({len(df_train)} samples)")
    print(f"Validation Period (15%): {df_val['Date'].min().date()} to {df_val['Date'].max().date()} ({len(df_val)} samples)")
    print(f"Test Period (15%):       {df_test['Date'].min().date()} to {df_test['Date'].max().date()} ({len(df_test)} samples)")
    
    print("\n" + "=" * 75)
    print("STEP 4: Training on Train Set & Benchmarking on VALIDATION Set (Model Selection)")
    print("=" * 75)
    registry = get_model_registry()
    val_records = []
    trained_models = {}
    
    for name, model in registry.items():
        # Fit exclusively on training data
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Evaluate on validation set for model selection
        y_val_pred = np.maximum(0, model.predict(X_val))
        val_m = compute_metrics(y_val, y_val_pred)
        val_m["Model"] = name
        val_records.append(val_m)
        
        print(f"-> [Validation] {name:<22} | MAE: ${val_m['MAE ($)']:>8,.2f} | RMSE: ${val_m['RMSE ($)']:>8,.2f} | WAPE: {val_m['WAPE (%)']:>5.2f}% | R2: {val_m['R2 Score']:>6.4f}")

    val_df = pd.DataFrame(val_records)[["Model", "MAE ($)", "RMSE ($)", "MAPE (%)", "WAPE (%)", "R2 Score"]]
    val_df = val_df.sort_values(by="WAPE (%)").reset_index(drop=True)
    val_metrics_path = os.path.join(metrics_dir, "validation_model_comparison.csv")
    val_df.to_csv(val_metrics_path, index=False)
    print(f"\nValidation comparison table saved to: {val_metrics_path}")
    
    # Select Champion Model ONLY from Validation WAPE
    champion_name = val_df.iloc[0]["Model"]
    champion_val_wape = val_df.iloc[0]["WAPE (%)"]
    champion_val_rmse = val_df.iloc[0]["RMSE ($)"]
    champion_model = trained_models[champion_name]
    
    print("\n" + "=" * 75)
    print(f"[CHAMPION MODEL SELECTED]: {champion_name} (Validation WAPE: {champion_val_wape:.2f}%)")
    print("=" * 75)
    
    print("\n" + "=" * 75)
    print("STEP 5: Unbiased Final Evaluation on Untouched Out-of-Time TEST Set")
    print("=" * 75)
    test_records = []
    test_predictions = {}
    
    for name, model in trained_models.items():
        y_test_pred = np.maximum(0, model.predict(X_test))
        test_predictions[name] = y_test_pred
        test_m = compute_metrics(y_test, y_test_pred)
        test_m["Model"] = name
        test_m["Is_Champion"] = (name == champion_name)
        test_records.append(test_m)
        
        marker = "[CHAMPION]" if name == champion_name else "[Candidate]"
        print(f"-> {marker:<12} {name:<20} | MAE: ${test_m['MAE ($)']:>8,.2f} | RMSE: ${test_m['RMSE ($)']:>8,.2f} | WAPE: {test_m['WAPE (%)']:>5.2f}% | R2: {test_m['R2 Score']:>6.4f}")

    test_df = pd.DataFrame(test_records)[["Model", "Is_Champion", "MAE ($)", "RMSE ($)", "MAPE (%)", "WAPE (%)", "R2 Score"]]
    test_df = test_df.sort_values(by="WAPE (%)").reset_index(drop=True)
    test_metrics_path = os.path.join(metrics_dir, "test_model_evaluation.csv")
    test_df.to_csv(test_metrics_path, index=False)
    
    # Save compatibility file model_comparison.csv
    val_df.to_csv(os.path.join(metrics_dir, "model_comparison.csv"), index=False)
    
    # Serialize champion model
    champion_save_path = os.path.join(models_dir, "best_forecasting_model.pkl")
    joblib.dump({
        "model": champion_model,
        "feature_cols": feature_cols,
        "model_name": champion_name,
        "val_rmse": champion_val_rmse,
        "val_wape": champion_val_wape
    }, champion_save_path)
    print(f"\nChampion model serialized to: {champion_save_path}")
    
    print("\n" + "=" * 75)
    print("STEP 6: Diagnostic Visualizations & Statistical Uncertainty Bands")
    print("=" * 75)
    df_test_eval = df_test.copy()
    df_test_eval["Predicted_Sales"] = test_predictions[champion_name]
    
    # Plot forecast vs actual with statistical 95% interval derived from validation RMSE
    plot_forecast_vs_actual(
        df_test_eval,
        os.path.join(figures_dir, "actual_vs_predicted_comparison.png"),
        model_name=champion_name,
        val_rmse=champion_val_rmse
    )
    plot_residual_diagnostics(df_test_eval, os.path.join(figures_dir, "residual_diagnostics.png"))
    plot_inventory_recommendations(df_test_eval, os.path.join(figures_dir, "inventory_reorder_recommendation.png"))
    
    # Feature importance
    if hasattr(champion_model, "feature_importances_"):
        fi_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": champion_model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        plot_feature_importance(fi_df, os.path.join(figures_dir, "feature_importance.png"))
        fi_df.to_csv(os.path.join(metrics_dir, "feature_importance.csv"), index=False)
    
    # Sliced department error breakdown on test set
    dept_error_df = analyze_error_by_segment(df_test_eval)
    dept_error_df.to_csv(os.path.join(metrics_dir, "department_error_breakdown.csv"), index=False)
    
    # Write Updated Business Report
    report_path = os.path.join(reports_dir, "forecasting_business_report.md")
    champ_test_row = test_df[test_df["Model"] == champion_name].iloc[0]
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"""# RetailPulse Forecast — Sales & Demand Forecasting Executive Report

## Executive Summary
This report presents the rigorous empirical validation of the **RetailPulse Forecast** machine learning demand planning engine.
To adhere strictly to professional time-series standards:
1. **70% Training Set**: Used strictly to fit model weights.
2. **15% Validation Set**: Used for candidate model benchmarking and champion selection.
3. **15% Out-of-Time Test Set**: Reserved for a single, unbiased final evaluation of the selected champion model.

The champion model selected on the validation set is **{champion_name}** (**{champion_val_wape:.2f}% Validation WAPE**).
On the untouched out-of-time test horizon, **{champion_name}** achieved an unbiased **WAPE of {champ_test_row['WAPE (%)']:.2f}%** and an **R² score of {champ_test_row['R2 Score']:.4f}**.

---

## 1. Validation Set Benchmarking (Model Selection Horizon)
*Evaluated on the chronological validation set (15% split, 600 records) to select the production champion:*

| Model | MAE ($) | RMSE ($) | MAPE (%) | WAPE (%) | R² Score | Selection Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
""" + "\n".join([f"| {r['Model']} | ${r['MAE ($)']:,.2f} | ${r['RMSE ($)']:,.2f} | {r['MAPE (%)']:.2f}% | {r['WAPE (%)']:.2f}% | {r['R2 Score']:.4f} | {'Champion' if r['Model']==champion_name else 'Candidate'} |" for _, r in val_df.iterrows()]) + f"""

---

## 2. Final Out-of-Time Test Evaluation (Unbiased Horizon)
*Evaluated on the completely untouched chronological test set (15% split, 600 records):*

| Model | Champion | MAE ($) | RMSE ($) | MAPE (%) | WAPE (%) | R² Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
""" + "\n".join([f"| {r['Model']} | {'Yes' if r['Is_Champion'] else 'No'} | ${r['MAE ($)']:,.2f} | ${r['RMSE ($)']:,.2f} | {r['MAPE (%)']:.2f}% | {r['WAPE (%)']:.2f}% | {r['R2 Score']:.4f} |" for _, r in test_df.iterrows()]) + f"""

---

## 3. Uncertainty Quantification & Inventory Policy
- **Statistical Prediction Interval**: The forecast band plotted on the test horizon is derived from the empirical validation residual standard error $\\sigma_{{\\text{{val}}}} = \\${champion_val_rmse:,.2f}$, constructing a statistically grounded $95\\%$ prediction interval ($\\hat{{y}} \\pm 1.96 \\cdot \\sigma_{{\\text{{val}}}}$).
- **Safety Stock Formula**: Calculated at a $95\\%$ Service Level Agreement ($Z = 1.645$) using residual forecast error standard deviation across a 2-week supplier lead time:
  $$\\text{{Safety Stock}} = Z_{{0.95}} \\times \\text{{RMSE}}_{{\\text{{residual}}}} \\times \\sqrt{{L}}$$
- **Reorder Point (ROP)**:
  $$\\text{{ROP}} = (\\text{{Forecasted Weekly Demand}} \\times L) + \\text{{Safety Stock}}$$
""")
    print(f"Executive Report generated: {report_path}")
    print("\nPipeline execution completed successfully with rigorous methodology!")

if __name__ == "__main__":
    run_pipeline()
