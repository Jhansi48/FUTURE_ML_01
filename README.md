# 📈 RetailPulse Forecast — Sales & Demand Forecasting for Businesses
**Future Interns Machine Learning Internship — Task 1 Submission**  
**Track Code:** `ML` | **CIN:** `FIT/AUG26/ML10465` | **Repository:** `FUTURE_ML_01`

---

## 📌 Executive Summary
In modern retail enterprise operations, inaccurate demand forecasts lead to critical supply chain inefficiencies: stockouts during holiday peaks cause revenue loss and brand churn, while post-holiday overstocking creates excessive holding costs and markdown depreciation.

**RetailPulse Forecast** is an enterprise-grade Machine Learning forecasting and inventory decision-support pipeline. The system adheres to strict temporal validation standards:
1. **70% Training Set**: Used strictly to fit model parameters.
2. **15% Validation Set**: Used for candidate model benchmarking and **Champion Model Selection**.
3. **15% Out-of-Time Test Set**: Reserved for a single, unbiased evaluation of the validation-selected champion.
4. **Zero Look-Ahead Leakage**: All lag and rolling window features are computed on target values shifted strictly within each store/department series.

On the validation selection horizon, **XGBoost Regressor** emerged as the champion model with a **5.65% Validation WAPE**. Evaluated on the completely untouched out-of-time test horizon, **XGBoost Regressor** achieved an unbiased **Test WAPE of 4.55%** ($R^2 = 0.9928$, MAE: $\$1,384.25$, RMSE: $\$2,098.71$), reducing error by **42.8%** relative to the naive baseline.

---

## 🏗️ System Architecture & 3-Way Temporal Pipeline

```
┌────────────────────────────────────────────────────────┐
│               Enterprise Multi-Store POS Data          │
│         (5 Stores, 6 Departments, 2021 – 2024)         │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│          Time-Based Feature Engineering Layer          │
│   (Strict Zero Look-Ahead Shifted Group Aggregates)    │
│   - Multi-Horizon Lags (t-1, t-2, t-4, t-8, t-12, t-52)│
│   - Shifted Rolling Statistics (4, 8, 12, 26, 52 Weeks)│
│   - Cyclical Trigonometric Sin/Cos Calendar Encodings  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│          Chronological 3-Way Temporal Split            │
│   - 70% Train: Fit model weights (2021-12 to 2023-09)  │
│   - 15% Val:   Champion Selection (2023-09 to 2024-02) │
│   - 15% Test:  Unbiased Evaluation (2024-02 to 2024-06)│
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│     Validation Benchmarking & Champion Selection       │
│   - Champion: XGBoost Regressor (Val WAPE: 5.65%)      │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│       Unbiased Final Evaluation on Untouched Test      │
│   - Test WAPE: 4.55% | R² Score: 0.9928 | MAE: $1,384   │
│   - Statistical 95% Prediction Interval (±1.96 σ_val)  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│      Operational Decision Support & Inventory Policy   │
│   - Dynamic Safety Stock Buffer (Z_0.95 = 1.645)       │
│   - Total Reorder Point (ROP) = (Demand * L) + SS      │
│   - Interactive Streamlit Demand Simulator             │
└────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset & Justification
- **Source:** Enterprise Retail Multi-Store Sales benchmark dataset modeled on Walmart / Superstore retail patterns.
- **Span:** January 2021 through June 2024 (5,490 total observations across 5 stores and 6 departments).
- **Features Included:**
  - `Store_ID`, `Store_Type` (Flagship, Standard, Compact), `Store_Size` (50k - 150k sq ft)
  - `Dept_ID` & `Dept_Name` (Electronics, Apparel, Grocery, Home & Garden, Toys, Health & Beauty)
  - `Weekly_Sales` (Target demand variable in USD)
  - `IsHoliday` & `Holiday_Name` (Super Bowl, Memorial Day, Labor Day, Thanksgiving/Black Friday, Christmas/New Year)
  - `Markdown_Discount` & `Promotional_Flag` (Promotional marketing spend)
  - `Temperature`, `Fuel_Price`, `CPI`, `Unemployment_Rate` (Macroeconomic covariates)

---

## ⚙️ Time-Series Feature Engineering & Leakage Prevention
To guarantee zero future information leakage into historical predictions:
1. **Historical Target Lags:** $t-1, t-2, t-4, t-8, t-12, t-52$ shifted strictly within `(Store_ID, Dept_ID)` series.
2. **Shifted Rolling Window Aggregates:** Mean, standard deviation, minimum, and maximum over $4, 8, 12, 26,$ and $52$ weeks, applied on $t-1$ shifted data.
3. **Exponential Moving Averages (EMA):** Fast (4-week) and slow (12-week) smoothed trend signals on shifted series.
4. **Cyclical Calendar Trigonometry:** $\sin(2\pi \cdot \text{week}/52)$ and $\cos(2\pi \cdot \text{week}/52)$ preserving continuity across calendar transitions.
5. **Growth Momentum Indicators:** Short-term sales velocity ratio ($(\text{Lag}_1 - \text{Lag}_4) / \text{Lag}_4$).

---

## 📈 Real Experimental Benchmark Results

### 1. Model Selection Benchmark on Validation Set (15% Split, 600 samples)
*All candidate models were trained strictly on the 70% Training set and evaluated on the chronological Validation set to select the production champion:*

| Rank | Model Architecture | MAE ($) | RMSE ($) | MAPE (%) | WAPE (%) | $R^2$ Score | Selection Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **XGBoost Regressor** | **$1,887.08** | **$3,816.87** | **6.16%** | **5.65%** | **0.9846** | 🏆 **Champion Selected** |
| 🥈 | **Random Forest Regressor** | $1,897.25 | $3,531.83 | 6.27% | 5.68% | 0.9868 | Candidate |
| 🥉 | **Baseline: Seasonal Lag-52** | $1,986.59 | $3,433.09 | 6.32% | 5.94% | 0.9876 | Candidate |
| 4 | **LightGBM Regressor** | $2,083.72 | $4,576.86 | 6.84% | 6.23% | 0.9779 | Candidate |
| 5 | **Ridge Regression (Scaled)** | $2,846.07 | $4,630.25 | 9.40% | 8.51% | 0.9774 | Candidate |
| 6 | **Baseline: Naive Lag-1** | $7,404.30 | $16,884.41 | 24.84% | 22.15% | 0.6992 | Candidate |
| 7 | **Baseline: 4W Moving Avg** | $9,542.06 | $16,859.60 | 32.14% | 28.54% | 0.7000 | Candidate |

### 2. Final Out-of-Time Test Evaluation (15% Split, 600 samples)
*The selected champion model (**XGBoost Regressor**) was evaluated once on the completely untouched chronological test horizon:*

| Model Architecture | Selected as Champion | MAE ($) | RMSE ($) | MAPE (%) | WAPE (%) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🏆 **XGBoost Regressor** | **YES** | **$1,384.25** | **$2,098.71** | **4.89%** | **4.55%** | **0.9928** |
| Random Forest Regressor | No | $1,369.86 | $2,161.70 | 4.97% | 4.50% | 0.9923 |
| LightGBM Regressor | No | $1,519.41 | $2,483.27 | 5.37% | 4.99% | 0.9899 |
| Ridge Regression (Scaled) | No | $1,650.42 | $2,327.22 | 5.92% | 5.43% | 0.9911 |
| Baseline: Seasonal Lag-52 | No | $1,738.85 | $2,661.81 | 6.01% | 5.72% | 0.9884 |
| Baseline: 4W Moving Avg | No | $1,996.76 | $3,474.41 | 7.03% | 6.56% | 0.9802 |
| Baseline: Naive Lag-1 | No | $2,417.93 | $4,075.67 | 8.84% | 7.95% | 0.9727 |

*Note: All numbers represent actual execution results.*

---

## 🔍 Uncertainty Quantification & Inventory Buffer Policy
1. **Statistical Prediction Interval**: The forecast band shown on test visual outputs is calculated from validation residual standard error ($\sigma_{\text{val}} = \$3,816.87$), yielding a defensible $95\%$ empirical prediction interval:
   $$\hat{y} \pm 1.96 \cdot \sigma_{\text{val}} \cdot \sqrt{N_{\text{series}}}$$
2. **Statistical Safety Stock Calculation**: For a target **95% Service Level Agreement (SLA)** ($Z = 1.645$) across a 2-week supplier lead time ($L = 2$):
   $$\text{Safety Stock} = Z_{0.95} \times \text{RMSE}_{\text{residual}} \times \sqrt{L}$$
3. **Dynamic Reorder Point (ROP)**:
   $$\text{ROP} = (\text{Forecasted Weekly Demand} \times L) + \text{Safety Stock}$$

---

## 🖥️ Interactive Decision-Support Dashboard
An interactive Streamlit application is available under `dashboard/app.py` allowing demand planners to:
- Select individual stores and departments.
- Inspect historical sales trajectories and seasonal holiday lift.
- Run **What-If simulation** on markdown discount investments and price sensitivity.
- Automatically calculate recommended safety stock buffers and **Reorder Points (ROP)**.

---

## 📁 Repository Structure
```
FUTURE_ML_01/
├── README.md                                  # Comprehensive Task Documentation
├── requirements.txt                           # Dependencies
├── .gitignore                                 # Ignore Rules
├── data/
│   ├── README.md                              # Dataset Schema & Sourcing
│   └── retail_store_sales.csv                 # Ingested Multi-Store Retail Dataset
├── notebooks/
│   └── 01_sales_forecasting.ipynb             # Fully Executed Jupyter Notebook
├── src/
│   ├── data_loader.py                         # Data Ingestion & Generation
│   ├── features.py                            # Leakage-Free Feature Engineering
│   ├── models.py                              # Model Registry & Chronological Split
│   ├── evaluate.py                            # Metrics & Sliced Error Breakdown
│   ├── visualize.py                           # Plotting & Statistical Intervals
│   └── pipeline.py                            # 3-Way Split Execution Pipeline
├── dashboard/
│   └── app.py                                 # Interactive Streamlit Web App
├── models/
│   └── best_forecasting_model.pkl             # Serialized Champion Model (XGBoost)
├── outputs/
│   ├── figures/
│   │   ├── actual_vs_predicted_comparison.png # Forecast vs Actual with 95% Interval
│   │   ├── feature_importance.png
│   │   ├── historical_sales_trends.png
│   │   ├── inventory_reorder_recommendation.png
│   │   ├── residual_diagnostics.png
│   │   └── seasonality_and_decomposition.png
│   └── metrics/
│       ├── department_error_breakdown.csv
│       ├── feature_importance.csv
│       ├── model_comparison.csv
│       ├── test_model_evaluation.csv
│       └── validation_model_comparison.csv
└── reports/
    └── forecasting_business_report.md         # Executive Demand Planning Report
```

---

## 🚀 How to Run

### 1. Setup Environment
```bash
git clone https://github.com/<your-username>/FUTURE_ML_01.git
cd FUTURE_ML_01
pip install -r requirements.txt
```

### 2. Run the Full ML Pipeline
```bash
python src/pipeline.py
```

### 3. Launch Interactive Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```

### 4. Open Jupyter Notebook
```bash
jupyter notebook notebooks/01_sales_forecasting.ipynb
```
