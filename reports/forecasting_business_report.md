# RetailPulse Forecast — Sales & Demand Forecasting Executive Report

## Executive Summary
This report presents the empirical validation of the **RetailPulse Forecast** machine learning demand planning engine.
The evaluation framework adheres strictly to statistical time-series best practices:
1. **70% Training Set**: Used strictly to fit model weights.
2. **15% Validation Set**: Used for candidate model benchmarking and champion selection.
3. **15% Out-of-Time Test Set**: Reserved for a single, unbiased final evaluation of the selected champion model.

The champion model selected strictly on validation performance is **XGBoost Regressor** (**5.65% Validation WAPE**).
On the untouched out-of-time test horizon, **XGBoost Regressor** achieved an unbiased **WAPE of 4.55%** and an **R² score of 0.9928**.

---

## 1. Validation Set Benchmarking (Model Selection Horizon)
*Evaluated on the chronological validation set (15% split, 600 records) to select the production champion:*

| Model | MAE ($) | RMSE ($) | MAPE (%) | WAPE (%) | R² Score | Selection Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| XGBoost Regressor | $1,887.08 | $3,816.87 | 5.49% | 5.65% | 0.9846 | Champion |
| Random Forest | $1,897.25 | $3,531.83 | 5.54% | 5.68% | 0.9868 | Candidate |
| Seasonal Lag-52 | $1,986.59 | $3,433.09 | 5.89% | 5.94% | 0.9876 | Candidate |
| LightGBM Regressor | $2,083.72 | $4,576.86 | 5.71% | 6.23% | 0.9779 | Candidate |
| Ridge Regression | $2,846.07 | $4,630.25 | 11.22% | 8.51% | 0.9774 | Candidate |
| Naive Lag-1 | $7,404.30 | $16,884.41 | 21.58% | 22.15% | 0.6992 | Candidate |
| Rolling 4W Moving Avg | $9,542.06 | $16,859.60 | 28.85% | 28.54% | 0.7000 | Candidate |

---

## 2. Final Out-of-Time Test Evaluation (Unbiased Horizon)
*Evaluated on the completely untouched chronological test set (15% split, 600 records):*

| Model | Champion | MAE ($) | RMSE ($) | MAPE (%) | WAPE (%) | R² Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Random Forest | No | $1,369.86 | $2,161.70 | 4.80% | 4.50% | 0.9923 |
| XGBoost Regressor | Yes | $1,384.25 | $2,098.71 | 4.86% | 4.55% | 0.9928 |
| LightGBM Regressor | No | $1,519.41 | $2,483.27 | 5.25% | 4.99% | 0.9899 |
| Ridge Regression | No | $1,650.42 | $2,327.22 | 7.88% | 5.43% | 0.9911 |
| Seasonal Lag-52 | No | $1,738.85 | $2,661.81 | 5.95% | 5.72% | 0.9884 |
| Rolling 4W Moving Avg | No | $1,996.76 | $3,474.41 | 6.61% | 6.56% | 0.9802 |
| Naive Lag-1 | No | $2,417.93 | $4,075.67 | 8.10% | 7.95% | 0.9727 |

---

## 3. Mathematical Formulation of Uncertainty & Inventory Buffer

### A. Aggregate Forecast Prediction Interval
The plotted demand forecast is an aggregation across all 30 store-department series: $\hat{Y}_{\text{agg}, t} = \sum_{i=1}^{30} \hat{y}_{i, t}$.
The uncertainty band is computed directly from the empirical sample standard deviation of aggregate weekly forecast residuals measured on the validation set ($\sigma_{\text{agg}, \text{val}} = \$33,820.44$):
$$\text{Lower Bound} = \max\left(0, \hat{Y}_{\text{agg}, t} - 1.96 \cdot \sigma_{\text{agg}, \text{val}}\right)$$
$$\text{Upper Bound} = \hat{Y}_{\text{agg}, t} + 1.96 \cdot \sigma_{\text{agg}, \text{val}}$$
This constitutes an empirical $95\%$ prediction interval under approximately normal aggregate forecast residuals.

### B. Lead-Time Inventory Safety Stock & Reorder Point (ROP)
Under standard supply chain inventory theory (Silver-Pyke-Peterson inventory model), demand uncertainty accumulates over the replenishment lead time ($L = 2\text{ weeks}$).
- **$\sigma_{\text{weekly}}$**: Sample standard deviation of weekly forecast errors for each individual department ($USD$).
- **Lead-Time Uncertainty Scaling**: For independent weekly errors over $L$ weeks, the variance scales as $\text{Var}(\text{Lead Time Error}) = L \cdot \sigma_{\text{weekly}}^2$, so the standard deviation of lead-time demand error is $\sigma_L = \sigma_{\text{weekly}} \cdot \sqrt{L}$.
- **Safety Stock at 95% Service Level** ($Z_{0.95} = 1.645$):
  $$\text{Safety Stock} = Z_{0.95} \times \sigma_{\text{weekly}} \times \sqrt{L} = 1.645 \times \sigma_{\text{weekly}} \times \sqrt{2}$$
- **Dynamic Reorder Point (ROP)**:
  $$\text{ROP} = (\text{Forecasted Weekly Demand} \times L) + \text{Safety Stock}$$
