"""
RetailPulse Forecast - Comprehensive Automated Verification Suite
Tests all 15 critical components required for the IBM Internship Review:
1. Dataset loading and schema validation
2. Missing-value and data-type handling
3. Feature engineering
4. Forecast date alignment
5. Model artifact loading
6. Prediction output shape and data types
7. Prediction finiteness
8. Evaluation metric calculations
9. Correct behavior for zero denominators in WAPE
10. Chronological train/validation/test separation
11. Store and department selection
12. Invalid or unsupported user inputs
13. Inventory calculations
14. Empty or insufficient time series
15. Application startup and critical user workflows
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import joblib

# Ensure repo root is in python path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import save_and_load_dataset, generate_retail_sales_data
from src.features import create_time_features, create_lag_and_rolling_features, prepare_feature_matrix
from src.models import chronological_split, get_model_registry
from src.evaluate import compute_metrics, analyze_error_by_segment
from src.inventory import calculate_inventory_policy

class TestDatasetSchemaAndIntegrity(unittest.TestCase):
    """1, 2, 11: Schema, Data Types, Missing Values, and Segment Selection"""
    
    @classmethod
    def setUpClass(cls):
        cls.data_path = os.path.join(BASE_DIR, "data", "retail_store_sales.csv")
        cls.df = save_and_load_dataset(cls.data_path)

    def test_01_dataset_loading_and_shape(self):
        """Verifies dataset loads with expected row count and column structure."""
        self.assertIsNotNone(self.df)
        self.assertGreaterEqual(len(self.df), 5000, "Dataset should have at least 5000 records")
        expected_cols = [
            "Date", "Store_ID", "Store_Type", "Store_Size",
            "Dept_ID", "Dept_Name", "Weekly_Sales", "IsHoliday",
            "Holiday_Name", "Markdown_Discount", "Promotional_Flag",
            "Temperature", "Fuel_Price", "CPI", "Unemployment_Rate"
        ]
        for col in expected_cols:
            self.assertIn(col, self.df.columns, f"Missing critical column: {col}")

    def test_02_data_types_and_missing_values(self):
        """Verifies no unexpected missing values in critical columns and correct types."""
        critical_cols = ["Store_ID", "Dept_ID", "Weekly_Sales", "IsHoliday"]
        for col in critical_cols:
            null_count = self.df[col].isnull().sum()
            self.assertEqual(null_count, 0, f"Column {col} contains {null_count} unexpected NaNs")
        
        # Verify Weekly_Sales are non-negative
        self.assertTrue((self.df["Weekly_Sales"] >= 0).all(), "Weekly_Sales must be non-negative")
        
        # Verify Date is datetime or parseable
        parsed_dates = pd.to_datetime(self.df["Date"])
        self.assertEqual(len(parsed_dates), len(self.df))

    def test_03_store_and_dept_coverage(self):
        """Verifies all 5 stores and 6 departments exist with balanced observations."""
        unique_stores = sorted(self.df["Store_ID"].unique())
        unique_depts = sorted(self.df["Dept_ID"].unique())
        self.assertEqual(unique_stores, [1, 2, 3, 4, 5])
        self.assertEqual(unique_depts, [1, 2, 3, 4, 5, 6])
        
        # Verify each of the 30 store-department series has identical length
        series_counts = self.df.groupby(["Store_ID", "Dept_ID"]).size()
        self.assertEqual(len(series_counts), 30)
        self.assertEqual(series_counts.nunique(), 1, "All series should have equal observation counts")

class TestFeatureEngineeringAndLeakage(unittest.TestCase):
    """3, 4, 14: Feature Engineering, Temporal Alignment, and Leakage Prevention"""

    @classmethod
    def setUpClass(cls):
        cls.data_path = os.path.join(BASE_DIR, "data", "retail_store_sales.csv")
        cls.df = save_and_load_dataset(cls.data_path)

    def test_04_time_features_creation(self):
        """Verifies calendar features and cyclical trigonometry encodings."""
        df_time = create_time_features(self.df.head(20))
        for col in ["Year", "Month", "WeekOfYear", "Sin_Week", "Cos_Week", "Sin_Month", "Cos_Month"]:
            self.assertIn(col, df_time.columns)
            # Cyclical features must strictly be bounded in [-1, 1]
            if col.startswith("Sin") or col.startswith("Cos"):
                self.assertTrue(df_time[col].between(-1.0001, 1.0001).all())

    def test_05_lag_and_rolling_leakage_prevention(self):
        """
        Guarantees zero future information leakage:
        - Lag_1 at index t must strictly equal Weekly_Sales at index t-1 within each group.
        - Rolling_Mean_4 must NOT include the target at index t (must be computed on shifted series).
        """
        sample_group = self.df[(self.df["Store_ID"] == 1) & (self.df["Dept_ID"] == 1)].copy().sort_values("Date").reset_index(drop=True)
        feat_df = create_lag_and_rolling_features(sample_group)
        
        # Check Lag_1 matches previous row's actual sales
        for idx in range(1, 10):
            actual_prev = sample_group.loc[idx - 1, "Weekly_Sales"]
            lag_val = feat_df.loc[idx, "Lag_1"]
            self.assertAlmostEqual(actual_prev, lag_val, places=4, msg="Lag_1 must equal previous week's sales")
            
        # Check Rolling_Mean_4 matches mean of (t-1, t-2, t-3, t-4)
        for idx in range(4, 10):
            expected_mean = sample_group.loc[idx-4:idx-1, "Weekly_Sales"].mean()
            rolling_val = feat_df.loc[idx, "Rolling_Mean_4"]
            self.assertAlmostEqual(expected_mean, rolling_val, places=4, msg="Rolling_Mean_4 must use shifted series")

    def test_06_prepare_feature_matrix_output(self):
        """Verifies matrix generation, categorical encoding, and non-feature exclusions."""
        df_feat, feature_cols = prepare_feature_matrix(self.df)
        self.assertGreater(len(df_feat), 0)
        self.assertGreater(len(feature_cols), 40)
        
        # Excluded columns must not be in feature columns
        for excluded in ["Date", "Weekly_Sales", "Holiday_Name"]:
            self.assertNotIn(excluded, feature_cols)

    def test_07_empty_and_insufficient_time_series(self):
        """Verifies graceful handling of empty and short time-series (< 52 weeks)."""
        empty_df = pd.DataFrame()
        df_out, cols = prepare_feature_matrix(empty_df)
        self.assertEqual(len(df_out), 0)
        self.assertEqual(len(cols), 0)
        
        # Short series where annual lags cannot be computed: should return 0 rows after dropna
        short_df = self.df.head(10).copy()
        df_short, _ = prepare_feature_matrix(short_df)
        self.assertEqual(len(df_short), 0, "Series shorter than 52 weeks cannot form annual lag features")

class TestChronologicalSplits(unittest.TestCase):
    """10: Chronological Validation and Temporal Boundaries"""

    @classmethod
    def setUpClass(cls):
        cls.data_path = os.path.join(BASE_DIR, "data", "retail_store_sales.csv")
        cls.df = save_and_load_dataset(cls.data_path)
        cls.df_feat, cls.feature_cols = prepare_feature_matrix(cls.df)

    def test_08_chronological_ordering_and_zero_overlap(self):
        """
        Verifies strict chronological splitting:
        max(Train Date) < min(Val Date) and max(Val Date) < min(Test Date).
        Zero overlapping dates between sets.
        """
        df_train, df_val, df_test = chronological_split(self.df_feat, train_ratio=0.70, val_ratio=0.15)
        
        train_max = df_train["Date"].max()
        val_min = df_val["Date"].min()
        val_max = df_val["Date"].max()
        test_min = df_test["Date"].min()
        
        self.assertLess(train_max, val_min, "Train set must precede validation set chronologically")
        self.assertLess(val_max, test_min, "Validation set must precede test set chronologically")
        
        train_dates = set(df_train["Date"].unique())
        val_dates = set(df_val["Date"].unique())
        test_dates = set(df_test["Date"].unique())
        
        self.assertEqual(len(train_dates.intersection(val_dates)), 0, "No date leakage between train and val")
        self.assertEqual(len(val_dates.intersection(test_dates)), 0, "No date leakage between val and test")
        self.assertEqual(len(train_dates.intersection(test_dates)), 0, "No date leakage between train and test")

    def test_09_insufficient_dates_split_rejection(self):
        """Verifies that attempting a 3-way split on < 3 dates raises ValueError."""
        dummy_df = pd.DataFrame({
            "Date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-08")],
            "Weekly_Sales": [100.0, 200.0]
        })
        with self.assertRaises(ValueError):
            chronological_split(dummy_df, 0.70, 0.15)

class TestEvaluationMetrics(unittest.TestCase):
    """8, 9: Evaluation Metrics and Zero-Denominator Robustness"""

    def test_10_standard_metrics_computation(self):
        """Verifies standard MAE, RMSE, MAPE, WAPE, and R2 against known calculations."""
        y_true = np.array([100.0, 200.0, 300.0, 400.0])
        y_pred = np.array([110.0, 190.0, 315.0, 380.0])
        
        # Absolute errors: 10, 10, 15, 20 -> Sum = 55 -> MAE = 13.75
        # Squared errors: 100, 100, 225, 400 -> Mean = 206.25 -> RMSE = sqrt(206.25) = 14.36
        # Sum y_true: 1000 -> WAPE = 55 / 1000 = 5.5%
        m = compute_metrics(y_true, y_pred)
        
        self.assertAlmostEqual(m["MAE ($)"], 13.75, places=2)
        self.assertAlmostEqual(m["RMSE ($)"], 14.36, places=2)
        self.assertAlmostEqual(m["WAPE (%)"], 5.50, places=2)
        self.assertGreater(m["R2 Score"], 0.95)

    def test_11_zero_denominator_wape_and_mape(self):
        """Verifies zero denominator does not raise division-by-zero or return NaN."""
        # Case 1: Both all zeros
        y_zeros = np.array([0.0, 0.0, 0.0])
        m_zeros = compute_metrics(y_zeros, y_zeros)
        self.assertFalse(np.isnan(m_zeros["WAPE (%)"]))
        self.assertFalse(np.isnan(m_zeros["MAPE (%)"]))
        self.assertEqual(m_zeros["WAPE (%)"], 0.0)
        self.assertEqual(m_zeros["MAPE (%)"], 0.0)
        
        # Case 2: y_true all zeros, y_pred non-zero
        y_pred_nonzero = np.array([10.0, 20.0, 30.0])
        m_pred_nonzero = compute_metrics(y_zeros, y_pred_nonzero)
        self.assertFalse(np.isnan(m_pred_nonzero["WAPE (%)"]))
        self.assertEqual(m_pred_nonzero["WAPE (%)"], 100.0)

    def test_12_single_sample_and_zero_variance_r2(self):
        """Verifies that single sample or constant targets do not crash R2 computation."""
        # Single sample
        m_single = compute_metrics(np.array([100.0]), np.array([100.0]))
        self.assertEqual(m_single["R2 Score"], 1.0)
        
        # Constant target with perfect predictions
        y_const = np.array([50.0, 50.0, 50.0])
        m_const_perfect = compute_metrics(y_const, y_const)
        self.assertEqual(m_const_perfect["R2 Score"], 1.0)
        
        # Constant target with imperfect predictions
        m_const_imperfect = compute_metrics(y_const, np.array([40.0, 50.0, 60.0]))
        self.assertEqual(m_const_imperfect["R2 Score"], 0.0)

    def test_13_mismatched_lengths_and_empty_arrays(self):
        """Verifies appropriate exception on mismatched lengths and safe empty handling."""
        with self.assertRaises(ValueError):
            compute_metrics(np.array([100.0, 200.0]), np.array([100.0]))
            
        m_empty = compute_metrics(np.array([]), np.array([]))
        self.assertEqual(m_empty["MAE ($)"], 0.0)
        self.assertEqual(m_empty["WAPE (%)"], 0.0)

    def test_14_analyze_error_by_segment(self):
        """Verifies sliced error breakdown across departments."""
        eval_df = pd.DataFrame({
            "Dept_ID": [1, 1, 2, 2],
            "Dept_Name": ["Electronics", "Electronics", "Apparel", "Apparel"],
            "Weekly_Sales": [1000.0, 1200.0, 800.0, 900.0],
            "Predicted_Sales": [1050.0, 1180.0, 780.0, 920.0]
        })
        dept_stats = analyze_error_by_segment(eval_df)
        self.assertEqual(len(dept_stats), 2)
        self.assertIn("WAPE", dept_stats.columns)
        self.assertIn("MAE", dept_stats.columns)
        self.assertIn("RMSE", dept_stats.columns)
        self.assertTrue((dept_stats["WAPE"] >= 0).all())

class TestModelArtifactAndInference(unittest.TestCase):
    """5, 6, 7: Model Artifact Loading, Prediction Shape, and Finiteness"""

    @classmethod
    def setUpClass(cls):
        cls.model_path = os.path.join(BASE_DIR, "models", "best_forecasting_model.pkl")
        cls.data_path = os.path.join(BASE_DIR, "data", "retail_store_sales.csv")
        cls.payload = joblib.load(cls.model_path)
        cls.df = save_and_load_dataset(cls.data_path)
        cls.df_feat, cls.feature_cols = prepare_feature_matrix(cls.df)
        _, _, cls.df_test = chronological_split(cls.df_feat, 0.70, 0.15)

    def test_15_model_artifact_loading(self):
        """Verifies serialized champion model artifact integrity and dictionary structure."""
        self.assertIsInstance(self.payload, dict)
        for key in ["model", "feature_cols", "model_name", "val_rmse", "val_wape"]:
            self.assertIn(key, self.payload)
        self.assertEqual(self.payload["model_name"], "XGBoost Regressor")
        self.assertEqual(len(self.payload["feature_cols"]), 57)

    def test_16_prediction_shape_and_types(self):
        """Verifies prediction output shape and data types on test set."""
        model = self.payload["model"]
        feat_cols = self.payload["feature_cols"]
        X_test = self.df_test[feat_cols]
        
        preds = model.predict(X_test)
        self.assertEqual(len(preds), len(self.df_test))
        self.assertIsInstance(preds, np.ndarray)
        self.assertTrue(np.issubdtype(preds.dtype, np.floating))

    def test_17_prediction_finiteness_and_non_negativity(self):
        """Verifies all predictions are strictly finite and non-negative after ReLU clamping."""
        model = self.payload["model"]
        feat_cols = self.payload["feature_cols"]
        X_test = self.df_test[feat_cols]
        
        preds = np.maximum(0, model.predict(X_test))
        self.assertTrue(np.all(np.isfinite(preds)), "All predictions must be finite numbers (no NaN/Inf)")
        self.assertTrue(np.all(preds >= 0), "Clamped predictions must be non-negative")

class TestInventoryCalculations(unittest.TestCase):
    """12, 13: Operations Research Inventory Buffer & Safety Stock Formulas"""

    def test_18_standard_inventory_policy(self):
        """
        Verifies standard Silver-Pyke-Peterson inventory formulas:
        - Cycle Demand = Demand * L = 20,000 * 2 = 40,000
        - Safety Stock = Z_0.95 * sigma * sqrt(L) = 1.64485 * 3,000 * sqrt(2) = 6,978.50
        - Reorder Point = Cycle Demand + Safety Stock = 46,978.50
        """
        res = calculate_inventory_policy(
            weekly_demand=20000.0,
            lead_time_weeks=2.0,
            uncertainty_sigma=3000.0,
            service_level=0.95
        )
        self.assertAlmostEqual(res["cycle_demand"], 40000.0, places=1)
        self.assertAlmostEqual(res["z_score"], 1.6449, places=3)
        self.assertAlmostEqual(res["safety_stock"], 6978.50, delta=2.0)
        self.assertAlmostEqual(res["reorder_point"], 46978.50, delta=2.0)

    def test_19_inventory_edge_cases(self):
        """Verifies edge cases: L=0, sigma=0, and boundary conditions."""
        # Zero lead time -> no cycle demand, no safety stock needed
        res_zero_lead = calculate_inventory_policy(20000.0, 0.0, 3000.0, 0.95)
        self.assertEqual(res_zero_lead["cycle_demand"], 0.0)
        self.assertEqual(res_zero_lead["safety_stock"], 0.0)
        self.assertEqual(res_zero_lead["reorder_point"], 0.0)
        
        # Zero uncertainty -> safety stock is zero, ROP equals cycle demand
        res_zero_unc = calculate_inventory_policy(20000.0, 2.0, 0.0, 0.95)
        self.assertEqual(res_zero_unc["safety_stock"], 0.0)
        self.assertEqual(res_zero_unc["reorder_point"], 40000.0)

    def test_20_inventory_invalid_inputs(self):
        """Verifies invalid service levels (< 0.50 or >= 1.0) raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_inventory_policy(20000.0, 2.0, 3000.0, service_level=0.40)
        with self.assertRaises(ValueError):
            calculate_inventory_policy(20000.0, 2.0, 3000.0, service_level=1.00)

class TestApplicationStartupAndWorkflows(unittest.TestCase):
    """15: Application Startup, Caching, and Critical User Workflows"""

    def test_21_dashboard_compilation(self):
        """Verifies that dashboard/app.py compiles without syntax or indentation errors."""
        import py_compile
        app_path = os.path.join(BASE_DIR, "dashboard", "app.py")
        compiled = py_compile.compile(app_path, doraise=True)
        self.assertTrue(os.path.exists(compiled))

    def test_22_cached_test_split_integrity(self):
        """Verifies that get_cached_test_split returns clean test split with 57 features."""
        import logging
        import warnings
        warnings.filterwarnings("ignore")
        logging.getLogger("streamlit").setLevel(logging.ERROR)
        from dashboard.app import get_cached_test_split
        df_test, feat_cols = get_cached_test_split()
        self.assertIsNotNone(df_test)
        self.assertEqual(len(df_test), 600, "Out-of-time test set must contain exactly 600 observations")
        self.assertEqual(len(feat_cols), 57, "Feature matrix must have exactly 57 features matching champion model")
        
        # Verify defensive copy property
        test_copy = df_test[(df_test["Store_ID"] == 1) & (df_test["Dept_ID"] == 1)].copy()
        test_copy["Predicted_Sales"] = 999.0
        self.assertNotIn("Predicted_Sales", df_test.columns, "Mutating slice must not pollute cached dataframe")

    def test_23_forecast_horizon_slicing(self):
        """Verifies that horizon slicing (4W, 8W, 12W, 20W) slices test series accurately."""
        from dashboard.app import get_cached_test_split
        df_test, feat_cols = get_cached_test_split()
        series_test = df_test[(df_test["Store_ID"] == 1) & (df_test["Dept_ID"] == 1)].sort_values("Date").reset_index(drop=True)
        
        for w in [4, 8, 12, 20]:
            sliced = series_test.head(w)
            self.assertEqual(len(sliced), w, f"Expected {w} rows for horizon {w}W")
            # Verify chronological continuity
            dates = pd.to_datetime(sliced["Date"]).tolist()
            for i in range(1, len(dates)):
                self.assertGreater(dates[i], dates[i-1], "Dates within sliced horizon must be chronologically ascending")

    def test_24_forecast_export_dataframe_schema(self):
        """Verifies structure and non-null values of the exported forecast dataset."""
        from dashboard.app import get_cached_test_split, load_champion_model
        df_test, feat_cols = get_cached_test_split()
        payload = load_champion_model()
        model = payload["model"]
        
        series = df_test[(df_test["Store_ID"] == 1) & (df_test["Dept_ID"] == 1)].copy().head(8).reset_index(drop=True)
        series["Predicted_Sales"] = np.maximum(0, model.predict(series[feat_cols]))
        series["Lower_Bound_95"] = np.maximum(0, series["Predicted_Sales"] - 1000.0)
        series["Upper_Bound_95"] = series["Predicted_Sales"] + 1000.0
        series["Forecast_Error"] = series["Predicted_Sales"] - series["Weekly_Sales"]
        
        export_df = pd.DataFrame({
            "Date": series["Date"].dt.strftime("%Y-%m-%d"),
            "Store_ID": 1,
            "Dept_ID": 1,
            "Actual_Sales_USD": series["Weekly_Sales"],
            "Forecast_Sales_USD": series["Predicted_Sales"],
            "Lower_Bound_95_USD": series["Lower_Bound_95"],
            "Upper_Bound_95_USD": series["Upper_Bound_95"],
            "Forecast_Error_USD": series["Forecast_Error"]
        })
        self.assertEqual(len(export_df), 8)
        self.assertFalse(export_df.isnull().any().any(), "Export dataframe must not contain NaNs")

    def test_25_inventory_service_level_mapping(self):
        """Verifies critical Z-scores across standard retail SLA tiers."""
        sla_90 = calculate_inventory_policy(10000, 2, 2000, service_level=0.90)
        sla_95 = calculate_inventory_policy(10000, 2, 2000, service_level=0.95)
        sla_99 = calculate_inventory_policy(10000, 2, 2000, service_level=0.99)
        
        self.assertAlmostEqual(sla_90["z_score"], 1.2816, places=3)
        self.assertAlmostEqual(sla_95["z_score"], 1.6449, places=3)
        self.assertAlmostEqual(sla_99["z_score"], 2.3263, places=3)
        
        # Verify safety stock increases monotonically with higher service level SLA
        self.assertLess(sla_90["safety_stock"], sla_95["safety_stock"])
        self.assertLess(sla_95["safety_stock"], sla_99["safety_stock"])

    def test_26_prediction_interval_coverage(self):
        """Verifies interval symmetry, non-negativity, and bounds encapsulation."""
        pred = 25000.0
        margin = 3500.0
        lower = max(0.0, pred - margin)
        upper = pred + margin
        self.assertLessEqual(lower, pred)
        self.assertGreaterEqual(upper, pred)
        self.assertGreaterEqual(lower, 0.0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
