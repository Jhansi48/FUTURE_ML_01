"""
RetailPulse Forecast - Data Ingestion & Loading Module
Source: Retail Multi-Store Sales Dataset (Simulated Enterprise Benchmark modeled on Walmart Retail Sales Data)
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple

def generate_retail_sales_data(
    start_date: str = "2021-01-01",
    end_date: str = "2024-06-30",
    n_stores: int = 5,
    n_departments: int = 6,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic multi-store, multi-department weekly retail sales dataset
    featuring trend, annual & quarterly seasonality, holiday surges, promotional markdowns,
    and macroeconomic covariates (CPI, Fuel Price, Unemployment Rate).
    """
    np.random.seed(random_state)
    dates = pd.date_range(start=start_date, end=end_date, freq="W-FRI")
    n_weeks = len(dates)

    records = []

    # Store base profiles (Store Type A: Flagship, B: Standard, C: Compact)
    store_types = {1: ("A", 150000, 1.4), 2: ("A", 140000, 1.3), 3: ("B", 90000, 1.0), 4: ("B", 85000, 0.95), 5: ("C", 50000, 0.65)}
    
    # Department profiles (1: Electronics, 2: Apparel, 3: Grocery, 4: Home & Garden, 5: Toys, 6: Health & Beauty)
    dept_profiles = {
        1: ("Electronics", 25000, 1.2, 0.25),  # High holiday surge
        2: ("Apparel", 18000, 1.0, 0.15),
        3: ("Grocery", 42000, 0.8, 0.05),     # Low volatility, steady
        4: ("Home & Garden", 15000, 1.1, 0.10), # Spring/Summer peak
        5: ("Toys", 12000, 1.5, 0.35),         # Huge Q4 peak
        6: ("Health & Beauty", 16000, 0.9, 0.08)
    }

    # Macro trend indices across weeks
    t = np.linspace(0, 1, n_weeks)
    macro_cpi = 210 + 25 * t + np.random.normal(0, 0.5, n_weeks)
    macro_fuel = 3.10 + 0.9 * np.sin(2 * np.pi * t * 2) + np.random.normal(0, 0.08, n_weeks)
    macro_unemp = 7.5 - 2.2 * t + np.random.normal(0, 0.1, n_weeks)

    # US Federal Holiday calendar weeks (Week numbers approx)
    for store_id, (stype, s_size, s_mult) in store_types.items():
        for dept_id, (dname, d_base, d_vol, d_surge) in dept_profiles.items():
            base_sales = d_base * s_mult * (s_size / 100000.0)
            
            for i, date in enumerate(dates):
                week_num = date.isocalendar().week
                month = date.month
                
                # Annual seasonality component
                annual_cycle = np.sin(2 * np.pi * (week_num - 10) / 52.0)
                
                # Department specific seasonality
                if dept_id == 5: # Toys - Massive Q4 boost
                    seasonal_factor = 1.0 + (0.8 if month in [11, 12] else -0.15)
                elif dept_id == 4: # Home & Garden - Spring boost
                    seasonal_factor = 1.0 + (0.4 if month in [4, 5, 6] else -0.2)
                elif dept_id == 1: # Electronics - Q4 & Back-to-school
                    seasonal_factor = 1.0 + (0.6 if month in [11, 12] else (0.2 if month in [8, 9] else -0.1))
                else:
                    seasonal_factor = 1.0 + 0.15 * annual_cycle
                
                # Holiday Flag & Impact
                is_holiday = False
                holiday_multiplier = 1.0
                holiday_name = "None"
                
                if week_num in [6]:
                    is_holiday = True
                    holiday_name = "Super Bowl"
                    holiday_multiplier = 1.12
                elif week_num in [21]:
                    is_holiday = True
                    holiday_name = "Memorial Day"
                    holiday_multiplier = 1.15
                elif week_num in [36]:
                    is_holiday = True
                    holiday_name = "Labor Day"
                    holiday_multiplier = 1.18
                elif week_num in [47, 48]:
                    is_holiday = True
                    holiday_name = "Thanksgiving / Black Friday"
                    holiday_multiplier = 1.65 + d_surge
                elif week_num in [51, 52]:
                    is_holiday = True
                    holiday_name = "Christmas / New Year"
                    holiday_multiplier = 1.50 + d_surge
                
                # Promotional markdowns (higher during holidays & Q4)
                promo_active = 1 if (is_holiday or np.random.rand() < 0.22) else 0
                markdown_amount = np.random.uniform(500, 4500) if promo_active else 0.0
                promo_boost = 1.0 + (markdown_amount / 15000.0) * (0.35 if promo_active else 0.0)

                # Macro impact
                temp = 60 + 22 * np.sin(2 * np.pi * (week_num - 15) / 52.0) + np.random.normal(0, 4)
                
                # Long term growth trend (+4% per year)
                growth_trend = 1.0 + 0.04 * (i / 52.0)
                
                # Random variation / noise
                noise = np.random.normal(1.0, 0.04 * d_vol)
                
                weekly_sales = (
                    base_sales * seasonal_factor * holiday_multiplier * promo_boost * growth_trend * noise
                )
                weekly_sales = max(100.0, weekly_sales)
                
                records.append({
                    "Date": date.strftime("%Y-%m-%d"),
                    "Store_ID": store_id,
                    "Store_Type": stype,
                    "Store_Size": s_size,
                    "Dept_ID": dept_id,
                    "Dept_Name": dname,
                    "Weekly_Sales": round(weekly_sales, 2),
                    "IsHoliday": int(is_holiday),
                    "Holiday_Name": holiday_name,
                    "Markdown_Discount": round(markdown_amount, 2),
                    "Promotional_Flag": promo_active,
                    "Temperature": round(temp, 1),
                    "Fuel_Price": round(macro_fuel[i], 3),
                    "CPI": round(macro_cpi[i], 3),
                    "Unemployment_Rate": round(macro_unemp[i], 2)
                })

    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by=["Store_ID", "Dept_ID", "Date"]).reset_index(drop=True)
    return df

def save_and_load_dataset(filepath: str) -> pd.DataFrame:
    """Ensures data directory exists and returns loaded dataset."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not os.path.exists(filepath):
        df = generate_retail_sales_data()
        df.to_csv(filepath, index=False)
        print(f"[DataLoader] Dataset generated and saved to {filepath} (Shape: {df.shape})")
    else:
        df = pd.read_csv(filepath)
        df["Date"] = pd.to_datetime(df["Date"])
        print(f"[DataLoader] Dataset loaded from {filepath} (Shape: {df.shape})")
    return df

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "retail_store_sales.csv")
    df = save_and_load_dataset(data_path)
    print(df.head())
