"""
RetailPulse Forecast - Interactive Decision-Support Dashboard
Streamlit web application for interactive demand simulation, inventory reorder planning,
and visual model exploration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib

st.set_page_config(page_title="RetailPulse Forecast", page_icon="📈", layout="wide")

# Paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(base_dir, "data", "retail_store_sales.csv")
model_path = os.path.join(base_dir, "models", "best_forecasting_model.pkl")
metrics_path = os.path.join(base_dir, "outputs", "metrics", "model_comparison.csv")

@st.cache_data
def load_data():
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    return None

@st.cache_data
def load_metrics():
    if os.path.exists(metrics_path):
        return pd.read_csv(metrics_path)
    return None

df = load_data()
metrics_df = load_metrics()

# Header
st.title("📈 RetailPulse Forecast — Sales & Demand Intelligence")
st.markdown("Enterprise Machine Learning solution for multi-echelon retail sales forecasting, inventory buffer optimization, and promotion sensitivity planning.")

# Sidebar Filters
st.sidebar.header("🕹️ Store & Department Controls")
if df is not None:
    stores = sorted(df["Store_ID"].unique())
    selected_store = st.sidebar.selectbox("Select Store", stores, index=0)
    
    depts = sorted(df[df["Store_ID"] == selected_store]["Dept_Name"].unique())
    selected_dept = st.sidebar.selectbox("Select Department", depts, index=0)
    
    filtered_df = df[(df["Store_ID"] == selected_store) & (df["Dept_Name"] == selected_dept)].copy()
    
    # KPIs Row
    st.subheader(f"📊 Historical Performance Overview — Store {selected_store} ({selected_dept})")
    col1, col2, col3, col4 = st.columns(4)
    avg_sales = filtered_df["Weekly_Sales"].mean()
    max_sales = filtered_df["Weekly_Sales"].max()
    holiday_avg = filtered_df[filtered_df["IsHoliday"] == 1]["Weekly_Sales"].mean()
    promo_avg = filtered_df[filtered_df["Promotional_Flag"] == 1]["Weekly_Sales"].mean()
    
    col1.metric("Average Weekly Sales", f"${avg_sales:,.2f}")
    col2.metric("Peak Weekly Sales", f"${max_sales:,.2f}")
    col3.metric("Holiday Week Avg", f"${holiday_avg:,.2f}", delta=f"{((holiday_avg-avg_sales)/avg_sales)*100:.1f}%")
    col4.metric("Promotional Lift", f"${promo_avg:,.2f}", delta=f"{((promo_avg-avg_sales)/avg_sales)*100:.1f}%")

    # Time series plot
    fig = px.line(
        filtered_df,
        x="Date",
        y="Weekly_Sales",
        title=f"Weekly Sales Trajectory — Store {selected_store} : {selected_dept}",
        labels={"Weekly_Sales": "Weekly Sales ($)", "Date": "Date"},
        color_discrete_sequence=["#1f77b4"]
    )
    
    # Mark holidays
    holiday_points = filtered_df[filtered_df["IsHoliday"] == 1]
    fig.add_trace(go.Scatter(
        x=holiday_points["Date"],
        y=holiday_points["Weekly_Sales"],
        mode="markers",
        name="Holiday Surge",
        marker=dict(color="#d62728", size=8, symbol="diamond")
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Model Evaluation Benchmark Tab
    st.divider()
    tab1, tab2, tab3 = st.tabs(["🏆 Benchmark Models", "🔮 Interactive What-If Simulator", "📦 Reorder & Safety Stock"])
    
    with tab1:
        st.subheader("Model Evaluation & Error Comparison")
        if metrics_df is not None:
            st.dataframe(metrics_df.style.highlight_min(subset=["WAPE (%)", "MAE ($)", "RMSE ($)"], color="#d4edda"), use_container_width=True)
            
            fig_bar = px.bar(
                metrics_df,
                x="Model",
                y="WAPE (%)",
                color="WAPE (%)",
                color_continuous_scale="Blues_r",
                title="Model Comparison by Weighted Absolute Percentage Error (WAPE % - Lower is Better)"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        st.subheader("Interactive Promo & Markdown Simulator")
        st.markdown("Simulate next week demand given pricing adjustments and promotional investment.")
        
        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            promo_active = st.radio("Apply Active Promotion Campaign?", ["No", "Yes"], index=1)
            markdown_input = st.slider("Markdown Discount Budget ($)", min_value=0, max_value=5000, value=1500, step=250)
        with sim_col2:
            is_holiday_week = st.radio("Is Next Week a Major Holiday?", ["No", "Yes"], index=0)
            target_growth = st.slider("Expected Market Growth (%)", min_value=-10, max_value=20, value=4, step=1)
            
        base_recent = filtered_df["Weekly_Sales"].iloc[-4:].mean()
        sim_multiplier = 1.0 + (0.45 if is_holiday_week == "Yes" else 0.0) + (0.22 if promo_active == "Yes" else 0.0) + (markdown_input / 20000.0) + (target_growth / 100.0)
        simulated_forecast = base_recent * sim_multiplier
        
        st.success(f"### Estimated Next Week Demand: **${simulated_forecast:,.2f}** (Base: ${base_recent:,.2f}, Adjusted Factor: {sim_multiplier:.2f}x)")
        
    with tab3:
        st.subheader("Dynamic Safety Stock & Inventory Buffer")
        st.markdown("Automated buffer calculation at a **95% Service Level Agreement (SLA)** to prevent out-of-stock scenarios.")
        
        lead_time_weeks = st.slider("Supplier Lead Time (Weeks)", min_value=1, max_value=6, value=2)
        demand_std = filtered_df["Weekly_Sales"].std()
        
        # Safety Stock formula: Z * sigma_demand * sqrt(lead_time)
        z_score = 1.645 # 95% service level
        safety_stock_dollars = z_score * demand_std * np.sqrt(lead_time_weeks)
        reorder_point = (avg_sales * lead_time_weeks) + safety_stock_dollars
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Cycle Demand (Lead Time)", f"${(avg_sales * lead_time_weeks):,.2f}")
        c2.metric("Safety Stock Buffer", f"${safety_stock_dollars:,.2f}")
        c3.metric("Total Reorder Point (ROP)", f"${reorder_point:,.2f}")
        
else:
    st.warning("Please run the pipeline script first to generate dataset and models.")
