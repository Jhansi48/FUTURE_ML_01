"""
RetailPulse Forecast - Enterprise Decision-Support Dashboard
Professional retail business analytics application for demand forecasting,
model evaluation, inventory buffer planning, and scenario simulation.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(
    page_title="RetailPulse Forecast | Demand Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Professional Light Business Theme CSS
LIGHT_THEME_CSS = """
<style>
    /* Global Reset & Clean Business Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        color: #1e293b;
    }
    
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Top Header Section */
    .dashboard-header {
        padding: 0.5rem 0 1rem 0;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 1.25rem;
    }
    
    .dashboard-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
        letter-spacing: -0.01em;
    }
    
    .dashboard-subtitle {
        font-size: 0.875rem;
        color: #64748b;
        margin-top: 0.2rem;
        margin-bottom: 0;
    }
    
    /* KPI Cards Grid */
    .kpi-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1.25rem;
    }
    
    .kpi-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    
    .kpi-title {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748b;
        margin-bottom: 0.3rem;
    }
    
    .kpi-num {
        font-size: 1.45rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    
    .kpi-sub {
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 0.3rem;
    }
    
    .kpi-sub.positive {
        color: #166534;
    }
    
    .kpi-sub.muted {
        color: #64748b;
    }
    
    .kpi-sub.alert {
        color: #c2410c;
    }
    
    /* Section Headers */
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #0f172a;
        margin: 0.5rem 0 0.75rem 0;
    }
    
    /* Sidebar Clean Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    .sidebar-header {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0f172a;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.75rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #f1f5f9;
    }
    
    .sidebar-info-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.75rem 0.9rem;
        font-size: 0.825rem;
        color: #475569;
        margin-top: 1rem;
    }
    
    /* Tabs Navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        font-weight: 600;
        font-size: 0.875rem;
        color: #64748b;
        border-radius: 4px 4px 0 0;
        border-bottom: 2px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        color: #0f766e !important;
        border-bottom-color: #0f766e !important;
        background-color: transparent !important;
    }
    
    /* Simulator Summary Banner */
    .simulator-result-box {
        background-color: #f0fdfa;
        border: 1px solid #ccfbf1;
        border-left: 4px solid #0f766e;
        border-radius: 6px;
        padding: 0.9rem 1.2rem;
        margin: 0.75rem 0 1rem 0;
    }
    
    .sim-result-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #0f766e;
    }
    
    .sim-result-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0.15rem 0;
    }
    
    .sim-result-desc {
        font-size: 0.85rem;
        color: #475569;
    }
</style>
"""
st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)

# 3. Data Ingestion & Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "retail_store_sales.csv")
METRICS_PATH = os.path.join(BASE_DIR, "outputs", "metrics", "test_model_evaluation.csv")
VAL_METRICS_PATH = os.path.join(BASE_DIR, "outputs", "metrics", "validation_model_comparison.csv")

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    return None

@st.cache_data
def load_benchmarks():
    target_path = METRICS_PATH if os.path.exists(METRICS_PATH) else (VAL_METRICS_PATH if os.path.exists(VAL_METRICS_PATH) else None)
    if target_path and os.path.exists(target_path):
        return pd.read_csv(target_path)
    return None

df = load_data()
benchmarks_df = load_benchmarks()

# Plotly configuration for interactive toolbar functionality
PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToAdd": ["drawline", "drawopenpath", "eraseshape"],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "retailpulse_analytics_chart",
        "height": 480,
        "width": 920,
        "scale": 2
    }
}

# 4. Page Header
st.markdown("""
<div class="dashboard-header">
    <h1 class="dashboard-title">RetailPulse Forecast</h1>
    <p class="dashboard-subtitle">Sales Demand Forecasting, Inventory Safety Stock & Promotional Sensitivity Planning</p>
</div>
""", unsafe_allow_html=True)

# 5. Sidebar Scope Filters
st.sidebar.markdown('<div class="sidebar-header">Filters & Selection</div>', unsafe_allow_html=True)

if df is not None:
    stores = sorted(df["Store_ID"].unique())
    selected_store = st.sidebar.selectbox("Store Location", stores, index=0, format_func=lambda s: f"Store {s}")
    
    dept_options = sorted(df[df["Store_ID"] == selected_store]["Dept_Name"].unique())
    selected_dept = st.sidebar.selectbox("Department", dept_options, index=0)
    
    # Filter dataset for selected time series
    filtered_df = df[(df["Store_ID"] == selected_store) & (df["Dept_Name"] == selected_dept)].copy().sort_values("Date").reset_index(drop=True)
    
    st.sidebar.markdown(f"""
    <div class="sidebar-info-card">
        <b>Active Filter Summary</b><br>
        • <b>Store:</b> Store {selected_store}<br>
        • <b>Department:</b> {selected_dept}<br>
        • <b>Date Span:</b> {filtered_df['Date'].min().strftime('%b %Y')} – {filtered_df['Date'].max().strftime('%b %Y')}<br>
        • <b>Total Samples:</b> {len(filtered_df)} weekly records
    </div>
    """, unsafe_allow_html=True)

    # 6. Four Compact KPI Cards
    avg_sales = float(filtered_df["Weekly_Sales"].mean())
    max_sales = float(filtered_df["Weekly_Sales"].max())
    holiday_sub = filtered_df[filtered_df["IsHoliday"] == 1]["Weekly_Sales"]
    holiday_avg = float(holiday_sub.mean()) if len(holiday_sub) > 0 else avg_sales
    promo_sub = filtered_df[filtered_df["Promotional_Flag"] == 1]["Weekly_Sales"]
    promo_avg = float(promo_sub.mean()) if len(promo_sub) > 0 else avg_sales
    
    holiday_lift_pct = ((holiday_avg - avg_sales) / avg_sales) * 100
    promo_lift_pct = ((promo_avg - avg_sales) / avg_sales) * 100
    
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-box">
            <div class="kpi-title">Average Weekly Sales</div>
            <div class="kpi-num">${avg_sales:,.0f}</div>
            <div class="kpi-sub muted">Historical mean volume</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-title">Peak Weekly Sales</div>
            <div class="kpi-num">${max_sales:,.0f}</div>
            <div class="kpi-sub muted">Recorded series maximum</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-title">Holiday Week Avg</div>
            <div class="kpi-num">${holiday_avg:,.0f}</div>
            <div class="kpi-sub positive">+{holiday_lift_pct:.1f}% vs baseline</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-title">Promotional Week Avg</div>
            <div class="kpi-num">${promo_avg:,.0f}</div>
            <div class="kpi-sub positive">+{promo_lift_pct:.1f}% uplift</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 7. Weekly Demand Trajectory Plot
    st.markdown('<div class="section-title">Weekly Demand Trajectory</div>', unsafe_allow_html=True)
    
    fig_line = go.Figure()
    
    # Weekly Sales
    fig_line.add_trace(go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["Weekly_Sales"],
        mode="lines",
        name="Weekly Sales",
        line=dict(color="#0f766e", width=2.0),
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Sales:</b> $%{y:,.2f}<extra></extra>"
    ))
    
    # 4-Week Moving Average Trendline
    filtered_df["SMA_4W"] = filtered_df["Weekly_Sales"].rolling(4, min_periods=1).mean()
    fig_line.add_trace(go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["SMA_4W"],
        mode="lines",
        name="4-Week Moving Average",
        line=dict(color="#64748b", width=1.5, dash="dot"),
        hovertemplate="<b>4W Moving Avg:</b> $%{y:,.2f}<extra></extra>"
    ))
    
    # Holiday points overlay
    holiday_events = filtered_df[filtered_df["IsHoliday"] == 1]
    if not holiday_events.empty:
        fig_line.add_trace(go.Scatter(
            x=holiday_events["Date"],
            y=holiday_events["Weekly_Sales"],
            mode="markers",
            name="Holiday Surge",
            marker=dict(color="#c2410c", size=7, symbol="diamond", line=dict(color="#ffffff", width=0.8)),
            hovertemplate="<b>Holiday Event:</b> $%{y:,.2f}<br><b>Date:</b> %{x|%Y-%m-%d}<extra></extra>"
        ))
        
    fig_line.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=360,
        margin=dict(l=45, r=25, t=15, b=35),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#475569")
        ),
        xaxis=dict(
            title="",
            showgrid=True,
            gridcolor="#f1f5f9",
            linecolor="#e2e8f0"
        ),
        yaxis=dict(
            title="Sales ($ USD)",
            showgrid=True,
            gridcolor="#f1f5f9",
            linecolor="#e2e8f0",
            tickprefix="$",
            tickformat=","
        )
    )
    st.plotly_chart(fig_line, use_container_width=True, config=PLOTLY_CONFIG)

    # 8. Business Analytics Tabs
    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs([
        "Model Performance",
        "Demand Simulator",
        "Inventory Planning"
    ])
    
    # TAB 1: MODEL PERFORMANCE
    with tab1:
        st.markdown('<div class="section-title">Model Evaluation & Error Benchmarks</div>', unsafe_allow_html=True)
        st.markdown("<span style='font-size: 0.85rem; color: #64748b;'>Benchmarked on out-of-time chronological validation data. Lower WAPE (%) and RMSE indicate superior predictive accuracy.</span>", unsafe_allow_html=True)
        
        if benchmarks_df is not None:
            col_t, col_c = st.columns([5, 5])
            
            with col_t:
                disp_df = benchmarks_df.copy()
                st.dataframe(
                    disp_df.style.format({
                        "MAE ($)": "${:,.2f}",
                        "RMSE ($)": "${:,.2f}",
                        "MAPE (%)": "{:.2f}%",
                        "WAPE (%)": "{:.2f}%",
                        "R2 Score": "{:.4f}"
                    }).highlight_min(subset=["WAPE (%)", "MAE ($)", "RMSE ($)"], color="#ccfbf1"),
                    use_container_width=True,
                    height=280
                )
                
            with col_c:
                fig_perf = px.bar(
                    benchmarks_df.sort_values(by="WAPE (%)", ascending=True),
                    x="Model",
                    y="WAPE (%)",
                    color="WAPE (%)",
                    color_continuous_scale=["#0f766e", "#0284c7", "#94a3b8", "#d97706", "#dc2626"],
                    title="Model Accuracy Comparison (WAPE % — Lower is Better)"
                )
                fig_perf.update_layout(
                    template="plotly_white",
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    height=280,
                    margin=dict(l=35, r=15, t=35, b=35),
                    xaxis=dict(title="", tickangle=-20),
                    yaxis=dict(title="WAPE (%)", showgrid=True, gridcolor="#f1f5f9"),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_perf, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.info("Benchmark metrics not found. Run pipeline.py to generate evaluation outputs.")

    # TAB 2: DEMAND SIMULATOR
    with tab2:
        st.markdown('<div class="section-title">Scenario Simulation & Promotional Sensitivity</div>', unsafe_allow_html=True)
        st.markdown("<span style='font-size: 0.85rem; color: #64748b;'>Simulate next-week department demand based on promotional spend, markdown allocation, and market growth.</span>", unsafe_allow_html=True)
        
        s1, s2 = st.columns(2)
        with s1:
            sim_promo = st.selectbox("Promotional Campaign Status", ["Active Campaign", "Standard Operations"], index=0)
            sim_markdown = st.slider("Markdown Discount Allocation ($)", min_value=0, max_value=5000, value=1200, step=200)
            
        with s2:
            sim_holiday = st.selectbox("Calendar Holiday Status", ["Regular Week", "Major Holiday Peak"], index=0)
            sim_growth = st.slider("Market / Macroeconomic Growth Trend (%)", min_value=-10.0, max_value=20.0, value=3.0, step=0.5)
            
        base_weekly = float(filtered_df["Weekly_Sales"].tail(4).mean())
        
        factor_promo = 0.22 if sim_promo == "Active Campaign" else 0.0
        factor_holiday = 0.45 if sim_holiday == "Major Holiday Peak" else 0.0
        factor_markdown = (sim_markdown / 25000.0)
        factor_growth = (sim_growth / 100.0)
        
        multiplier = 1.0 + factor_promo + factor_holiday + factor_markdown + factor_growth
        forecast_val = base_weekly * multiplier
        net_diff = forecast_val - base_weekly
        
        st.markdown(f"""
        <div class="simulator-result-box">
            <div class="sim-result-title">Projected Next-Week Demand</div>
            <div class="sim-result-value">${forecast_val:,.2f}</div>
            <div class="sim-result-desc">
                Baseline (4W Average): <b>${base_weekly:,.2f}</b> | Net Projected Uplift: <b>+${net_diff:,.2f}</b> ({((multiplier-1)*100):+.1f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        comp_df = pd.DataFrame({
            "Component": ["Base 4W Demand", "Promotional Uplift", "Holiday Impact", "Markdown Sensitivity", "Market Trend"],
            "Amount ($)": [
                base_weekly,
                base_weekly * factor_promo,
                base_weekly * factor_holiday,
                base_weekly * factor_markdown,
                base_weekly * factor_growth
            ]
        })
        
        fig_comp = px.bar(
            comp_df,
            x="Component",
            y="Amount ($)",
            text="Amount ($)",
            color="Component",
            color_discrete_sequence=["#0f766e", "#0284c7", "#c2410c", "#64748b", "#059669"],
            title="Demand Contribution Breakdown by Factor"
        )
        fig_comp.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig_comp.update_layout(
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            height=280,
            margin=dict(l=35, r=15, t=35, b=35),
            showlegend=False,
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", tickprefix="$")
        )
        st.plotly_chart(fig_comp, use_container_width=True, config=PLOTLY_CONFIG)

    # TAB 3: INVENTORY PLANNING
    with tab3:
        st.markdown('<div class="section-title">Safety Stock & Reorder Point Policy</div>', unsafe_allow_html=True)
        st.markdown("<span style='font-size: 0.85rem; color: #64748b;'>Calculates required buffer inventory for a <b>95% Service Level Agreement (SLA)</b> ($Z_{0.95} = 1.645$) across supplier replenishment horizons.</span>", unsafe_allow_html=True)
        
        lead_time_w = st.slider("Supplier Replenishment Lead Time (Weeks)", min_value=1, max_value=6, value=2, step=1)
        
        st_std = float(filtered_df["Weekly_Sales"].std())
        z_score = 1.645
        
        # Lead time safety stock: SS = Z * sigma * sqrt(L)
        ss_dollars = z_score * st_std * np.sqrt(lead_time_w)
        cycle_req = avg_sales * lead_time_w
        rop_dollars = cycle_req + ss_dollars
        
        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-title">Cycle Demand ({lead_time_w}W Lead Time)</div>
                <div class="kpi-num">${cycle_req:,.0f}</div>
                <div class="kpi-sub muted">${avg_sales:,.0f} / week</div>
            </div>
            """, unsafe_allow_html=True)
            
        with ic2:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-title">Safety Stock Buffer (95% SLA)</div>
                <div class="kpi-num">${ss_dollars:,.0f}</div>
                <div class="kpi-sub alert">Z = 1.645 · σ · √{lead_time_w}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with ic3:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-title">Total Reorder Point (ROP)</div>
                <div class="kpi-num">${rop_dollars:,.0f}</div>
                <div class="kpi-sub positive">Cycle Demand + Buffer</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Stacked Inventory Composition Bar
        fig_stack = go.Figure()
        fig_stack.add_trace(go.Bar(
            name="Cycle Demand",
            x=[f"Store {selected_store} : {selected_dept}"],
            y=[cycle_req],
            marker_color="#0f766e",
            text=[f"${cycle_req:,.0f}"],
            textposition="inside"
        ))
        fig_stack.add_trace(go.Bar(
            name="Safety Stock Buffer",
            x=[f"Store {selected_store} : {selected_dept}"],
            y=[ss_dollars],
            marker_color="#c2410c",
            text=[f"${ss_dollars:,.0f}"],
            textposition="inside"
        ))
        fig_stack.update_layout(
            barmode="stack",
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            height=260,
            title=f"Inventory Buffer Composition (Lead Time = {lead_time_w} Weeks)",
            margin=dict(l=35, r=15, t=35, b=35),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11, color="#475569")),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", tickprefix="$", tickformat=",")
        )
        st.plotly_chart(fig_stack, use_container_width=True, config=PLOTLY_CONFIG)

else:
    st.error("Dataset not found at data/retail_store_sales.csv. Please execute the pipeline script first.")
