"""
RetailPulse Forecast - Enterprise Decision-Support Dashboard
Streamlit business analytics interface for multi-echelon sales forecasting,
inventory safety-stock planning, model benchmarking, and promotion sensitivity simulation.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration with professional metadata
st.set_page_config(
    page_title="RetailPulse Forecast | Enterprise Demand Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise Light Theme CSS
CUSTOM_CSS = """
<style>
    /* Global typography & clean background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #1e293b;
    }
    
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Header styling */
    .main-header {
        padding: 0.8rem 0 1.2rem 0;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 1.2rem;
    }
    
    .main-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.02em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .main-subtitle {
        font-size: 0.95rem;
        color: #64748b;
        margin-top: 0.25rem;
        margin-bottom: 0;
    }
    
    /* KPI Card styling */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        border-top: 3px solid #0f766e;
    }
    
    .kpi-card.amber {
        border-top-color: #d97706;
    }
    
    .kpi-card.teal {
        border-top-color: #0f766e;
    }
    
    .kpi-card.blue {
        border-top-color: #2563eb;
    }
    
    .kpi-card.slate {
        border-top-color: #475569;
    }
    
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.35rem;
    }
    
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    
    .kpi-delta {
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 0.35rem;
    }
    
    .kpi-delta.positive {
        color: #0d9488;
    }
    
    .kpi-delta.neutral {
        color: #64748b;
    }
    
    /* Section containers */
    .content-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    
    .section-heading {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.8rem;
    }
    
    /* Sidebar customization */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f172a;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 0.5rem;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        color: #64748b;
        border-radius: 6px 6px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        color: #0f766e !important;
        border-bottom: 2px solid #0f766e !important;
        background-color: transparent !important;
    }
    
    /* Simulator result banner */
    .sim-banner {
        background-color: #f0fdfa;
        border: 1px solid #ccfbf1;
        border-left: 4px solid #0f766e;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# File Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "retail_store_sales.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_forecasting_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "outputs", "metrics", "test_model_evaluation.csv")
VAL_METRICS_PATH = os.path.join(BASE_DIR, "outputs", "metrics", "validation_model_comparison.csv")

@st.cache_data
def load_dataset():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    return None

@st.cache_data
def load_metrics_data():
    metrics_file = METRICS_PATH if os.path.exists(METRICS_PATH) else (VAL_METRICS_PATH if os.path.exists(VAL_METRICS_PATH) else None)
    if metrics_file and os.path.exists(metrics_file):
        return pd.read_csv(metrics_file)
    return None

df = load_dataset()
metrics_df = load_metrics_data()

# Plotly default configuration for interactive toolbars
PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToAdd": ["drawline", "drawopenpath", "eraseshape"],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "retailpulse_chart",
        "height": 520,
        "width": 960,
        "scale": 2
    }
}

# Header Banner
st.markdown("""
<div class="main-header">
    <h1 class="main-title">RetailPulse Forecast</h1>
    <p class="main-subtitle">Enterprise Demand Planning, Inventory Safety Stock & Promotion Sensitivity Intelligence</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.markdown("## Scope & Filter Controls")
if df is not None:
    stores = sorted(df["Store_ID"].unique())
    selected_store = st.sidebar.selectbox("Store Location", stores, index=0, format_func=lambda s: f"Store {s} (Flagship)" if s == 1 else f"Store {s}")
    
    depts = sorted(df[df["Store_ID"] == selected_store]["Dept_Name"].unique())
    selected_dept = st.sidebar.selectbox("Merchandise Department", depts, index=0)
    
    # Filter dataset for selected series
    filtered_df = df[(df["Store_ID"] == selected_store) & (df["Dept_Name"] == selected_dept)].copy().sort_values("Date").reset_index(drop=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    **Active Series Summary:**
    - **Store ID:** {selected_store}
    - **Department:** {selected_dept}
    - **Date Range:** {filtered_df['Date'].min().strftime('%b %Y')} – {filtered_df['Date'].max().strftime('%b %Y')}
    - **Observations:** {len(filtered_df)} weekly records
    """)

    # 1. Historical Performance KPI Cards
    avg_sales = float(filtered_df["Weekly_Sales"].mean())
    max_sales = float(filtered_df["Weekly_Sales"].max())
    holiday_sales = filtered_df[filtered_df["IsHoliday"] == 1]["Weekly_Sales"]
    holiday_avg = float(holiday_sales.mean()) if len(holiday_sales) > 0 else avg_sales
    promo_sales = filtered_df[filtered_df["Promotional_Flag"] == 1]["Weekly_Sales"]
    promo_avg = float(promo_sales.mean()) if len(promo_sales) > 0 else avg_sales
    
    holiday_lift_pct = ((holiday_avg - avg_sales) / avg_sales) * 100
    promo_lift_pct = ((promo_avg - avg_sales) / avg_sales) * 100
    
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card teal">
            <div class="kpi-label">Average Weekly Sales</div>
            <div class="kpi-value">${avg_sales:,.0f}</div>
            <div class="kpi-delta neutral">Base baseline volume</div>
        </div>
        <div class="kpi-card blue">
            <div class="kpi-label">Peak Weekly Sales</div>
            <div class="kpi-value">${max_sales:,.0f}</div>
            <div class="kpi-delta neutral">Historical maximum</div>
        </div>
        <div class="kpi-card amber">
            <div class="kpi-label">Holiday Week Average</div>
            <div class="kpi-value">${holiday_avg:,.0f}</div>
            <div class="kpi-delta positive">+{holiday_lift_pct:.1f}% vs baseline</div>
        </div>
        <div class="kpi-card slate">
            <div class="kpi-label">Promotional Campaign Lift</div>
            <div class="kpi-value">${promo_avg:,.0f}</div>
            <div class="kpi-delta positive">+{promo_lift_pct:.1f}% uplift</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Interactive Time-Series Chart
    st.markdown('<div class="section-heading">Weekly Demand Trajectory & Seasonal Surges</div>', unsafe_allow_html=True)
    
    fig_ts = go.Figure()
    
    # Primary weekly sales line
    fig_ts.add_trace(go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["Weekly_Sales"],
        mode="lines",
        name="Weekly Sales ($)",
        line=dict(color="#0f766e", width=2.2),
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Sales:</b> $%{y:,.2f}<extra></extra>"
    ))
    
    # Rolling 4-week moving average
    filtered_df["SMA_4W"] = filtered_df["Weekly_Sales"].rolling(4, min_periods=1).mean()
    fig_ts.add_trace(go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["SMA_4W"],
        mode="lines",
        name="4-Week Moving Avg",
        line=dict(color="#64748b", width=1.5, dash="dot"),
        hovertemplate="<b>4W Trend:</b> $%{y:,.2f}<extra></extra>"
    ))
    
    # Overlay Holiday Points
    holiday_pts = filtered_df[filtered_df["IsHoliday"] == 1]
    if not holiday_pts.empty:
        fig_ts.add_trace(go.Scatter(
            x=holiday_pts["Date"],
            y=holiday_pts["Weekly_Sales"],
            mode="markers",
            name="Holiday Event",
            marker=dict(color="#d97706", size=8, symbol="diamond", line=dict(color="#ffffff", width=1)),
            hovertemplate="<b>Holiday Surge:</b> $%{y:,.2f}<br><b>Date:</b> %{x|%Y-%m-%d}<extra></extra>"
        ))
        
    fig_ts.update_layout(
        template="plotly_white",
        margin=dict(l=50, r=30, t=20, b=40),
        height=380,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            title="",
            showgrid=True,
            gridcolor="#f1f5f9",
            linecolor="#cbd5e1"
        ),
        yaxis=dict(
            title="Weekly Sales ($ USD)",
            showgrid=True,
            gridcolor="#f1f5f9",
            linecolor="#cbd5e1",
            tickprefix="$",
            tickformat=","
        )
    )
    st.plotly_chart(fig_ts, use_container_width=True, config=PLOTLY_CONFIG)

    # 3. Decision-Support Analytics Tabs
    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs([
        "Model Benchmarks & Selection",
        "What-If Demand Simulator",
        "Inventory Buffer & Reorder Point"
    ])
    
    # TAB 1: MODEL BENCHMARKS
    with tab1:
        st.markdown('<div class="section-heading">Empirical Model Benchmarking & Accuracy Comparison</div>', unsafe_allow_html=True)
        st.markdown("All candidate models are evaluated using rigorous out-of-time chronological validation. Lower WAPE (%) indicates superior forecasting precision.")
        
        if metrics_df is not None:
            col_tbl, col_chart = st.columns([5, 5])
            
            with col_tbl:
                # Format dataframe for clean display
                display_metrics = metrics_df.copy()
                st.dataframe(
                    display_metrics.style.format({
                        "MAE ($)": "${:,.2f}",
                        "RMSE ($)": "${:,.2f}",
                        "MAPE (%)": "{:.2f}%",
                        "WAPE (%)": "{:.2f}%",
                        "R2 Score": "{:.4f}"
                    }).highlight_min(subset=["WAPE (%)", "MAE ($)", "RMSE ($)"], color="#ccfbf1"),
                    use_container_width=True,
                    height=280
                )
                
            with col_chart:
                fig_bar = px.bar(
                    metrics_df.sort_values(by="WAPE (%)", ascending=True),
                    x="Model",
                    y="WAPE (%)",
                    color="WAPE (%)",
                    color_continuous_scale=["#0f766e", "#38bdf8", "#94a3b8", "#f59e0b", "#ef4444"],
                    title="Weighted Absolute Percentage Error (WAPE % - Lower is Better)"
                )
                fig_bar.update_layout(
                    template="plotly_white",
                    margin=dict(l=40, r=20, t=40, b=40),
                    height=280,
                    xaxis=dict(title="", tickangle=-25),
                    yaxis=dict(title="WAPE (%)", showgrid=True, gridcolor="#f1f5f9"),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_bar, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.info("Metrics comparison table not found. Run pipeline.py to generate benchmark results.")

    # TAB 2: WHAT-IF DEMAND SIMULATOR
    with tab2:
        st.markdown('<div class="section-heading">Scenario Simulation & Promotional Sensitivity</div>', unsafe_allow_html=True)
        st.markdown("Simulate next-week store-department demand based on active promotional campaigns, markdown budget allocations, and calendar events.")
        
        sim_c1, sim_c2 = st.columns(2)
        with sim_c1:
            promo_choice = st.selectbox("Promotional Campaign Status", ["Active Campaign", "Standard Operations"], index=0)
            markdown_budget = st.slider("Markdown Discount Allocation ($)", min_value=0, max_value=5000, value=1200, step=200)
            
        with sim_c2:
            holiday_choice = st.selectbox("Calendar Holiday Status", ["Regular Week", "Major Holiday Peak"], index=0)
            macro_growth = st.slider("Macroeconomic / Market Trend (%)", min_value=-10.0, max_value=20.0, value=3.0, step=0.5)
            
        # Calculation logic
        base_demand = float(filtered_df["Weekly_Sales"].tail(4).mean())
        
        promo_factor = 0.22 if promo_choice == "Active Campaign" else 0.0
        holiday_factor = 0.45 if holiday_choice == "Major Holiday Peak" else 0.0
        markdown_factor = (markdown_budget / 25000.0)
        growth_factor = (macro_growth / 100.0)
        
        total_multiplier = 1.0 + promo_factor + holiday_factor + markdown_factor + growth_factor
        simulated_sales = base_demand * total_multiplier
        net_uplift = simulated_sales - base_demand
        
        st.markdown(f"""
        <div class="sim-banner">
            <div style="font-size: 0.85rem; font-weight: 600; color: #0f766e; text-transform: uppercase;">Simulated Forecast Output</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #0f172a; margin-top: 0.2rem;">${simulated_sales:,.2f}</div>
            <div style="font-size: 0.9rem; color: #475569; margin-top: 0.35rem;">
                Baseline (4W Avg): <b>${base_demand:,.2f}</b> | Net Projected Uplift: <b>+${net_uplift:,.2f}</b> ({((total_multiplier-1)*100):+.1f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Breakdown chart
        breakdown_df = pd.DataFrame({
            "Component": ["Base 4W Demand", "Promotional Lift", "Holiday Impact", "Markdown Sensitivity", "Market Trend"],
            "Value ($)": [
                base_demand,
                base_demand * promo_factor,
                base_demand * holiday_factor,
                base_demand * markdown_factor,
                base_demand * growth_factor
            ]
        })
        fig_breakdown = px.bar(
            breakdown_df,
            x="Component",
            y="Value ($)",
            text="Value ($)",
            color="Component",
            color_discrete_sequence=["#0f766e", "#38bdf8", "#d97706", "#64748b", "#0284c7"],
            title="Projected Demand Contribution Breakdown"
        )
        fig_breakdown.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig_breakdown.update_layout(
            template="plotly_white",
            height=280,
            margin=dict(l=40, r=20, t=40, b=40),
            showlegend=False,
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", tickprefix="$")
        )
        st.plotly_chart(fig_breakdown, use_container_width=True, config=PLOTLY_CONFIG)

    # TAB 3: REORDER & SAFETY STOCK
    with tab3:
        st.markdown('<div class="section-heading">Lead-Time Safety Stock & Dynamic Reorder Point Policy</div>', unsafe_allow_html=True)
        st.markdown("Operations research inventory buffer calculation configured for a **95% Service Level Agreement (SLA)** ($Z_{0.95} = 1.645$) across supplier replenishment horizons.")
        
        lead_time = st.slider("Supplier Replenishment Lead Time (Weeks)", min_value=1, max_value=6, value=2, step=1)
        
        # Standard deviation of weekly demand residuals
        demand_std = float(filtered_df["Weekly_Sales"].std())
        z_sla = 1.645 # 95% cycle service level
        
        # Lead time safety stock: SS = Z * sigma * sqrt(L)
        safety_stock = z_sla * demand_std * np.sqrt(lead_time)
        cycle_demand = avg_sales * lead_time
        reorder_point = cycle_demand + safety_stock
        
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.markdown(f"""
            <div class="kpi-card slate">
                <div class="kpi-label">Cycle Demand ({lead_time}W Lead Time)</div>
                <div class="kpi-value">${cycle_demand:,.0f}</div>
                <div class="kpi-delta neutral">${avg_sales:,.0f} / week</div>
            </div>
            """, unsafe_allow_html=True)
            
        with rc2:
            st.markdown(f"""
            <div class="kpi-card amber">
                <div class="kpi-label">Safety Stock Buffer (95% SLA)</div>
                <div class="kpi-value">${safety_stock:,.0f}</div>
                <div class="kpi-delta positive">Z = 1.645 · σ · √{lead_time}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with rc3:
            st.markdown(f"""
            <div class="kpi-card teal">
                <div class="kpi-label">Total Reorder Point (ROP)</div>
                <div class="kpi-value">${reorder_point:,.0f}</div>
                <div class="kpi-delta positive">Cycle Demand + Buffer</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Policy Stack Bar
        fig_inv = go.Figure()
        fig_inv.add_trace(go.Bar(
            name="Cycle Demand",
            x=[f"Store {selected_store} : {selected_dept}"],
            y=[cycle_demand],
            marker_color="#0f766e",
            text=[f"${cycle_demand:,.0f}"],
            textposition="inside"
        ))
        fig_inv.add_trace(go.Bar(
            name="Safety Stock Buffer",
            x=[f"Store {selected_store} : {selected_dept}"],
            y=[safety_stock],
            marker_color="#d97706",
            text=[f"${safety_stock:,.0f}"],
            textposition="inside"
        ))
        fig_inv.update_layout(
            barmode="stack",
            template="plotly_white",
            height=260,
            title=f"Inventory Buffer Composition for Replenishment Horizon (Lead Time = {lead_time} Weeks)",
            margin=dict(l=40, r=20, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", tickprefix="$", tickformat=",")
        )
        st.plotly_chart(fig_inv, use_container_width=True, config=PLOTLY_CONFIG)

else:
    st.error("Dataset not found at data/retail_store_sales.csv. Please execute the pipeline script first.")
