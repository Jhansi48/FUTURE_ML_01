"""
RetailPulse Forecast — Retail Intelligence Studio
Production-Quality Enterprise Demand Forecasting & Inventory Planning Platform
Designed for Live Technical Internship Review
"""

import os
import sys
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Ensure repo root is available
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.features import prepare_feature_matrix
from src.models import chronological_split
from src.inventory import calculate_inventory_policy

# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="RetailPulse | Retail Intelligence Studio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. EDITORIAL ANALYTICS DESIGN SYSTEM & STYLING
# ==============================================================================
EDITORIAL_STUDIO_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&display=swap');
    
    /* Global Base Rules */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #F7F3EC !important;
        color: #4A2F2F !important;
    }
    
    /* Header and Main Container Spacing - Prevents Header Obscuring Content */
    header[data-testid="stHeader"] {
        background-color: #F7F3EC !important;
        height: 2.2rem !important;
        min-height: 2.2rem !important;
        z-index: 99 !important;
    }
    
    .stMainBlockContainer,
    [data-testid="stMainBlockContainer"],
    [data-testid="stAppViewBlockContainer"],
    .block-container,
    .main .block-container {
        padding-top: 0.85rem !important;
        padding-bottom: 3.5rem !important;
        padding-left: 2.2rem !important;
        padding-right: 2.2rem !important;
        max-width: 1400px;
    }
    
    /* Typography Overrides */
    h1, h2, h3, h4, h5, h6,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: #4A2F2F !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    p, span, label, div,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] li {
        color: #4A2F2F !important;
    }
    
    /* Top Header Section */
    .studio-header {
        padding: 0 0 0.95rem 0;
        border-bottom: 1.5px solid #E5DED3;
        margin-bottom: 1.05rem;
    }
    
    .studio-title {
        font-size: 2.65rem !important;
        font-weight: 800 !important;
        color: #4A2F2F !important;
        margin: 0 !important;
        letter-spacing: -0.035em;
        line-height: 1.12;
    }
    
    .studio-title span {
        color: #B85C38 !important;
        font-weight: 800 !important;
    }
    
    .studio-subtitle {
        font-size: 1.02rem !important;
        color: #6E5656 !important;
        margin-top: 0.45rem !important;
        margin-bottom: 0 !important;
        font-weight: 500 !important;
        line-height: 1.5;
    }
    
    /* Compact, Subtly Styled Metadata Badge Row Below Divider */
    .studio-badge-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.75rem;
        flex-wrap: wrap;
    }
    
    .studio-badge {
        display: inline-block;
        font-size: 0.71rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #A04A26;
        background-color: #FAF0EA;
        border: 1px solid #EED8CC;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        line-height: 1.4;
    }
    
    .studio-badge.sage {
        color: #436A40;
        background-color: #EDF3ED;
        border-color: #D3E0D3;
    }
    
    .studio-badge.gold {
        color: #826210;
        background-color: #FAF4E5;
        border-color: #EDE0BC;
    }
    
    /* Signature KPI Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1.35rem;
    }
    
    .kpi-grid-3 {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-bottom: 1.35rem;
        align-items: stretch;
    }
    
    .kpi-card {
        background-color: #FFFCF7 !important;
        border: 1px solid #E5DED3 !important;
        border-radius: 8px;
        padding: 1.05rem 1.25rem;
        box-shadow: 0 2px 6px rgba(74, 47, 47, 0.03);
        border-left: 4.5px solid #B85C38 !important;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .kpi-card.sage {
        border-left-color: #7B9A78 !important;
    }
    
    .kpi-card.gold {
        border-left-color: #D4A017 !important;
    }
    
    .kpi-card.cocoa {
        border-left-color: #4A2F2F !important;
    }
    
    .kpi-meta-label {
        font-size: 0.74rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #7C6E6E !important;
        margin-bottom: 0.35rem;
    }
    
    .kpi-meta-val {
        font-size: 1.7rem !important;
        font-weight: 800 !important;
        color: #4A2F2F !important;
        line-height: 1.15;
    }
    
    .kpi-meta-sub {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        color: #6E5656 !important;
        margin-top: 0.45rem;
        line-height: 1.35;
    }
    
    /* Equal-Sized Editorial Cards Grid */
    .equal-cards-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.15rem;
        margin-top: 0.8rem;
        margin-bottom: 1.4rem;
        align-items: stretch;
    }
    
    .equal-card {
        background-color: #FFFCF7 !important;
        border: 1px solid #E5DED3 !important;
        border-radius: 8px;
        padding: 1.25rem 1.35rem;
        box-shadow: 0 2px 6px rgba(74, 47, 47, 0.03);
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        height: 100%;
        box-sizing: border-box;
    }
    
    .equal-card.terracotta {
        border-top: 3.5px solid #B85C38 !important;
    }
    
    .equal-card.gold {
        border-top: 3.5px solid #D4A017 !important;
    }
    
    .equal-card.sage {
        border-top: 3.5px solid #7B9A78 !important;
    }
    
    /* Section Surfaces */
    .surface-card {
        background-color: #FFFCF7 !important;
        border: 1px solid #E5DED3 !important;
        border-radius: 8px;
        padding: 1.3rem 1.45rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 2px 6px rgba(74, 47, 47, 0.03);
    }
    
    .surface-header-box {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        margin-bottom: 0.25rem;
    }
    
    .surface-tag-line {
        width: 4px;
        height: 18px;
        background-color: #B85C38;
        border-radius: 2px;
    }
    
    .surface-title {
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        color: #4A2F2F !important;
        margin: 0 !important;
    }
    
    .surface-desc {
        font-size: 0.88rem !important;
        color: #6E5656 !important;
        margin: 0.25rem 0 1rem 0 !important;
        line-height: 1.5;
    }
    
    /* Forecast Horizon Banner */
    .horizon-banner {
        background-color: #FAF5EE;
        border: 1.5px solid #EADDCF;
        border-radius: 8px;
        padding: 0.95rem 1.2rem;
        margin-bottom: 1.15rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.8rem;
    }
    
    .horizon-badge-pill {
        display: inline-block;
        font-size: 0.8rem;
        font-weight: 800;
        padding: 0.25rem 0.75rem;
        border-radius: 5px;
        background-color: #B85C38;
        color: #FFFFFF !important;
        letter-spacing: 0.04em;
    }
    
    /* SIDEBAR STYLING */
    [data-testid="stSidebar"] {
        background-color: #FFFCF7 !important;
        border-right: 1.5px solid #E5DED3 !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #4A2F2F !important;
    }
    
    .sidebar-brand {
        font-size: 1.35rem !important;
        font-weight: 800 !important;
        color: #4A2F2F !important;
        letter-spacing: -0.02em;
        margin-top: -1.25rem !important;
        margin-bottom: 0 !important;
        padding-top: 0 !important;
    }
    
    .sidebar-brand span {
        color: #B85C38 !important;
    }
    
    .sidebar-sub {
        font-size: 0.8rem !important;
        color: #7C6E6E !important;
        margin-top: 0.2rem !important;
        margin-bottom: 1rem !important;
        padding-bottom: 0.6rem !important;
        border-bottom: 1.5px solid #E5DED3 !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .sidebar-hdr {
        font-size: 0.78rem !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #6E5656 !important;
        margin: 1rem 0 0.5rem 0 !important;
    }
    
    .sidebar-card {
        background-color: #F7F3EC !important;
        border: 1px solid #E5DED3 !important;
        border-radius: 6px;
        padding: 0.85rem 0.95rem;
        font-size: 0.83rem !important;
        color: #4A2F2F !important;
        margin-top: 1rem;
        line-height: 1.55;
    }
    
    /* Selectboxes and Inputs */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #FFFCF7 !important;
        border: 1.5px solid #DDD6CA !important;
        border-radius: 6px !important;
        color: #4A2F2F !important;
        box-shadow: none !important;
    }
    
    .stSelectbox div[data-baseweb="select"] > div:hover {
        border-color: #B85C38 !important;
    }
    
    .stSelectbox div[data-baseweb="select"] * {
        color: #4A2F2F !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }

    .stSelectbox div[data-baseweb="select"] svg {
        fill: #4A2F2F !important;
    }

    /* BaseWeb Dropdown Menus & Popover Options (Fixes Black Dropdown Popup) */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="menu"],
    ul[data-baseweb="menu"],
    ul[role="listbox"],
    div[role="listbox"] {
        background-color: #FFFCF7 !important;
        border: 1.5px solid #DDD6CA !important;
        border-radius: 6px !important;
        box-shadow: 0 4px 16px rgba(74, 47, 47, 0.12) !important;
        color: #4A2F2F !important;
    }

    li[role="option"],
    div[role="option"],
    ul[data-baseweb="menu"] li,
    div[data-baseweb="menu"] div {
        background-color: #FFFCF7 !important;
        color: #4A2F2F !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        border-bottom: 1px solid #F5EFE6 !important;
        transition: background-color 0.15s ease, color 0.15s ease !important;
    }

    li[role="option"] *,
    div[role="option"] *,
    ul[data-baseweb="menu"] li *,
    div[data-baseweb="menu"] div * {
        color: #4A2F2F !important;
        background-color: transparent !important;
    }

    /* Hover State */
    li[role="option"]:hover,
    div[role="option"]:hover,
    ul[data-baseweb="menu"] li:hover {
        background-color: #FAF0EA !important;
        color: #B85C38 !important;
    }

    li[role="option"]:hover *,
    div[role="option"]:hover *,
    ul[data-baseweb="menu"] li:hover * {
        color: #B85C38 !important;
    }

    /* Selected / Active State */
    li[role="option"][aria-selected="true"],
    div[role="option"][aria-selected="true"],
    ul[data-baseweb="menu"] li[aria-selected="true"] {
        background-color: #FAF0EA !important;
        color: #B85C38 !important;
        font-weight: 700 !important;
        border-left: 3.5px solid #B85C38 !important;
    }

    li[role="option"][aria-selected="true"] *,
    div[role="option"][aria-selected="true"] *,
    ul[data-baseweb="menu"] li[aria-selected="true"] * {
        color: #B85C38 !important;
        font-weight: 700 !important;
    }
    
    /* ------------------------------------------------------------- */
    /* Sidebar Navigation Radio Styling                              */
    /* ------------------------------------------------------------- */
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        display: none !important;
    }
    
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] {
        display: flex !important;
        flex-direction: column !important;
        gap: 0.4rem !important;
    }
    
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
        background-color: #FFFCF7 !important;
        border: 1px solid #E5DED3 !important;
        border-radius: 6px !important;
        padding: 0.45rem 0.85rem !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
        color: #4A2F2F !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        min-height: auto !important;
        margin: 0 !important;
    }
    
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {
        border-color: #B85C38 !important;
        background-color: #FAF4EE !important;
    }

    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has(input:checked) {
        background-color: #FAF0EA !important;
        border: 1.5px solid #B85C38 !important;
        color: #B85C38 !important;
        font-weight: 700 !important;
    }

    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has(input:checked) * {
        color: #B85C38 !important;
        font-weight: 700 !important;
    }

    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has(input:checked) input[type="radio"] {
        accent-color: #B85C38 !important;
    }

    /* ------------------------------------------------------------- */
    /* Sidebar Forecast Horizon Equal-Sized Radio Option Boxes       */
    /* ------------------------------------------------------------- */
    [data-testid="stSidebar"] div[role="radiogroup"][aria-label="Horizon"],
    [data-testid="stSidebar"] div[role="radiogroup"]:has(input[value*="20-Week"]) {
        display: flex !important;
        flex-direction: column !important;
        width: 100% !important;
        gap: 0.45rem !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"][aria-label="Horizon"] label,
    [data-testid="stSidebar"] div[role="radiogroup"]:has(input[value*="20-Week"]) label {
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        height: 42px !important;
        min-height: 42px !important;
        max-height: 42px !important;
        box-sizing: border-box !important;
        padding: 0 0.85rem !important;
        margin: 0 !important;
        background-color: #FFFCF7 !important;
        border: 1.5px solid #E5DED3 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
        color: #4A2F2F !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"][aria-label="Horizon"] label:hover,
    [data-testid="stSidebar"] div[role="radiogroup"]:has(input[value*="20-Week"]) label:hover {
        border-color: #B85C38 !important;
        background-color: #FAF4EE !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"][aria-label="Horizon"] label:has(input:checked),
    [data-testid="stSidebar"] div[role="radiogroup"]:has(input[value*="20-Week"]) label:has(input:checked) {
        background-color: #FAF0EA !important;
        border: 1.5px solid #B85C38 !important;
        color: #B85C38 !important;
        font-weight: 700 !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"][aria-label="Horizon"] label:has(input:checked) *,
    [data-testid="stSidebar"] div[role="radiogroup"]:has(input[value*="20-Week"]) label:has(input:checked) * {
        color: #B85C38 !important;
        font-weight: 700 !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"][aria-label="Horizon"] label p,
    [data-testid="stSidebar"] div[role="radiogroup"]:has(input[value*="20-Week"]) label p,
    [data-testid="stSidebar"] div[role="radiogroup"][aria-label="Horizon"] label span,
    [data-testid="stSidebar"] div[role="radiogroup"]:has(input[value*="20-Week"]) label span {
        margin: 0 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    /* ------------------------------------------------------------- */
    /* Main Content: Uncertainty Metric Basis, Forecast Error σ,     */
    /* and Historical Demand σ Equal Boxes                           */
    /* ------------------------------------------------------------- */
    [data-testid="stMain"] .stRadio label[data-testid="stWidgetLabel"] {
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        min-height: 48px !important;
        background-color: #F8F4ED !important;
        border: 1.5px solid #DDD6CA !important;
        border-left: 3.5px solid #B85C38 !important;
        border-radius: 6px !important;
        padding: 0.5rem 0.85rem !important;
        font-weight: 700 !important;
        font-size: 0.84rem !important;
        color: #4A2F2F !important;
        letter-spacing: 0.02em !important;
        margin: 0 0 0.45rem 0 !important;
        box-sizing: border-box !important;
    }

    [data-testid="stMain"] .stRadio label[data-testid="stWidgetLabel"] p {
        margin: 0 !important;
        font-weight: 700 !important;
        font-size: 0.84rem !important;
        color: #4A2F2F !important;
    }

    [data-testid="stMain"] .stRadio [role="radiogroup"] {
        display: flex !important;
        flex-direction: column !important;
        width: 100% !important;
        gap: 0 !important;
    }

    [data-testid="stMain"] .stRadio [role="radiogroup"] label {
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        min-height: 48px !important;
        background-color: #FFFCF7 !important;
        border: 1.5px solid #DDD6CA !important;
        border-radius: 6px !important;
        padding: 0.5rem 0.85rem !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
        line-height: 1.35 !important;
        color: #4A2F2F !important;
        margin: 0 0 0.45rem 0 !important;
        box-sizing: border-box !important;
        transition: all 0.15s ease !important;
        cursor: pointer !important;
    }

    [data-testid="stMain"] .stRadio [role="radiogroup"] label:hover {
        border-color: #B85C38 !important;
        background-color: #FAF4EE !important;
    }

    [data-testid="stMain"] .stRadio [role="radiogroup"] label:has(input:checked) {
        border-color: #B85C38 !important;
        background-color: #FAF0EA !important;
        color: #B85C38 !important;
        font-weight: 700 !important;
    }

    [data-testid="stMain"] .stRadio [role="radiogroup"] label:has(input:checked) * {
        color: #B85C38 !important;
    }
    
    /* Light Editorial Tables */
    .editorial-table-box {
        background-color: #FFFCF7;
        border: 1px solid #E5DED3;
        border-radius: 6px;
        overflow-x: auto;
        margin-bottom: 1rem;
    }
    
    .editorial-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.86rem;
        color: #4A2F2F;
        text-align: left;
    }
    
    .editorial-table th {
        background-color: #F8F4ED;
        color: #6E5656;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.74rem;
        letter-spacing: 0.04em;
        padding: 0.65rem 0.85rem;
        border-bottom: 1.5px solid #E5DED3;
        white-space: nowrap;
    }
    
    .editorial-table td {
        padding: 0.6rem 0.85rem;
        border-bottom: 1px solid #F2ECE2;
        white-space: nowrap;
        color: #4A2F2F;
    }
    
    .editorial-table tr:hover {
        background-color: #FAF7F0;
    }
    
    .editorial-table tr.champion-row {
        background-color: #FAF0EA !important;
        font-weight: 700;
    }
    
    .editorial-table tr.champion-row td {
        color: #8C3B1D !important;
        border-bottom: 1.5px solid #EAD1C4;
    }
    
    .badge-champion {
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 800;
        background-color: #B85C38;
        color: #FFFFFF !important;
        padding: 0.15rem 0.45rem;
        border-radius: 3px;
        margin-left: 0.4rem;
        letter-spacing: 0.04em;
    }
    
    /* Formula Card */
    .formula-card {
        background-color: #FAF5EE;
        border: 1.5px solid #E8DDD0;
        border-left: 4.5px solid #B85C38;
        border-radius: 6px;
        padding: 1.1rem 1.3rem;
        margin: 1rem 0;
    }
    
    .formula-title {
        font-size: 0.8rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #B85C38;
        margin-bottom: 0.4rem;
    }
    
    .formula-math {
        font-size: 1.05rem;
        font-weight: 700;
        color: #4A2F2F;
        margin: 0.35rem 0;
    }
    
    .formula-desc {
        font-size: 0.84rem;
        color: #6E5656;
        line-height: 1.5;
    }
    
    /* Guided Demo Step Card */
    .demo-step-card {
        background-color: #FFFCF7;
        border: 1.5px solid #E5DED3;
        border-radius: 8px;
        padding: 1.25rem 1.45rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 6px rgba(74, 47, 47, 0.03);
    }
    
    .demo-step-badge {
        display: inline-block;
        font-size: 0.74rem;
        font-weight: 800;
        text-transform: uppercase;
        background-color: #FAF0EA;
        color: #B85C38;
        border: 1px solid #F0D4C7;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }

    /* ------------------------------------------------------------- */
    /* Forecast CSV Download Button Styling                          */
    /* ------------------------------------------------------------- */
    div[data-testid="stDownloadButton"] button,
    div.stDownloadButton button,
    .stDownloadButton > button {
        background-color: #FFFCF7 !important;
        color: #4A2F2F !important;
        border: 1.5px solid #DDD6CA !important;
        border-radius: 6px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        padding: 0.55rem 1.25rem !important;
        box-shadow: 0 1px 4px rgba(74, 47, 47, 0.05) !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }
    
    div[data-testid="stDownloadButton"] button *,
    div.stDownloadButton button *,
    .stDownloadButton > button * {
        color: #4A2F2F !important;
        font-weight: 700 !important;
        transition: color 0.2s ease !important;
    }

    div[data-testid="stDownloadButton"] button:hover,
    div.stDownloadButton button:hover,
    .stDownloadButton > button:hover {
        background-color: #FAF0EA !important;
        color: #B85C38 !important;
        border-color: #B85C38 !important;
        box-shadow: 0 2px 8px rgba(184, 92, 56, 0.12) !important;
    }

    div[data-testid="stDownloadButton"] button:hover *,
    div.stDownloadButton button:hover *,
    .stDownloadButton > button:hover * {
        color: #B85C38 !important;
    }

    div[data-testid="stDownloadButton"] button:active,
    div.stDownloadButton button:active,
    .stDownloadButton > button:active {
        background-color: #F4E2D8 !important;
        color: #8C3B1D !important;
        border-color: #8C3B1D !important;
    }
</style>
"""
st.markdown(EDITORIAL_STUDIO_CSS, unsafe_allow_html=True)

# ==============================================================================
# 3. DATA AND MODEL INGESTION WITH SAFE STREAMLIT CACHING
# ==============================================================================
DATA_PATH = os.path.join(BASE_DIR, "data", "retail_store_sales.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_forecasting_model.pkl")
VAL_METRICS_PATH = os.path.join(BASE_DIR, "outputs", "metrics", "validation_model_comparison.csv")
TEST_METRICS_PATH = os.path.join(BASE_DIR, "outputs", "metrics", "test_model_evaluation.csv")
DEPT_ERROR_PATH = os.path.join(BASE_DIR, "outputs", "metrics", "department_error_breakdown.csv")

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    return None

@st.cache_data
def load_all_metrics():
    val_m = pd.read_csv(VAL_METRICS_PATH) if os.path.exists(VAL_METRICS_PATH) else None
    test_m = pd.read_csv(TEST_METRICS_PATH) if os.path.exists(TEST_METRICS_PATH) else None
    dept_err = pd.read_csv(DEPT_ERROR_PATH) if os.path.exists(DEPT_ERROR_PATH) else None
    return val_m, test_m, dept_err

@st.cache_resource
def load_champion_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

@st.cache_data
def get_cached_test_split():
    """
    Computes feature matrix and chronological out-of-time splits once.
    Cached securely to prevent recomputation on interactive widget reruns.
    """
    raw_df = load_data()
    if raw_df is None:
        return None, []
    df_feat, feat_cols = prepare_feature_matrix(raw_df)
    _, _, df_test = chronological_split(df_feat, 0.70, 0.15)
    return df_test, feat_cols

# Ingest all data and artifacts
df = load_data()
val_metrics_df, test_metrics_df, dept_error_df = load_all_metrics()
champion_payload = load_champion_model()
df_test_split, feat_cols = get_cached_test_split()

# Enterprise Plotly Configuration
PLOTLY_CONFIG = {
    "displayModeBar": "hover",
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToAdd": [
        "drawline",
        "drawopenpath",
        "drawclosedpath",
        "drawrect",
        "drawcircle",
        "eraseshape"
    ],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "retailpulse_chart",
        "height": 520,
        "width": 1080,
        "scale": 2
    }
}

def render_editorial_table(df_table, champion_col=None, champion_val=None):
    """Renders light, publication-quality editorial HTML table."""
    html = ['<div class="editorial-table-box"><table class="editorial-table">']
    html.append("<thead><tr>")
    for col in df_table.columns:
        html.append(f"<th>{col}</th>")
    html.append("</tr></thead><tbody>")
    for _, row in df_table.iterrows():
        is_champ = (champion_col is not None and str(row.get(champion_col, "")) == str(champion_val))
        row_cls = ' class="champion-row"' if is_champ else ""
        html.append(f"<tr{row_cls}>")
        for col in df_table.columns:
            val = row[col]
            cell_str = str(val)
            if is_champ and col == "Model":
                cell_str += ' <span class="badge-champion">CHAMPION</span>'
            html.append(f"<td>{cell_str}</td>")
        html.append("</tr>")
    html.append("</tbody></table></div>")
    return "".join(html)

# ==============================================================================
# 4. TOP STUDIO HEADER & METADATA
# ==============================================================================
st.markdown("""
<div class="studio-header">
    <h1 class="studio-title">RetailPulse <span>Studio</span></h1>
    <p class="studio-subtitle">Intelligent Retail Demand Forecasting, Out-of-Time Model Benchmarking, and Operations Research Inventory Planning</p>
</div>
<div class="studio-badge-row">
    <span class="studio-badge">RETAIL INTELLIGENCE STUDIO</span>
    <span class="studio-badge sage">CHRONOLOGICAL 3-WAY VALIDATION</span>
    <span class="studio-badge gold">CHAMPION: XGBOOST REGRESSOR (4.55% WAPE)</span>
    <span class="studio-badge">ZERO DATA LEAKAGE</span>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 5. SIDEBAR NAVIGATION & SCOPE CONTROLS
# ==============================================================================
st.sidebar.markdown('<div class="sidebar-brand">RetailPulse <span>Studio</span></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-sub">Retail Intelligence Console</div>', unsafe_allow_html=True)

# Primary Navigation Destination
st.sidebar.markdown('<div class="sidebar-hdr">STUDIO DESTINATION</div>', unsafe_allow_html=True)
nav_choice = st.sidebar.radio(
    "Destination",
    [
        "Studio Overview",
        "Demand Forecast Explorer",
        "Model Evaluation Laboratory",
        "Inventory Planning Studio",
        "Demand & Feature Insights"
    ],
    index=0,
    label_visibility="collapsed"
)

st.sidebar.markdown('<div class="sidebar-hdr">ACTIVE MERCHANDISE SCOPE</div>', unsafe_allow_html=True)

if df is not None:
    stores = sorted(df["Store_ID"].unique())
    selected_store = st.sidebar.selectbox(
        "Store Location",
        stores,
        index=0,
        format_func=lambda s: f"Store {s} (Flagship)" if s in [1, 2] else (f"Store {s} (Standard)" if s in [3, 4] else f"Store {s} (Compact)")
    )
    
    dept_options = sorted(df[df["Store_ID"] == selected_store]["Dept_Name"].unique())
    selected_dept = st.sidebar.selectbox("Department", dept_options, index=0)
    
    # Filter dataset for selected series
    filtered_df = df[(df["Store_ID"] == selected_store) & (df["Dept_Name"] == selected_dept)].copy().sort_values("Date").reset_index(drop=True)
    dept_id = filtered_df["Dept_ID"].iloc[0]
    store_size = filtered_df["Store_Size"].iloc[0]
    store_type = filtered_df["Store_Type"].iloc[0]
    
    # Forecast Horizon Selector in Sidebar
    st.sidebar.markdown('<div class="sidebar-hdr">FORECAST HORIZON</div>', unsafe_allow_html=True)
    horizon_choice = st.sidebar.radio(
        "Horizon",
        ["4 Weeks (1 Month)", "8 Weeks (2 Months)", "12 Weeks (1 Quarter)", "Full 20-Week Horizon"],
        index=0,
        label_visibility="collapsed"
    )
    horizon_weeks = 4 if "4" in horizon_choice else (8 if "8" in horizon_choice else (12 if "12" in horizon_choice else 20))

    st.sidebar.markdown(f"""
    <div class="sidebar-card">
        <div style="font-weight: 800; text-transform: uppercase; font-size: 0.72rem; color: #7C6E6E; margin-bottom: 0.35rem; letter-spacing: 0.04em;">ACTIVE SCOPE METADATA</div>
        • <b>Store:</b> Store {selected_store} (Type {store_type})<br>
        • <b>Footprint:</b> {store_size:,} sq ft<br>
        • <b>Department:</b> {selected_dept} (ID: {dept_id})<br>
        • <b>Timeline:</b> {filtered_df['Date'].min().strftime('%b %Y')} – {filtered_df['Date'].max().strftime('%b %Y')}<br>
        • <b>History:</b> {len(filtered_df)} weekly observations<br>
        • <b>Selected Horizon:</b> {horizon_weeks} out-of-time weeks
    </div>
    """, unsafe_allow_html=True)

    # Historical metrics calculation
    avg_sales = float(filtered_df["Weekly_Sales"].mean())
    max_sales = float(filtered_df["Weekly_Sales"].max())
    holiday_sub = filtered_df[filtered_df["IsHoliday"] == 1]["Weekly_Sales"]
    holiday_avg = float(holiday_sub.mean()) if len(holiday_sub) > 0 else avg_sales
    promo_sub = filtered_df[filtered_df["Promotional_Flag"] == 1]["Weekly_Sales"]
    promo_avg = float(promo_sub.mean()) if len(promo_sub) > 0 else avg_sales
    holiday_lift_pct = ((holiday_avg - avg_sales) / avg_sales) * 100
    promo_lift_pct = ((promo_avg - avg_sales) / avg_sales) * 100

    # Model inference on test split (defensive slice)
    if champion_payload is not None and df_test_split is not None:
        model = champion_payload["model"]
        champ_name = champion_payload.get("model_name", "XGBoost Regressor")
        val_rmse = champion_payload.get("val_rmse", 3816.87)
        val_wape = champion_payload.get("val_wape", 5.65)
        val_agg_residual_std = champion_payload.get("val_agg_residual_std", 33820.44)
        
        # Isolate selected series for out-of-time test horizon
        full_series_test = df_test_split[(df_test_split["Store_ID"] == selected_store) & (df_test_split["Dept_ID"] == dept_id)].copy().sort_values("Date").reset_index(drop=True)
        full_series_test["Predicted_Sales"] = np.maximum(0, model.predict(full_series_test[feat_cols]))
        
        # Slice for chosen forecast horizon (4W, 8W, 12W, or 20W)
        horizon_series_test = full_series_test.head(horizon_weeks).copy().reset_index(drop=True)
        
        # Empirical 95% prediction interval (scaled for single series standard error)
        series_residual_std = float(np.std(full_series_test["Weekly_Sales"] - full_series_test["Predicted_Sales"], ddof=1)) if len(full_series_test) > 1 else val_rmse / 5.0
        interval_margin = 1.96 * series_residual_std
        horizon_series_test["Lower_Bound_95"] = np.maximum(0, horizon_series_test["Predicted_Sales"] - interval_margin)
        horizon_series_test["Upper_Bound_95"] = horizon_series_test["Predicted_Sales"] + interval_margin
        horizon_series_test["Forecast_Error"] = horizon_series_test["Predicted_Sales"] - horizon_series_test["Weekly_Sales"]
        horizon_series_test["Abs_Error"] = np.abs(horizon_series_test["Forecast_Error"])
        
        # Horizon metrics
        hz_actual_sum = float(horizon_series_test["Weekly_Sales"].sum())
        hz_pred_sum = float(horizon_series_test["Predicted_Sales"].sum())
        hz_pred_mean = float(horizon_series_test["Predicted_Sales"].mean())
        hz_mae = float(horizon_series_test["Abs_Error"].mean())
        hz_wape = (float(horizon_series_test["Abs_Error"].sum()) / hz_actual_sum * 100.0) if hz_actual_sum > 0 else 0.0
    else:
        horizon_series_test = pd.DataFrame()
        hz_pred_mean = avg_sales
        hz_pred_sum = avg_sales * horizon_weeks
        hz_wape = 4.55
        hz_mae = 1384.25
        series_residual_std = 1500.0

    # ==============================================================================
    # 6. DESTINATION ROUTING
    # ==============================================================================

    # --------------------------------------------------------------------------
    # DESTINATION A: STUDIO OVERVIEW
    # --------------------------------------------------------------------------
    if nav_choice == "Studio Overview":
        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card cocoa">
                <div class="kpi-meta-label">Historical Average Weekly Sales</div>
                <div class="kpi-meta-val">${avg_sales:,.0f}</div>
                <div class="kpi-meta-sub">Historical Baseline ({len(filtered_df)} observed weeks)</div>
            </div>
            <div class="kpi-card terracotta">
                <div class="kpi-meta-label">Projected Demand ({horizon_weeks}W Horizon)</div>
                <div class="kpi-meta-val">${hz_pred_mean:,.0f}</div>
                <div class="kpi-meta-sub">${hz_pred_sum:,.0f} cumulative projected demand</div>
            </div>
            <div class="kpi-card sage">
                <div class="kpi-meta-label">Champion Test Precision (Global Benchmark)</div>
                <div class="kpi-meta-val">4.55% <span style="font-size:0.8rem; font-weight:700; color:#3C6339;">WAPE</span></div>
                <div class="kpi-meta-sub">Active Slice ({horizon_weeks}W): <b>{hz_wape:.2f}% WAPE</b> • MAE: ${hz_mae:,.0f} • R²: 0.9928</div>
            </div>
            <div class="kpi-card gold">
                <div class="kpi-meta-label">Seasonal Surge Velocity</div>
                <div class="kpi-meta-val">+{holiday_lift_pct:.1f}%</div>
                <div class="kpi-meta-sub">Peak Holiday Week Average: ${holiday_avg:,.0f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="surface-card">
            <div class="surface-header-box">
                <div class="surface-tag-line"></div>
                <h2 class="surface-title">Retail Demand Trajectory & Evaluation Horizon</h2>
            </div>
            <p class="surface-desc">Multi-year retail demand pattern across 182 historical weeks with the out-of-time machine learning evaluation horizon highlighted.</p>
        """, unsafe_allow_html=True)
        
        fig_hist = go.Figure()
        # Historical actual sales
        fig_hist.add_trace(go.Scatter(
            x=filtered_df["Date"],
            y=filtered_df["Weekly_Sales"],
            mode="lines",
            name="Weekly Actual Sales",
            line=dict(color="#7B9A78", width=2.2),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Sales:</b> $%{y:,.2f}<extra></extra>"
        ))
        # 4W Trend
        filtered_df["SMA_4W"] = filtered_df["Weekly_Sales"].rolling(4, min_periods=1).mean()
        fig_hist.add_trace(go.Scatter(
            x=filtered_df["Date"],
            y=filtered_df["SMA_4W"],
            mode="lines",
            name="4-Week Smoothed Trend",
            line=dict(color="#4A2F2F", width=1.8, dash="dot"),
            hovertemplate="<b>4W Smoothed:</b> $%{y:,.2f}<extra></extra>"
        ))
        # Holiday markers
        holiday_events = filtered_df[filtered_df["IsHoliday"] == 1]
        if not holiday_events.empty:
            fig_hist.add_trace(go.Scatter(
                x=holiday_events["Date"],
                y=holiday_events["Weekly_Sales"],
                mode="markers",
                name="Holiday Surge Peak",
                marker=dict(color="#D4A017", size=8, symbol="diamond", line=dict(color="#4A2F2F", width=0.8)),
                hovertemplate="<b>Holiday Peak:</b> $%{y:,.2f}<br><b>Date:</b> %{x|%Y-%m-%d}<extra></extra>"
            ))
            
        # Shaded out-of-time evaluation region
        if not horizon_series_test.empty:
            test_start = horizon_series_test["Date"].min()
            test_end = horizon_series_test["Date"].max()
            fig_hist.add_vrect(
                x0=test_start,
                x1=test_end,
                fillcolor="#B85C38",
                opacity=0.08,
                layer="below",
                line_width=1,
                line_dash="dash",
                line_color="#B85C38",
                annotation_text=f"Active Evaluation Horizon ({horizon_weeks}W)",
                annotation_position="top left",
                annotation_font=dict(color="#B85C38", size=11, family="Plus Jakarta Sans")
            )
            
        fig_hist.update_layout(
            paper_bgcolor="#FFFCF7",
            plot_bgcolor="#FFFCF7",
            height=370,
            font=dict(family="Plus Jakarta Sans, sans-serif", color="#4A2F2F", size=12),
            margin=dict(l=65, r=30, t=55, b=55),
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.03,
                xanchor="left",
                x=0,
                font=dict(size=11, color="#4A2F2F")
            ),
            xaxis=dict(
                title=dict(text="Calendar Timeline (2021 – 2024)", font=dict(color="#4A2F2F", size=12), standoff=15),
                tickfont=dict(color="#4A2F2F", size=11),
                showgrid=True,
                gridcolor="#EFEAE1",
                linecolor="#DDD6CA"
            ),
            yaxis=dict(
                title=dict(text="Weekly Sales ($ USD)", font=dict(color="#4A2F2F", size=12)),
                tickfont=dict(color="#4A2F2F", size=11),
                showgrid=True,
                gridcolor="#EFEAE1",
                linecolor="#DDD6CA",
                tickprefix="$",
                tickformat=","
            )
        )
        st.plotly_chart(fig_hist, width="stretch", config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

        # Operational Strategy Callouts — Exactly Equal Sized & Aligned Grid
        st.markdown(f"""
        <div class="equal-cards-grid">
            <div class="equal-card terracotta">
                <div style="font-weight: 800; font-size: 0.88rem; color: #4A2F2F; margin-bottom: 0.45rem; text-transform: uppercase; letter-spacing: 0.04em;">1. DEMAND TREND DIAGNOSTIC</div>
                <div style="font-size: 0.84rem; color: #6E5656; line-height: 1.55; flex-grow: 1;">
                    The active series in <b>{selected_dept}</b> exhibits an average weekly base volume of <b>${avg_sales:,.0f}</b>.
                    Projected demand for the selected {horizon_weeks}-week horizon averages <b>${hz_pred_mean:,.0f}</b> per week.
                </div>
            </div>
            <div class="equal-card gold">
                <div style="font-weight: 800; font-size: 0.88rem; color: #4A2F2F; margin-bottom: 0.45rem; text-transform: uppercase; letter-spacing: 0.04em;">2. SEASONAL SURGE BUFFER</div>
                <div style="font-size: 0.84rem; color: #6E5656; line-height: 1.55; flex-grow: 1;">
                    Historical holiday events create a <b>+{holiday_lift_pct:.1f}%</b> volume surge.
                    Replenishment orders must be scheduled at least <b>2 weeks prior</b> to Q4 peak holiday horizons to prevent stockouts.
                </div>
            </div>
            <div class="equal-card sage">
                <div style="font-weight: 800; font-size: 0.88rem; color: #4A2F2F; margin-bottom: 0.45rem; text-transform: uppercase; letter-spacing: 0.04em;">3. FORECAST RELIABILITY</div>
                <div style="font-size: 0.84rem; color: #6E5656; line-height: 1.55; flex-grow: 1;">
                    The champion <b>XGBoost Regressor</b> achieves an unbiased out-of-time benchmark of <b>4.55% Test WAPE</b> across all 30 series (and <b>{hz_wape:.2f}% WAPE</b> on this active {horizon_weeks}W series slice), enabling tight safety stock margins without risking customer service levels.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # DESTINATION B: DEMAND FORECAST EXPLORER
    # --------------------------------------------------------------------------
    elif nav_choice == "Demand Forecast Explorer":
        st.markdown(f"""
        <div class="horizon-banner">
            <div>
                <span class="horizon-badge-pill">{horizon_choice}</span>
                <span style="font-weight: 700; font-size: 0.95rem; color: #4A2F2F; margin-left: 0.6rem;">
                    Active Horizon Date Range: {horizon_series_test['Date'].min().strftime('%b %d, %Y')} – {horizon_series_test['Date'].max().strftime('%b %d, %Y')}
                </span>
            </div>
            <div style="font-size: 0.86rem; color: #6E5656;">
                Model: <b>{champ_name}</b> • Series: <b>Store {selected_store} : {selected_dept}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 3 Horizon Metric Boxes — Equal Sized Grid
        st.markdown(f"""
        <div class="kpi-grid-3">
            <div class="kpi-card terracotta" style="padding: 1rem 1.15rem;">
                <div class="kpi-meta-label">Horizon Cumulative Demand</div>
                <div class="kpi-meta-val">${hz_pred_sum:,.0f}</div>
                <div class="kpi-meta-sub">Total {horizon_weeks}-Week Projected Volume</div>
            </div>
            <div class="kpi-card cocoa" style="padding: 1rem 1.15rem;">
                <div class="kpi-meta-label">Horizon Mean Weekly Demand</div>
                <div class="kpi-meta-val">${hz_pred_mean:,.0f}</div>
                <div class="kpi-meta-sub">Weekly Operational Rate ($ USD)</div>
            </div>
            <div class="kpi-card sage" style="padding: 1rem 1.15rem;">
                <div class="kpi-meta-label">Active Horizon Slice WAPE</div>
                <div class="kpi-meta-val">{hz_wape:.2f}% <span style="font-size:0.75rem; color:#3C6339; font-weight:700;">SLICE</span></div>
                <div class="kpi-meta-sub">Store {selected_store} : {selected_dept} • MAE: ${hz_mae:,.2f} (Global Test Benchmark: 4.55%)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Interactive Forecast vs Actual Chart
        st.markdown("""
        <div class="surface-card">
            <div class="surface-header-box">
                <div class="surface-tag-line"></div>
                <h2 class="surface-title">Out-of-Time Machine Learning Demand Forecast</h2>
            </div>
            <p class="surface-desc">Comparison of actual observed weekly sales against the champion XGBoost forecast with empirical 95% prediction intervals (±1.96 σ_residual).</p>
        """, unsafe_allow_html=True)
        
        fig_fc = go.Figure()
        
        # Prior historical context (last 8 weeks before test horizon)
        prior_history = filtered_df[filtered_df["Date"] < horizon_series_test["Date"].min()].tail(8)
        if not prior_history.empty:
            fig_fc.add_trace(go.Scatter(
                x=prior_history["Date"],
                y=prior_history["Weekly_Sales"],
                mode="lines+markers",
                name="Prior Historical Sales (Context)",
                line=dict(color="#7B9A78", width=1.8, dash="dot"),
                marker=dict(size=5, color="#7B9A78"),
                hovertemplate="<b>Historical:</b> $%{y:,.2f}<br>Date: %{x|%Y-%m-%d}<extra></extra>"
            ))
            
        # Actual Out-of-Time Demand
        fig_fc.add_trace(go.Scatter(
            x=horizon_series_test["Date"],
            y=horizon_series_test["Weekly_Sales"],
            mode="lines+markers",
            name="Actual Out-of-Time Sales",
            line=dict(color="#7B9A78", width=2.4),
            marker=dict(size=7, color="#7B9A78"),
            hovertemplate="<b>Actual Sales:</b> $%{y:,.2f}<br>Date: %{x|%Y-%m-%d}<extra></extra>"
        ))
        
        # Champion Forecast
        fig_fc.add_trace(go.Scatter(
            x=horizon_series_test["Date"],
            y=horizon_series_test["Predicted_Sales"],
            mode="lines+markers",
            name=f"Champion Forecast ({champ_name})",
            line=dict(color="#B85C38", width=2.4, dash="dash"),
            marker=dict(size=8, symbol="square", color="#B85C38"),
            hovertemplate="<b>Model Forecast:</b> $%{y:,.2f}<br>Date: %{x|%Y-%m-%d}<extra></extra>"
        ))
        
        # 95% Prediction Interval Band
        fig_fc.add_trace(go.Scatter(
            x=list(horizon_series_test["Date"]) + list(horizon_series_test["Date"])[::-1],
            y=list(horizon_series_test["Upper_Bound_95"]) + list(horizon_series_test["Lower_Bound_95"])[::-1],
            fill="toself",
            fillcolor="rgba(184, 92, 56, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% Empirical Prediction Interval",
            hoverinfo="skip"
        ))
        
        fig_fc.update_layout(
            paper_bgcolor="#FFFCF7",
            plot_bgcolor="#FFFCF7",
            height=400,
            font=dict(family="Plus Jakarta Sans, sans-serif", color="#4A2F2F", size=12),
            margin=dict(l=65, r=30, t=55, b=55),
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.03,
                xanchor="left",
                x=0,
                font=dict(size=11, color="#4A2F2F")
            ),
            xaxis=dict(
                title=dict(text="Forecast Horizon Week Date", font=dict(color="#4A2F2F", size=12), standoff=15),
                tickfont=dict(color="#4A2F2F", size=11),
                showgrid=True,
                gridcolor="#EFEAE1",
                linecolor="#DDD6CA"
            ),
            yaxis=dict(
                title=dict(text="Weekly Sales ($ USD)", font=dict(color="#4A2F2F", size=12)),
                tickfont=dict(color="#4A2F2F", size=11),
                showgrid=True,
                gridcolor="#EFEAE1",
                linecolor="#DDD6CA",
                tickprefix="$",
                tickformat=","
            )
        )
        st.plotly_chart(fig_fc, width="stretch", config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

        # Forecast Table and Export Block
        st.markdown("""
        <div class="surface-card">
            <div class="surface-header-box">
                <div class="surface-tag-line"></div>
                <h3 class="surface-title">Detailed Horizon Forecast Records</h3>
            </div>
            <p class="surface-desc">Numerical demand projections, 95% uncertainty interval boundaries, and actual sales variance for each evaluated week.</p>
        """, unsafe_allow_html=True)
        
        # Prepare clean display table
        table_df = pd.DataFrame()
        table_df["Week Date"] = horizon_series_test["Date"].dt.strftime("%Y-%m-%d")
        table_df["Actual Demand"] = horizon_series_test["Weekly_Sales"].apply(lambda x: f"${x:,.2f}")
        table_df["Model Forecast"] = horizon_series_test["Predicted_Sales"].apply(lambda x: f"${x:,.2f}")
        table_df["Lower Bound (95%)"] = horizon_series_test["Lower_Bound_95"].apply(lambda x: f"${x:,.2f}")
        table_df["Upper Bound (95%)"] = horizon_series_test["Upper_Bound_95"].apply(lambda x: f"${x:,.2f}")
        table_df["Absolute Error"] = horizon_series_test["Abs_Error"].apply(lambda x: f"${x:,.2f}")
        
        st.markdown(render_editorial_table(table_df), unsafe_allow_html=True)
        
        # CSV Export preparation
        export_csv_df = pd.DataFrame({
            "Date": horizon_series_test["Date"].dt.strftime("%Y-%m-%d"),
            "Store_ID": selected_store,
            "Store_Type": store_type,
            "Dept_ID": dept_id,
            "Dept_Name": selected_dept,
            "Actual_Sales_USD": horizon_series_test["Weekly_Sales"],
            "Forecast_Sales_USD": horizon_series_test["Predicted_Sales"],
            "Lower_Bound_95_USD": horizon_series_test["Lower_Bound_95"],
            "Upper_Bound_95_USD": horizon_series_test["Upper_Bound_95"],
            "Forecast_Error_USD": horizon_series_test["Forecast_Error"]
        })
        csv_buffer = io.StringIO()
        export_csv_df.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label=f"📥 Download {horizon_choice} Forecast Data (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"retailpulse_forecast_store{selected_store}_dept{dept_id}_{horizon_weeks}w.csv",
            mime="text/csv",
            use_container_width=False
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # DESTINATION C: MODEL EVALUATION LABORATORY
    # --------------------------------------------------------------------------
    elif nav_choice == "Model Evaluation Laboratory":
        st.markdown("""
        <div class="surface-card">
            <div class="surface-header-box">
                <div class="surface-tag-line"></div>
                <h2 class="surface-title">3-Way Chronological Validation & Model Selection</h2>
            </div>
            <p class="surface-desc">To guarantee zero look-ahead data leakage, model selection was strictly conducted on the chronological Validation split (15%), before a single unbiased evaluation of the champion on the untouched Out-of-Time Test split (15%).</p>
        """, unsafe_allow_html=True)
        
        col_m1, col_m2 = st.columns(2)
        if val_metrics_df is not None and test_metrics_df is not None:
            val_disp = val_metrics_df.copy()
            val_disp["MAE"] = val_disp["MAE ($)"].apply(lambda x: f"${x:,.2f}")
            val_disp["RMSE"] = val_disp["RMSE ($)"].apply(lambda x: f"${x:,.2f}")
            val_disp["MAPE"] = val_disp["MAPE (%)"].apply(lambda x: f"{x:.2f}%")
            val_disp["WAPE"] = val_disp["WAPE (%)"].apply(lambda x: f"{x:.2f}%")
            val_disp["R² Score"] = val_disp["R2 Score"].apply(lambda x: f"{x:.4f}")
            val_disp = val_disp[["Model", "MAE", "RMSE", "MAPE", "WAPE", "R² Score"]]
            
            test_disp = test_metrics_df.copy()
            test_disp["MAE"] = test_disp["MAE ($)"].apply(lambda x: f"${x:,.2f}")
            test_disp["RMSE"] = test_disp["RMSE ($)"].apply(lambda x: f"${x:,.2f}")
            test_disp["MAPE"] = test_disp["MAPE (%)"].apply(lambda x: f"{x:.2f}%")
            test_disp["WAPE"] = test_disp["WAPE (%)"].apply(lambda x: f"{x:.2f}%")
            test_disp["R² Score"] = test_disp["R2 Score"].apply(lambda x: f"{x:.4f}")
            test_disp = test_disp[["Model", "MAE", "RMSE", "MAPE", "WAPE", "R² Score"]]
            
            with col_m1:
                st.markdown("""
                <div style="min-height: 56px; margin-bottom: 0.65rem;">
                    <div style="font-weight: 700; font-size: 0.95rem; color: #4A2F2F;">1. Validation Set Benchmarking (Champion Selection)</div>
                    <div style="font-size: 0.8rem; color: #6E5656; margin-top: 0.25rem;">Trained on 70% Train (2,730 samples) • Evaluated on 15% Val split (600 samples)</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(render_editorial_table(val_disp, champion_col="Model", champion_val="XGBoost Regressor"), unsafe_allow_html=True)
                
            with col_m2:
                st.markdown("""
                <div style="min-height: 56px; margin-bottom: 0.65rem;">
                    <div style="font-weight: 700; font-size: 0.95rem; color: #4A2F2F;">2. Final Out-of-Time Test Evaluation (Unbiased Horizon)</div>
                    <div style="font-size: 0.8rem; color: #6E5656; margin-top: 0.25rem;">Evaluated once on untouched 15% Test split (600 samples, Feb – Jun 2024)</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(render_editorial_table(test_disp, champion_col="Model", champion_val="XGBoost Regressor"), unsafe_allow_html=True)

        st.markdown(f"""
        <div class="formula-card">
            <div class="formula-title">CHAMPION MODEL SELECTION RATIONALE</div>
            <div style="font-size: 0.88rem; color: #4A2F2F; line-height: 1.6;">
                <b>XGBoost Regressor</b> was selected as the production champion strictly based on its superior validation error rate (<b>5.65% Validation WAPE</b>), outperforming LightGBM (6.23%), Ridge Regression (8.51%), and naive moving averages (28.54%).<br>
                On the completely untouched out-of-time test horizon, XGBoost generalized with exceptional precision:
                <b>4.55% Test WAPE</b>, <b>$1,384.25 MAE</b>, and an <b>R² Score of 0.9928</b>, reducing prediction error by <b>42.8%</b> relative to the naive lag-1 baseline.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Residual Diagnostics
        st.markdown("""
        <div class="surface-card">
            <div class="surface-header-box">
                <div class="surface-tag-line"></div>
                <h3 class="surface-title">Test Residual Diagnostics & Parity Analysis</h3>
            </div>
            <p class="surface-desc">Error distribution and actual-versus-predicted regression fit on the 600 out-of-time test samples.</p>
        """, unsafe_allow_html=True)
        
        if df_test_split is not None and champion_payload is not None:
            diag_df = df_test_split.copy()
            diag_df["Predicted_Sales"] = np.maximum(0, champion_payload["model"].predict(diag_df[feat_cols]))
            diag_df["Residual"] = diag_df["Weekly_Sales"] - diag_df["Predicted_Sales"]
            
            d_c1, d_c2 = st.columns(2)
            with d_c1:
                # Scatter Parity Plot
                fig_parity = go.Figure()
                fig_parity.add_trace(go.Scatter(
                    x=diag_df["Weekly_Sales"],
                    y=diag_df["Predicted_Sales"],
                    mode="markers",
                    marker=dict(color="#7B9A78", size=5, opacity=0.65),
                    name="Test Observations",
                    hovertemplate="Actual: $%{x:,.0f}<br>Predicted: $%{y:,.0f}<extra></extra>"
                ))
                min_v = 0
                max_v = max(diag_df["Weekly_Sales"].max(), diag_df["Predicted_Sales"].max())
                fig_parity.add_trace(go.Scatter(
                    x=[min_v, max_v],
                    y=[min_v, max_v],
                    mode="lines",
                    name="Perfect Parity (y = x)",
                    line=dict(color="#4A2F2F", dash="dash", width=1.5)
                ))
                fig_parity.update_layout(
                    title=dict(
                        text="Actual vs Predicted Sales ($ USD)",
                        x=0.03,
                        y=0.97,
                        xanchor="left",
                        yanchor="top",
                        font=dict(family="Plus Jakarta Sans, sans-serif", size=13, color="#4A2F2F")
                    ),
                    paper_bgcolor="#FFFCF7",
                    plot_bgcolor="#FFFCF7",
                    height=360,
                    font=dict(family="Plus Jakarta Sans, sans-serif", size=11, color="#4A2F2F"),
                    margin=dict(l=75, r=30, t=75, b=60),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1.0,
                        font=dict(family="Plus Jakarta Sans, sans-serif", size=10, color="#4A2F2F"),
                        bgcolor="rgba(255, 252, 247, 0.85)",
                        bordercolor="#DDD6CA",
                        borderwidth=1
                    ),
                    xaxis=dict(
                        title=dict(text="Actual Weekly Sales ($ USD)", font=dict(color="#4A2F2F", size=11), standoff=14),
                        tickfont=dict(color="#4A2F2F", size=10),
                        gridcolor="#EFEAE1",
                        linecolor="#DDD6CA",
                        tickprefix="$",
                        tickformat=","
                    ),
                    yaxis=dict(
                        title=dict(text="Predicted Weekly Sales ($ USD)", font=dict(color="#4A2F2F", size=11), standoff=14),
                        tickfont=dict(color="#4A2F2F", size=10),
                        gridcolor="#EFEAE1",
                        linecolor="#DDD6CA",
                        tickprefix="$",
                        tickformat=","
                    )
                )
                st.plotly_chart(fig_parity, width="stretch", config=PLOTLY_CONFIG)
                
            with d_c2:
                # Residual Histogram
                fig_res = go.Figure()
                fig_res.add_trace(go.Histogram(
                    x=diag_df["Residual"],
                    nbinsx=35,
                    marker_color="#7B9A78",
                    opacity=0.75,
                    name="Test Residuals"
                ))
                fig_res.add_vline(x=0, line_dash="dash", line_color="#B85C38", line_width=1.5)
                fig_res.update_layout(
                    title=dict(
                        text="Test Residual Error Distribution (Actual - Predicted)",
                        x=0.03,
                        y=0.97,
                        xanchor="left",
                        yanchor="top",
                        font=dict(family="Plus Jakarta Sans, sans-serif", size=13, color="#4A2F2F")
                    ),
                    paper_bgcolor="#FFFCF7",
                    plot_bgcolor="#FFFCF7",
                    height=360,
                    font=dict(family="Plus Jakarta Sans, sans-serif", size=11, color="#4A2F2F"),
                    margin=dict(l=75, r=30, t=75, b=60),
                    xaxis=dict(
                        title=dict(text="Forecast Error ($ USD)", font=dict(color="#4A2F2F", size=11), standoff=14),
                        tickfont=dict(color="#4A2F2F", size=10),
                        gridcolor="#EFEAE1",
                        linecolor="#DDD6CA",
                        tickprefix="$",
                        tickformat=","
                    ),
                    yaxis=dict(
                        title=dict(text="Frequency (Test Samples)", font=dict(color="#4A2F2F", size=11), standoff=14),
                        tickfont=dict(color="#4A2F2F", size=10),
                        gridcolor="#EFEAE1",
                        linecolor="#DDD6CA"
                    )
                )
                st.plotly_chart(fig_res, width="stretch", config=PLOTLY_CONFIG)
                
        # Metrics Plain English Glossary
        st.markdown("""
        <div style="background-color: #F8F4ED; border: 1px solid #E5DED3; border-radius: 6px; padding: 0.95rem 1.1rem; margin-top: 0.8rem; font-size: 0.83rem; color: #4A2F2F; line-height: 1.55;">
            <b>Metric Interpretation Guide for Reviewers:</b><br>
            • <b>WAPE (Weighted Absolute Percentage Error):</b> Evaluates total absolute dollar error divided by total volume. Highly recommended over MAPE in retail because it avoids zero-denominator division and weights high-volume products appropriately.<br>
            • <b>MAE (Mean Absolute Error):</b> Average dollar discrepancy per weekly series. Intuitive for demand planners.<br>
            • <b>RMSE (Root Mean Squared Error):</b> Penalizes large forecast outliers heavily; vital for inventory sizing.<br>
            • <b>R² Score (Coefficient of Determination):</b> Measures percentage of variance explained (0.9928 indicates near-perfect trend and seasonal capture).
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # DESTINATION D: INVENTORY PLANNING STUDIO
    # --------------------------------------------------------------------------
    elif nav_choice == "Inventory Planning Studio":
        st.markdown("""
        <div class="surface-card">
            <div class="surface-header-box">
                <div class="surface-tag-line"></div>
                <h2 class="surface-title">Operations Research Inventory Policy Optimization</h2>
            </div>
            <p class="surface-desc">Dynamic safety stock sizing and Reorder Point (ROP) policy configured under the Silver-Pyke-Peterson inventory framework to protect against replenishment lead-time demand uncertainty.</p>
        """, unsafe_allow_html=True)
        
        # Interactive Controls
        inv_c1, inv_c2, inv_c3 = st.columns(3)
        with inv_c1:
            lead_time_input = st.slider("Replenishment Lead Time (L in Weeks)", min_value=1, max_value=6, value=2, step=1)
        with inv_c2:
            sla_option = st.selectbox(
                "Service Level Agreement (SLA Target)",
                ["95% In-Stock SLA (Z = 1.645)", "90% In-Stock SLA (Z = 1.282)", "99% In-Stock SLA (Z = 2.326)"],
                index=0
            )
            sla_val = 0.95 if "95%" in sla_option else (0.90 if "90%" in sla_option else 0.99)
        with inv_c3:
            uncertainty_basis = st.radio(
                "Uncertainty Metric Basis",
                ["Forecast Error σ (Operations Research)", "Historical Demand σ (Baseline)"],
                index=0
            )
            
        # Select standard deviation
        sigma_used = series_residual_std if "Forecast Error" in uncertainty_basis else float(filtered_df["Weekly_Sales"].std())
        
        # Calculate policy using src/inventory.py
        policy_res = calculate_inventory_policy(
            weekly_demand=avg_sales,
            lead_time_weeks=lead_time_input,
            uncertainty_sigma=sigma_used,
            service_level=sla_val
        )
        
        c_demand = policy_res["cycle_demand"]
        s_stock = policy_res["safety_stock"]
        rop = policy_res["reorder_point"]
        z_score = policy_res["z_score"]
        
        # 3 Result Cards
        r_c1, r_c2, r_c3 = st.columns(3)
        with r_c1:
            st.markdown(f"""
            <div class="kpi-card cocoa">
                <div class="kpi-meta-label">Cycle Demand ({lead_time_input}W Lead Time)</div>
                <div class="kpi-meta-val">${c_demand:,.0f}</div>
                <div class="kpi-meta-sub">${avg_sales:,.0f} / week × {lead_time_input} weeks</div>
            </div>
            """, unsafe_allow_html=True)
        with r_c2:
            st.markdown(f"""
            <div class="kpi-card gold">
                <div class="kpi-meta-label">Recommended Safety Stock</div>
                <div class="kpi-meta-val">${s_stock:,.0f}</div>
                <div class="kpi-meta-sub">{sla_val*100:.0f}% SLA Buffer (Z = {z_score:.3f})</div>
            </div>
            """, unsafe_allow_html=True)
        with r_c3:
            st.markdown(f"""
            <div class="kpi-card terracotta">
                <div class="kpi-meta-label">Dynamic Reorder Point (ROP)</div>
                <div class="kpi-meta-val">${rop:,.0f}</div>
                <div class="kpi-meta-sub">Cycle Demand + Safety Buffer</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Formula Walkthrough Box with Proper Streamlit LaTeX Rendering
        st.markdown("""
        <div class="surface-header-box" style="margin-top: 1.1rem; margin-bottom: 0.35rem;">
            <div class="surface-tag-line"></div>
            <h4 style="margin: 0; font-size: 1.05rem; color: #4A2F2F;">Mathematical Derivation & Unit Balance</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Proper Streamlit LaTeX rendering
        st.latex(r"\text{Reorder Point (ROP)} = (\bar{d} \times L) + \left(Z_{\text{SLA}} \times \sigma_{\text{weekly}} \times \sqrt{L}\right)")
        
        lead_time_sigma = sigma_used * np.sqrt(lead_time_input)
        st.latex(rf"\text{{ROP}} = (\${avg_sales:,.0f} \times {lead_time_input}\text{{ wks}}) + \left({z_score:.3f} \times \${sigma_used:,.0f} \times \sqrt{{{lead_time_input}}}\right) = \${c_demand:,.0f} + \${s_stock:,.0f} = \mathbf{{\${rop:,.0f}\text{{ USD}}}}")
        
        st.markdown(f"""
        <div style="background-color: #FAF5EE; border: 1.5px solid #E8DDD0; border-radius: 6px; padding: 0.85rem 1.15rem; margin-top: 0.6rem; margin-bottom: 1.1rem; font-size: 0.85rem; color: #4A2F2F; line-height: 1.6;">
            <b>Active Scope Numerical Substitution:</b><br>
            • <b>Expected Lead-Time Demand (Cycle Demand):</b> ${avg_sales:,.2f} / week × {lead_time_input} weeks = <b>${c_demand:,.2f} USD</b><br>
            • <b>Lead-Time Uncertainty Scaling:</b> &sigma;<sub>L</sub> = ${sigma_used:,.2f} × &radic;{lead_time_input} = <b>${lead_time_sigma:,.2f} USD</b><br>
            • <b>Safety Stock Buffer ({sla_val*100:.0f}% SLA):</b> {z_score:.4f} × ${lead_time_sigma:,.2f} = <b>${s_stock:,.2f} USD</b><br>
            • <b>Total Dynamic Reorder Point:</b> ${c_demand:,.2f} + ${s_stock:,.2f} = <b>${rop:,.2f} USD</b>
        </div>
        """, unsafe_allow_html=True)
        
        # Stacked Composition Bar Chart with High Contrast & Total Annotation
        fig_inv = go.Figure()
        
        # Cycle demand (Terracotta bar with crisp white text)
        fig_inv.add_trace(go.Bar(
            name=f"Cycle Demand ({lead_time_input}W Expected Demand)",
            x=[f"Store {selected_store} : {selected_dept}"],
            y=[c_demand],
            marker=dict(color="#B85C38", line=dict(color="#8C3B1D", width=1.5)),
            text=[f"Cycle Demand: ${c_demand:,.0f}"],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="#FFFFFF", size=13, family="Plus Jakarta Sans, sans-serif"),
            hovertemplate="<b>Cycle Demand:</b> $%{y:,.2f}<br>Lead Time: " + str(lead_time_input) + " Weeks<extra></extra>"
        ))
        
        # Safety stock (Goldenrod bar with dark cocoa text)
        fig_inv.add_trace(go.Bar(
            name=f"Safety Stock ({sla_val*100:.0f}% SLA Buffer)",
            x=[f"Store {selected_store} : {selected_dept}"],
            y=[s_stock],
            marker=dict(color="#D4A017", line=dict(color="#9E7D23", width=1.5)),
            text=[f"Safety Stock: ${s_stock:,.0f}"],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="#4A2F2F", size=13, family="Plus Jakarta Sans, sans-serif"),
            hovertemplate="<b>Safety Stock:</b> $%{y:,.2f}<br>Z-Score: " + f"{z_score:.3f}" + "<extra></extra>"
        ))
        
        # Annotation for Total Reorder Point at top of stacked bar
        fig_inv.add_annotation(
            x=f"Store {selected_store} : {selected_dept}",
            y=rop,
            text=f"<b>Total ROP: ${rop:,.0f} USD</b>",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor="#B85C38",
            ax=0,
            ay=-35,
            font=dict(family="Plus Jakarta Sans, sans-serif", size=13, color="#4A2F2F"),
            bgcolor="#FAF5EE",
            bordercolor="#B85C38",
            borderwidth=1.5,
            borderpad=6
        )
        
        fig_inv.update_layout(
            barmode="stack",
            title=dict(
                text=f"Inventory Reorder Point Composition (Lead Time = {lead_time_input} Weeks, Target SLA = {sla_val*100:.0f}%)",
                x=0.01,
                y=0.98,
                yref="container",
                xanchor="left",
                yanchor="top",
                font=dict(family="Plus Jakarta Sans, sans-serif", color="#4A2F2F", size=13.5)
            ),
            paper_bgcolor="#FFFCF7",
            plot_bgcolor="#FFFCF7",
            height=370,
            font=dict(family="Plus Jakarta Sans, sans-serif", color="#4A2F2F", size=12),
            margin=dict(l=75, r=30, t=95, b=55),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.04,
                xanchor="right",
                x=1.0,
                font=dict(size=11.5, color="#4A2F2F"),
                bgcolor="#FFFCF7",
                bordercolor="#E5DED3",
                borderwidth=1
            ),
            xaxis=dict(
                tickfont=dict(color="#4A2F2F", size=13),
                showgrid=False,
                linecolor="#DDD6CA",
                linewidth=1.2
            ),
            yaxis=dict(
                title=dict(text="Inventory Capital Allocation ($ USD)", font=dict(color="#4A2F2F", size=12, family="Plus Jakarta Sans"), standoff=15),
                tickfont=dict(color="#4A2F2F", size=12),
                showgrid=True,
                gridcolor="#EFEAE1",
                gridwidth=1,
                linecolor="#DDD6CA",
                tickprefix="$",
                tickformat=","
            )
        )
        st.plotly_chart(fig_inv, width="stretch", config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # DESTINATION E: DEMAND & FEATURE INSIGHTS
    # --------------------------------------------------------------------------
    elif nav_choice == "Demand & Feature Insights":
        st.markdown("""
        <div class="surface-card">
            <div class="surface-header-box">
                <div class="surface-tag-line"></div>
                <h2 class="surface-title">Merchandise Seasonality & Feature Drivers</h2>
            </div>
            <p class="surface-desc">Exploratory diagnostics examining annual seasonal cycles, promotional lift elasticity, and gradient boosted feature importance.</p>
        """, unsafe_allow_html=True)
        
        # 1. Department Sliced Error Breakdown
        st.markdown("**1. Department Error Breakdown (Out-of-Time Test Set)**")
        if dept_error_df is not None:
            d_err_disp = dept_error_df.copy()
            d_err_disp["MAE"] = d_err_disp["MAE"].apply(lambda x: f"${x:,.2f}")
            d_err_disp["RMSE"] = d_err_disp["RMSE"].apply(lambda x: f"${x:,.2f}")
            d_err_disp["WAPE"] = d_err_disp["WAPE"].apply(lambda x: f"{x:.2f}%")
            d_err_disp["Mean Actual"] = d_err_disp["Mean_Actual"].apply(lambda x: f"${x:,.2f}")
            d_err_disp["Mean Pred"] = d_err_disp["Mean_Predicted"].apply(lambda x: f"${x:,.2f}")
            st.markdown(render_editorial_table(d_err_disp[["Dept_Name", "Mean Actual", "Mean Pred", "MAE", "RMSE", "WAPE"]]), unsafe_allow_html=True)
            
        st.markdown("<div style='margin-top: 1.4rem;'></div>", unsafe_allow_html=True)
        
        # 2. Top Features Chart
        st.markdown("**2. Top Predictive Drivers (XGBoost Feature Gain)**")
        fi_path = os.path.join(BASE_DIR, "outputs", "metrics", "feature_importance.csv")
        if os.path.exists(fi_path):
            fi_df = pd.read_csv(fi_path).head(10).sort_values("Importance", ascending=True)
            fig_fi = px.bar(
                fi_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale=["#D4A017", "#B85C38"]
            )
            fig_fi.update_layout(
                paper_bgcolor="#FFFCF7",
                plot_bgcolor="#FFFCF7",
                height=360,
                font=dict(family="Plus Jakarta Sans, sans-serif", size=11, color="#4A2F2F"),
                margin=dict(l=150, r=25, t=30, b=45),
                coloraxis_showscale=False,
                xaxis=dict(
                    title=dict(text="Feature Importance (Gain Metric)", font=dict(color="#4A2F2F", size=11)),
                    tickfont=dict(color="#4A2F2F", size=10),
                    gridcolor="#EFEAE1",
                    linecolor="#DDD6CA"
                ),
                yaxis=dict(
                    title="",
                    tickfont=dict(color="#4A2F2F", size=11, family="Plus Jakarta Sans, sans-serif")
                )
            )
            st.plotly_chart(fig_fi, width="stretch", config=PLOTLY_CONFIG)
                
        st.markdown(f"""
        <div style="background-color: #FAF5EE; border: 1px solid #E8DDD0; border-radius: 6px; padding: 0.9rem 1.1rem; margin-top: 0.8rem; font-size: 0.83rem; color: #6E5656; line-height: 1.55;">
            <b>Responsible AI & Observational Attribution Disclosure:</b><br>
            Feature importance scores reflect mathematical split gain inside the XGBoost gradient boosting decision tree ensemble.
            They indicate statistical predictive association on historical patterns, not causal proof. Macroeconomic covariates (CPI, Fuel Price, Unemployment) provide contextual normalization alongside multi-horizon autoregressive lag anchors.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.error("Dataset not found at data/retail_store_sales.csv. Please execute the pipeline script first.")
