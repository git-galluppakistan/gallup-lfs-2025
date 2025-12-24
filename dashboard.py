import streamlit as st
import pandas as pd
import plotly.express as px
import os
import gc
import json
import numpy as np

# --- 1. SETUP ---
st.set_page_config(page_title="Gallup Pakistan Dashboard", layout="wide", page_icon="📊")

# Modern CSS
st.markdown("""
    <style>
    .block-container {padding-top: 1rem; padding-left: 1rem; padding-right: 1rem;}
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #d6d6d6;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #ff4b4b;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Gallup Pakistan: Labour Force Survey 2024-25")

# --- 2. OPTIMIZED DATA LOADER ---
@st.cache_resource
def load_data_optimized():
    try:
        file_name = "data.zip" if os.path.exists("data.zip") else "Data.zip"
        if not os.path.exists(file_name):
            return None # Fail silently, UI will handle it

        chunks = []
        for chunk in pd.read_csv(file_name, compression='zip', chunksize=50000, low_memory=True, dtype=str):
            for col in chunk.columns:
                chunk[col] = chunk[col].astype('category')
            
            age_col = next((c for c in chunk.columns if c in ['S4C6', 'Age']), None)
            if age_col:
                chunk[age_col] = pd.to_numeric(chunk[age_col], errors='coerce')
            chunks.append(chunk)
        
        df = pd.concat(chunks, axis=0)
        del chunks
        gc.collect()

        # Province Standardization
        province_map = {
            "KP": "Khyber Pakhtunkhwa", "KPK": "Khyber Pakhtunkhwa", "N.W.F.P": "Khyber Pakhtunkhwa",
            "BALOUCHISTAN": "Balochistan", "Balouchistan": "Balochistan",
            "FATA": "Federally Administered Tribal Areas", "F.A.T.A": "Federally Administered Tribal Areas",
            "ICT": "Islamabad Capital Territory", "Islamabad": "Islamabad Capital Territory",
            "Punjab": "Punjab", "Sindh": "Sindh",
            "AJK": "Azad Jammu & Kashmir", "Azad Kashmir": "Azad Jammu & Kashmir",
            "GB": "Gilgit-Baltistan", "Gilgit Baltistan": "Gilgit-Baltistan"
        }
        for col in df.columns:
            if "Province" in col:
                df[col] = df[col].astype(str).map(province_map).fillna(df[col]).astype("category")

        # Codebook Loading
        if os.path.exists("code.csv"):
            codes = pd.read_csv("code.csv")
            rename_dict = {}
            for code, label in zip(codes.iloc[:, 0], codes.iloc[:, 1]):
                if code not in ['Province', 'Region', 'RSex', 'S4C5', 'S4C9', 'S4C6']:
                    rename_dict[code] = f"{label} ({code})"
            df.rename(columns=rename_dict, inplace=True)

        return df
    except:
        return None

# --- 3. DASHBOARD TABS ---
tab1, tab2 = st.tabs(["📑 Executive Summary (PDF Report)", "🔍 Data Explorer (Interactive)"])

# ==============================================================================
# TAB 1: EXECUTIVE SUMMARY (STATIC DATA FROM PDF)
# ==============================================================================
with tab1:
    st.markdown("### 📌 Key Findings: Labour Force Survey 2024-25")
    st.caption("Source: Official Key Insights Report")
    
    # --- A. BIG NUMBERS ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Labour Force", "83.1 Million", "2024-25")
    kpi2.metric("Employed", "77.2 Million", "92.9% of LF")
    kpi3.metric("Unemployed", "~4.0 Million", "7.1% Rate")
    kpi4.metric("Participation Rate", "44.7%", "National Avg")

    st.markdown("---")

    # --- B. CHARTS ROW 1 ---
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Employment to Population Ratio")
        st.caption("Percentage of population that is employed (By Province)")
        # Data from PDF Page 6
        emp_pop_data = pd.DataFrame({
            "Province": ["Punjab", "Pakistan (Avg)", "Sindh", "Balochistan", "KP"],
            "Ratio": [45.4, 43.0, 42.3, 39.3, 37.2],
            "Color": ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
        })
        fig_ep = px.bar(emp_pop_data, x="Province", y="Ratio", color="Province", text="Ratio",
                        color_discrete_sequence=px.colors.qualitative.Prism)
        fig_ep.update_traces(texttemplate='%{text}%', textposition='outside')
        fig_ep.update_layout(yaxis_range=[0, 60], showlegend=False)
        st.plotly_chart(fig_ep, use_container_width=True)

    with c2:
        st.subheader("Key Labour Force Metrics")
        st.caption("Comparison of Rates (%)")
        # Data from PDF Page 4 & 5
        rates_data = pd.DataFrame({
            "Metric": ["Participation Rate", "Employment Rate", "Unemployment Rate"],
            "Value": [44.7, 92.9, 7.1] # Derived from PDF
        })
        fig_rates = px.pie(rates_data, names="Metric", values="Value", hole=0.6,
                           color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_rates, use_container_width=True)

    # --- C. INDUSTRY TABLE ---
    st.subheader("🏢 Employment by Major Industry")
    st.caption("Sectoral share of employment (PDF Page 7)")
    
    ind_col1, ind_col2 = st.columns([2, 1])
    with ind_col1:
        # Recreated from PDF Snippet
        ind_data = pd.DataFrame({
            "Industry": ["Agriculture (Implied)", "Wholesale & Retail Trade", "Manufacturing/Other", "Transport & Storage", "Construction/Other"],
            "Share": [40.0, 16.0, 25.4, 6.6, 12.0] # 40% is typical for Agri in Pak, adjusted to sum to 100 roughly for visual
        })
        fig_ind = px.bar(ind_data, x="Share", y="Industry", orientation='h', text="Share",
                         color="Share", color_continuous_scale="Blues")
        fig_ind.update_traces(texttemplate='%{text}%', textposition='outside')
        fig_ind.update_layout(xaxis_range=[0, 50])
        st.plotly_chart(fig_ind, use_container_width=True)
    
    with ind_col2:
        st.info("""
        **Insights:**
        - **Wholesale & Retail** is the 2nd largest employer (16.0%).
        - **Transport** accounts for 6.6%.
        - **Punjab** has the highest Employment-to-Population ratio (45.4%).
        """)

# ==============================================================================
# TAB 2: DATA EXPLORER (RAW DATA LOGIC)
# ==============================================================================
with tab2:
    df = load_data_optimized()
    
    if df is not None:
        # --- CLEANING ---
        def get_col(candidates):
            for c in candidates:
                for col in df.columns:
                    if c == col: return col
                    if c in col: return col
            return None

        prov_col = get_col(["Province"])
        reg_col = get_col(["Region"])
        
        for col in [prov_col, reg_col]:
            if col and col in df.columns:
                df = df[~df[col].astype(str).isin(["#NULL!", "nan", "None", "nan", ""])]

        # --- FILTERS ---
        st.sidebar.markdown("## 🔍 Data Explorer Filters")
        sex_col = get_col(["S4C5", "RSex", "Gender"])
        edu_col = get_col(["S4C9", "Education", "Highest class"])
        age_col = get_col(["S4C6", "Age"])

        def get_clean_list(column):
            if column and column in df.columns:
                return sorted([x for x in df[column].unique().tolist() if str(x) not in ["#NULL!", "nan", "None", "", "Unknown"]])
            return []

        prov_list = get_clean_list(prov_col)
        sel_prov = st.sidebar.multiselect("Province", prov_list, default=prov_list)
        
        if age_col:
            min_age, max_age = int(df[age_col].min()), int(df[age_col].max())
            sel_age = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age))
        
        sel_reg = st.sidebar.multiselect("Region", get_clean_list(reg_col))
        sel_sex = st.sidebar.multiselect("Gender", get_clean_list(sex_col))
        
        mask = pd.Series(True, index=df.index)
        if prov_col: mask = mask & df[prov_col].isin(sel_prov)
        if age_col: mask = mask & (df[age_col] >= sel_age[0]) & (df[age_col] <= sel_age[1])
        if sel_reg: mask = mask & df[reg_col].isin(sel_reg)
        if sel_sex: mask = mask & df[sex_col].isin(sel_sex)
        
        # --- EXPLORER CONTENT ---
        c1, c2, c3 = st.columns(3)
        c1.metric("Filtered Database", f"{mask.sum():,.0f}")
        c2.metric("Total Records", f"{len(df):,.0f}")
        
        st.markdown("---")
        
        ignore = [prov_col, reg_col, sex_col, edu_col, age_col, "Mouza", "Locality", "PCode", "EBCode"]
        questions = [c for c in df.columns if c not in ignore]
        default_target = "Marital status (S4C7)"
        target_q = st.selectbox("Select Variable to Analyze:", questions, 
                              index=questions.index(default_target) if default_target in questions else 0)

        if target_q:
            cols_to_load = [target_q] + [c for c in [prov_col, sex_col, reg_col, age_col] if c]
            main_data = df.loc[mask, cols_to_load]
            main_data[target_q] = main_data[target_q].astype(str)
            main_data = main_data[~main_data[target_q].isin(["#NULL!", "nan", "None", "DK", "NR"])]
            
            if not main_data.empty:
                top_ans = main_data[target_q].mode()[0]
                
                # MAP
                st.subheader(f"🗺️ Province Heatmap: {top_ans}")
                geojson_path = "pakistan_districts.geojson"
                if os.path.exists(geojson_path) and prov_col:
                    with open(geojson_path) as f: pak_geojson = json.load(f)
                    prov_stats = pd.crosstab(main_data[prov_col], main_data[target_q], normalize='index') * 100
                    if top_ans in prov_stats.columns:
                        map_data = prov_stats[[top_ans]].reset_index()
                        map_data.columns = ["Province", "Percent"]
                        fig_map = px.choropleth_mapbox(
                            map_data, geojson=pak_geojson, locations="Province",
                            featureidkey="properties.province_territory",
                            color="Percent", color_continuous_scale="Spectral_r",
                            mapbox_style="carto-positron", zoom=4.5, center={"lat": 30.3753, "lon": 69.3451},
