import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import gc
import json
import numpy as np

# --- 1. SETUP ---
st.set_page_config(page_title="Gallup Pakistan Dashboard", layout="wide", page_icon="📊")

# --- CSS ---
st.markdown("""
    <style>
    .block-container {padding-top: 1rem;}
    div[data-testid="metric-container"] {
        background-color: #ffffff; border: 1px solid #d6d6d6;
        padding: 10px; border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Gallup Pakistan: Labour Force Survey 2024-25")

# --- 2. DATA LOADERS ---
@st.cache_data(ttl=3600)
def load_geojson_dist():
    path = "pakistan_districts.geojson" 
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return None

@st.cache_data(ttl=3600)
def load_geojson_prov():
    path = "pakistan_provinces.geojson" 
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return None

@st.cache_resource(show_spinner="Loading Data...", ttl="2h")
def load_data():
    try:
        # A. Load Main Data
        file_name = "data.zip" if os.path.exists("data.zip") else "Data.zip"
        if not os.path.exists(file_name): return None, "Data file missing"

        chunks = []
        for chunk in pd.read_csv(file_name, compression='zip', chunksize=50000, low_memory=True, dtype=str):
            for col in chunk.columns:
                chunk[col] = chunk[col].astype('category')
            chunks.append(chunk)
        
        df = pd.concat(chunks, axis=0)
        del chunks
        gc.collect()

        # B. Load Mappings
        possible_files = ["district_mapping.csv", "DSTT.xlsx - Sheet1.csv", "lahore-district-mapping-file.xlsx - Lahore.csv"]
        dfs_to_merge = []
        for f in possible_files:
            if os.path.exists(f):
                temp = pd.read_csv(f, dtype=str)
                if "PCode" in temp.columns and "District" in temp.columns:
                    dfs_to_merge.append(temp)
        
        if dfs_to_merge and "PCode" in df.columns:
            combined_map = pd.concat(dfs_to_merge, ignore_index=True)
            
            # CLEANING: Match PCode formats
            combined_map["PCode"] = combined_map["PCode"].astype(str).str.split('.').str[0].str.strip()
            df["PCode"] = df["PCode"].astype(str).str.split('.').str[0].str.strip()

            dist_map = combined_map.drop_duplicates(subset="PCode").set_index("PCode")["District"].to_dict()
            
            # Manual Fixes
            dist_map['352'] = 'LAHORE'
            dist_map['201'] = 'LAHORE'
            dist_map['25121030'] = 'LAHORE'

            # Map & FORCE UPPER CASE to match GeoJSON
            df["District"] = df["PCode"].map(dist_map)
            df["District"] = df["District"].astype(str).str.upper().str.strip()
            
            # Convert back to category
            df["District"] = df["District"].astype('category')

        # C. Global Fixes
        if "S4C81" in df.columns:
            df["S4C81"] = df["S4C81"].astype(str).replace({"1": "Yes", "2": "No", "Yes' 2'No": "Yes"}).astype("category")

        # D. Codebook Rename
        if os.path.exists("code.csv"):
            codes = pd.read_csv("code.csv")
            rename_dict = {}
            for code, label in zip(codes.iloc[:, 0], codes.iloc[:, 1]):
                if code not in ['Province', 'Region', 'RSex', 'S4C5', 'S4C9', 'S4C6', 'District']:
                    rename_dict[code] = f"{label} ({code})"
            df.rename(columns=rename_dict, inplace=True)

        return df, "Success"

    except Exception as e:
        return None, str(e)

df, status = load_data()
pak_dist_json = load_geojson_dist()
pak_prov_json = load_geojson_prov()

# --- 3. SESSION STATE FOR RESET ---
if 'reset_trigger' not in st.session_state:
    st.session_state['reset_trigger'] = False

def reset_filters():
    st.session_state['prov_key'] = []
    st.session_state['dist_key'] = []
    st.session_state['reg_key'] = []
    st.session_state['sex_key'] = []
    st.session_state['edu_key'] = []

# --- 4. DASHBOARD MAIN ---
if df is not None:
    # --- TABS ---
    tab1, tab2 = st.tabs(["📑 Executive Summary", "🔍 Data Explorer (Full Dashboard)"])

    # === TAB 1: SUMMARY ===
    with tab1:
        st.markdown("### 📌 Key Findings: Labour Force Survey 2024-25")
        st.caption("Source: Official Key Insights Report")
        
        st.link_button("📥 Download Full Questionnaire (PDF)", "https://www.pbs.gov.pk/wp-content/uploads/2020/07/Questionnaire-of-LFS-2024-25-Final.pdf")
        st.markdown("---")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Labour Force", "83.1 Million", "2024-25")
        kpi2.metric("Employed", "77.2 Million", "92.9% of LF")
        kpi3.metric("Unemployed", "~4.0 Million", "7.1% Rate")
        kpi4.metric("Participation Rate", "44.7%", "National Avg")

        st.markdown("---")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Employment Ratio")
            emp_data = pd.DataFrame({"Province": ["Punjab", "Sindh", "Balochistan", "KP"], "Ratio": [45.4, 42.3, 39.3, 37.2]})
            fig = px.bar(emp_data, x="Province", y="Ratio", color="Province", text="Ratio")
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            st.subheader("Key Metrics")
            pie_data = pd.DataFrame({"Metric": ["Employed", "Unemployed"], "Value": [92.9, 7.1]})
            fig = px.pie(pie_data, names="Metric", values="Value", hole=0.5)
            st.plotly_chart(fig, use_container_width=True)

    # === TAB 2: EXPLORER ===
    with tab2:
        # FILTERS
        st.sidebar.markdown("## 🔍 Filters")
        if st.sidebar.button("🔄 Reset All Filters", on_click=reset_filters): st.rerun()
        
        prov = st.sidebar.multiselect("Province", df["Province"].unique(), key='prov_key')
        
        # MASK
        mask = pd.Series(True, index=df.index)
        if prov: mask = mask & df["Province"].isin(prov)
        
        # HEADER
        c1, c2, c3 = st.columns(3)
        c1.metric("Filtered Database", f"{mask.sum():,.0f}")
        c2.metric("Total Records", f"{len(df):,.0f}")
        c3.metric("Selection Share", f"{(mask.sum()/len(df)*100):.1f}%")
        st.markdown("---")

        # SELECTOR
        questions = [c for c in df.columns if c not in ["PCode", "District", "Province", "Region", "S4C6"]]
        default_target = "Marital Status (S4C7)"
        try:
            def_idx = questions.index(default_target)
        except:
            def_idx = 0
            
        target = st.selectbox("Select Variable:", questions, index=def_idx)
        
        # DATA PREP
        main_data = df.loc[mask].copy()
        main_data[target] = main_data[target].astype(str)
        main_data = main_data[~main_data[target].isin(["#NULL!", "nan", "None", "DK", "NR"])]
        
        # Check S4C81 cleanup again
        if "S4C81" in target:
             main_data[target] = main_data[target].replace({"1": "Yes", "2": "No", "Yes' 2'No": "Yes"})

        if not main_data.empty:
            opts = sorted(main_data[target].unique())
            mode_val = main_data[target].mode()[0]
            if mode_val not in opts: mode_val = opts[0]
            
            map_choice = st.selectbox("Select Answer to Map:", opts, index=opts.index(mode_val))
            
            # --- MAPS ---
            m1, m2 = st.columns(2)
            
            # PROVINCE MAP
            with m1:
                st.subheader(f"Province: {map_choice}")
                if pak_prov_json:
                    p_stats = pd.crosstab(main_data["Province"], main_data[target], normalize='index') * 100
                    if map_choice in p_stats.columns:
                        p_map = p_stats[[map_choice]].reset_index()
                        p_map.columns = ["Province", "Percent"]
                        
                        fig = px.choropleth_mapbox(
                            p_map, geojson=pak_prov_json, locations="Province",
                            featureidkey="properties.shapeName",
                            color="Percent", color_continuous_scale="Spectral_r",
                            mapbox_style="carto-positron", zoom=4.5, center={"lat": 30.3753, "lon": 69.3451},
                            opacity=0.7
                        )
                        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=400)
                        st.plotly_chart(fig, use_container_width=True)
            
            # DISTRICT MAP (FIXED!)
            with m2:
                st.subheader(f"District: {map_choice}")
                if pak_dist_json and "District" in main_data.columns:
                    d_stats = pd.crosstab(main_data["District"], main_data[target], normalize='index') * 100
                    
                    if map_choice in d_stats.columns:
                        d_map = d_stats[[map_choice]].reset_index()
                        d_map.columns = ["District", "Percent"]
                        
                        # DEBUG CHECK: Ensure names match GeoJSON upper case
                        d_map["District"] = d_map["District"].astype(str).str.upper()

                        fig = px.choropleth_mapbox(
                            d_map, geojson=pak_dist_json, locations="District",
                            featureidkey="properties.districts", # <--- THE CORRECT KEY
                            color="Percent", color_continuous_scale="Spectral_r",
                            mapbox_style="carto-positron", zoom=4.5, center={"lat": 30.3753, "lon": 69.3451},
                            opacity=0.7
                        )
                        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"No data for '{map_choice}'")
                else:
                    st.warning("District column missing or GeoJSON not loaded.")

            # --- CHARTS ---
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**📊 Bar Chart**")
                counts = main_data[target].value_counts().reset_index()
                counts.columns = ["Answer", "Count"]
                counts["%"] = (counts["Count"] / counts["Count"].sum() * 100)
                fig = px.bar(counts, x="Answer", y="%", text=counts["%"].apply(lambda x: f"{x:.1f}%"))
                st.plotly_chart(fig, use_container_width=True)
            
            with c2:
                st.markdown("**📋 Data Table**")
                if "District" in main_data.columns:
                    d_stats = pd.crosstab(main_data["District"], main_data[target], normalize='index') * 100
                    if map_choice in d_stats.columns:
                        st.dataframe(d_stats.sort_values(by=map_choice, ascending=False).style.format("{:.1f}%"), use_container_width=True)

else:
    st.error(f"Data Load Failed: {status}")
