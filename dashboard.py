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
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px;
        gap: 1px; padding-top: 10px; padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 2px solid #ff4b4b; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Gallup Pakistan: Labour Force Survey 2024-25")

# --- 2. SAFE DATA LOADER ---
@st.cache_data
def load_data_optimized():
    try:
        file_name = "data.zip" if os.path.exists("data.zip") else "Data.zip"
        
        if not os.path.exists(file_name):
            return None

        chunks = []
        for chunk in pd.read_csv(file_name, compression='zip', chunksize=50000, low_memory=True, dtype=str):
            for col in chunk.columns:
                chunk[col] = chunk[col].astype('category')
            
            age_col = next((c for c in chunk.columns if c in ['S4C6', 'Age']), None)
            if age_col:
                chunk[age_col] = pd.to_numeric(chunk[age_col], errors='coerce')
            chunks.append(chunk)
        
        if not chunks:
            return None

        df = pd.concat(chunks, axis=0)
        del chunks
        gc.collect()

        # --- PROVINCE NAME STANDARDIZATION ---
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

        # Codebook
        if os.path.exists("code.csv"):
            codes = pd.read_csv("code.csv")
            rename_dict = {}
            for code, label in zip(codes.iloc[:, 0], codes.iloc[:, 1]):
                if code not in ['Province', 'Region', 'RSex', 'S4C5', 'S4C9', 'S4C6']:
                    rename_dict[code] = f"{label} ({code})"
            df.rename(columns=rename_dict, inplace=True)

        return df

    except Exception as e:
        st.error(f"⚠️ Data Loading Error: {e}")
        return None

df = load_data_optimized()

# --- 3. DASHBOARD TABS ---
try:
    tab1, tab2 = st.tabs(["📑 Executive Summary", "🔍 Data Explorer (Full Dashboard)"])

    # ==============================================================================
    # TAB 1: EXECUTIVE SUMMARY
    # ==============================================================================
    with tab1:
        st.markdown("### 📌 Key Findings: Labour Force Survey 2024-25")
        st.caption("Source: Official Key Insights Report")
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Labour Force", "83.1 Million", "2024-25")
        kpi2.metric("Employed", "77.2 Million", "92.9% of LF")
        kpi3.metric("Unemployed", "~4.0 Million", "7.1% Rate")
        kpi4.metric("Participation Rate", "44.7%", "National Avg")

        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Employment to Population Ratio")
            emp_pop_data = pd.DataFrame({
                "Province": ["Punjab", "Pakistan (Avg)", "Sindh", "Balochistan", "KP"],
                "Ratio": [45.4, 43.0, 42.3, 39.3, 37.2]
            })
            fig_ep = px.bar(emp_pop_data, x="Province", y="Ratio", color="Province", text="Ratio",
                            color_discrete_sequence=px.colors.qualitative.Prism)
            fig_ep.update_traces(texttemplate='%{text}%', textposition='outside')
            fig_ep.update_layout(yaxis_range=[0, 60], showlegend=False)
            st.plotly_chart(fig_ep, use_container_width=True)

        with c2:
            st.subheader("Key Labour Force Metrics")
            rates_data = pd.DataFrame({
                "Metric": ["Participation Rate", "Employment Rate", "Unemployment Rate"],
                "Value": [44.7, 92.9, 7.1]
            })
            fig_rates = px.pie(rates_data, names="Metric", values="Value", hole=0.6,
                               color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_rates, use_container_width=True)

        st.subheader("🏢 Employment by Major Industry")
        ind_col1, ind_col2 = st.columns([2, 1])
        with ind_col1:
            ind_data = pd.DataFrame({
                "Industry": ["Agriculture", "Wholesale & Retail", "Manufacturing", "Transport", "Construction", "Other"],
                "Share": [40.0, 16.0, 25.4, 6.6, 11.0, 1.0] 
            })
            fig_ind = px.bar(ind_data, x="Share", y="Industry", orientation='h', text="Share",
                             color="Share", color_continuous_scale="Blues")
            fig_ind.update_traces(texttemplate='%{text}%', textposition='outside')
            fig_ind.update_layout(xaxis_range=[0, 50])
            st.plotly_chart(fig_ind, use_container_width=True)
        
        with ind_col2:
            st.info("**Key Insight:** Wholesale & Retail Trade is the 2nd largest employer after Agriculture.")

    # ==============================================================================
    # TAB 2: DATA EXPLORER (CUSTOM AGE GROUPS)
    # ==============================================================================
    with tab2:
        if df is not None:
            # --- CLEANING & COLUMNS ---
            def get_col(candidates):
                for c in candidates:
                    for col in df.columns:
                        if c == col: return col
                        if c in col: return col
                return None

            prov_col = get_col(["Province"])
            reg_col = get_col(["Region"])
            sex_col = get_col(["S4C5", "RSex", "Gender"])
            edu_col = get_col(["S4C9", "Education", "Highest class"])
            age_col = get_col(["S4C6", "Age"])
            
            # --- FILTERS ---
            st.sidebar.markdown("## 🔍 Data Explorer Filters")
            def get_clean_list(column):
                if column and column in df.columns:
                    return sorted([x for x in df[column].unique().tolist() if str(x) not in ["#NULL!", "nan", "None", "", "Unknown"]])
                return []

            prov_list = get_clean_list(prov_col)
            sel_prov = st.sidebar.multiselect("Province", prov_list, default=prov_list)
            
            if age_col:
                min_age, max_age = int(df[age_col].min()), int(df[age_col].max())
                sel_age = st.sidebar.slider("Age Range (Filter)", min_age, max_age, (min_age, max_age))
            
            sel_reg = st.sidebar.multiselect("Region", get_clean_list(reg_col))
            sel_sex = st.sidebar.multiselect("Gender", get_clean_list(sex_col))
            sel_edu = st.sidebar.multiselect("Education", get_clean_list(edu_col))
            
            # --- APPLY FILTERS ---
            mask = pd.Series(True, index=df.index)
            if prov_col: mask = mask & df[prov_col].isin(sel_prov)
            if age_col: mask = mask & (df[age_col] >= sel_age[0]) & (df[age_col] <= sel_age[1])
            if sel_reg: mask = mask & df[reg_col].isin(sel_reg)
            if sel_sex: mask = mask & df[sex_col].isin(sel_sex)
            if sel_edu: mask = mask & df[edu_col].isin(sel_edu)
            
            # --- HEADER ---
            c1, c2, c3 = st.columns(3)
            c1.metric("Filtered Database", f"{mask.sum():,.0f}")
            c2.metric("Total Records", f"{len(df):,.0f}")
            c3.metric("Selection Share", f"{(mask.sum()/len(df)*100):.1f}%")
            
            st.markdown("---")
            
            # --- QUESTION SELECTOR ---
            ignore = [prov_col, reg_col, sex_col, edu_col, age_col, "Mouza", "Locality", "PCode", "EBCode"]
            questions = [c for c in df.columns if c not in ignore]
            default_target = "Marital Status (S4C7)"
            target_q = st.selectbox("Select Variable to Analyze:", questions, 
                                  index=questions.index(default_target) if default_target in questions else 0)

            if target_q:
                cols_to_load = [target_q] + [c for c in [prov_col, sex_col, reg_col, age_col, edu_col] if c]
                main_data = df.loc[mask, cols_to_load]
                main_data[target_q] = main_data[target_q].astype(str)
                main_data = main_data[~main_data[target_q].isin(["#NULL!", "nan", "None", "DK", "NR"])]
                
                if not main_data.empty:
                    top_ans = main_data[target_q].mode()[0]
                    
                    # 1. MAP
                    st.subheader(f"🗺️ Province Heatmap: {top_ans}")
                    geojson_path = "pakistan_provinces.geojson"
                    
                    if os.path.exists(geojson_path) and prov_col:
                        with open(geojson_path) as f: pak_geojson = json.load(f)
                        
                        prov_stats = pd.crosstab(main_data[prov_col], main_data[target_q], normalize='index') * 100
                        if top_ans in prov_stats.columns:
                            map_data = prov_stats[[top_ans]].reset_index()
                            map_data.columns = ["Province", "Percent"]
                            
                            fig_map = px.choropleth_mapbox(
                                map_data, geojson=pak_geojson, locations="Province",
                                featureidkey="properties.shapeName",
                                color="Percent", color_continuous_scale="Spectral_r",
                                mapbox_style="carto-positron", zoom=4.5, center={"lat": 30.3753, "lon": 69.3451},
                                opacity=0.7
                            )
                            
                            centroids = pd.DataFrame([
                                {"Province": "Punjab", "Lat": 30.8, "Lon": 72.5},
                                {"Province": "Sindh", "Lat": 26.0, "Lon": 68.5},
                                {"Province": "Balochistan", "Lat": 28.5, "Lon": 65.5},
                                {"Province": "Khyber Pakhtunkhwa", "Lat": 34.5, "Lon": 72.0},
                                {"Province": "Gilgit-Baltistan", "Lat": 35.8, "Lon": 74.5},
                                {"Province": "Azad Jammu & Kashmir", "Lat": 34.0, "Lon": 73.8},
                                {"Province": "Islamabad Capital Territory", "Lat": 33.7, "Lon": 73.1}
                            ])
                            
                            label_data = pd.merge(centroids, map_data, on="Province", how="inner")
                            if not label_data.empty:
                                fig_map.add_trace(go.Scattermapbox(
                                    lat=label_data["Lat"], lon=label_data["Lon"], mode='text',
                                    text=label_data.apply(lambda x: f"<b>{x['Percent']:.1f}%</b>", axis=1),
                                    textfont=dict(size=14, color='black'), showlegend=False, hoverinfo='none'
                                ))

                            fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=500)
                            st.plotly_chart(fig_map, use_container_width=True)
                    else:
                         st.warning("⚠️ Map file missing. Please ensure 'pakistan_provinces.geojson' is uploaded.")

                    # 2. CHARTS ROW 1
                    col1, col2, col3 = st.columns([1.5, 1, 1])

                    with col1:
                        st.markdown("**📊 Overall Results (%)**")
                        counts = main_data[target_q].value_counts().reset_index()
                        counts.columns = ["Answer", "Count"]
                        counts["%"] = (counts["Count"] / counts["Count"].sum() * 100).fillna(0)
                        fig_bar = px.bar(counts, x="Answer", y="%", color="Answer", 
                                        text=counts["%"].apply(lambda x: f"{x:.1f}%"),
                                        color_discrete_sequence=px.colors.qualitative.Bold)
                        st.plotly_chart(fig_bar, use_container_width=True)

                    with col2:
                        st.markdown("**🗺️ By Province (Stacked)**")
                        if prov_col:
                            prov_grp = main_data.groupby([prov_col, target_q], observed=True).size().reset_index(name='Count')
                            fig_prov = px.bar(prov_grp, x=prov_col, y="Count", color=target_q, barmode="stack")
                            fig_prov.update_layout(showlegend=False)
                            st.plotly_chart(fig_prov, use_container_width=True)

                    with col3:
                        st.markdown("**🚻 By Gender**")
                        if sex_col:
                            g_counts = main_data[sex_col].value_counts().reset_index()
                            g_counts.columns = ["Gender", "Count"]
                            fig_pie = px.pie(g_counts, names="Gender", values="Count", hole=0.5)
                            fig_pie.update_layout(showlegend=False)
                            st.plotly_chart(fig_pie, use_container_width=True)

                    # 3. CHARTS ROW 2 (Updated Age Logic)
                    col4, col5 = st.columns([1, 1.5])

                    with col4:
                        st.markdown("**🏙️ By Region**")
                        if reg_col:
                            reg_counts = main_data[reg_col].value_counts().reset_index()
                            reg_counts.columns = ["Region", "Count"]
                            fig_reg = px.pie(reg_counts, names="Region", values="Count", 
                                          color_discrete_sequence=px.colors.qualitative.Set3)
                            st.plotly_chart(fig_reg, use_container_width=True)

                    with col5:
                        st.markdown("**📈 Age Trends (%)**")
                        if age_col:
                            # --- CUSTOM AGE GROUP LOGIC ---
                            chart_data = main_data.copy()
                            # Updated bins and labels as requested
                            bins = [0, 4, 5, 9, 12, 15, 18, 24, 30, 40, 50, 60, 65, 200]
                            labels = ['0-4', '4-5', '5-9', '9-12', '12-15', '15-18', '18-24', '25-30', '30-40', '40-50', '50-60', '60-65', '65+']
                            
                            chart_data['AgeGrp'] = pd.cut(chart_data[age_col], bins=bins, labels=labels, include_lowest=True)
                            
                            age_grp = chart_data.groupby(['AgeGrp', target_q], observed=True).size().reset_index(name='Count')
                            age_totals = age_grp.groupby('AgeGrp', observed=True)['Count'].transform('sum')
                            age_grp['%'] = (age_grp['Count'] / age_totals * 100).fillna(0)
                            
                            fig_age = px.area(age_grp, x="AgeGrp", y="%", color=target_q, markers=True)
                            st.plotly_chart(fig_age, use_container_width=True)

                    # 4. TABLES
                    st.markdown("---")
                    st.subheader("📋 Detailed Data View")
                    t1, t2 = st.columns(2)
                    
                    with t1:
                        st.caption("Overall Response Count")
                        st.dataframe(counts, use_container_width=True, hide_index=True)
                        
                    with t2:
                        st.caption(f"Province Breakdown (Top % {top_ans})")
                        if prov_col:
                            prov_pivot = pd.crosstab(main_data[prov_col], main_data[target_q], normalize='index') * 100
                            if not prov_pivot.empty:
                                prov_pivot = prov_pivot.sort_values(by=top_ans, ascending=False)
                                st.dataframe(prov_pivot.style.format("{:.1f}%"), use_container_width=True)

        else:
            st.warning("⚠️ Data file not found. Please upload 'data.zip' to GitHub.")

except Exception as e:
    st.error(f"🚨 Critical Dashboard Error: {e}")

