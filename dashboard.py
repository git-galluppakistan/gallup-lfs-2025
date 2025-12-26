import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import gc
import json
import numpy as np

# --- 1. SETUP & SESSION STATE ---
st.set_page_config(page_title="Gallup Pakistan Dashboard", layout="wide", page_icon="📊")

if 'reset_trigger' not in st.session_state:
    st.session_state['reset_trigger'] = False

def reset_filters():
    st.session_state['prov_key'] = []
    st.session_state['dist_key'] = []
    st.session_state['reg_key'] = []
    st.session_state['sex_key'] = []
    st.session_state['edu_key'] = []

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

# --- 2. MEMORY-OPTIMIZED DATA LOADERS ---
@st.cache_data(ttl=3600)
def load_geojson_prov():
    path = "pakistan_provinces.geojson"
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return None

@st.cache_data(ttl=3600)
def load_geojson_dist():
    path = "pakistan_districts.geojson" 
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return None

@st.cache_resource(show_spinner="Loading Heavy Dataset...", ttl="2h")
def load_data_optimized():
    df = None
    try:
        # 1. LOAD MAIN DATA
        file_name = "data.zip" if os.path.exists("data.zip") else "Data.zip"
        if not os.path.exists(file_name): 
            st.error("❌ 'data.zip' file missing from repository.")
            return None

        chunks = []
        for chunk in pd.read_csv(file_name, compression='zip', chunksize=50000, low_memory=True, dtype=str):
            chunk.columns = chunk.columns.str.strip()
            for col in chunk.columns:
                chunk[col] = chunk[col].astype('category')
            age_col = next((c for c in chunk.columns if c in ['S4C6', 'Age']), None)
            if age_col: chunk[age_col] = pd.to_numeric(chunk[age_col], errors='coerce')
            chunks.append(chunk)
        
        if not chunks: return None
        df = pd.concat(chunks, axis=0)
        del chunks
        gc.collect()

        # --- PRE-PROCESSING ---
        
        # 2. Province Map (Standard)
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

        # 3. DISTRICT MAPPING (SAFE MODE)
        # We wrap this in a separate try/except so if mapping fails, the Dashboard STILL LOADS.
        try:
            possible_files = [
                "district_mapping.csv", 
                "DSTT.xlsx - Sheet1.csv", 
                "lahore-district-mapping-file.xlsx - Lahore.csv" # Ensure this matches GitHub EXACTLY
            ]
            
            dfs_to_merge = []
            for f in possible_files:
                if os.path.exists(f):
                    temp = pd.read_csv(f, dtype=str)
                    if "PCode" in temp.columns and "District" in temp.columns:
                        dfs_to_merge.append(temp)
            
            if dfs_to_merge:
                combined_map_df = pd.concat(dfs_to_merge, ignore_index=True)
                dist_map = combined_map_df.drop_duplicates(subset="PCode").set_index("PCode")["District"].to_dict()
                
                # Manual Fallback for Lahore if files miss it
                dist_map['352'] = 'Lahore'
                dist_map['201'] = 'Lahore'
                
                if "PCode" in df.columns:
                    df["District"] = df["PCode"].astype(str).map(dist_map).astype('category')
                    
        except Exception as e:
            # If mapping fails, just print a warning but KEEP GOING
            st.warning(f"⚠️ District mapping failed partially (Dashboard still running): {e}")

        # 4. Global Value Fixes
        target_cols = ['S4C81', 'S4C82'] 
        for c in target_cols:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip().replace({
                    "1": "Yes", "1.0": "Yes", "01": "Yes", "Yes' 2'No": "Yes",
                    "2": "No",  "2.0": "No",  "02": "No"
                }).astype("category")

        # 5. Codebook Rename
        if os.path.exists("code.csv"):
            codes = pd.read_csv("code.csv")
            rename_dict = {}
            for code, label in zip(codes.iloc[:, 0], codes.iloc[:, 1]):
                if code not in ['Province', 'Region', 'RSex', 'S4C5', 'S4C9', 'S4C6', 'District']:
                    rename_dict[code] = f"{label} ({code})"
            df.rename(columns=rename_dict, inplace=True)

        return df

    except Exception as e:
        st.error(f"🚨 CRITICAL DATA LOAD ERROR: {e}")
        return None

df = load_data_optimized()
pak_prov_json = load_geojson_prov()
pak_dist_json = load_geojson_dist()
gc.collect()

# --- 3. DASHBOARD TABS ---
if df is not None:
    try:
        tab1, tab2 = st.tabs(["📑 Executive Summary", "🔍 Data Explorer (Full Dashboard)"])

        # ==============================================================================
        # TAB 1: EXECUTIVE SUMMARY
        # ==============================================================================
        with tab1:
            st.markdown("### 📌 Key Findings: Labour Force Survey 2024-25")
            st.caption("Source: Official Key Insights Report")
            
            st.link_button(
                "📥 Download Full Questionnaire (PDF)", 
                "https://www.pbs.gov.pk/wp-content/uploads/2020/07/Questionnaire-of-LFS-2024-25-Final.pdf",
                help="Click to open the official PBS Questionnaire PDF"
            )
            st.markdown("---")

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
                    "Province": ["Pakistan (Avg)", "Punjab", "Sindh", "Balochistan", "KP"],
                    "Ratio": [43.0, 45.4, 42.3, 39.3, 37.2]
                })
                fig_ep = px.bar(emp_pop_data, x="Province", y="Ratio", text="Ratio",
                                color="Province", 
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
            ind_data = pd.DataFrame({
                "Industry": ["Agriculture", "Manufacturing", "Wholesale & Retail", "Construction", "Transport", "Other"],
                "Share": [40.0, 25.4, 16.0, 11.0, 6.6, 1.0] 
            })
            ind_data = ind_data.sort_values(by="Share", ascending=True)
            fig_ind = px.bar(ind_data, x="Share", y="Industry", orientation='h', text="Share",
                            color="Share", color_continuous_scale="Blues")
            fig_ind.update_traces(texttemplate='%{text}%', textposition='outside')
            fig_ind.update_layout(xaxis_range=[0, 50], height=400)
            st.plotly_chart(fig_ind, use_container_width=True)
            st.info("**Key Insight:** Manufacturing is the 2nd largest employer (25.4%), followed by Wholesale & Retail Trade (16.0%).")

        # ==============================================================================
        # TAB 2: DATA EXPLORER
        # ==============================================================================
        with tab2:
            # --- HELPER FUNCTIONS ---
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
            dist_col = "District"
            
            # --- FILTERS ---
            st.sidebar.markdown("## 🔍 Data Explorer Filters")
            
            if st.sidebar.button("🔄 Reset All Filters", on_click=reset_filters):
                st.rerun()

            def get_clean_list(column):
                if column and column in df.columns:
                    return sorted([x for x in df[column].unique().tolist() if str(x) not in ["#NULL!", "nan", "None", "", "Unknown"]])
                return []

            prov_list = get_clean_list(prov_col)
            sel_prov = st.sidebar.multiselect("Province", prov_list, default=prov_list, key='prov_key')

            if age_col:
                min_age, max_age = int(df[age_col].min()), int(df[age_col].max())
                sel_age = st.sidebar.slider("Age Range (Filter)", min_age, max_age, (min_age, max_age))

            sel_dist = []
            if dist_col in df.columns:
                valid_dist_mask = (df[prov_col] != "Balochistan")
                if sel_prov:
                    valid_dist_mask = valid_dist_mask & df[prov_col].isin(sel_prov)
                valid_districts = sorted([
                    x for x in df[valid_dist_mask][dist_col].unique().tolist() 
                    if str(x) not in ["#NULL!", "nan", "None", "", "Unknown"]
                ])
                sel_dist = st.sidebar.multiselect("District (Excl. Balochistan)", valid_districts, key='dist_key')

            sel_reg = st.sidebar.multiselect("Region", get_clean_list(reg_col), key='reg_key')
            sel_sex = st.sidebar.multiselect("Gender", get_clean_list(sex_col), key='sex_key')
            sel_edu = st.sidebar.multiselect("Education", get_clean_list(edu_col), key='edu_key')
            
            # --- APPLY FILTERS ---
            mask = pd.Series(True, index=df.index)
            if prov_col: mask = mask & df[prov_col].isin(sel_prov)
            if age_col: mask = mask & (df[age_col] >= sel_age[0]) & (df[age_col] <= sel_age[1])
            if sel_dist and dist_col in df.columns: mask = mask & df[dist_col].isin(sel_dist)
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
            ignore = [prov_col, reg_col, sex_col, edu_col, age_col, "Mouza", "Locality", "PCode", "EBCode", "District"]
            questions = [c for c in df.columns if c not in ignore]
            default_target = "Marital status (S4C7)"
            target_q = st.selectbox("Select Variable to Analyze:", questions, 
                                  index=questions.index(default_target) if default_target in questions else 0)

            if target_q:
                st.markdown(f"### 🧐 Analysis of: {target_q}")
                st.markdown("---")

                cols_to_load = [target_q] + [c for c in [prov_col, sex_col, reg_col, age_col, edu_col, dist_col] if c]
                main_data = df.loc[mask, cols_to_load]
                main_data[target_q] = main_data[target_q].astype(str)
                main_data = main_data[~main_data[target_q].isin(["#NULL!", "nan", "None", "DK", "NR"])]
                
                if "S4C81" in target_q or "S4C82" in target_q:
                    main_data[target_q] = main_data[target_q].astype(str).replace({
                        "1": "Yes", "1.0": "Yes", "01": "Yes", "Yes' 2'No": "Yes",
                        "2": "No",  "2.0": "No",  "02": "No"
                    })

                if not main_data.empty:
                    unique_options = sorted(main_data[target_q].unique())
                    default_opt = main_data[target_q].mode()[0]
                    map_choice = st.selectbox("Select Answer to Map:", unique_options, index=unique_options.index(default_opt) if default_opt in unique_options else 0)
                    
                    # --- SIDE-BY-SIDE MAPS ---
                    map_col1, map_col2 = st.columns(2)
                    
                    # 1. PROVINCE MAP
                    with map_col1:
                        st.subheader(f"Province: {map_choice}")
                        if pak_prov_json and prov_col:
                            prov_stats = pd.crosstab(main_data[prov_col], main_data[target_q], normalize='index') * 100
                            if map_choice in prov_stats.columns:
                                map_data = prov_stats[[map_choice]].reset_index()
                                map_data.columns = ["Province", "Percent"]
                                
                                fig_map = px.choropleth_mapbox(
                                    map_data, geojson=pak_prov_json, locations="Province",
                                    featureidkey="properties.shapeName",
                                    color="Percent", color_continuous_scale="Spectral_r",
                                    mapbox_style="carto-positron", zoom=4.5, center={"lat": 30.3753, "lon": 69.3451},
                                    opacity=0.7
                                )
                                fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=400)
                                st.plotly_chart(fig_map, use_container_width=True)
                            else:
                                st.info("No data for this option.")
                        else:
                            st.warning("⚠️ Province Map Missing")

                    # 2. DISTRICT MAP
                    with map_col2:
                        st.subheader(f"District: {map_choice}")
                        if pak_dist_json and dist_col in main_data.columns:
                            dist_stats = pd.crosstab(main_data[dist_col], main_data[target_q], normalize='index') * 100
                            if map_choice in dist_stats.columns:
                                d_map_data = dist_stats[[map_choice]].reset_index()
                                d_map_data.columns = ["District", "Percent"]
                                
                                fig_d_map = px.choropleth_mapbox(
                                    d_map_data, geojson=pak_dist_json, locations="District",
                                    featureidkey="properties.shapeName", 
                                    color="Percent", color_continuous_scale="Spectral_r",
                                    mapbox_style="carto-positron", zoom=4.5, center={"lat": 30.3753, "lon": 69.3451},
                                    opacity=0.7
                                )
                                fig_d_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=400)
                                st.plotly_chart(fig_d_map, use_container_width=True)
                            else:
                                st.info("No data for this option.")
                        else:
                             st.warning("⚠️ Please upload 'pakistan_districts.geojson'")

                    # 3. CHARTS ROW 1
                    col1, col2, col3 = st.columns([1.5, 1, 1])

                    with col1:
                        st.markdown("**📊 Overall Results (%)**")
                        counts = main_data[target_q].value_counts().reset_index()
                        counts.columns = ["Answer", "Count"]
                        counts["%"] = (counts["Count"] / counts["Count"].sum() * 100).fillna(0)
                        fig_bar = px.bar(counts, x="Answer", y="%", color="Answer", 
                                        text=counts["%"].apply(lambda x: f"{x:.1f}%"),
                                        color_discrete_sequence=px.colors.qualitative.Bold)
                        fig_bar.update_layout(showlegend=False, title_text="")
                        st.plotly_chart(fig_bar, use_container_width=True)

                    with col2:
                        st.markdown("**🗺️ By Province (Percentage)**")
                        if prov_col:
                            prov_grp = main_data.groupby([prov_col, target_q], observed=True).size().reset_index(name='Count')
                            prov_totals = prov_grp.groupby(prov_col, observed=True)['Count'].transform('sum')
                            prov_grp['%'] = (prov_grp['Count'] / prov_totals * 100).fillna(0)
                            fig_prov = px.bar(prov_grp, x=prov_col, y="%", color=target_q, barmode="stack")
                            fig_prov.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.2), yaxis_title="%", title_text="")
                            st.plotly_chart(fig_prov, use_container_width=True)

                    with col3:
                        st.markdown("**🚻 By Gender**")
                        if sex_col:
                            g_counts = main_data[sex_col].value_counts().reset_index()
                            g_counts.columns = ["Gender", "Count"]
                            fig_pie = px.pie(g_counts, names="Gender", values="Count", hole=0.5)
                            fig_pie.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.1), title_text="")
                            st.plotly_chart(fig_pie, use_container_width=True)

                    # 4. CHARTS ROW 2
                    col4, col5 = st.columns([1, 1.5])

                    with col4:
                        st.markdown("**🏙️ By Region**")
                        if reg_col:
                            reg_counts = main_data[reg_col].value_counts().reset_index()
                            reg_counts.columns = ["Region", "Count"]
                            fig_reg = px.pie(reg_counts, names="Region", values="Count", 
                                          color_discrete_sequence=px.colors.qualitative.Set3)
                            fig_reg.update_layout(title_text="")
                            st.plotly_chart(fig_reg, use_container_width=True)

                    with col5:
                        st.markdown("**📈 Age Trends (%)**")
                        if age_col:
                            chart_data = main_data.copy()
                            bins = [0, 4, 5, 9, 12, 15, 18, 24, 30, 40, 50, 60, 65, 200]
                            labels = ['0-4', '4-5', '5-9', '9-12', '12-15', '15-18', '18-24', '25-30', '30-40', '40-50', '50-60', '60-65', '65+']
                            
                            chart_data['AgeGrp'] = pd.cut(chart_data[age_col], bins=bins, labels=labels, right=False)
                            
                            age_grp = chart_data.groupby(['AgeGrp', target_q], observed=True).size().reset_index(name='Count')
                            age_totals = age_grp.groupby('AgeGrp', observed=True)['Count'].transform('sum')
                            age_grp['%'] = (age_grp['Count'] / age_totals * 100).fillna(0)
                            
                            age_grp['AgeGrp'] = pd.Categorical(age_grp['AgeGrp'], categories=labels, ordered=True)
                            age_grp = age_grp.sort_values('AgeGrp')
                            
                            fig_age = px.area(age_grp, x="AgeGrp", y="%", color=target_q, markers=True,
                                            category_orders={"AgeGrp": labels}) 
                            fig_age.update_xaxes(type='category') 
                            fig_age.update_layout(title_text="", showlegend=True, legend=dict(orientation="h", y=-0.2))
                            st.plotly_chart(fig_age, use_container_width=True)

                    # 5. TABLES
                    st.markdown("---")
                    st.subheader("📋 Detailed Data View")
                    t1, t2 = st.columns(2)
                    
                    with t1:
                        st.caption("Overall Response Count")
                        st.dataframe(counts, use_container_width=True, hide_index=True)
                        
                    with t2:
                        st.caption(f"Province Breakdown (Showing % for: {map_choice})")
                        if prov_col:
                            prov_pivot = pd.crosstab(main_data[prov_col], main_data[target_q], normalize='index') * 100
                            if not prov_pivot.empty:
                                if map_choice in prov_pivot.columns:
                                    prov_pivot = prov_pivot.sort_values(by=map_choice, ascending=False)
                                st.dataframe(prov_pivot.style.format("{:.1f}%"), use_container_width=True)

    except Exception as e:
        st.error(f"🚨 Critical Dashboard Error: {e}")
else:
    st.error("⚠️ Data failed to load. Please check file formatting.")
