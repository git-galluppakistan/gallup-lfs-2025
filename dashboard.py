import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import gc
import json
import numpy as np
import traceback
from io import BytesIO

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

# CRITICAL FIX: Changed to cache_data and added defragmentation steps
@st.cache_data(show_spinner="Loading Data...", ttl=7200)
def load_data():
    try:
        # A. Load Main Data
        file_name = "data.zip" if os.path.exists("data.zip") else "Data.zip"
        if not os.path.exists(file_name): return None, "Data file missing"

        # Load in chunks to manage memory during read
        chunks = []
        for chunk in pd.read_csv(file_name, compression='zip', chunksize=50000, low_memory=False, dtype=str):
            chunk.columns = chunk.columns.str.strip()
            
            # Numeric Age conversion per chunk
            age_col = next((c for c in chunk.columns if c in ['S4C6', 'Age']), None)
            if age_col: 
                chunk[age_col] = pd.to_numeric(chunk[age_col], errors='coerce')
            
            chunks.append(chunk)
        
        # Combine chunks
        df = pd.concat(chunks, axis=0)
        del chunks
        gc.collect()

        # --- CRITICAL MEMORY FIX 1: DEFRAGMENTATION ---
        # This prevents the "DataFrame is highly fragmented" warning that crashes the app
        df = df.copy()

        # --- B. PROVINCE STANDARDIZATION ---
        province_map = {
            "KP": "Khyber Pakhtunkhwa", "KPK": "Khyber Pakhtunkhwa", "N.W.F.P": "Khyber Pakhtunkhwa",
            "BALOUCHISTAN": "Balochistan", "Balouchistan": "Balochistan",
            "FATA": "Federally Administered Tribal Areas", "F.A.T.A": "Federally Administered Tribal Areas",
            "ICT": "Islamabad Capital Territory", "Islamabad": "Islamabad Capital Territory",
            "Punjab": "Punjab", "Sindh": "Sindh",
            "AJK": "Azad Jammu & Kashmir", "Azad Kashmir": "Azad Jammu & Kashmir",
            "GB": "Gilgit-Baltistan", "Gilgit Baltistan": "Gilgit-Baltistan"
        }
        
        # Apply map only to relevant columns
        for col in df.columns:
            if "Province" in col:
                df[col] = df[col].map(province_map).fillna(df[col]).astype("category")

        # --- C. DISTRICT MAPPING ---
        possible_files = ["district_mapping.csv", "DSTT.xlsx - Sheet1.csv", "lahore-district-mapping-file.xlsx - Lahore.csv"]
        dfs_to_merge = []
        for f in possible_files:
            if os.path.exists(f):
                temp = pd.read_csv(f, dtype=str)
                if "PCode" in temp.columns and "District" in temp.columns:
                    dfs_to_merge.append(temp)
        
        if dfs_to_merge and "PCode" in df.columns:
            combined_map = pd.concat(dfs_to_merge, ignore_index=True)
            dist_map = combined_map.drop_duplicates(subset="PCode").set_index("PCode")["District"].to_dict()
            
            # Manual Fixes
            dist_map['352'] = 'LAHORE'
            dist_map['201'] = 'LAHORE'
            dist_map['25121030'] = 'LAHORE'

            # --- CRITICAL MEMORY FIX 2: PRE-MAPPING COPY ---
            df = df.copy() 

            # Map & FORCE UPPER CASE
            df["District"] = df["PCode"].astype(str).map(dist_map)
            df["District"] = df["District"].fillna("Unknown")
            df["District"] = df["District"].astype(str).str.upper().str.strip()
            df["District"] = df["District"].astype('category')

        # D. Global Value Fixes
        if "S4C81" in df.columns:
            df["S4C81"] = df["S4C81"].astype(str).replace({"1": "Yes", "2": "No", "Yes' 2'No": "Yes"}).astype("category")

        # E. Codebook Rename
        if os.path.exists("code.csv"):
            codes = pd.read_csv("code.csv")
            rename_dict = {}
            for code, label in zip(codes.iloc[:, 0], codes.iloc[:, 1]):
                if code not in ['Province', 'Region', 'RSex', 'S4C5', 'S4C9', 'S4C6', 'District']:
                    rename_dict[code] = f"{label} ({code})"
            df.rename(columns=rename_dict, inplace=True)
        
        # Final Defrag before returning to app
        df = df.copy()
        return df, "Success"

    except Exception as e:
        return None, f"{str(e)} : {traceback.format_exc()}"

# --- LOAD DATA ---
df, status = load_data()
pak_dist_json = load_geojson_dist()
pak_prov_json = load_geojson_prov()

# --- 3. SESSION STATE & RESET LOGIC ---
if 'reset_trigger' not in st.session_state:
    st.session_state['reset_trigger'] = False

def reset_filters():
    # We delete the keys so Streamlit re-initializes widgets with their 'default' values
    keys_to_clear = ['prov_key', 'dist_key', 'reg_key', 'sex_key', 'edu_key']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

# --- 4. EXCEL EXPORT (Robust) ---
def to_excel(df_input):
    output = BytesIO()
    try:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_input.to_excel(writer, index=True, sheet_name='Sheet1')
    except:
        with pd.ExcelWriter(output) as writer:
            df_input.to_excel(writer, index=True, sheet_name='Sheet1')
    return output.getvalue()

# --- 5. DASHBOARD MAIN ---
if df is not None:
    # --- TABS ---
    tab1, tab2 = st.tabs(["📑 Executive Summary", "🔍 Data Explorer (Full Dashboard)"])

    # === TAB 1: SUMMARY ===
    with tab1:
        try:
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
                if "Province" in df.columns:
                    emp_counts = df["Province"].value_counts().reset_index()
                    emp_counts.columns = ["Province", "Count"]
                    fig = px.bar(emp_counts, x="Province", y="Count", color="Province", text="Count")
                    st.plotly_chart(fig, use_container_width=True)
            
            with c2:
                st.subheader("Key Metrics")
                pie_data = pd.DataFrame({"Metric": ["Employed", "Unemployed"], "Value": [92.9, 7.1]})
                fig = px.pie(pie_data, names="Metric", values="Value", hole=0.5)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
             st.error(f"Error in Tab 1: {e}")

    # === TAB 2: EXPLORER ===
    with tab2:
        try:
            # FILTERS
            st.sidebar.markdown("## 🔍 Data Explorer Filters")
            if st.sidebar.button("🔄 Reset All Filters", on_click=reset_filters): st.rerun()

            def get_clean_list(column):
                if column and column in df.columns:
                    return sorted([x for x in df[column].unique().tolist() if str(x) not in ["#NULL!", "nan", "None", "", "Unknown"]])
                return []

            prov_col = "Province"
            prov_list = get_clean_list(prov_col)
            sel_prov = st.sidebar.multiselect("Province", prov_list, default=prov_list, key='prov_key')

            age_col = next((c for c in df.columns if c in ['S4C6', 'Age']), None)
            if age_col:
                min_age, max_age = int(df[age_col].min()), int(df[age_col].max())
                sel_age = st.sidebar.slider("Age Range (Filter)", min_age, max_age, (min_age, max_age))

            dist_col = "District"
            sel_dist = []
            if dist_col in df.columns:
                valid_dist_mask = (df[prov_col] != "Balochistan")
                if sel_prov:
                    valid_dist_mask = valid_dist_mask & df[prov_col].isin(sel_prov)
                valid_districts = sorted([x for x in df[valid_dist_mask][dist_col].unique().tolist() if str(x) not in ["#NULL!", "nan", "None", "", "Unknown", "nan", "UNKNOWN"]])
                sel_dist = st.sidebar.multiselect("District (Excl. Balochistan)", valid_districts, key='dist_key')

            # Helper for other filters
            def get_col(candidates):
                for c in candidates:
                    for col in df.columns:
                        if c == col: return col
                        if c in col: return col
                return None

            reg_col = get_col(["Region"])
            sex_col = get_col(["S4C5", "RSex", "Gender"])
            edu_col = get_col(["S4C9", "Education", "Highest class"])

            sel_reg = st.sidebar.multiselect("Region", get_clean_list(reg_col), key='reg_key')
            sel_sex = st.sidebar.multiselect("Gender", get_clean_list(sex_col), key='sex_key')
            sel_edu = st.sidebar.multiselect("Education", get_clean_list(edu_col), key='edu_key')

            # MASK
            mask = pd.Series(True, index=df.index)
            if prov_col: mask = mask & df[prov_col].isin(sel_prov)
            if age_col: mask = mask & (df[age_col] >= sel_age[0]) & (df[age_col] <= sel_age[1])
            if sel_dist and dist_col in df.columns: mask = mask & df[dist_col].isin(sel_dist)
            if sel_reg: mask = mask & df[reg_col].isin(sel_reg)
            if sel_sex: mask = mask & df[sex_col].isin(sel_sex)
            if sel_edu: mask = mask & df[edu_col].isin(sel_edu)
            
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
            
            target = st.selectbox("Select Variable to Analyze:", questions, index=def_idx)
            
            # DATA PREP (Efficient)
            main_data = df.loc[mask].copy()
            gc.collect()

            main_data[target] = main_data[target].astype(str)
            main_data = main_data[~main_data[target].isin(["#NULL!", "nan", "None", "DK", "NR"])]
            
            if "S4C81" in target:
                 main_data[target] = main_data[target].replace({"1": "Yes", "2": "No", "Yes' 2'No": "Yes"})

            if not main_data.empty:
                opts = sorted(main_data[target].unique())
                if len(opts) > 0:
                    mode_val = main_data[target].mode()[0]
                    if mode_val not in opts: mode_val = opts[0]
                    map_choice = st.selectbox("Select Answer to Map:", opts, index=opts.index(mode_val))
                else:
                    map_choice = None
                
                # --- MAPS ---
                m1, m2 = st.columns(2)
                
                # PROVINCE MAP
                with m1:
                    if map_choice:
                        st.subheader(f"Province: {map_choice}")
                        if pak_prov_json:
                            p_stats = pd.crosstab(main_data["Province"], main_data[target], normalize='index') * 100
                            if map_choice in p_stats.columns:
                                p_map = p_stats[[map_choice]].reset_index()
                                p_map.columns = ["Province", "Percent"]
                                
                                # Updated to choropleth_map for Plotly 6.5 compatibility
                                fig = px.choropleth_map(
                                    p_map, geojson=pak_prov_json, locations="Province",
                                    featureidkey="properties.shapeName",
                                    color="Percent", color_continuous_scale="Spectral_r",
                                    map_style="carto-positron", zoom=4.5, center={"lat": 30.3753, "lon": 69.3451},
                                    opacity=0.7
                                )
                                # LABELS
                                centroids = pd.DataFrame([
                                    {"Province": "Punjab", "Lat": 31.1, "Lon": 72.7},
                                    {"Province": "Sindh", "Lat": 25.9, "Lon": 68.5},
                                    {"Province": "Balochistan", "Lat": 28.5, "Lon": 65.1},
                                    {"Province": "Khyber Pakhtunkhwa", "Lat": 34.9, "Lon": 72.3},
                                    {"Province": "Islamabad Capital Territory", "Lat": 33.7, "Lon": 73.1},
                                    {"Province": "Gilgit-Baltistan", "Lat": 35.8, "Lon": 74.5},
                                    {"Province": "Azad Jammu & Kashmir", "Lat": 33.7, "Lon": 73.8}
                                ])
                                
                                label_data = pd.merge(centroids, p_map, on="Province", how="inner")
                                if not label_data.empty:
                                    # Updated to scatter_map
                                    fig.add_trace(go.Scattermap(
                                        lat=label_data["Lat"], lon=label_data["Lon"], mode='text',
                                        text=label_data.apply(lambda x: f"{x['Percent']:.1f}%", axis=1),
                                        textfont=dict(size=14, color='black'), showlegend=False, hoverinfo='none'
                                    ))
                                fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=400)
                                st.plotly_chart(fig, use_container_width=True)
                
                # DISTRICT MAP
                with m2:
                    if map_choice:
                        st.subheader(f"District: {map_choice}")
                        if pak_dist_json and "District" in main_data.columns:
                            d_stats = pd.crosstab(main_data["District"], main_data[target], normalize='index') * 100
                            
                            if map_choice in d_stats.columns:
                                d_map = d_stats[[map_choice]].reset_index()
                                d_map.columns = ["District", "Percent"]
                                d_map["District"] = d_map["District"].astype(str).str.upper().str.strip()

                                # Updated to choropleth_map for Plotly 6.5 compatibility
                                fig = px.choropleth_map(
                                    d_map, geojson=pak_dist_json, locations="District",
                                    featureidkey="properties.districts",
                                    color="Percent", color_continuous_scale="Spectral_r",
                                    map_style="carto-positron", zoom=4.5, center={"lat": 30.3753, "lon": 69.3451},
                                    opacity=0.7
                                )
                                fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=400)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning(f"No data for '{map_choice}'")
                        else:
                            st.warning("District column missing or GeoJSON not loaded.")

                # --- CHARTS ---
                c1, c2, c3 = st.columns([1.5, 1, 1])
                with c1:
                    st.markdown("**📊 Overall (%)**")
                    counts = main_data[target].value_counts().reset_index()
                    counts.columns = ["Answer", "Count"]
                    counts["%"] = (counts["Count"] / counts["Count"].sum() * 100)
                    fig = px.bar(counts, x="Answer", y="%", text=counts["%"].apply(lambda x: f"{x:.1f}%"))
                    st.plotly_chart(fig, use_container_width=True)
                
                with c2:
                    st.markdown("**🗺️ By Province (%)**")
                    if prov_col:
                        p_grp = main_data.groupby([prov_col, target], observed=True).size().reset_index(name='Count')
                        p_tot = p_grp.groupby(prov_col, observed=True)['Count'].transform('sum')
                        p_grp['%'] = (p_grp['Count'] / p_tot * 100).fillna(0)
                        fig = px.bar(p_grp, x=prov_col, y="%", color=target, barmode="stack")
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

                with c3:
                    st.markdown("**🚻 By Gender**")
                    if sex_col:
                        g_counts = main_data[sex_col].value_counts().reset_index()
                        g_counts.columns = ["Gender", "Count"]
                        fig = px.pie(g_counts, names="Gender", values="Count", hole=0.5)
                        fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.1))
                        st.plotly_chart(fig, use_container_width=True)

                # --- TABLES ---
                st.markdown("---")
                t_head, t_btn = st.columns([4, 1])
                t_head.subheader("📋 Detailed Data View")
                
                if map_choice and "District" in main_data.columns:
                    d_stats = pd.crosstab(main_data["District"], main_data[target], normalize='index') * 100
                    if map_choice in d_stats.columns:
                        final_table = d_stats.sort_values(by=map_choice, ascending=False)
                        
                        # EXCEL BUTTON
                        try:
                            excel_data = to_excel(final_table)
                            t_btn.download_button(
                                label="📥 Download Excel",
                                data=excel_data,
                                file_name=f'gallup_data_{map_choice}.xlsx',
                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                            )
                        except Exception as e:
                            t_btn.error("Export Failed")
                        
                        st.dataframe(final_table.style.format("{:.1f}%"), use_container_width=True)
        except Exception as e:
            st.error(f"Error in Tab 2: {e}")
            st.code(traceback.format_exc())

else:
    st.error(f"Data Load Failed: {status}")
