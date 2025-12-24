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
        background-color: #f0f2f6;
        border: 1px solid #d6d6d6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Gallup Pakistan: National LFS Survey 2024-25")

# --- 2. OPTIMIZED DATA LOADER ---
@st.cache_resource
def load_data_optimized():
    try:
        # A. File Check
        file_name = "data.zip" if os.path.exists("data.zip") else "Data.zip"
        if not os.path.exists(file_name):
            st.error(f"File not found: {file_name}")
            return None

        # B. Chunk Load
        chunks = []
        for chunk in pd.read_csv(file_name, compression='zip', chunksize=50000, low_memory=True, dtype=str):
            for col in chunk.columns:
                chunk[col] = chunk[col].astype('category')
            
            # AGE FIX
            age_col = next((c for c in chunk.columns if c in ['S4C6', 'Age']), None)
            if age_col:
                chunk[age_col] = pd.to_numeric(chunk[age_col], errors='coerce')

            chunks.append(chunk)
        
        df = pd.concat(chunks, axis=0)
        del chunks
        gc.collect()

        # --- C. PROVINCE NAME STANDARDIZATION ---
        # This matches your Data to the GeoJSON keys
        province_map = {
            "KP": "Khyber Pakhtunkhwa",
            "KPK": "Khyber Pakhtunkhwa",
            "N.W.F.P": "Khyber Pakhtunkhwa",
            "BALOUCHISTAN": "Balochistan",
            "Balouchistan": "Balochistan",
            "FATA": "Federally Administered Tribal Areas",
            "F.A.T.A": "Federally Administered Tribal Areas",
            "ICT": "Islamabad",
            "Islamabad Capital Territory": "Islamabad",
            "Punjab": "Punjab",
            "Sindh": "Sindh",
            "AJK": "Azad Kashmir",
            "GB": "Gilgit Baltistan"
        }
        
        # Apply standard names to all Province columns found
        for col in df.columns:
            if "Province" in col:
                df[col] = df[col].astype(str).replace(province_map).astype("category")

        # D. Load Codebook
        if os.path.exists("code.csv"):
            codes = pd.read_csv("code.csv")
            rename_dict = {}
            for code, label in zip(codes.iloc[:, 0], codes.iloc[:, 1]):
                if code not in ['Province', 'Region', 'RSex', 'S4C5', 'S4C9', 'S4C6']:
                    rename_dict[code] = f"{label} ({code})"
            df.rename(columns=rename_dict, inplace=True)

        return df

    except Exception as e:
        st.error(f"Error: {e}")
        return None

df = load_data_optimized()

# --- 3. DASHBOARD LOGIC ---
if df is not None:
    
    # --- GLOBAL DATA CLEANING ---
    def get_col(candidates):
        for c in candidates:
            for col in df.columns:
                if c == col: return col
                if c in col: return col
        return None

    prov_col = get_col(["Province"])
    reg_col = get_col(["Region"])
    
    # Clean bad rows
    for col in [prov_col, reg_col]:
        if col and col in df.columns:
            df = df[~df[col].astype(str).isin(["#NULL!", "nan", "None", "nan", ""])]

    # --- SIDEBAR FILTERS ---
    st.sidebar.title("🔍 Filter Panel")

    sex_col = get_col(["S4C5", "RSex", "Gender"])
    edu_col = get_col(["S4C9", "Education", "Highest class"])
    age_col = get_col(["S4C6", "Age"])

    def get_clean_list(column):
        if column and column in df.columns:
            return sorted([x for x in df[column].unique().tolist() if str(x) not in ["#NULL!", "nan", "None", "", "Unknown"]])
        return []

    # 1. Province Filter
    prov_list = get_clean_list(prov_col)
    sel_prov = st.sidebar.multiselect("Province", prov_list, default=prov_list)
    
    # 2. Age Range
    if age_col:
        min_age, max_age = int(df[age_col].min()), int(df[age_col].max())
        sel_age = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age))
    
    # 3. Region & Gender
    sel_reg = st.sidebar.multiselect("Region", get_clean_list(reg_col))
    sel_sex = st.sidebar.multiselect("Gender", get_clean_list(sex_col))
    
    # --- FILTER MASK ---
    mask = pd.Series(True, index=df.index)
    if prov_col: mask = mask & df[prov_col].isin(sel_prov)
    if age_col: mask = mask & (df[age_col] >= sel_age[0]) & (df[age_col] <= sel_age[1])
    if sel_reg: mask = mask & df[reg_col].isin(sel_reg)
    if sel_sex: mask = mask & df[sex_col].isin(sel_sex)
        
    filtered_count = mask.sum()

    # --- KPI CARDS ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Database", f"{len(df):,.0f}")
    c2.metric("Filtered Respondents", f"{filtered_count:,.0f}")
    c3.metric("Selection Share", f"{(filtered_count/len(df)*100):.1f}%")
    
    st.markdown("---")

    # --- MAIN QUESTION SELECTION ---
    ignore = [prov_col, reg_col, sex_col, edu_col, age_col, "Mouza", "Locality", "PCode", "EBCode"]
    questions = [c for c in df.columns if c not in ignore]
    
    default_target = "Marital Status (S4C7)"
    default_index = questions.index(default_target) if default_target in questions else 0
    target_q = st.selectbox("Select Question to Analyze:", questions, index=default_index)

    if target_q:
        # Prepare Data
        cols_to_load = [target_q] + [c for c in [prov_col, sex_col, reg_col, age_col] if c]
        main_data = df.loc[mask, cols_to_load]
        
        # --- CHART CLEANING ---
        main_data[target_q] = main_data[target_q].astype(str)
        main_data = main_data[~main_data[target_q].isin(["#NULL!", "nan", "None", "DK", "NR"])]
        
        if not main_data.empty:
            top_ans = main_data[target_q].mode()[0]
        else:
            top_ans = "N/A"
            st.warning("No data available after filtering.")

        # ==========================================================
        # ROW 1: THE HERO MAP (PROVINCE LEVEL)
        # ==========================================================
        st.subheader("🗺️ Geographic Distribution (By Province)")
        st.caption(f"**Red** = High Percentage | **Blue** = Low Percentage (Showing data for: '{top_ans}')")
        
        geojson_path = "pakistan_districts.geojson"
        
        if os.path.exists(geojson_path) and prov_col:
            with open(geojson_path) as f:
                pak_geojson = json.load(f)
            
            # Calculate Province Stats
            prov_stats = pd.crosstab(main_data[prov_col], main_data[target_q], normalize='index') * 100
            
            if top_ans in prov_stats.columns:
                map_data = prov_stats[[top_ans]].reset_index()
                map_data.columns = ["Province", "Percent"]
                
                # DRAW MAP
                # Note: We use the SAME District GeoJSON, but map to "province_territory"
                # This colors all districts in a province the same color.
                fig_map = px.choropleth_mapbox(
                    map_data, 
                    geojson=pak_geojson, 
                    locations="Province",
                    featureidkey="properties.province_territory", # <-- The Key Change
                    color="Percent", 
                    color_continuous_scale="Spectral_r", 
                    mapbox_style="carto-positron",
                    zoom=4.5, center = {"lat": 30.3753, "lon": 69.3451},
                    opacity=0.7, 
                    labels={'Percent': f'% {top_ans}'}
                )
                fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=500)
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.warning("Not enough data to map.")
        else:
            st.warning("⚠️ Map file missing.")

        # ==========================================================
        # ROW 2: STANDARD CHARTS
        # ==========================================================
        st.markdown("---")
        col1, col2, col3 = st.columns([1.5, 1, 1])

        with col1:
            st.markdown("**📊 Overall Results (%)**")
            counts = main_data[target_q].value_counts().reset_index()
            counts.columns = ["Answer", "Count"]
            total = counts["Count"].sum()
            counts["%"] = (counts["Count"] / total * 100).fillna(0)
            
            fig1 = px.bar(counts, x="Answer", y="%", color="Answer", 
                          text=counts["%"].apply(lambda x: f"{x:.1f}%"),
                          template="plotly_white", 
                          color_discrete_sequence=px.colors.qualitative.Bold)
            fig1.update_layout(showlegend=True)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.markdown("**🗺️ By Province (%)**")
            if prov_col:
                prov_grp = main_data.groupby([prov_col, target_q], observed=True).size().reset_index(name='Count')
                prov_totals = prov_grp.groupby(prov_col, observed=True)['Count'].transform('sum')
                prov_grp['%'] = (prov_grp['Count'] / prov_totals * 100).fillna(0)
                
                fig2 = px.bar(prov_grp, x=prov_col, y="%", color=target_q,
                              template="plotly_white", barmode="stack")
                fig2.update_layout(showlegend=True, yaxis_title="%")
                st.plotly_chart(fig2, use_container_width=True)

        with col3:
            st.markdown("**🚻 By Gender**")
            if sex_col:
                gender_counts = main_data[sex_col].value_counts().reset_index()
                gender_counts.columns = ["Gender", "Count"]
                fig3 = px.pie(gender_counts, names="Gender", values="Count", hole=0.5,
                              color_discrete_sequence=px.colors.qualitative.Pastel)
                fig3.update_layout(showlegend=True, legend=dict(orientation="h"))
                st.plotly_chart(fig3, use_container_width=True)

        # ==========================================================
        # ROW 3: REGION & AGE
        # ==========================================================
        col4, col5 = st.columns([1, 1.5])

        with col4:
            st.markdown("**🏙️ By Region**")
            if reg_col:
                reg_counts = main_data[reg_col].value_counts().reset_index()
                reg_counts.columns = ["Region", "Count"]
                fig4 = px.pie(reg_counts, names="Region", values="Count", 
                              color_discrete_sequence=px.colors.qualitative.Set3)
                fig4.update_layout(showlegend=True, legend=dict(orientation="h"))
                st.plotly_chart(fig4, use_container_width=True)

        with col5:
            st.markdown("**📈 Age Trends (%)**")
            if age_col:
                chart_data = main_data.copy()
                chart_data['AgeGrp'] = pd.cut(chart_data[age_col], bins=[0,18,30,45,60,100], labels=['<18','18-30','31-45','46-60','60+'])
                age_grp = chart_data.groupby(['AgeGrp', target_q], observed=True).size().reset_index(name='Count')
                age_totals = age_grp.groupby('AgeGrp', observed=True)['Count'].transform('sum')
                age_grp['%'] = (age_grp['Count'] / age_totals * 100).fillna(0)
                
                fig5 = px.area(age_grp, x="AgeGrp", y="%", color=target_q,
                               template="plotly_white", markers=True)
                fig5.update_layout(showlegend=True, yaxis_title="%")
                st.plotly_chart(fig5, use_container_width=True)

        # ==========================================================
        # ROW 5: TABLES
        # ==========================================================
        st.markdown("---")
        t1, t2 = st.columns(2)
        
        with t1:
            st.subheader("📋 Overall Data")
            counts["%"] = counts["%"].map("{:.1f}%".format)
            st.dataframe(counts, use_container_width=True, hide_index=True)
            
        with t2:
            st.subheader(f"🗺️ Province Rankings (Top % {top_ans})")
            if prov_col:
                dist_pivot = pd.crosstab(main_data[prov_col], main_data[target_q], normalize='index') * 100
                if not dist_pivot.empty:
                    dist_pivot = dist_pivot.sort_values(by=top_ans, ascending=False).head(50)
                    dist_display = dist_pivot.applymap(lambda x: f"{x:.1f}%")
                    st.dataframe(dist_display, use_container_width=True)

else:
    st.info("Awaiting Data...")

