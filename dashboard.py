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

# --- 2. AGGRESSIVE CLEANING FUNCTIONS ---
def clean_pcode(x):
    """Removes .0, spaces, and converts to string"""
    if pd.isna(x) or str(x).lower() in ['nan', 'none', '']:
        return None
    s = str(x).strip()
    if "." in s:
        s = s.split(".")[0]
    return s

# --- 3. DATA LOADERS ---
@st.cache_data(ttl=3600)
def load_geojson_dist():
    path = "pakistan_districts.geojson" 
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
            # Optimize columns
            for col in chunk.columns:
                chunk[col] = chunk[col].astype('category')
            # Clean PCode IMMEDIATELY upon load
            if "PCode" in chunk.columns:
                chunk["PCode"] = chunk["PCode"].apply(clean_pcode).astype('category')
            chunks.append(chunk)
        
        df = pd.concat(chunks, axis=0)
        del chunks
        gc.collect()

        # B. Load & Merge Mappings (Aggressive Fix)
        possible_files = ["district_mapping.csv", "DSTT.xlsx - Sheet1.csv", "lahore-district-mapping-file.xlsx - Lahore.csv"]
        dfs_to_merge = []
        for f in possible_files:
            if os.path.exists(f):
                temp = pd.read_csv(f, dtype=str)
                if "PCode" in temp.columns and "District" in temp.columns:
                    # Clean Mapping PCode to match Data PCode
                    temp["PCode"] = temp["PCode"].apply(clean_pcode)
                    dfs_to_merge.append(temp)
        
        if dfs_to_merge and "PCode" in df.columns:
            combined_map = pd.concat(dfs_to_merge, ignore_index=True)
            # Create Dictionary
            dist_map = combined_map.drop_duplicates(subset="PCode").set_index("PCode")["District"].to_dict()
            
            # Manual Fixes
            dist_map['352'] = 'Lahore'
            dist_map['201'] = 'Lahore'
            dist_map['25121030'] = 'Lahore' # From your uploaded file

            # Map & Title Case
            df["District"] = df["PCode"].map(dist_map)
            df["District"] = df["District"].astype(str).str.title().str.strip() # "LAHORE" -> "Lahore"
            
            # Remove "Nan" strings
            df.loc[df["District"] == "Nan", "District"] = np.nan
            df["District"] = df["District"].astype('category')

        # C. Global Fixes
        if "S4C81" in df.columns:
            df["S4C81"] = df["S4C81"].astype(str).replace({"1": "Yes", "2": "No", "Yes' 2'No": "Yes"}).astype("category")

        return df, "Success"

    except Exception as e:
        return None, str(e)

df, status = load_data()
pak_dist_json = load_geojson_dist()

# --- 4. DEBUGGER SECTION (Use this to find the error!) ---
with st.expander("🛠️ DEBUGGER (Expand if Map is Blank)", expanded=False):
    c1, c2, c3 = st.columns(3)
    if df is not None:
        c1.write("**Data PCodes (Sample):**")
        c1.write(df['PCode'].unique().tolist()[:5])
        
        c1.write("**Mapped Districts (Sample):**")
        if "District" in df.columns:
            c1.write(df['District'].unique().tolist()[:5])
        else:
            c1.error("No District Column Created!")

    if pak_dist_json:
        c2.write("**GeoJSON Keys Found:**")
        # Check the first feature to see what properties exist
        props = pak_dist_json['features'][0]['properties']
        c2.write(list(props.keys()))
        
        c2.write("**GeoJSON District Names (Sample):**")
        # Try to find the name key
        name_key = next((k for k in props.keys() if k in ['districts', 'DISTRICT', 'shapeName', 'NAME_2', 'name']), None)
        if name_key:
            sample_names = [f['properties'][name_key] for f in pak_dist_json['features'][:5]]
            c2.write(sample_names)
            st.session_state['geo_key'] = name_key # Store for main app
        else:
            c2.error("Could not find a Name key in GeoJSON!")
            st.session_state['geo_key'] = None

# --- 5. DASHBOARD MAIN ---
if df is not None:
    # FILTERS
    st.sidebar.header("Filters")
    if st.sidebar.button("Reset Filters"): st.rerun()
    
    prov = st.sidebar.multiselect("Province", df["Province"].unique())
    
    # Filter Data
    mask = pd.Series(True, index=df.index)
    if prov: mask = mask & df["Province"].isin(prov)
    
    # SELECTOR
    questions = [c for c in df.columns if c not in ["PCode", "District", "Province", "Region", "S4C6"]]
    target = st.selectbox("Select Variable:", questions, index=0)
    
    # DATA PREP
    main_data = df.loc[mask].copy()
    main_data = main_data[~main_data[target].isin(["#NULL!", "nan", "None"])]
    
    if not main_data.empty:
        # DROPDOWN FOR MAP
        opts = sorted(main_data[target].unique())
        map_choice = st.selectbox("Select Answer to Map:", opts)
        
        # --- MAP LOGIC ---
        st.subheader(f"District Map: {map_choice}")
        
        if pak_dist_json and "District" in main_data.columns:
            # Group by District
            d_stats = pd.crosstab(main_data["District"], main_data[target], normalize='index') * 100
            
            if map_choice in d_stats.columns:
                map_df = d_stats[[map_choice]].reset_index()
                map_df.columns = ["District", "Percent"]
                
                # USE DYNAMIC KEY FROM DEBUGGER
                feature_key = st.session_state.get('geo_key', 'properties.shapeName')
                
                fig = px.choropleth_mapbox(
                    map_df, geojson=pak_dist_json, locations="District",
                    featureidkey=f"properties.{feature_key}", # Dynamic Key!
                    color="Percent", color_continuous_scale="Spectral_r",
                    mapbox_style="carto-positron", zoom=4.5, center={"lat": 30.3753, "lon": 69.3451},
                    opacity=0.7
                )
                fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No data found for '{map_choice}' in any district.")
        else:
            st.warning("District column missing or GeoJSON not loaded.")
            
        # --- TABLES ---
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Detailed Breakdown")
            st.dataframe(d_stats.style.format("{:.1f}%"), use_container_width=True)

else:
    st.error(f"Data Load Failed: {status}")
