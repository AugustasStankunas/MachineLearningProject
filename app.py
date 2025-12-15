import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import joblib
import torch
from PIL import Image
import io
import re

from image_processing import img_model, transform, device
from group import ARTIFACT_DIR, MODEL_PATH, PCA_PATH, COLUMNS_PATH

# --- 1. Konfigūracija ir Profesionalus Dizainas (CSS) ---
st.set_page_config(
    page_title="EstateAI Pro | Pietų Ispanija",
    page_icon="🏢",
    layout="wide", # Naudojame visą plotį
    initial_sidebar_state="expanded"
)

# Custom CSS - čia vyksta visa magija
st.markdown("""
    <style>
    /* Pagrindinis fonas */
    .stApp {
        background-color: #f4f7f6;
    }
    
    /* Paslepiame standartinį meniu ir footerį švaresniam vaizdui */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Konteinerių (Cards) stilius */
    .css-1r6slb0, .css-12oz5g7 {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    /* Antraštės stilius */
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        color: #1E293B;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-family: 'Helvetica Neue', sans-serif;
        color: #64748B;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Mygtuko stilius */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        border: none;
        padding: 0.8rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(16, 185, 129, 0.2);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(16, 185, 129, 0.3);
        color: white;
    }
    
    /* Metrikos (Kainos) stilius */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        color: #059669;
        font-weight: 800;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1rem;
        color: #64748B;
    }
    
    /* Sidebar stilius */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Formų antraštės */
    .form-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #334155;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Pagalbinės funkcijos ---
@st.cache_resource
def load_models():
    if not MODEL_PATH.exists() or not PCA_PATH.exists() or not COLUMNS_PATH.exists():
        return None, None, None
    model = joblib.load(MODEL_PATH)
    pca = joblib.load(PCA_PATH)
    with open(COLUMNS_PATH, "r", encoding="utf-8") as f:
        model_columns = json.load(f)
    return model, pca, model_columns

@st.cache_data
def load_dropdown_data():
    csv_path = "data/properties.csv"
    if not os.path.exists(csv_path):
        return None, None, None, None, None, None
    df = pd.read_csv(csv_path, header=None, encoding="utf-8")
    df.columns = ["reference", "location", "price", "title", "bedrooms", "bathrooms", "indoor_sqm", "outdoor_sqm", "features"]
    
    def extract_property_type(title):
        if pd.isna(title): return "Unknown"
        t = str(title).strip()
        t = re.sub(r"\s+", " ", t)
        m = re.match(r"^\d+\s+Bedrooms?\s+(.*)$", t, flags=re.IGNORECASE)
        if m: return m.group(1).strip()
        m = re.match(r"^\d+\s+(.*)$", t)
        if m: return m.group(1).strip()
        return t

    df["property_type"] = df["title"].apply(extract_property_type)
    df["features_list"] = df["features"].fillna("").str.split("|")
    all_features = sorted(set(f.strip() for sub in df["features_list"] for f in sub if f.strip()))
    unique_locations = sorted(df["location"].dropna().unique())
    unique_property_types = sorted(df["property_type"].dropna().unique())
    location_freq = df["location"].value_counts(normalize=True)
    type_freq = df["property_type"].value_counts(normalize=True)
    
    return unique_locations, unique_property_types, all_features, location_freq, type_freq

def merge_rare_categories(value, freq_series, min_freq=0.01, other_label="Other"):
    if value in freq_series.index and freq_series[value] >= min_freq:
        return value
    return other_label

# --- 3. UI Struktūra ---

def main():
    # Krovimas
    model, pca, model_columns = load_models()
    locs, types, feats, loc_freq, type_freq = load_dropdown_data()

    if not model or not locs:
        st.error("Klaida: Nerasti modelio failai arba data/properties.csv. Įsitikinkite, kad `group.py` buvo paleistas.")
        st.stop()

    # Sidebar: Informacija
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/609/609803.png", width=80)
        st.markdown("### EstateAI Pro")
        st.info(
            """
            **Apie įrankį:**
            Ši sistema naudoja dirbtinį intelektą (XGBoost + Computer Vision), 
            kad įvertintų nekilnojamojo turto kainą Pietų Ispanijoje.
            
            **Kaip naudotis:**
            1. Užpildykite NT parametrus.
            2. Įkelkite nuotraukas (nebūtina).
            3. Spauskite "Skaičiuoti kainą".
            """
        )
        st.markdown("---")
        st.caption("© 2024 AI Real Estate Analytics v1.2")

    # Main Area
    st.markdown('<div class="main-header">Rinkos Kainos Analizė</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Pietų Ispanijos nekilnojamojo turto vertinimo algoritmas</div>', unsafe_allow_html=True)

    # Sukuriame dviejų stulpelių išdėstymą (2/3 forma, 1/3 rezultatai vėliau)
    col_left, col_right = st.columns([2, 1], gap="large")

    with col_left:
        # 1 Kortelė: Pagrindiniai duomenys
        with st.container():
            st.markdown('<div class="form-header">📍 Vieta ir Tipas</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                location = st.selectbox("Miestas / Regionas", locs, index=locs.index("Marbella") if "Marbella" in locs else 0)
            with c2:
                property_type = st.selectbox("Objekto tipas", types, index=types.index("Apartment") if "Apartment" in types else 0)

        st.write("") # Tarpas

        # 2 Kortelė: Detalės
        with st.container():
            st.markdown('<div class="form-header">📐 Specifikacijos</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                bedrooms = st.number_input("Miegamieji", 0, 30, 3)
            with c2:
                bathrooms = st.number_input("Vonios", 0, 20, 2)
            with c3:
                indoor_sqm = st.number_input("Vidaus m²", 0.0, 5000.0, 120.0, step=10.0)
            with c4:
                outdoor_sqm = st.number_input("Lauko m²", 0.0, 5000.0, 50.0, step=10.0)
            
            st.markdown("---")
            with st.expander("➕ Papildomi privalumai (Baseinas, Garažas...)"):
                selected_features = st.multiselect("Pasirinkite privalumus:", feats)

        st.write("") # Tarpas

        # 3 Kortelė: Vaizdai
        with st.container():
            st.markdown('<div class="form-header">📸 Vizualinė Analizė</div>', unsafe_allow_html=True)
            uploaded_files = st.file_uploader("Įkelkite nuotraukas (AI analizuos interjero kokybę)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
            
            if uploaded_files:
                st.caption(f"Pasirinkta nuotraukų: {len(uploaded_files)}")
                # Rodyti miniatiūras
                cols = st.columns(6)
                for i, file in enumerate(uploaded_files[:6]):
                    cols[i].image(file, use_container_width=True)

    # Dešinysis stulpelis - Veiksmas ir Rezultatas
    with col_right:
        st.markdown("<br>", unsafe_allow_html=True) # Šiek tiek nuleidžiame žemyn
        
        # Action Card
        with st.container():
            st.info("💡 **Patarimas:** Tiksliausiam rezultatui įkelkite bent 3 vidaus nuotraukas.")
            
            predict_btn = st.button("🚀 Skaičiuoti Kainą", use_container_width=True)

        if predict_btn:
            with st.spinner("AI analizuoja duomenis ir nuotraukas..."):
                try:
                    # --- DUOMENŲ PARUOŠIMAS (Logika išlieka ta pati) ---
                    location_clean = merge_rare_categories(location, loc_freq, other_label="Other_location")
                    property_type_clean = merge_rare_categories(property_type, type_freq, other_label="Other_property_type")

                    data = {
                        "bedrooms": [bedrooms], "bathrooms": [bathrooms],
                        "indoor_sqm": [indoor_sqm], "outdoor_sqm": [outdoor_sqm],
                    }
                    X_new = pd.DataFrame(data)

                    X_new[f"location_{location_clean}"] = 1
                    X_new[f"property_type_{property_type_clean}"] = 1

                    for col in [c for c in model_columns if c.startswith("feat_")]:
                        feat_name = col[5:]
                        X_new[col] = 1 if feat_name in selected_features else 0

                    # Vaizdai
                    embs = []
                    if uploaded_files:
                        for file in uploaded_files[:8]:
                            file.seek(0)
                            bytes_data = file.read()
                            img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
                            x = transform(img).unsqueeze(0).to(device)
                            with torch.no_grad():
                                emb = img_model(x).cpu().numpy().squeeze()
                            embs.append(emb)

                    if embs:
                        emb_mean = np.mean(embs, axis=0)
                    else:
                        emb_mean = np.zeros(img_model.num_features)

                    emb_red = pca.transform([emb_mean])[0]
                    for i, val in enumerate(emb_red):
                        X_new[f"img_{i}"] = val

                    X_new = X_new.reindex(columns=model_columns, fill_value=0)

                    # Prognozė
                    pred_log = model.predict(X_new)[0]
                    predicted_price = np.expm1(pred_log)

                    # --- REZULTATO ATVAIZDAVIMAS ---
                    st.success("✅ Skaičiavimas baigtas!")
                    
                    st.markdown("---")
                    st.metric(label="Rinkos vertė", value=f"€ {predicted_price:,.0f}")
                    
                    st.caption("Ši kaina yra algoritmo prognozė, paremta panašiais objektais rinkoje.")

                except Exception as e:
                    st.error(f"Klaida: {str(e)}")

if __name__ == "__main__":
    main()