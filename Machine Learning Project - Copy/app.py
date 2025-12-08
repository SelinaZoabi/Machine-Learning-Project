import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

# -------------------------
# Load model
# -------------------------
model = xgb.XGBClassifier()
model.load_model("best_xgb.json")   # ⬅️ Make sure this is your saved XGBoost model


# -------------------------
# Training feature order 
# -------------------------
FEATURE_ORDER = [
    'Primary Type', 'Description', 'Location Description', 'Arrest',
    'IUCR', 'FBI Code', 'Beat', 'District', 'Ward', 'Community Area',
    'X Coordinate', 'Y Coordinate', 'Latitude', 'Longitude', 'year',
    'month', 'day', 'hour', 'dayofweek', 'is_weekend', 'is_night',
    'month_sin', 'month_cos', 'lat_lon_sum', 'lat_lon_diff',
    'district_risk', 'hour_risk'
]


# -------------------------
# Feature Engineering
# -------------------------
def build_features(data):
    df = pd.DataFrame([data])

    # ---- 1) Convert categorical -> category codes
    cat_cols = ['Primary Type', 'Description', 'Location Description', 'IUCR', 'FBI Code']
    for c in cat_cols:
        df[c] = df[c].astype("category").cat.codes

    # ---- 2) Convert numeric
    num_cols = [c for c in df.columns if c not in cat_cols]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- 3) Feature engineering
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_night'] = ((df['hour'] >= 20) | (df['hour'] <= 6)).astype(int)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['lat_lon_sum'] = df['Latitude'] + df['Longitude']
    df['lat_lon_diff'] = df['Latitude'] - df['Longitude']

    # ---- 4) Recreate district_risk & hour_risk
    df['district_risk'] = df['District'] / 25
    df['hour_risk'] = df['hour'] / 23

    # ---- 5) Final order
    df = df[FEATURE_ORDER]

    return df

# Streamlit UI
st.set_page_config(page_title="Chicago Crime Predictor", layout="wide")
st.title("🔍 Chicago Crime Risk Prediction")
st.caption("Model used: XGBoost (Optuna-tuned, F1 ≈ 0.80)")


with st.sidebar:
    st.header("Input Features")

    primary_type = st.text_input("Primary Type", "THEFT")
    desc = st.text_input("Description", "SIMPLE")
    loc_desc = st.text_input("Location Description", "STREET")
    iucr = st.text_input("IUCR Code", "0810")
    fbi = st.text_input("FBI Code", "08B")

    arrest = st.selectbox("Arrest", [0, 1])
    beat = st.number_input("Beat", 0, 9999, 111)
    district = st.number_input("District", 0, 25, 11)
    ward = st.number_input("Ward", 0, 50, 10)
    comm = st.number_input("Community Area", 0, 77, 25)

    x = st.number_input("X Coordinate", value=0.0)
    y = st.number_input("Y Coordinate", value=0.40)
    lat = st.number_input("Latitude", value=41.87)
    lon = st.number_input("Longitude", value=-87.65)

    year = st.slider("Year", 2001, 2004)
    month = st.slider("Month", 1, 12, 6)
    day = st.slider("Day", 1, 31, 15)
    hour = st.slider("Hour", 0, 23, 13)
    dow = st.slider("Day of Week (Mon=0)", 0, 6, 3)

    predict = st.button("Predict Crime Risk")


if predict:
    row = {
        'Primary Type': primary_type,
        'Description': desc,
        'Location Description': loc_desc,
        'Arrest': arrest,
        'IUCR': iucr,
        'FBI Code': fbi,
        'Beat': beat,
        'District': district,
        'Ward': ward,
        'Community Area': comm,
        'X Coordinate': x,
        'Y Coordinate': y,
        'Latitude': lat,
        'Longitude': lon,
        'year': year,
        'month': month,
        'day': day,
        'hour': hour,
        'dayofweek': dow
    }

    X = build_features(row)
    prob = model.predict_proba(X)[0][1]
    pred = int(prob >= 0.5)

    st.subheader("Prediction Result")
    st.metric("Risk of Incident (Probability):", f"{prob:.2%}")
    st.write("Prediction:", "**YES**" if pred == 1 else "**NO**")
