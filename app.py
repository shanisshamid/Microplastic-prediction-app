import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page config FIRST
st.set_page_config(page_title="MPs Prediction", layout="wide")

# Background image
RAW_LINK = "https://raw.githubusercontent.com/shanisshamid/Microplastic-prediction-app/main/river%20wallpaper%203.jpg"

def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({image_url});
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background(RAW_LINK)

# Load model and scaler
@st.cache_resource
def load_assets():
    model = joblib.load("champion_gradientboost_model.pkl")
    scaler = joblib.load("scaler_aug.pkl")
    return model, scaler

model, scaler = load_assets()

# UI
st.title("💧 Microplastic Concentration Predictor for Penang Rivers")
st.markdown("Enter physicochemical water quality parameters to estimate **microplastic concentration (particles/L)**.")

with st.form("prediction_form"):
    st.header("Water Quality Inputs")

    temp = st.number_input("Temperature (°C)", value=28.0)
    ph = st.number_input("pH", value=7.0)
    do = st.number_input("DO (mg/L)", value=6.5)
    cdc = st.number_input("CDC (µS/cm)", value=500.0)
    turb = st.number_input("Turbidity (NTUs)", value=15.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    # EXACT feature order used in training
    input_df = pd.DataFrame(
        [[temp, ph, do, cdc, turb]],
        columns=[
            "Temperature (°C)",
            "pH",
            "DO(mg/L)",
            "CDC(µs/cm)",
            "Turbidity(NTUs)"
        ]
    )

    # Scale
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]

    st.success("Prediction Complete")
    st.markdown(f"### **{prediction:,.2f} particles/L**")
