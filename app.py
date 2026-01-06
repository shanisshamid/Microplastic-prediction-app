import streamlit as st
import joblib
import pandas as pd

st.title("Microplastic Concentration Predictor (Penang Rivers)")

# Load model and scaler
model = joblib.load("champion_gradientboost_model.pkl")
scaler = joblib.load("scaler_aug.pkl")

st.subheader("Water Quality Inputs")

temp = st.number_input("Temperature (°C)", value=28.0)
ph = st.number_input("pH", value=7.0)
do = st.number_input("DO (mg/L)", value=6.5)
cdc = st.number_input("CDC (µS/cm)", value=500.0)
turb = st.number_input("Turbidity (NTU)", value=10.0)

if st.button("Predict"):
    X = pd.DataFrame(
        [[temp, ph, do, cdc, turb]],
        columns=[
            "Temperature (°C)",
            "pH",
            "DO(mg/L)",
            "CDC(µs/cm)",
            "Turbidity(NTUs)"
        ]
    )

    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]

    st.success(f"Predicted Microplastic Concentration: {pred:.2f} particles/L")
