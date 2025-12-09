# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. Configuration & Styling (KEEPING THE BACKGROUND IMAGE) ---

# Raw link for the background image
RAW_LINK = "https://raw.githubusercontent.com/shanisshamid/Microplastic-prediction-app/main/river%20wallpaper.jpg"

def set_background(image_url):
    """Injects custom CSS to set the background image."""
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

set_background(RAW_LINK) # Apply the background immediately
st.set_page_config(layout="wide") 


# --- 2. Load Assets and Define Features ---

# Feature names MUST match the exact order your model was trained on!
FINAL_FEATURE_NAMES = ['Temperature (°C)', 'pH', 'DO(mg/L)', 'CDC(µs/cm)', 'Turbidity(NTUs)']

@st.cache_resource 
def load_assets():
    """Loads the trained model and scaler only once."""
    try:
        model = joblib.load('xgb_champion_model.joblib')
        scaler = joblib.load('scaler_for_prediction.joblib')
        return model, scaler
    except FileNotFoundError:
        st.error("Model or Scaler files not found. Ensure 'xgb_champion_model.joblib' and 'scaler_for_prediction.joblib' are in the directory.")
        return None, None

model, scaler = load_assets()

# --- 3. Streamlit Interface (REVERTED TO WORKING LAYOUT) ---

if model is not None:
    
    st.title("💧 Microplastic Concentration Predictor for Penang River")
    st.markdown("Enter sensor readings to get a prediction from the **Reliable XGBoost Champion Model**.")

    # --- Input Form (Using simple st.form structure) ---
    with st.form("prediction_form"):
        st.header("🔬 Key Sensor Inputs")
        
        # --- SIMPLE INPUTS (Vertical Stacked, the original working format) ---
        
        # 1. Temperature (°C) - Low Importance
        temp = st.number_input('Temperature (°C)', min_value=0.0, max_value=50.0, value=25.0, help="Lowest importance feature.")
        # 2. pH - Medium Importance
        ph = st.number_input('pH', min_value=0.0, max_value=14.0, value=7.5, step=0.1)
        # 3. DO (mg/L) - Medium Importance
        do = st.number_input('DO (mg/L)', min_value=0.0, max_value=20.0, value=8.0, step=0.1)
        # 4. CDC (µs/cm) - HIGHEST IMPORTANCE (77%)
        cdc = st.number_input('CDC (µs/cm) - Conductivity (Critical)', min_value=0.0, max_value=1500.0, value=500.0, help="This is the most critical feature (77% importance).")
        # 5. Turbidity (NTUs) - High Importance
        turbidity = st.number_input('Turbidity (NTUs)', min_value=0.0, max_value=100.0, value=10.0)
        
        st.markdown("---")
        # Use st.form_submit_button inside the form
        submitted = st.form_submit_button("🚀 Predict Microplastic Concentration", type="primary", use_container_width=True)


    # --- 4. Prediction Logic (Runs ONLY when submitted is True) ---
    if submitted:
        
        # 1. Collect inputs in the EXACT training order
        user_input_values = [
            temp,       # 1. Temperature (°C)
            ph,         # 2. pH
            do,         # 3. DO(mg/L)
            cdc,        # 4. CDC(µs/cm)
            turbidity   # 5. Turbidity(NTUs)
        ]
        
        # 2. Create DataFrame
        input_data = pd.DataFrame([user_input_values], columns=FINAL_FEATURE_NAMES)
        
        # 3. Scale the input data
        scaled_input = scaler.transform(input_data)
        
        # 4. Generate prediction
        prediction = model.predict(scaled_input)[0]

        # 5. Display the result
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.success("✅ Prediction Result:"):
            st.markdown(f"The estimated Microplastic concentration is:")
            st.markdown(f"## **{prediction:,.2f} Particles/L**")
            st.caption("Prediction is based on the robust XGBoost model (R²: 0.88).")