# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. Background Image Function ---
# Raw link for the background image (Corrected link from previous steps)
RAW_LINK = "https://raw.githubusercontent.com/shanisshamid/Microplastic-prediction-app/main/river%20wallpaper.jpg"

def set_background(image_url):
    """
    Injects custom CSS to set the background of the main Streamlit app container.
    """
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({image_url});
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: black; /* Ensure text is visible over the image */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- CALL THE BACKGROUND FUNCTION ---
set_background(RAW_LINK)
# ------------------------------------

# --- 2. Load the Champion Model and Scaler ---
@st.cache_resource 
def load_assets():
    try:
        model = joblib.load('xgb_champion_model.joblib')
        scaler = joblib.load('scaler_for_prediction.joblib')
        
        # FINALIZED FEATURE LIST (MUST match the exact order your model was trained on!)
        # IMPORTANT: If your model was ONLY trained on these 5 features, use this list.
        final_feature_names = ['Temperature (°C)', 'pH', 'DO(mg/L)', 'CDC(µs/cm)', 'Turbidity(NTUs)']
        
        return model, scaler, final_feature_names
    except FileNotFoundError:
        st.error("Model or Scaler files not found. Please ensure 'xgb_champion_model.joblib' and 'scaler_for_prediction.joblib' are in the directory.")
        return None, None, None

model, scaler, feature_names = load_assets()

# --- 3. Streamlit Interface ---

if model is not None:
    
    st.title("💧 Microplastic Concentration Predictor for Penang River")
    st.markdown("Enter sensor readings to get a prediction from the **Reliable XGBoost Champion Model**.")

    # Input Form
    with st.form("prediction_form"):
        st.header("Key Sensor Inputs")
        
        # NOTE: The variable order here MUST match the 'final_feature_names' list above!
        
        # 1. Temperature (°C) - Low Importance
        temp = st.number_input('Temperature (°C)', min_value=0.0, max_value=50.0, value=25.0, help="Lowest importance feature.")
        # 2. pH - Medium Importance
        ph = st.number_input('pH', min_value=0.0, max_value=14.0, value=7.5, step=0.1)
        # 3. DO (mg/L) - Medium Importance
        do = st.number_input('DO (mg/L)', min_value=0.0, max_value=20.0, value=8.0, step=0.1)
        # 4. CDC (µs/cm) - HIGHEST IMPORTANCE (77%)
        cdc = st.number_input('CDC (µs/cm) - Conductivity', min_value=0.0, max_value=1500.0, value=500.0, help="This is the most critical feature (77% importance).")
        # 5. Turbidity (NTUs) - High Importance
        turbidity = st.number_input('Turbidity (NTUs)', min_value=0.0, max_value=100.0, value=10.0)
        
        submitted = st.form_submit_button("Predict Microplastic Concentration")

    # --- 4. Prediction Logic ---
    if submitted:
        # Create a single row DataFrame from user input (ORDER IS CRUCIAL!)
        # The list of values must be in the exact order of 'feature_names'
        user_input_values = [temp, ph, do, cdc, turbidity]
        
        # Convert to DataFrame using the correct feature names
        input_data = pd.DataFrame([user_input_values], columns=feature_names)
        
        # Scale the input data using the saved scaler
        scaled_input = scaler.transform(input_data)
        
        # Generate prediction
        prediction = model.predict(scaled_input)[0]
        
        # Display Result
        st.success("Prediction Complete! The estimated Microplastic concentration is:")
        st.markdown(f"## **{prediction:,.2f} Particles/L**")
        st.caption("Prediction is based on the robust XGBoost model.")