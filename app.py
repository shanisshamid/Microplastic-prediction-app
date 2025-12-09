# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. Styling Configuration (NEW) ---

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

set_background(RAW_LINK) # Apply the background image
st.set_page_config(layout="wide") 


# --- 2. Load the Champion Model and Scaler (ORIGINAL WORKING CODE) ---
@st.cache_resource 
def load_assets():
    try:
        model = joblib.load('xgb_champion_model.joblib')
        scaler = joblib.load('scaler_for_prediction.joblib')
        # Placeholder Feature names (Used for reference only, prediction uses order)
        feature_names = ['Feature_A', 'Feature_B', 'Feature_C', 'pH', 'CDC(µs/cm)', 'Turbidity(NTUs)', 'DO(mg/L)', 'Temperature (°C)']
        return model, scaler, feature_names
    except FileNotFoundError:
        st.error("Model or Scaler files not found. Please ensure 'xgb_champion_model.joblib' and 'scaler_for_prediction.joblib' are in the directory.")
        return None, None, None

model, scaler, feature_names = load_assets()

if model is not None:
    
    st.title("💧 Microplastic Concentration Predictor for Penang River")
    st.markdown("Enter sensor readings to get a prediction from the **Reliable XGBoost Champion Model**.")

    # --- 3. Input Form (ORIGINAL WORKING CODE STRUCTURE) ---
    with st.form("prediction_form"):
        st.header("Key Sensor Inputs")
        
        # 1. Temperature (°C)
        temp = st.number_input('Temperature (°C)', min_value=0.0, max_value=100.0, value=25.0, help="Lowest importance feature.")
        # 2. pH
        ph = st.number_input('pH', min_value=0.0, max_value=14.0, value=7.5, step=0.1)
        # 3. DO (mg/L)
        do = st.number_input('DO (mg/L)', min_value=0.0, max_value=100.0, value=8.0, step=0.1)
        # 4. CDC (µs/cm)
        cdc = st.number_input('CDC (µs/cm) - Conductivity', min_value=0.0, max_value=100000.0, value=500.0, help="This is the most critical feature (77% importance).")
        # 5. Turbidity (NTUs)
        turbidity = st.number_input('Turbidity (NTUs)', min_value=0.0, max_value=10000.0, value=10.0)
        
        # Form submission button
        submitted = st.form_submit_button("Predict Microplastic Concentration")

    # --- 4. Prediction Logic (ORIGINAL WORKING CODE) ---
    if submitted:
        # Create a single row DataFrame from user input (ORDER IS CRUCIAL!)
        # Order of values must match the order of columns provided to the DataFrame!
        
        # NOTE: Your input list order is [temp, ph, do, cdc, turbidity]
        # Your input column list is ['Temperature (°C)', 'pH', 'DO(mg/L)', 'CDC(µs/cm)', 'Turbidity(NTUs)']
        
        user_input_values = [temp, ph, do, cdc, turbidity] 
        input_column_names = ['Temperature (°C)', 'pH', 'DO(mg/L)', 'CDC(µs/cm)', 'Turbidity(NTUs)']
        
        # Assuming your model was trained on the order: Temp, pH, DO, CDC, Turbidity (as per your current code)
        input_data = pd.DataFrame([user_input_values], columns=input_column_names) 
        
        # Scale the input data using the saved scaler
        scaled_input = scaler.transform(input_data)
        
        # Generate prediction
        prediction = model.predict(scaled_input)[0]
        
        # Display Result
        st.success("Prediction Complete! The estimated Microplastic concentration is:")
        st.markdown(f"## **{prediction:,.2f} Particles/L**")
        st.caption("Prediction is based on the robust XGBoost model.")