# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np # Needed for potential array handling

# --- 1. Load the Champion Model and Scaler ---
@st.cache_resource 
def load_assets():
    # Ensure these files are in the same directory as app.py
    try:
        model = joblib.load('xgb_champion_model.joblib')
        scaler = joblib.load('scaler_for_prediction.joblib')
        # The feature names must be in the exact order used during training (Importance order is ignored here)
        feature_names = ['Feature_A', 'Feature_B', 'Feature_C', 'pH', 'CDC(µs/cm)', 'Turbidity(NTUs)', 'DO(mg/L)', 'Temperature (°C)']
        return model, scaler, feature_names
    except FileNotFoundError:
        st.error("Model or Scaler files not found. Please ensure 'xgb_champion_model.joblib' and 'scaler_for_prediction.joblib' are in the directory.")
        return None, None, None

model, scaler, feature_names = load_assets()

if model is not None:
    
    st.title("💧 Microplastic Concentration Predictor for Penang River")
    st.markdown("Enter sensor readings to get a prediction from the **Reliable XGBoost Champion Model**.")

    # --- 2. Input Form (Use your 5 most important features for simplicity) ---
    with st.form("prediction_form"):
        st.header("Key Sensor Inputs")
        
        # Use inputs based on your actual feature names, ensuring the order matches your full list (feature_names) 
        # For simplicity, let's assume your original features were: Temp, pH, DO, CDC, Turbidity
        
        # NOTE: You MUST replace these placeholder feature names and input ranges
        # with your actual features and realistic min/max values.
        
        # 1. Temperature (°C) - Low Importance
        temp = st.number_input('Temperature (°C)', min_value=0.0, max_value=50.0, value=25.0, help="Lowest importance feature.")
        # 2. pH - Medium Importance
        ph = st.number_input('pH', min_value=0.0, max_value=14.0, value=7.5, step=0.1)
        # 3. DO (mg/L) - Medium Importance
        do = st.number_input('DO (mg/L)', min_value=0.0, max_value=20.0, value=8.0, step=0.1)
        # 4. CDC (µs/cm) - HIGHEST IMPORTANCE (77%)
        cdc = st.number_input('CDC (µs/cm) - Conductivity', min_value=10.0, max_value=1500.0, value=500.0, help="This is the most critical feature (77% importance).")
        # 5. Turbidity (NTUs) - High Importance
        turbidity = st.number_input('Turbidity (NTUs)', min_value=0.0, max_value=100.0, value=10.0)
        
        # Note: If you have other features ('Feature_A', 'Feature_B', etc.), you must add input boxes for them here.
        # Ensure the order of variables matches your training data's feature order!

        submitted = st.form_submit_button("Predict Microplastic Concentration")

    # --- 3. Prediction Logic ---
    if submitted:
        # Create a single row DataFrame from user input (ORDER IS CRUCIAL!)
        # Use only the values from the input fields in the exact order of your feature_names list
        user_input = np.array([[temp, ph, do, cdc, turbidity]]) 
        input_data = pd.DataFrame(user_input, columns=['Temperature (°C)', 'pH', 'DO(mg/L)', 'CDC(µs/cm)', 'Turbidity(NTUs)']) # Adjust columns as necessary
        
        # Scale the input data using the saved scaler
        scaled_input = scaler.transform(input_data)
        
        # Generate prediction
        prediction = model.predict(scaled_input)[0]
        
        # Display Result
        st.success("Prediction Complete! The estimated Microplastic concentration is:")
        st.markdown(f"## **{prediction:,.2f} Particles/L**")
        st.caption("Prediction is based on the robust XGBoost model.")