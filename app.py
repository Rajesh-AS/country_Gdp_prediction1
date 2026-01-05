import streamlit as st
import numpy as np
import joblib
import os

# Page config
st.set_page_config(page_title="Country GDP Predictor", page_icon="🌍")

st.title("🌍 Country GDP Prediction App")
st.write("Predict GDP per capita based on socio-economic indicators")

# Load model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "country_gdpp_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Input fields
child_mort = st.number_input("Child Mortality", value=20.0)
exports = st.number_input("Exports (%)", value=40.0)
health = st.number_input("Health Spending (%)", value=7.0)
imports = st.number_input("Imports (%)", value=45.0)
income = st.number_input("Income", value=12000.0)
inflation = st.number_input("Inflation (%)", value=4.0)
life_expec = st.number_input("Life Expectancy", value=72.0)
total_fer = st.number_input("Total Fertility", value=2.1)

if st.button("Predict GDP"):
    input_data = np.array([[child_mort, exports, health, imports,
                            income, inflation, life_expec, total_fer]])
    
    prediction = model.predict(input_data)[0]

    # ✅ Prevent negative GDP
    prediction = max(0, prediction)

    st.success(f"💰 Predicted GDP per capita: {prediction:.2f}")
