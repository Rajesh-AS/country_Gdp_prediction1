import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Country GDP Predictor", page_icon="🌍")

st.title("🌍 Country GDP Prediction App")
st.write("Predict GDP per capita based on socio-economic indicators")

# ---------------------------
# Safe model loading
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "country_gdpp_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ---------------------------
# Inputs
# ---------------------------
child_mort = st.number_input("Child Mortality", 0.0)
exports = st.number_input("Exports (%)", 0.0)
health = st.number_input("Health Spending (%)", 0.0)
imports = st.number_input("Imports (%)", 0.0)
income = st.number_input("Income", 0.0)
inflation = st.number_input("Inflation (%)", 0.0)
life_expec = st.number_input("Life Expectancy", 0.0)
total_fer = st.number_input("Total Fertility", 0.0)

if st.button("Predict GDP"):
    input_data = np.array([[child_mort, exports, health, imports,
                            income, inflation, life_expec, total_fer]])
    prediction = model.predict(input_data)
    st.success(f"💰 Predicted GDP per capita: {prediction[0]:.2f}")
