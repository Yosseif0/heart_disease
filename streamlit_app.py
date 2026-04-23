import streamlit as st
import pandas as pd
import joblib

# ======================
# LOAD MODEL FILES
# ======================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# ======================
# UI TITLE
# ======================
st.title("❤️ Heart Disease Prediction App")

st.write("Fill in the patient details below:")

# ======================
# MAPPINGS (FOR HUMAN UI)
# ======================
binary_map = {
    0: "No",
    1: "Yes"
}

gender_map = {
    0: "Female",
    1: "Male"
}

chestpain_map = {
    0: "Type 0 (Typical Angina)",
    1: "Type 1 (Atypical Angina)",
    2: "Type 2 (Non-anginal Pain)",
    3: "Type 3 (Asymptomatic)"
}

slope_map = {
    0: "Upsloping",
    1: "Flat",
    2: "Downsloping",
    3: "Severe"
}

vessels_map = {
    0: "0 vessels",
    1: "1 vessel",
    2: "2 vessels",
    3: "3 vessels"
}

# ======================
# INPUT FIELDS
# ======================

age = st.number_input("Age", 1, 120)

gender = st.selectbox(
    "Gender",
    options=[0, 1],
    format_func=lambda x: gender_map[x]
)

chestpain = st.selectbox(
    "Chest Pain Type",
    options=[0, 1, 2, 3],
    format_func=lambda x: chestpain_map[x]
)

restingBP = st.number_input("Resting Blood Pressure")

serumcholestrol = st.number_input("Serum Cholesterol")

fastingbloodsugar = st.selectbox(
    "Fasting Blood Sugar",
    options=[0, 1],
    format_func=lambda x: binary_map[x]
)

restingecg = st.selectbox(
    "Resting ECG",
    options=[0, 1],
    format_func=lambda x: binary_map[x]
)

maxheartrate = st.number_input("Max Heart Rate")

exerciseangia = st.selectbox(
    "Exercise Angina",
    options=[0, 1],
    format_func=lambda x: binary_map[x]
)

oldpeak = st.number_input("Oldpeak")

slope = st.selectbox(
    "Slope",
    options=[0, 1, 2, 3],
    format_func=lambda x: slope_map[x]
)

noofvessels = st.selectbox(
    "No of Major Vessels",
    options=[0, 1, 2, 3],
    format_func=lambda x: vessels_map[x]
)

# ======================
# PREDICTION
# ======================

if st.button("Predict"):

    # IMPORTANT: MUST MATCH training column order
    new_data = pd.DataFrame([[
        age,
        gender,
        chestpain,
        restingBP,
        serumcholestrol,
        fastingbloodsugar,
        restingecg,
        maxheartrate,
        exerciseangia,
        oldpeak,
        slope,
        noofvessels
    ]], columns=columns)

    # scale input
    new_data_scaled = scaler.transform(new_data)

    # predict
    prediction = model.predict(new_data_scaled)

    # output
    if prediction[0] == 1:
        st.error("⚠️ High risk of heart disease")
    else:
        st.success("✅ Low risk of heart disease")
