import streamlit as st
import pandas as pd
import joblib

# ======================
# LOAD MODEL FILES
# ======================
# Ensure these files are in the same directory as your script
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# ======================
# UI TITLE & STYLING
# ======================
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")
st.title("❤️ Heart Disease Prediction App")
st.write("Please select the patient's clinical parameters below to predict heart disease risk.")

# ======================
# MAPPINGS (FOR HUMAN-FRIENDLY UI)
# ======================
gender_map = {0: "Female", 1: "Male"}

binary_map = {0: "No / False", 1: "Yes / True"}

chestpain_map = {
    0: "0: Typical Angina",
    1: "1: Atypical Angina",
    2: "2: Non-anginal Pain",
    3: "3: Asymptomatic"
}

slope_map = {
    0: "0: Upsloping",
    1: "1: Flat",
    2: "2: Downsloping",
    3: "3: Severe/Other"
}

vessels_map = {
    0: "0 Vessels",
    1: "1 Vessel",
    2: "2 Vessels",
    3: "3 Vessels"
}

# ======================
# INPUT FIELDS (Organized in Columns)
# ======================
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=40)
    
    gender = st.selectbox(
        "Gender",
        options=[1, 0],
        format_func=lambda x: gender_map[x]
    )
    
    chestpain = st.selectbox(
        "Chest Pain Type",
        options=[0, 1, 2, 3],
        format_func=lambda x: chestpain_map[x]
    )
    
    restingBP = st.number_input("Resting Blood Pressure (mm Hg)", value=120)
    
    serumcholestrol = st.number_input("Serum Cholestrol (mg/dl)", value=200)
    
    fastingbloodsugar = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        options=[0, 1],
        format_func=lambda x: binary_map[x]
    )

with col2:
    restingrelectro = st.selectbox(
        "Resting Electrocardiographic Results",
        options=[0, 1],
        format_func=lambda x: "Normal" if x == 0 else "ST-T Wave Abnormality"
    )
    
    maxheartrate = st.number_input("Maximum Heart Rate Achieved", value=150)
    
    exerciseangia = st.selectbox(
        "Exercise Induced Angina",
        options=[0, 1],
        format_func=lambda x: binary_map[x]
    )
    
    oldpeak = st.number_input("Oldpeak (ST depression)", value=0.0, format="%.1f")
    
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        options=[0, 1, 2, 3],
        format_func=lambda x: slope_map[x]
    )
    
    noofmajorvessels = st.selectbox(
        "Number of Major Vessels Colored by Flourosopy",
        options=[0, 1, 2, 3],
        format_func=lambda x: vessels_map[x]
    )

# ======================
# PREDICTION LOGIC
# ======================
st.write("---")
if st.button("Predict Heart Health Status"):

    # 1. Prepare raw data (order must match training columns)
    input_data = pd.DataFrame([[
        age, gender, chestpain, restingBP, serumcholestrol, 
        fastingbloodsugar, restingrelectro, maxheartrate, 
        exerciseangia, oldpeak, slope, noofmajorvessels
    ]], columns=columns)

    # 2. Scale the data
    input_scaled = scaler.transform(input_data)

    # 3. Predict
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled) # Optional: show probability

    # 4. Display Results (Target 0 = Healthy, Target 1 = Risk)
    if prediction[0] == 1:
        st.error(f"### Result: ⚠️ High Risk of Heart Disease")
        st.info(f"Confidence Level: {prediction_proba[0][1]*100:.2f}%")
    else:
        st.success(f"### Result: ✅ Low Risk of Heart Disease")
        st.info(f"Confidence Level: {prediction_proba[0][0]*100:.2f}%")

st.caption("Note: This app is for informational purposes and is not a substitute for professional medical advice.")
