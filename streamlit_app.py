import streamlit as st
import pandas as pd
import joblib


model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# ======================
# UI TITLE & STYLING
# ======================
st.title("❤️ Heart Disease Prediction ")
st.write("select the patient's clinical parameters to predict heart disease risk.")

gender_map = {0: "Female", 1: "Male"}

chestpain_map = {
    0:"type 0: typical angina",
    1:"type 1: atypical angina",
    2:"type 2: non-anginal pain",
    3:"type 3: asymptomatic)"   
            }

fastingbloodsugar_map = {0:"False", 1:"True"}

restingrelectro_map = {
    0:"type 0: normal",
    1:"type 1: having ST-T wave abnormality (T wave inversions",
    2:"type 2: showing probable or definite left ventricular",
}

exerciseangia_map = {
    0: "no",
    1 :"yes"
}

slope_map = {
    1: "type 1: upsloping",
    2: "type 2: flat",
    3: "type 3: downsloping"
}

noofmajorvessels_map = {
    0: "0",
    1: "1",
    2: "2",
    3: "3"
}

# ======================
# INPUTS
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
    format_func=lambda x: fastingbloodsugar_map[x]
)

restingrelectro = st.selectbox(
    "Resting electrocardiogram results",
    options=[0, 1, 2],
    format_func=lambda x: restingrelectro_map[x]
)

maxheartrate = st.number_input("Maximum heart rate")

exerciseangia = st.selectbox(
    "Exercise induced angina ",
    options=[0, 1],
    format_func=lambda x: exerciseangia_map[x]
)

oldpeak = st.number_input("Oldpeak")

slope = st.selectbox(
    "Slope of the peak exercise ST segment ",
    options=[1, 2,3],
    format_func=lambda x: slope_map[x]
)

noofmajorvessels = st.selectbox(
    "Number of major vessels ",
    options=[0, 1, 2, 3],
    format_func=lambda x: noofmajorvessels_map[x]
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
