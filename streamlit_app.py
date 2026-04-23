import streamlit as st
import pandas as pd
import joblib

# load saved files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("❤️ Heart Disease Prediction App")

st.write("Enter patient data:")

# 🔹 inputs (IMPORTANT: match your dataset columns)
age = st.number_input("Age")
sex = st.selectbox("Sex", ["Male", "Female"])
bp = st.number_input("Resting BP")
chol = st.number_input("Cholesterol")
hr = st.number_input("Max Heart Rate")
oldpeak = st.number_input("Oldpeak")

# convert sex
sex = 1 if sex == "Male" else 0

if st.button("Predict"):

    # create dataframe in SAME column order
    new_data = pd.DataFrame([[age, sex, bp, chol, hr, oldpeak]], columns=columns)

    # scale
    new_data_scaled = scaler.transform(new_data)

    # predict
    prediction = model.predict(new_data_scaled)

    if prediction[0] == 1:
        st.error("⚠️ High risk of heart disease")
    else:
        st.success("✅ Low risk")
