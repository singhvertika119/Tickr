# heart_app.py

import streamlit as st
import pandas as pd
import joblib

# Load model and column order
model = joblib.load("heart_model_pipeline.pkl")
columns = joblib.load("heart_columns.pkl")

# UI Title
st.title("💓 Heart Disease Prediction App")
st.write("Enter the patient's details to predict the risk of heart disease.")

# Sidebar Inputs
age = st.sidebar.slider("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex", ("Male", "Female"))
cp = st.sidebar.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 400, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.sidebar.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0)
slope = st.sidebar.selectbox("Slope of ST Segment", [0, 1, 2])
ca = st.sidebar.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.sidebar.selectbox("Thalassemia", [0, 1, 2, 3])

# Manual input dict
input_data = {
    'age': age,
    'sex': 1 if sex == 'Male' else 0,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal
}

# Fill any missing columns (if one-hot encoded)
for col in columns:
    if col not in input_data:
        input_data[col] = 0

# Convert to DataFrame and reorder
input_df = pd.DataFrame([input_data])[columns]

# Predict
if st.button("🔍 Predict Heart Disease"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("🩺 Prediction Result:")
    st.success("High Risk of Heart Disease 💔" if prediction == 1 else "Low Risk ✅")

    st.subheader("📊 Risk Probability:")
    st.info(f"{round(probability * 100, 2)} %")
