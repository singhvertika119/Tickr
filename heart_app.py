import streamlit as st
import requests

st.title("💓 Heart Disease Prediction App (via FastAPI)")

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

API_URL = "http://127.0.0.1:8000/predict/"

if st.button("🔍 Predict Heart Disease"):
    payload = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            st.subheader("🩺 Prediction Result:")
            st.success(data["risk"])
            st.subheader("📊 Risk Probability:")
            st.info(f"{data['risk_probability']} %")
        else:
            st.error(f"API Error: {response.text}")
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
