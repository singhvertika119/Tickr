from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Initialize FastAPI
app = FastAPI(title="Heart Disease Prediction API")

# Load model
model = joblib.load("heart_model_pipeline.pkl")
columns = joblib.load("heart_columns.pkl")

# Pydantic model for request validation
class PatientData(BaseModel):
    age: int
    sex: str  
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# Endpoint for prediction
@app.post("/predict/")
def predict_heart_disease(data: PatientData):
    # Convert sex to numeric
    sex_num = 1 if data.sex.lower() == "male" else 0

    # Prepare input dict
    input_data = {
        'age': data.age,
        'sex': sex_num,
        'cp': data.cp,
        'trestbps': data.trestbps,
        'chol': data.chol,
        'fbs': data.fbs,
        'restecg': data.restecg,
        'thalach': data.thalach,
        'exang': data.exang,
        'oldpeak': data.oldpeak,
        'slope': data.slope,
        'ca': data.ca,
        'thal': data.thal
    }

    # Fill missing columns (one-hot)
    for col in columns:
        if col not in input_data:
            input_data[col] = 0

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])[columns]

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    result = {
        "prediction": int(prediction),
        "risk_probability": round(probability * 100, 2),
        "risk": "High Risk 💔" if prediction == 1 else "Low Risk ✅"
    }
    return result
