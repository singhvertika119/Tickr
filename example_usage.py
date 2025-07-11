import joblib
import pandas as pd

# Load the model
model = joblib.load("heart_model_rf.pkl")

print("\nEnter patient details:")

data = {
    "age": int(input("Age: ")),
    "sex": int(input("Sex (1=Male, 0=Female): ")),
    "cp": int(input("Chest Pain Type (0–3): ")),
    "trestbps": int(input("Resting Blood Pressure: ")),
    "chol": int(input("Serum Cholesterol (mg/dl): ")),
    "fbs": int(input("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False): ")),
    "restecg": int(input("Resting ECG (0–2): ")),
    "thalach": int(input("Max Heart Rate Achieved: ")),
    "exang": int(input("Exercise Induced Angina (1=Yes, 0=No): ")),
    "oldpeak": float(input("Oldpeak (ST depression): ")),
    "slope": int(input("Slope of ST segment (0–2): ")),
    "ca": int(input("No. of major vessels (0–3): ")),
    "thal": int(input("Thal (1=Normal, 2=Fixed defect, 3=Reversible defect): "))
}

# Convert input to DataFrame
input_df = pd.DataFrame([data])

# One-hot encode input to match model training
input_df = pd.get_dummies(input_df)

# Re-align columns to match training data (padding missing columns if needed)
model_columns = model.feature_names_in_
for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model_columns]

# Make prediction
pred = model.predict(input_df)[0]
probs = model.predict_proba(input_df)[0][1]

# Print result
print("\nPrediction:")
if pred == 1:
    print("Likely to have heart disease")

else:
    print("No sign of heart disease")

print("Probability of Heart Disease: {:.2f}%".format(probs * 100))
