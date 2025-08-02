# 🫀 Heart Disease Prediction using Machine Learning

A machine learning project that predicts whether a person is at risk of heart disease based on key health parameters. Built using Scikit-learn, Streamlit, and XGBoost with interpretability and deployment in mind.

![Heart Disease](https://img.shields.io/badge/healthcare-ML--powered-red) ![status](https://img.shields.io/badge/status-deployed-blue)

---

## 🚀 Features

- 🧠 Machine Learning ( RandomForestClassifier )
- 📊 SHAP for model explainability
- 💉 Clean and structured healthcare dataset
- 🧪 Hyperparameter tuning
- 🌐 Streamlit Web App for real-time predictions
- 💾 Joblib model persistence
- 🔬 Ready for deployment

---

## 📊 Dataset

- **Source**: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- **Target**: `target` (0 = no disease, 1 = disease)
- **Records**: 303 patients
- **Features**:
  - age, sex, chest pain type (cp), resting blood pressure (trestbps), cholesterol (chol)
  - fasting blood sugar (fbs), resting ECG (restecg), max heart rate (thalach)
  - exercise-induced angina (exang), ST depression (oldpeak), slope, number of vessels (ca), thalassemia (thal)

---

## 🧠 Model Pipeline

- `StandardScaler` to normalize inputs
- `RandomForestClassifier` for classification
- Pipeline saved using `joblib`

---

## 👧 Author

Made with ❤️ by Vertika Singh
B.Tech CSE (AIML) | JSS Academy of Technical Education, Noida
