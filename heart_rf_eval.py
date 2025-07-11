import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("heart-disease.csv")
df.rename(columns={"condition": "target"}, inplace=True)
x = pd.get_dummies(df.drop('target', axis=1))
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=42, test_size=0.2, stratify=y)

# Tuning
best_params = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 1
}

# Train the model with best parameters
rf = RandomForestClassifier(**best_params, random_state=42)
rf.fit(x_train, y_train)

# Predictions and Probabalities
y_pred = rf.predict(x_test)
y_prob = rf.predict_proba(x_test)[:, 1]

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\n--- Confusion Matrix ---")
print(cm)

# Classification report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# ROC-AOC Score
roc_aoc = roc_auc_score(y_test, y_prob)
print(f"ROC-AOC Score: {roc_aoc:.4f}")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_aoc:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Heart Disease Prediction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
