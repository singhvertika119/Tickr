import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load and preprocess data
df = pd.read_csv("heart-disease.csv")
df.rename(columns={"condition": "target"}, inplace=True)
x = pd.get_dummies(df.drop("target", axis=1))
y = df["target"]

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Train best Random Forest model
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
rf.fit(x_train, y_train)

# Get feature importance
importance = rf.feature_importances_
feature_names = x.columns

# Create a dataframe
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

# Show top 10 features
print("\n--- Top 10 Important Features ---")
print(importance_df.head(10))

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df.head(10),
            x='Importance', y='Feature', palette='mako')
plt.title("Top 10 Important Features in Heart Disease Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
