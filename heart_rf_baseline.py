import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('heart-disease.csv')

df.rename(columns={"condition": "target"}, inplace=True)

x = df.drop('target', axis=1)
y = df['target']

# Handle categorical columns
x = pd.get_dummies(x)

# Split into training amd test data
x_train, x_test, y_train, y_test = train_test_split(
    # stratify maitains class balance
    x, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)

# Predit on test data
y_pred = rf.predict(x_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
