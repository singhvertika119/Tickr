import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
import json

# Load and preprocess data
df = pd.read_csv("heart-disease.csv")
df.rename(columns={"condition": "target"}, inplace=True)
x = pd.get_dummies(df.drop("target", axis=1))
y = df["target"]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Train best RF model
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
rf.fit(x_train, y_train)

# Save the model
joblib.dump(rf, "heart_model_pipeline.pkl")

# Load the model back
loaded_model = joblib.load("heart_model_pipeline.pkl")

# read output
data = json.loads(sys.argv[1])
features = data["features"]

# Make prediction with the loaded model
y_pred = loaded_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

joblib.dump(list(x_train.columns), 'heart_columns.pkl')

# make prediction 
prediction = loaded_model.predict([features]).tolist()

print(json.dumps({"prediction": int(prediction[0])}))

