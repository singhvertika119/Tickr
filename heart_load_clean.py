import pandas as pd

# Load dataset
df = pd.read_csv('heart-disease.csv')

# Rename target column
df.rename(columns={"condition": "target"}, inplace=True)

# Display first few rows
print("First five rows of the dataset")
print(df.head())

# Dataset shape
print("Shape of dataset")
print(df.shape)

# Check for missing values
print("Missing valuees")
print(df.isnull().sum())

# Datatypes and basic information
print("Data info")
print(df.info())

# Target class distribution
print("\n--- Target Class Distribution ---")
print(df['target'].value_counts())
