import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

# Load dataset
df = pd.read_csv("country_data.csv")

print("Dataset Columns:", df.columns)

# Drop non-numeric column
df_numeric = df.drop(columns=['country'])

# Features and target
X = df_numeric.drop(columns=['gdpp'])
y = df_numeric['gdpp']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/country_gdpp_model.pkl")

print("✅ Model trained successfully and saved!")
