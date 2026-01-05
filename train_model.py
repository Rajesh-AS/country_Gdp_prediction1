import pandas as pd
import joblib
import os
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("country_data.csv")

# Drop non-numeric column
df = df.drop(columns=["country"])

# Features and target
X = df.drop(columns=["gdpp"])
y = df["gdpp"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/country_gdpp_model.pkl")

print("✅ Model trained and saved successfully")
