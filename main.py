# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib

# Step 2: Load dataset
df = pd.read_csv("dataset.csv")

# Step 3: Show first rows
print("Dataset Head:")
print(df.head())

# Step 4: Check shape
print("\nDataset Shape:", df.shape)

# Step 5: Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Step 6: Define feature columns and target column
X = df.drop("Predicted Score", axis=1)
y = df["Predicted Score"]

# Step 7: Identify categorical and numerical columns
categorical_cols = ["Home/Away", "Pitch Condition", "Weather"]
numerical_cols = ["Match ID", "Overs Played", "Wickets Lost", "Run Rate", "Opponent Strength"]

# Step 8: Preprocessing (One-hot encoding categorical columns)
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

# Step 9: Split dataset into training & testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 10: Build ML Model
model = RandomForestRegressor(n_estimators=200, random_state=42)

# Step 11: Create pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# Step 12: Train the model
pipeline.fit(X_train, y_train)

# Step 13: Predict values
y_pred = pipeline.predict(X_test)

# Step 14: Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Step 15: Save model
joblib.dump(pipeline, "models/score_prediction_model.pkl")

print("\nModel saved successfully in models/score_prediction_model.pkl")

# Step 16: Visualization (Actual vs Predicted)
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Actual vs Predicted Cricket Score")
plt.savefig("outputs/actual_vs_predicted.png")
plt.show()

print("\nGraph saved in outputs/actual_vs_predicted.png")
