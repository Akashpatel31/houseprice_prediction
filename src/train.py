import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from preprocess import load_and_preprocess_data

# Ensure the models directory exists
models_dir = "../models"
os.makedirs(models_dir, exist_ok=True)

# Load and preprocess data
X, y, scaler, label_encoder = load_and_preprocess_data("../data/house_prices.csv")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")

# Save the model and encoders
joblib.dump(model, f"{models_dir}/house_price_model.pkl")
joblib.dump(scaler, f"{models_dir}/scaler.pkl")
joblib.dump(label_encoder, f"{models_dir}/label_encoder.pkl")
print("Model saved successfully!")
