from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the trained model and scaler
model = joblib.load("../models/house_price_model.pkl") 
scaler = joblib.load("../models/scaler.pkl")

# Initialize the label encoder for the 'neighborhood' feature
neighborhood_encoder = LabelEncoder()
neighborhood_encoder.fit(["Downtown", "Suburb", "CityCenter"])  # Add all possible neighborhood labels here

app = FastAPI()

# Define the request model
class HouseRequest(BaseModel):
    square_feet: float
    bedrooms: int
    bathrooms: int
    year_built: int
    neighborhood: str
    bankvaluation: float

@app.get("/")
async def read_root():
    return {"message": "Welcome to the house price prediction API!"}

@app.post("/predict")
async def predict_price(request: HouseRequest):
    # Convert the input data to the format expected by the model
    neighborhood_encoded = neighborhood_encoder.transform([request.neighborhood])[0]  # Encode 'neighborhood'
    
    input_data = np.array([
        request.square_feet,
        request.bedrooms,
        request.bathrooms,
        request.year_built,
        neighborhood_encoded,
        request.bankvaluation
    ]).reshape(1, -1)
    # Normalize the input data using the same scaler
    input_data = scaler.transform(input_data)
    
    # Get the prediction
    prediction = model.predict(input_data)
    
    return {"predicted_price": prediction[0]}
