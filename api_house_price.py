"""
FastAPI Web API for House Price Prediction
Provides REST endpoints for house price predictions with Swagger documentation
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import joblib
import tensorflow as tf
import uvicorn
from typing import Optional
import os

# Initialize FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    description="ML-powered API for predicting California house prices using a Neural Network model",
    version="1.0.0",
    docs_url="/",  # Swagger UI at root
    redoc_url="/redoc"
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
MODEL_PATH = os.getenv("MODEL_PATH", "House Price Prediction/house_price_model.h5")
SCALER_PATH = os.getenv("SCALER_PATH", "House Price Prediction/house_price_scaler.pkl")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None


# Request/Response models
class HouseFeatures(BaseModel):
    """Input features for house price prediction"""
    longitude: float = Field(..., example=-122.23, description="Longitude coordinate")
    latitude: float = Field(..., example=37.88, description="Latitude coordinate")
    housing_median_age: float = Field(..., ge=1, le=52, example=25, description="Median age of houses in the block")
    total_rooms: int = Field(..., ge=1, example=880, description="Total number of rooms")
    total_bedrooms: int = Field(..., ge=1, example=129, description="Total number of bedrooms")
    population: int = Field(..., ge=1, example=322, description="Population in the block")
    households: int = Field(..., ge=1, example=126, description="Number of households")
    median_income: float = Field(..., ge=0, example=8.3252, description="Median income (in 10,000s)")

    class Config:
        json_schema_extra = {
            "example": {
                "longitude": -122.23,
                "latitude": 37.88,
                "housing_median_age": 25,
                "total_rooms": 880,
                "total_bedrooms": 129,
                "population": 322,
                "households": 126,
                "median_income": 8.3252
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predicted_price: float = Field(..., description="Predicted house price in USD")
    price_per_room: float = Field(..., description="Price per room")
    price_per_bedroom: float = Field(..., description="Price per bedroom")
    price_per_household_member: float = Field(..., description="Price per household member")
    status: str = Field(default="success", description="Status of the prediction")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    version: str


# API Endpoints
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint to verify API is running
    """
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_house_price(features: HouseFeatures):
    """
    Predict house price based on input features
    
    - **longitude**: Geographic longitude coordinate
    - **latitude**: Geographic latitude coordinate  
    - **housing_median_age**: Median age of houses in block
    - **total_rooms**: Total number of rooms
    - **total_bedrooms**: Total number of bedrooms
    - **population**: Population in the block
    - **households**: Number of households
    - **median_income**: Median income (in 10,000s USD)
    
    Returns predicted house price and additional metrics.
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server configuration.")
    
    try:
        # Prepare input data
        input_data = np.array([[
            features.longitude,
            features.latitude,
            features.housing_median_age,
            features.total_rooms,
            features.total_bedrooms,
            features.population,
            features.households,
            features.median_income
        ]])
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = float(model.predict(input_scaled)[0][0])
        
        # Calculate additional metrics
        price_per_room = prediction / features.total_rooms
        price_per_bedroom = prediction / features.total_bedrooms
        price_per_household_member = prediction / features.households
        
        return {
            "predicted_price": round(prediction, 2),
            "price_per_room": round(price_per_room, 2),
            "price_per_bedroom": round(price_per_bedroom, 2),
            "price_per_household_member": round(price_per_household_member, 2),
            "status": "success"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(houses: list[HouseFeatures]):
    """
    Predict prices for multiple houses in batch
    
    Accepts a list of house features and returns predictions for all.
    Maximum 100 houses per request.
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    if len(houses) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 houses per batch request")
    
    try:
        predictions = []
        for house in houses:
            input_data = np.array([[
                house.longitude, house.latitude, house.housing_median_age,
                house.total_rooms, house.total_bedrooms, house.population,
                house.households, house.median_income
            ]])
            
            input_scaled = scaler.transform(input_data)
            prediction = float(model.predict(input_scaled)[0][0])
            
            predictions.append({
                "predicted_price": round(prediction, 2),
                "price_per_room": round(prediction / house.total_rooms, 2),
                "price_per_bedroom": round(prediction / house.total_bedrooms, 2)
            })
        
        return {
            "predictions": predictions,
            "count": len(predictions),
            "status": "success"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info", tags=["Model"])
async def model_info():
    """
    Get information about the loaded model
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "Neural Network (TensorFlow/Keras)",
        "input_features": 8,
        "feature_names": [
            "longitude", "latitude", "housing_median_age", "total_rooms",
            "total_bedrooms", "population", "households", "median_income"
        ],
        "output": "median_house_value (USD)",
        "preprocessing": "StandardScaler normalization",
        "status": "loaded"
    }


# Run the application
if __name__ == "__main__":
    # Train model if it doesn't exist
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("⚠️  Model files not found. Training model...")
        import subprocess
        subprocess.run(["python", "train_models.py"])
    
    # Start server
    uvicorn.run(
        "api_house_price:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
