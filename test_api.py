"""
Test Client for House Price Prediction API
Demonstrates how to interact with the deployed API
"""
import requests
import json

# Configuration
API_URL = "http://localhost:8000"  # Change to your deployed URL
# Examples:
# API_URL = "https://your-app.up.railway.app"
# API_URL = "https://house-price-api.onrender.com"
# API_URL = "http://your-ec2-ip:8000"


def test_health_check():
    """Test the health check endpoint"""
    print("=" * 60)
    print("Testing Health Check Endpoint")
    print("=" * 60)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_single_prediction():
    """Test single house price prediction"""
    print("=" * 60)
    print("Testing Single Prediction")
    print("=" * 60)
    
    # Sample house data
    house_data = {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 25,
        "total_rooms": 880,
        "total_bedrooms": 129,
        "population": 322,
        "households": 126,
        "median_income": 8.3252
    }
    
    print(f"Input Data: {json.dumps(house_data, indent=2)}")
    
    response = requests.post(
        f"{API_URL}/predict",
        json=house_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nPrediction Result:")
        print(f"  Predicted Price: ${result['predicted_price']:,.2f}")
        print(f"  Price per Room: ${result['price_per_room']:,.2f}")
        print(f"  Price per Bedroom: ${result['price_per_bedroom']:,.2f}")
        print(f"  Price per Household Member: ${result['price_per_household_member']:,.2f}")
    else:
        print(f"Error: {response.text}")
    
    print()


def test_batch_prediction():
    """Test batch predictions"""
    print("=" * 60)
    print("Testing Batch Prediction")
    print("=" * 60)
    
    # Multiple houses
    houses = [
        {
            "longitude": -122.23,
            "latitude": 37.88,
            "housing_median_age": 25,
            "total_rooms": 880,
            "total_bedrooms": 129,
            "population": 322,
            "households": 126,
            "median_income": 8.3252
        },
        {
            "longitude": -122.45,
            "latitude": 37.75,
            "housing_median_age": 35,
            "total_rooms": 1200,
            "total_bedrooms": 180,
            "population": 450,
            "households": 150,
            "median_income": 6.5
        },
        {
            "longitude": -122.10,
            "latitude": 37.95,
            "housing_median_age": 15,
            "total_rooms": 2500,
            "total_bedrooms": 350,
            "population": 800,
            "households": 300,
            "median_income": 12.5
        }
    ]
    
    print(f"Predicting prices for {len(houses)} houses...")
    
    response = requests.post(
        f"{API_URL}/predict/batch",
        json=houses,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nBatch Predictions ({result['count']} houses):")
        for i, prediction in enumerate(result['predictions'], 1):
            print(f"\n  House {i}:")
            print(f"    Predicted Price: ${prediction['predicted_price']:,.2f}")
            print(f"    Price per Room: ${prediction['price_per_room']:,.2f}")
            print(f"    Price per Bedroom: ${prediction['price_per_bedroom']:,.2f}")
    else:
        print(f"Error: {response.text}")
    
    print()


def test_model_info():
    """Test model information endpoint"""
    print("=" * 60)
    print("Testing Model Info Endpoint")
    print("=" * 60)
    
    response = requests.get(f"{API_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def run_all_tests():
    """Run all tests"""
    try:
        test_health_check()
        test_single_prediction()
        test_batch_prediction()
        test_model_info()
        
        print("=" * 60)
        print("All Tests Completed Successfully! ✅")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Error: Could not connect to API at {API_URL}")
        print("Please make sure the API is running and the URL is correct.")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    print("\n🏠 House Price Prediction API - Test Client\n")
    run_all_tests()
