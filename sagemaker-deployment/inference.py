"""
SageMaker Inference Script for Heart Disease Prediction
This script handles predictions for the deployed model
"""
import joblib
import json
import numpy as np
import os

# Global variable to hold the model
model = None


def model_fn(model_dir):
    """
    Load the model for inference
    Called once when the endpoint is created
    """
    global model
    model_path = os.path.join(model_dir, 'heart_disease_model.pkl')
    model = joblib.load(model_path)
    return model


def input_fn(request_body, content_type='application/json'):
    """
    Deserialize and prepare the prediction input
    """
    if content_type == 'application/json':
        input_data = json.loads(request_body)
        
        # Expected features in order
        feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        
        # Convert to numpy array in correct order
        if isinstance(input_data, dict):
            # Single prediction
            features = [input_data[feature] for feature in feature_names]
            return np.array([features])
        elif isinstance(input_data, list):
            # Batch prediction
            features = [[item[feature] for feature in feature_names] for item in input_data]
            return np.array(features)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """
    Make predictions using the loaded model
    """
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)
    
    return {
        'predictions': predictions.tolist(),
        'probabilities': probabilities.tolist()
    }


def output_fn(prediction, accept='application/json'):
    """
    Serialize the prediction output
    """
    if accept == 'application/json':
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
