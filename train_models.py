"""
Script to train and save ML models for deployment
This script trains both Heart Disease and House Price Prediction models
"""
import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def train_heart_disease_model():
    """Train and save Heart Disease Prediction model"""
    print("Training Heart Disease Prediction model...")
    
    # Load dataset
    df = pd.read_csv('Heart Disease Predction/Dataset/heart_cleveland_upload.csv')
    
    # Separate features and target
    X = df.drop('condition', axis=1)
    y = df['condition']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create and train AdaBoost model
    base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)
    model = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Heart Disease Model - Train Accuracy: {train_score:.4f}")
    print(f"Heart Disease Model - Test Accuracy: {test_score:.4f}")
    
    # Save model
    os.makedirs('Heart Disease Predction', exist_ok=True)
    joblib.dump(model, 'Heart Disease Predction/heart_disease_model.pkl')
    print("Heart Disease model saved successfully!")
    
    return model

def train_house_price_model():
    """Train and save House Price Prediction model"""
    print("\nTraining House Price Prediction model...")
    
    # Load dataset
    df = pd.read_csv('House Price Prediction/datasets/housing.csv')
    
    # Handle missing values
    df = df.dropna()
    
    # Select features (excluding ocean_proximity if it exists)
    feature_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
                      'total_bedrooms', 'population', 'households', 'median_income']
    
    X = df[feature_columns]
    y = df['median_house_value']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build neural network
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Train model
    print("Training neural network... (this may take a few minutes)")
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate
    train_loss, train_mae = model.evaluate(X_train_scaled, y_train, verbose=0)
    test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    
    print(f"\nHouse Price Model - Train MAE: ${train_mae:,.2f}")
    print(f"House Price Model - Test MAE: ${test_mae:,.2f}")
    
    # Save model and scaler
    os.makedirs('House Price Prediction', exist_ok=True)
    model.save('House Price Prediction/house_price_model.h5')
    joblib.dump(scaler, 'House Price Prediction/house_price_scaler.pkl')
    print("House Price model and scaler saved successfully!")
    
    return model, scaler

def main():
    """Main function to train all models"""
    print("=" * 60)
    print("ML Models Training Script")
    print("=" * 60)
    
    try:
        # Check if models already exist
        heart_model_exists = os.path.exists('Heart Disease Predction/heart_disease_model.pkl')
        house_model_exists = os.path.exists('House Price Prediction/house_price_model.h5')
        house_scaler_exists = os.path.exists('House Price Prediction/house_price_scaler.pkl')
        
        if heart_model_exists and house_model_exists and house_scaler_exists:
            print("All models already exist. Skipping training.")
            return
        
        # Train Heart Disease model
        if not heart_model_exists:
            train_heart_disease_model()
        else:
            print("Heart Disease model already exists. Skipping...")
        
        # Train House Price model
        if not (house_model_exists and house_scaler_exists):
            train_house_price_model()
        else:
            print("House Price model already exists. Skipping...")
        
        print("\n" + "=" * 60)
        print("All models trained and saved successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during model training: {e}")
        raise

if __name__ == "__main__":
    main()
