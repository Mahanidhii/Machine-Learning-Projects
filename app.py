import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# Page configuration
st.set_page_config(
    page_title="ML Models Showcase",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("Machine Learning Models Showcase")
st.markdown("---")

# Sidebar for model selection
st.sidebar.title("Select Model")
model_choice = st.sidebar.radio(
    "Choose a prediction model:",
    ["🏠 House Price Prediction", "❤️ Heart Disease Prediction"]
)

# House Price Prediction
if model_choice == "House Price Prediction":
    st.header("House Price Prediction")
    st.markdown("Predict house prices based on various features using a Neural Network.")
    
    try:
        # Load model and scaler
        house_model = tf.keras.models.load_model('House Price Prediction/house_price_model.h5')
        house_scaler = joblib.load('House Price Prediction/house_price_scaler.pkl')
        
        st.success("Model loaded successfully!")
        
        # Create input form
        st.subheader("Enter House Details:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            longitude = st.number_input("Longitude", value=-122.23, format="%.2f")
            latitude = st.number_input("Latitude", value=37.88, format="%.2f")
            housing_median_age = st.slider("Housing Median Age", 1, 52, 25)
            total_rooms = st.number_input("Total Rooms", min_value=1, value=880, step=1)
            total_bedrooms = st.number_input("Total Bedrooms", min_value=1, value=129, step=1)
        
        with col2:
            population = st.number_input("Population", min_value=1, value=322, step=1)
            households = st.number_input("Households", min_value=1, value=126, step=1)
            median_income = st.number_input("Median Income (in 10,000s)", min_value=0.0, value=8.3252, format="%.4f")
        
        if st.button("Predict House Price", key="house_predict"):
            # Prepare input data
            input_data = np.array([[
                longitude, latitude, housing_median_age, total_rooms,
                total_bedrooms, population, households, median_income
            ]])
            
            # Scale the input
            input_scaled = house_scaler.transform(input_data)
            
            # Make prediction
            prediction = house_model.predict(input_scaled)[0][0]
            
            # Display result
            st.markdown("---")
            st.success(f"### Predicted House Price: ${prediction:,.2f}")
            
            # Additional insights
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Price per Room", f"${prediction/total_rooms:,.2f}")
            with col2:
                st.metric("Price per Bedroom", f"${prediction/total_bedrooms:,.2f}")
            with col3:
                st.metric("Price per Household Member", f"${prediction/households:,.2f}")
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please make sure you have run the House Price Prediction notebook and saved the model first.")

# Heart Disease Prediction
elif model_choice == "❤️ Heart Disease Prediction":
    st.header("❤️Heart Disease Prediction")
    st.markdown("Predict the presence of heart disease using an AdaBoost classifier.")
    
    try:
        # Load model
        heart_model = joblib.load('Heart Disease Predction/heart_disease_model.pkl')
        
        st.success("Model loaded successfully!")
        
        # Create input form
        st.subheader("Enter Patient Details:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Age", 20, 80, 50)
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3], 
                            format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 120)
            chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 400, 200)
        
        with col2:
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], 
                             format_func=lambda x: "No" if x == 0 else "Yes")
            restecg = st.selectbox("Resting ECG Results", [0, 1, 2],
                                 format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x])
            thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
            exang = st.selectbox("Exercise Induced Angina", [0, 1],
                               format_func=lambda x: "No" if x == 0 else "Yes")
        
        with col3:
            oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
            slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2],
                               format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
            ca = st.selectbox("Number of Major Vessels (0-3) Colored by Fluoroscopy", [0, 1, 2, 3])
            thal = st.selectbox("Thalassemia", [0, 1, 2, 3],
                              format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x])
        
        if st.button("🔮 Predict Heart Disease", key="heart_predict"):
            # Prepare input data
            input_data = pd.DataFrame({
                'age': [age],
                'sex': [sex],
                'cp': [cp],
                'trestbps': [trestbps],
                'chol': [chol],
                'fbs': [fbs],
                'restecg': [restecg],
                'thalach': [thalach],
                'exang': [exang],
                'oldpeak': [oldpeak],
                'slope': [slope],
                'ca': [ca],
                'thal': [thal]
            })
            
            # Make prediction
            prediction = heart_model.predict(input_data)[0]
            prediction_proba = heart_model.predict_proba(input_data)[0]
            
            # Display result
            st.markdown("---")
            
            if prediction == 1:
                st.error("### Heart Disease Detected")
                st.warning("The model predicts the presence of heart disease. Please consult with a healthcare professional.")
            else:
                st.success("### No Heart Disease Detected")
                st.info("The model predicts no heart disease. However, regular check-ups are always recommended.")
            
            # Show probabilities
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Probability of No Disease", f"{prediction_proba[0]*100:.2f}%")
            with col2:
                st.metric("Probability of Disease", f"{prediction_proba[1]*100:.2f}%")
            
            # Progress bar for visualization
            st.progress(prediction_proba[1])
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please make sure you have run the Heart Disease Prediction notebook and saved the model first.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit | Machine Learning Projects</p>
</div>
""", unsafe_allow_html=True)
