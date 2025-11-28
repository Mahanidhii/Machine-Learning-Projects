# Streamlit App - ML Models Showcase

This Streamlit application showcases two machine learning models:
1. **House Price Prediction** - Neural Network model
2. **Heart Disease Prediction** - AdaBoost Classifier

## 📋 Prerequisites

Before running the Streamlit app, you **MUST** train and save the models first:

### Step 1: Train and Save Models

1. **House Price Prediction Model:**
   - Open `House Price Prediction/House Price Prediction.ipynb`
   - Run all cells in the notebook
   - Make sure the new cell that saves the model is executed (it will save `house_price_model.h5` and `house_price_scaler.pkl`)

2. **Heart Disease Prediction Model:**
   - Open `Heart Disease Predction/Heart Disease Prediction.ipynb`
   - Run all cells in the notebook
   - Make sure the new cell that saves the model is executed (it will save `heart_disease_model.pkl`)

### Step 2: Install Streamlit

If you haven't installed streamlit yet, run:
```bash
pip install streamlit
```

Or install all dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Running the App

Once both models are trained and saved, run the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 🎯 Using the App

### House Price Prediction
- Select "🏠 House Price Prediction" from the sidebar
- Enter house details (longitude, latitude, age, rooms, etc.)
- Click "🔮 Predict House Price"
- View the predicted price and additional metrics

### Heart Disease Prediction
- Select "❤️ Heart Disease Prediction" from the sidebar
- Enter patient details (age, sex, blood pressure, cholesterol, etc.)
- Click "🔮 Predict Heart Disease"
- View the prediction result and probability scores

## 📁 File Structure

```
Machine-Learning-Projects/
├── app.py                                    # Streamlit app
├── House Price Prediction/
│   ├── House Price Prediction.ipynb         # Training notebook
│   ├── house_price_model.h5                 # Saved model (generated)
│   └── house_price_scaler.pkl               # Saved scaler (generated)
├── Heart Disease Predction/
│   ├── Heart Disease Prediction.ipynb       # Training notebook
│   └── heart_disease_model.pkl              # Saved model (generated)
└── requirements.txt                          # Dependencies
```

## ⚠️ Troubleshooting

**Error: "Model not found"**
- Make sure you've run the training notebooks completely
- Verify that the model files are created in the correct directories

**Error: "Module not found"**
- Install missing dependencies: `pip install -r requirements.txt`

## 🎨 Features

- Interactive UI with sidebar navigation
- Real-time predictions
- Probability scores for heart disease prediction
- Additional metrics and insights
- Responsive design with columns layout
