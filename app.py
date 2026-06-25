import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

st.title("🚗 Road Accident Severity Predictor")

@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 50000
    X = pd.DataFrame({
        'Temperature(F)': np.random.uniform(-20, 120, n),
        'Humidity(%)': np.random.uniform(0, 100, n),
        'Visibility(mi)': np.random.uniform(0, 10, n),
        'Wind_Speed(mph)': np.random.uniform(0, 100, n),
        'Pressure(in)': np.random.uniform(28, 32, n),
        'Hour': np.random.randint(0, 24, n),
        'Month': np.random.randint(1, 12, n),
        'Is_Night': np.random.randint(0, 2, n),
        'Is_Weekend': np.random.randint(0, 2, n),
    })
    y = np.random.choice([1,2,3,4], n, p=[0.09,0.80,0.08,0.03])
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

st.write("Enter the conditions below to predict accident severity")

temperature = st.slider("Temperature (F)", -20, 120, 70)
humidity = st.slider("Humidity (%)", 0, 100, 50)
visibility = st.slider("Visibility (mi)", 0, 10, 5)
wind_speed = st.slider("Wind Speed (mph)", 0, 100, 10)
pressure = st.slider("Pressure (in)", 28, 32, 30)
hour = st.slider("Hour of Day", 0, 23, 12)
month = st.slider("Month", 1, 12, 6)
is_night = st.selectbox("Is it Night?", [0, 1])
is_weekend = st.selectbox("Is it Weekend?", [0, 1])

if st.button("Predict Severity"):
    features = pd.DataFrame([[temperature, humidity, visibility,
                              wind_speed, pressure, hour,
                              month, is_night, is_weekend]],
                            columns=['Temperature(F)', 'Humidity(%)', 
                                    'Visibility(mi)', 'Wind_Speed(mph)',
                                    'Pressure(in)', 'Hour', 'Month', 
                                    'Is_Night', 'Is_Weekend'])
    prediction = model.predict(features)[0]
    labels = {1:"Minor 🟢", 2:"Moderate 🟡", 3:"Serious 🔴", 4:"Fatal ⚫"}
    st.success(f"Predicted Severity: {labels[prediction]}")