import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureCreator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['AverageCloud'] = X[['Cloud9am', 'Cloud3pm']].mean(axis=1)
        X['AveragePressure'] = X[['Pressure9am', 'Pressure3pm']].mean(axis=1)

        final_features = [
            'Sunshine',
            'Humidity3pm',
            'AverageCloud',
            'Temp3pm',
            'Rainfall',
            'Evaporation',
            'AveragePressure',
            'Location',
            'WindDir9am',
            'RainToday',
        ]
        return X[final_features]

# Load the model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'xgboost_model.pkl')
    try:
        return pickle.load(open(model_path, 'rb'))
    except FileNotFoundError:
        st.error("Model file not found. Please check the path.")
        return None

# Define location and wind direction options
LOCATIONS = [
    'Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree',
    'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond',
    'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown',
    'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat',
    'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura',
    'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns',
    'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa',
    'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport',
    'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston',
    'AliceSprings', 'Darwin', 'Katherine', 'Uluru'
]

WIND_DIRECTIONS = [
    'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'
]

def validate_numeric_input(value, min_val, max_val, field_name):
    try:
        num_value = float(value)
        if min_val <= num_value <= max_val:
            return num_value
        else:
            st.error(f"{field_name} must be between {min_val} and {max_val}")
            return None
    except ValueError:
        st.error(f"{field_name} must be a number")
        return None

# Page configuration
st.set_page_config(page_title="Rain Prediction", page_icon="🌧️")

# Title and description
st.title('🌧️ Rain Prediction')
st.markdown("""
This app predicts whether it will rain tomorrow based on today's weather conditions.
Please fill in all the fields below with current weather data.
""")

# Load model
model = load_model()

if model:
    # Create three columns for layout
    col1, col2, col3 = st.columns(3)

    with col1:
        location = st.selectbox('Location', LOCATIONS)
        
        sunshine = st.number_input(
            'Sunshine (hours)',
            min_value=0.0,
            max_value=24.0,
            help="Total hours of sunshine today"
        )
        
        cloud = st.number_input(
            'Cloud Coverage',
            min_value=0.0,
            max_value=100.0,
            help="Fraction of sky obscured by cloud (measured in oktas)"
        )

    with col2:
        humidity = st.number_input(
            'Humidity at 3pm (%)',
            min_value=0.0,
            max_value=100.0,
            help="Current humidity percentage"
        )
        
        temp = st.number_input(
            'Temperature at 3pm (°C)',
            min_value=-20.0,
            max_value=50.0,
            help="Current temperature in Celsius"
        )
        
        pressure = st.number_input(
            'Pressure (hPa)',
            min_value=900.0,
            max_value=1100.0,
            help="Average pressure in hectopascals"
        )

    with col3:
        rainfall = st.number_input(
            'Rainfall (mm)',
            min_value=0.0,
            max_value=500.0,
            help="Total rainfall today in millimeters"
        )
        
        evaporation = st.number_input(
            'Evaporation (mm)',
            min_value=0.0,
            max_value=100.0,
            help="Evaporation in millimeters"
        )
        
        wind_dir = st.selectbox(
            'Wind Direction at 9am',
            WIND_DIRECTIONS
        )

    rain_today = st.radio(
        "Is it raining today?",
        ("Yes", "No"),
        horizontal=True
    )

    # Predict button
    if st.button('Predict Rain Tomorrow', type='primary'):
        try:
            # Prepare input data
            input_data = pd.DataFrame({
                'Location': [location],
                'Sunshine': [sunshine],
                'Humidity3pm': [humidity],
                'AverageCloud': [cloud],
                'Temp3pm': [temp],
                'Rainfall': [rainfall],
                'Evaporation': [evaporation],
                'AveragePressure': [pressure],
                'WindDir9am': [wind_dir],
                'RainToday': [rain_today]
            })

            # Make prediction
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0][1]

            # Display result with custom styling
            if prediction[0] == 1:
                st.markdown("""
                    <div style='background-color: #FFE4E1; padding: 20px; border-radius: 10px;'>
                        <h3 style='color: #FF6B6B; margin: 0;'>☔ Rain Expected Tomorrow</h3>
                        <p style='color: #FF3333;'>Don't forget your umbrella! There's a {:.1f}% chance of rain.</p>
                    </div>
                """.format(probability * 100), unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style='background-color: #E1FFE4; padding: 20px; border-radius: 10px;'>
                        <h3 style='color: #6BFF6B; margin: 0;'>🌞 No Rain Expected Tomorrow</h3>
                        <p style='color: #4CAF50;'>Clear skies ahead! Only a {:.1f}% chance of rain.</p>
                    </div>
                """.format(probability * 100), unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

    # Add information about the model
    with st.expander("About this predictor"):
        st.markdown("""
        This rain prediction model uses XGBoost algorithm trained on Australian weather data.
        The model takes into account various weather parameters to predict the likelihood of rain tomorrow.
        
        **Features used:**
        - Location and wind direction
        - Temperature and humidity measurements
        - Cloud coverage and sunshine hours
        - Rainfall and evaporation data
        - Atmospheric pressure
        """)