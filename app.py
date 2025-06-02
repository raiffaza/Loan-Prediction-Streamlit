import streamlit as st
import pandas as pd
import joblib
import requests
from io import BytesIO

# Function to load files directly from GitHub
def load_file_from_github(url):
    # Send GET request to fetch the file
    response = requests.get(url)
    
    # Check if request was successful (status code 200)
    if response.status_code == 200:
        print("File downloaded successfully")
    else:
        print("Error downloading the file")
        return None

    # Load the content of the file into joblib
    return joblib.load(BytesIO(response.content))

# URLs for model and scaler files hosted on GitHub
model_url = "https://github.com/raiffaza/Food-Delivery-Times-Prediction/raw/main/xgb_tuned_model.pkl"
scaler_url = "https://github.com/raiffaza/Food-Delivery-Times-Prediction/raw/main/scaler.pkl"

# Load model and scaler from GitHub
model = load_file_from_github(model_url)
scaler = load_file_from_github(scaler_url)

if model and scaler:
    # Model and scaler loaded successfully
    print("Model and scaler loaded successfully")
else:
    # Failed to load model and scaler
    print("Error loading model and scaler")

# Get the exact feature names and order expected by the model
expected_columns = model.feature_names_in_.tolist()

# Identify numeric features (must match training data)
numeric_features = ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs']

def make_prediction(distance_km, prep_time, courier_exp, weather, traffic, time_of_day, vehicle):
    # Initialize all features to 0 (scalar, not list)
    data = {col: 0 for col in expected_columns}

    # Fill numeric features
    data['Distance_km'] = distance_km
    data['Preparation_Time_min'] = prep_time
    data['Courier_Experience_yrs'] = courier_exp

    # Set selected categorical features to 1
    weather_col = f'Weather_{weather}'
    if weather_col in data:
        data[weather_col] = 1

    traffic_col = f'Traffic_Level_{traffic}'
    if traffic_col in data:
        data[traffic_col] = 1

    time_col = f'Time_of_Day_{time_of_day}'
    if time_col in data:
        data[time_col] = 1

    vehicle_col = f'Vehicle_Type_{vehicle}'
    if vehicle_col in data:
        data[vehicle_col] = 1

    # Convert to DataFrame with the expected columns
    input_df = pd.DataFrame([data], columns=expected_columns)

    # Scale only numeric features
    input_df[numeric_features] = scaler.transform(input_df[numeric_features])

    # Make prediction
    prediction = model.predict(input_df)[0]
    return prediction

# Set page config
st.set_page_config(
    page_title="Uber Eats Delivery Time Prediction",
    page_icon="🍔",
    layout="centered"  # Set layout to centered for all content
)

# --- HEADER --- 
# Centered Logo (Ensure this image is in the same directory as app.py)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("uber eats.png", width=250)  # Ensure this image is in the same directory as app.py

# Title and Subtitle with white text
st.markdown("<h1 style='color: white; text-align: center;'>Uber Eats Delivery Time Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='color: white; text-align: center;'>Fast • Secure • Intelligent</h3>", unsafe_allow_html=True)

# --- ABOUT SECTION ---
st.markdown("""
## About Uber Eats
Uber Eats is revolutionizing food delivery by leveraging cutting-edge machine learning techniques to improve customer satisfaction and operational efficiency.

This app uses a trained machine learning model to predict the delivery time based on parameters like distance, weather, traffic, and more.
""", unsafe_allow_html=True)

# Display the Business Problem Image (wide and centered)
st.image("uber eats business problem.jpeg", use_container_width=True)

# --- PURPOSE OF THE APP SECTION ---
st.markdown("""
## Purpose of this Website
This platform allows users to input specific delivery parameters and instantly receive a **data-driven estimate** of the delivery time, powered by an advanced **XGBoost machine learning model**.

### 🎯 Objectives:
- Real-time predictions based on input data
- Improved resource allocation for Uber Eats operations
- More accurate delivery time predictions to optimize customer experience
""", unsafe_allow_html=True)

# Display the Company Profile Image (wide and centered)
st.image("uber eats company profile.jpeg", use_container_width=True)

# --- INPUT FORM SECTION ---
st.markdown("### Enter Delivery Details", unsafe_allow_html=True)

with st.form("delivery_form"):
    distance_km = st.number_input("Distance (km)", min_value=0.0, format="%.2f", help="Distance between restaurant and delivery address.")
    prep_time = st.number_input("Preparation Time (minutes)", min_value=0, help="Time taken to prepare the order.")
    courier_exp = st.number_input("Courier Experience (years)", min_value=0.0, format="%.1f", help="Years of delivery courier experience.")

    st.markdown("<span style='color:white; font-weight:bold;'>Weather Condition</span>", unsafe_allow_html=True)
    weather = st.selectbox("", ['Windy', 'Clear', 'Foggy', 'Rainy', 'Snowy'])

    st.markdown("<span style='color:white; font-weight:bold;'>Traffic Level</span>", unsafe_allow_html=True)
    traffic = st.selectbox("", ['Low', 'Medium', 'High'])

    st.markdown("<span style='color:white; font-weight:bold;'>Time of Day</span>", unsafe_allow_html=True)
    time_of_day = st.selectbox("", ['Afternoon', 'Evening', 'Night', 'Morning'])

    st.markdown("<span style='color:white; font-weight:bold;'>Vehicle Type</span>", unsafe_allow_html=True)
    vehicle = st.selectbox("", ['Scooter', 'Bike', 'Car'])

    submit = st.form_submit_button("Predict Delivery Time")

# --- PREDICTION RESULT SECTION ---
if submit:
    # Call prediction function when form is submitted
    predicted_time = make_prediction(distance_km, prep_time, courier_exp, weather, traffic, time_of_day, vehicle)
    
    # Display result in an attractive manner with white text
    st.markdown("<h2 style='color: white; text-align: center;'>📊 Prediction Result</h2>", unsafe_allow_html=True)
    st.success(f"✅ Estimated Delivery Time: {predicted_time:.2f} minutes", icon="💨")

    # Additional Info (Explanation) centered
    st.markdown("""
    ---
    ### 🔍 Explanation of Features

    - `Distance (km)` (numeric): Distance between the restaurant and the delivery address.
    - `Preparation Time (minutes)` (numeric): Time taken to prepare the order.
    - `Courier Experience (years)` (numeric): Years of experience of the courier.
    - `Weather Condition` (categorical): Weather during the delivery (Windy, Clear, etc.).
    - `Traffic Level` (categorical): Traffic conditions during the delivery (Low, Medium, High).
    - `Time of Day` (categorical): Time of day during the delivery (Morning, Afternoon, Evening, Night).
    - `Vehicle Type` (categorical): Type of vehicle used by the courier (Scooter, Bike, Car).
    """, unsafe_allow_html=True)
