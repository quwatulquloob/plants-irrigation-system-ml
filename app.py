import streamlit as st
import joblib
import numpy as np
from PIL import Image
import io
import base64

# Set up the Streamlit page configuration
st.set_page_config(page_title="SMART IRRIGATION SYSTEM", page_icon="🌿", layout="centered")

# Load the saved scaler and model
scaler = joblib.load('model/scaler.pkl')
model = joblib.load('model/random_forest_model.pkl')

# Display the logo image in the center
logo_path = "images/irigation logo.jpg"  # Replace with the actual path to your logo image
logo = Image.open(logo_path)

# Function to display image in the center with increased size
def display_centered_image(image, caption="", width="800px", height="auto"):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{img_str}" alt="{caption}" style="max-width: {width}; height: {height};">
        </div>
        """,
        unsafe_allow_html=True,
    )

# Example usage:
display_centered_image(logo, width="800px", height="500px")

# Display the project title
st.title("SMART IRRIGATION SYSTEM")

# Define the features for input with ranges
st.subheader("Enter the following feature values:")

# Input sliders for features
temperature = st.slider("Temperature", min_value=0, max_value=100, value=25, step=1)
humidity = st.slider("Humidity", min_value=0, max_value=100, value=50, step=1)
water_level = st.slider("Water Level", min_value=0, max_value=100, value=30, step=1)
N = st.slider("Nitrogen (N)", min_value=0, max_value=255, value=100, step=1)
P = st.slider("Phosphorus (P)", min_value=0, max_value=255, value=100, step=1)
K = st.slider("Potassium (K)", min_value=0, max_value=255, value=100, step=1)

# Add a button for prediction
if st.button("Predict"):
    # Convert user input to numpy array and reshape to match model input shape
    try:
        user_data = np.array([temperature, humidity, water_level, N, P, K]).reshape(1, -1)

        # Standardize the user input using the loaded scaler
        user_data_scaled = scaler.transform(user_data)

        # Make prediction using the loaded model
        prediction = model.predict(user_data_scaled)
        prediction_proba = model.predict_proba(user_data_scaled) if hasattr(model, "predict_proba") else None

        # Output the prediction
        st.subheader("Actuator Predictions")
        classification_targets = ['Fan_actuator_ON', 'Watering_plant_pump_ON', 'Water_pump_actuator_ON']

        if prediction_proba is not None:  # Multi-class classification
            # Ensure prediction is in the correct shape
            prediction_proba = np.array(prediction_proba)
            for i, target in enumerate(classification_targets):
                proba = prediction_proba[0, i]
                status = "✔️ ON" if proba > 0.5 else "❌ OFF"
                st.markdown(f"**{target.replace('_', ' ')}:** {status}")
        else:  # Binary classification or other
            for target, pred in zip(classification_targets, prediction[0]):
                status = "✔️ ON" if pred == 1 else "❌ OFF"
                st.markdown(f"**{target.replace('_', ' ')}:** {status}")

    except ValueError:
        st.error("Please enter valid numeric values for all fields.")
