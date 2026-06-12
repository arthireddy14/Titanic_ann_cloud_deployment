import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# -----------------------------
# Load Model
# -----------------------------

model = load_model("titanic_ann_model.h5")

# -----------------------------
# Streamlit Page Config
# -----------------------------

st.set_page_config(
    page_title="Titanic Survival Prediction",
    layout="centered"
)

# -----------------------------
# Header Section
# -----------------------------

st.title("🚢 Titanic Survival Prediction System")

st.subheader(
    "Deep Learning Based Passenger Survival Prediction"
)

# -----------------------------
# Project Description
# -----------------------------

st.markdown("""
This application predicts whether a passenger
would survive during the Titanic disaster using:

- Artificial Neural Networks (ANN)
- TensorFlow/Keras
- Streamlit Deployment

The model uses:
- Passenger Class
- Age
- Fare

Titanic Survival Prediction System is a Deep Learning-based web application that predicts the likelihood of a passenger surviving a Titanic-like emergency situation. The model uses passenger information such as Passenger Class (Pclass), Age, and Fare to estimate survival probability. Historical Titanic data shows that passengers in higher classes, younger individuals, and those with higher ticket fares generally had better chances of survival. The application is powered by an Artificial Neural Network (ANN) built using TensorFlow and deployed through Streamlit, providing real-time predictions and visualizations of survival probability.
""")

# -----------------------------
# Input Form
# -----------------------------

st.header("Passenger Details")

col1, col2, col3 = st.columns(3)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        ["1-Higher class","2-Middle class","3-Lower class"]
    )

with col2:
    age = st.slider(
        "Age",
        1,
        80,
        24
    )

with col3:
    fare = st.number_input(
        "Fare",
        min_value=0.0,
        value=50.0
    )

# -----------------------------
# Data Preprocessing
# -----------------------------

# Training dataset min/max assumptions
pclass=int(pclass[0])
pclass_norm = (pclass - 1) / (3 - 1)
age_norm = (age - 0) / (80 - 0)
fare_norm = (fare - 0) / (512 - 0)

input_data = np.array([
    [pclass_norm, age_norm, fare_norm]
])

# -----------------------------
# Prediction Button
# -----------------------------

if st.button("Predict Survival"):

    prediction = model.predict(input_data)

    probability = prediction[0][0]

    # -------------------------
    # Prediction Logic
    # -------------------------

    if probability > 0.5:
        result = "✅ Survived"
    else:
        result = "❌ Not Survived"

    # -------------------------
    # Output Section
    # -------------------------

    st.header("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Prediction",
            result
        )

    with col2:
        st.metric(
            "Survival Probability",
            f"{probability:.2f}"
        )

    # -------------------------
    # Confidence Score
    # -------------------------

    confidence = max(probability, 1 - probability)

    st.metric(
        "Confidence Score",
        f"{confidence:.2f}"
    )

    # -------------------------
    # Visualization
    # -------------------------

    st.header("Prediction Visualization")

    labels = ['Survival', 'Non-Survival']

    values = [
        probability,
        1 - probability
    ]

    fig, ax = plt.subplots(figsize=(4,3))

    ax.bar(labels, values)

    st.pyplot(fig)