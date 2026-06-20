import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

st.set_page_config(page_title="DDoS Detection System")

st.title("DDoS Attack Detection System")

st.write("Upload a network traffic CSV file.")

# Load model
model = tf.keras.models.load_model("ddos_model.h5")

# Load scaler and columns
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file, low_memory=False)

    data.columns = data.columns.str.strip()

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    if st.button("Detect Attack"):

        # Remove Label if present
        if "Label" in data.columns:
            data = data.drop("Label", axis=1)

        # Match training columns
        data = data.reindex(
            columns=columns,
            fill_value=0
        )

        # Numeric conversion
        data = data.apply(
            pd.to_numeric,
            errors="coerce"
        )

        # Clean values
        data.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True
        )

        data.fillna(0, inplace=True)

        # Scale
        data_scaled = scaler.transform(data)

        # Predict
        prediction = model.predict(data_scaled)

        attack_ratio = np.mean(prediction > 0.5)

        st.write(
            f"Attack Ratio: {attack_ratio:.2f}"
        )

        if attack_ratio > 0.30:
            st.error("⚠ DDoS Attack Detected")
        else:
            st.success("✅ Normal Network Traffic")