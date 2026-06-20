import pandas as pd
import numpy as np
import glob
import tensorflow as tf
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Loading dataset...")

# Read all CSV files
files = glob.glob("archive/*.csv")

data_list = []

for file in files:
    print("Reading:", file)

    try:
        df = pd.read_csv(file, nrows=20000, low_memory=False)

        df.columns = df.columns.str.strip()

        if "Label" not in df.columns:
            continue

        data_list.append(df)

    except Exception as e:
        print("Error:", e)

# Combine files
data = pd.concat(data_list, ignore_index=True)

print("Dataset Loaded")

# Keep only BENIGN and DDoS
data = data[data["Label"].isin(["BENIGN", "DDoS"])]

# Convert labels
data["Label"] = data["Label"].map({
    "BENIGN": 0,
    "DDoS": 1
})

# Features and labels
X = data.drop("Label", axis=1)
y = data["Label"]

# Save columns for dashboard
joblib.dump(X.columns.tolist(), "columns.pkl")

# Convert to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Handle missing values
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)

# Scale
scaler = StandardScaler()

X = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Model...")

# Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)

loss, accuracy = model.evaluate(X_test, y_test)

print("\nAccuracy:", accuracy)

# Save model
model.save("ddos_model.h5")

print("Model Saved Successfully")