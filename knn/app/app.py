from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

app = Flask(__name__)

# Get absolute path to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data/plant_health_data.csv')
df = pd.read_csv(DATA_PATH)

# List of numeric columns to interpolate
interpolator_columns = [
    "Soil_Moisture", "Ambient_Temperature", "Soil_Temperature", "Humidity",
    "Light_Intensity", "Soil_pH", "Nitrogen_Level", "Phosphorus_Level",
    "Potassium_Level", "Chlorophyll_Content", "Electrochemical_Signal"
]

# Build interpolation functions
interpolators = {}

# Convert Timestamp hour of day (0 to 24 (float))
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour + df['Timestamp'].dt.minute / 60.0
plant_ids = df["Plant_ID"].unique().tolist()

# Load artifacts once at startup
knn = joblib.load(os.path.join(MODEL_DIR, "knn_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
selected_features = joblib.load(os.path.join(MODEL_DIR, "selected_features.pkl"))

def build_interpolators_for_plant(plant_id):
    """Build interpolation functions only for the selected plant (raw data)."""
    df_plant = df[df["Plant_ID"] == plant_id]
    interpolators = {}

    for col in interpolator_columns:
        # Raw points
        hourly = df_plant[["Hour", col]].dropna().copy()

        # Sorted, no duplicates
        hourly = hourly.sort_values("Hour")
        hourly = hourly.drop_duplicates(subset="Hour", keep="first")

        if len(hourly) < 1:
            continue

        x = hourly["Hour"].values
        y = hourly[col].values

        kind = "cubic" if len(x) >= 4 else "linear"

        interpolators[col] = interp1d(
            x, y,
            kind=kind,
        )

    return interpolators

@app.route("/")
def index():
    return render_template(
        "index.html",
        features=list(selected_features)
    )

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])[selected_features]
    df_scaled = scaler.transform(df.values)
    pred = knn.predict(df_scaled)
    label = label_encoder.inverse_transform(pred)[0]
    return jsonify({"Predicted_Health_Status": label})

@app.route("/day_cycle/<int:plant_id>")
def day_cycle(plant_id):
    if plant_id not in plant_ids:
        return jsonify({"error": "Invalid plant ID"}), 400

    df_plant = df[df["Plant_ID"] == plant_id]

    if df_plant.empty:
        return jsonify({"error": "No data for this plant"}), 400

    min_hour = df_plant["Hour"].min()
    max_hour = df_plant["Hour"].max()

    hours = np.linspace(min_hour, max_hour, 200)
    interpolators = build_interpolators_for_plant(plant_id)
    result = {"Hour": hours.tolist()}

    for col in interpolator_columns:
        if col in interpolators:
            result[col] = interpolators[col](hours).tolist()
        else:
            result[col] = None

    return jsonify(result)

@app.route("/plant_ids")
def get_plant_ids():
    return jsonify(sorted(plant_ids))

if __name__ == "__main__":
    app.run(debug=True)
