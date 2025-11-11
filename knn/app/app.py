from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Get absolute path to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Load artifacts once at startup
knn = joblib.load(os.path.join(MODEL_DIR, "knn_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
selected_features = joblib.load(os.path.join(MODEL_DIR, "selected_features.pkl"))
feature_ranges = joblib.load(os.path.join(MODEL_DIR, "feature_ranges.pkl"))

@app.route("/")
def index():
    return render_template(
        "index.html",
        features=list(selected_features),
        ranges=feature_ranges
    )

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])[selected_features]
    df_scaled = scaler.transform(df)
    pred = knn.predict(df_scaled)
    label = label_encoder.inverse_transform(pred)[0]
    return jsonify({"Predicted_Health_Status": label})

if __name__ == "__main__":
    app.run(debug=True)
