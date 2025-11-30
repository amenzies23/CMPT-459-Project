import numpy as np
import pandas as pd

from typing import Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(filepath: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, LabelEncoder, StandardScaler]:
    df = pd.read_csv(filepath)
    df = df.drop(columns=["Timestamp", "Plant_ID"], errors="ignore")

    # Separate features and target
    X = df.drop(columns=["Plant_Health_Status"])
    y = df["Plant_Health_Status"]

    # Handle missing values
    X = X.fillna(X.mean())

    # Encode target
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled, y_encoded, encoder, scaler