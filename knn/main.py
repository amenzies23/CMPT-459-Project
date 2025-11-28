import argparse
import joblib
import random

import pandas as pd
import numpy as np

from typing import List
from sklearn.calibration import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)

from core.preprocessing import load_and_preprocess_data
from core.outlier_detection import remove_outliers
from core.feature_selection import select_features
from core.clustering import perform_clustering

from core.classification import (
    classify_with_knn,
    classify_with_knn_without_hyperparameter,
)
from core.visualization import (
    visualize_ground_truth,
    visualize_knn_decision_boundary,
)

from utils import divider

def main():
    parser = argparse.ArgumentParser(
        description="Plant Health Classification Pipeline"
    )
    parser.add_argument(
        "--no-hyperparameter-tuning",
        action="store_true",
        help="Disable hyperparameter tuning for KNN (use default parameters).",
    )
    args = parser.parse_args()
    
    # Set random seeds
    seed = 42
    random.seed(seed) # Python
    np.random.seed(seed) # NumPy

    divider("1: Load & Preprocess Data")
    X_raw, X_scaled, y, label_encoder, scaler = load_and_preprocess_data(
        "../data/plant_health_data.csv"
    )

    divider("2: Outlier Detection (LOF)")
    X_clean, y_clean = remove_outliers(X_scaled, y)

    divider("3: Feature Selection (Mutual Information)")
    X_selected, selected_features = select_features(X_clean, y_clean, X_raw.columns, k=8)

    selected_scaler = StandardScaler()
    X_selected_scaled = selected_scaler.fit_transform(X_selected)

    divider("4A: Ground Truth Visualization")
    visualize_ground_truth(X_selected_scaled, y_clean, label_encoder)

    divider("4B: DBSCAN Clustering")
    perform_clustering(X_selected_scaled)

    divider("4C: KNN Decision Boundary Visualization")
    visualize_knn_decision_boundary(X_selected_scaled, y_clean, n_neighbors=5)

    divider("5: Classification (k-NN)")
    if args.no_hyperparameter_tuning:
        print("Running KNN without hyperparameter tuning...")
        knn = classify_with_knn_without_hyperparameter(
            X_selected_scaled, y_clean, label_encoder
        )
    else:
        print("Running KNN with hyperparameter tuning...")
        knn = classify_with_knn(X_selected_scaled, y_clean, label_encoder)
        
    divider("6: Saving KNN, Label Encoder, Scaler, Selected Features for Inference")
    save_model(
        knn=knn, 
        label_encoder=label_encoder, 
        scaler=selected_scaler, 
        selected_features=selected_features,
    )

def save_model(
    knn: KNeighborsClassifier,
    label_encoder: LabelEncoder,
    scaler: StandardScaler,
    selected_features: List[str],
) -> None:
    path = './model'
    # Save trained model and label encoder
    joblib.dump(knn, f"{path}/knn_model.pkl")
    joblib.dump(label_encoder, f"{path}/label_encoder.pkl")
    joblib.dump(scaler, f"{path}/scaler.pkl")
    joblib.dump(selected_features, f"{path}/selected_features.pkl")

    print("Model and label encoder saved.")

if __name__ == "__main__":
    main()