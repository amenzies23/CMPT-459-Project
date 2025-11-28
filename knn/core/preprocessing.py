import numpy as np
import pandas as pd

from typing import Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(
    train_path: str,
    test_path: str
)  -> Tuple[pd.DataFrame, pd.DataFrame, 
            np.ndarray, np.ndarray,
            np.ndarray, np.ndarray,
            LabelEncoder, 
            StandardScaler]:
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    drop_cols = ["Timestamp", "Plant_ID"]
    df_train = df_train.drop(columns=drop_cols, errors="ignore")
    df_test = df_test.drop(columns=drop_cols, errors="ignore")

    # Separate features and labels
    X_train = df_train.drop(columns=["Plant_Health_Status"])
    y_train = df_train["Plant_Health_Status"]

    X_test = df_test.drop(columns=["Plant_Health_Status"])
    y_test = df_test["Plant_Health_Status"]

    # Handle missing values
    # (Use train mean for BOTH)
    train_means = X_train.mean()
    X_train = X_train.fillna(train_means)
    X_test = X_test.fillna(train_means)

    # Encode labels (fit on train)
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test)

    # Scale numerical features (fit on train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train, X_test,
        y_train_enc, y_test_enc,
        X_train_scaled, X_test_scaled,
        encoder, 
        scaler
    )