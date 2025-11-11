import pandas as pd
from typing import Tuple
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif

def select_features(
    X: np.ndarray, 
    y: np.ndarray, 
    feature_names: pd.Index, 
    k: int = 8
) -> Tuple[np.ndarray, pd.Index]:
    selector = SelectKBest(mutual_info_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = feature_names[selector.get_support()]
    print("[Feature Selection] Selected features:", list(selected_features))
    return X_selected, selected_features