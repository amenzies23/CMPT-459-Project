# Alex Menzies - 301563620
# Isolation forest outlier detection + removal
# This follows the same outlier detection analysis I did in the `./isolation_forest.py` file, but 
# I was having a hard time exporting those functions to my random forest classifier. So I added this file
import pandas as pd
from sklearn.ensemble import IsolationForest


def isolation_forest_mask(
    X: pd.DataFrame,
    n_estimators: int = 200,
    contamination: str | float = 0.1,
    random_state: int = 42,
):
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    iso.fit(X)

    preds = iso.predict(X)  # 1 = inlier, -1 = outlier
    inlier_mask = preds == 1
    scores = iso.decision_function(X)

    return inlier_mask, scores, iso


def remove_outliers(
    X: pd.DataFrame,
    inlier_mask,
    y: pd.Series | None = None,
):
    X_clean = X.loc[inlier_mask].copy()

    if y is None:
        return X_clean, None

    y_clean = y.loc[X_clean.index].copy()
    return X_clean, y_clean