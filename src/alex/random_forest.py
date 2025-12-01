# Alex Menzies - 301563620
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, label_binarize

# Ensure src/ is on sys.path so we can import project modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocessing import preprocess, get_labels
from alex.outlier_utils import isolation_forest_mask, remove_outliers 
from alex.plots import plot_confusion_matrix, plot_roc_curve

"""
    This file uses a random forest classifier to classify the plant health dataset in 5 different scenarios:
    
    I first run the dataset through a preprocessing pipeline, to clean the data, and normalize values. Then
    I invoke my outlier detection (Isolation forest) functions to remove outliers. Then we go through these 5 scenarios:
    
    1. A baseline random forest, this does not use any feature selection or hyperparameter tuning
    2. A random forest with feature selection, but no hyperparameter tuning
    3. A random forest without feature selection, but with hyperparameter tuning
    4. A random forest with both feature selection and hyperparameter tuning
    5. After finding the best performing classifier, I ran this classifier without outlier removal, to see the
        if the outlier detection improves, or lessens the scores.
    
    The results are all displayed in the console, showing the training and testing accuracy of the model,
    the full classification report, the cross-validation scores, and the AUC values.
    
    Plots showing the ROC curve and the confusion matrices are also saved in `./plots`
"""

def report_train_test_accuracy(model, X_train, y_train, X_test, y_test, label: str):
    """Print simple train and test accuracy for a fitted model."""
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"[{label}] Train accuracy: {train_acc:.4f}")
    print(f"[{label}] Test accuracy:  {test_acc:.4f}")
    return train_acc, test_acc


def apply_mutual_information_selection(X: pd.DataFrame, y: np.ndarray, k: int | None):
    """Mutual information feature selection."""
    n_features = X.shape[1]
    if k is None:
        k = min(5, n_features)

    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    return X_selected, selector

# Random Forest variants
def rf_baseline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
):
    # Scenario 1: RF baseline (no FS, no tuning) with constrained depth
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print("\n[Scenario 1: Baseline RF] Classification Report")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(
        cm,
        class_names=label_encoder.classes_,
        title="Scenario 1: RF Baseline - Confusion Matrix",
        filename="rf_s1_baseline_cm.png",
    )

    train_acc, test_acc = report_train_test_accuracy(
        rf, X_train, y_train, X_test, y_test, label="Scenario 1 RF Baseline"
    )

    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
    print("Fold scores:", cv_scores)
    print("Mean:", cv_scores.mean(), "Std:", cv_scores.std())
    print("[Scenario 1 CV Accuracy]:", round(cv_scores.mean(), 4))

    auc_value = None
    if hasattr(rf, "predict_proba"):
        auc_value = plot_roc_curve(
            rf, X_test, y_test, label_encoder, title_suffix="_s1_baseline"
        )
        print(f"[Scenario 1 RF Baseline] Test AUC (macro): {auc_value:.4f}")

    return rf, train_acc, test_acc, auc_value


def rf_mi_only(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
):
    # Scenario 2: RF with MI feature selection (no tuning)
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)

    X_train_mi, selector = apply_mutual_information_selection(X_train_df, y_train, k=5)
    X_test_mi = selector.transform(X_test_df)

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_mi, y_train)
    y_pred = rf.predict(X_test_mi)

    print("\n[Scenario 2: RF + MI] Classification Report")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(
        cm,
        class_names=label_encoder.classes_,
        title="Scenario 2: RF + MI - Confusion Matrix",
        filename="rf_s2_mi_cm.png",
    )

    train_acc, test_acc = report_train_test_accuracy(
        rf, X_train_mi, y_train, X_test_mi, y_test, label="Scenario 2 RF + MI"
    )

    cv_scores = cross_val_score(
        rf, X_train_mi, y_train, cv=5, scoring="accuracy", n_jobs=-1
    )
    print("Fold scores:", cv_scores)
    print("Mean:", cv_scores.mean(), "Std:", cv_scores.std())
    print("[Scenario 2 CV Accuracy]:", round(cv_scores.mean(), 4))

    auc_value = None
    if hasattr(rf, "predict_proba"):
        auc_value = plot_roc_curve(
            rf, X_test_mi, y_test, label_encoder, title_suffix="_s2_mi"
        )
        print(f"[Scenario 2 RF + MI] Test AUC (macro): {auc_value:.4f}")

    return rf, train_acc, test_acc, auc_value


def rf_tuned_only(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
):
    # Scenario 3: RF with hyperparameter tuning (no feature selection)
    base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    param_grid = {
        "n_estimators": [100, 200, 400],
        "max_depth": [5, 8, 12, None],
        "min_samples_split": [5, 10, 20],
        "min_samples_leaf": [2, 5, 10, 20],
        "max_features": ["sqrt"],
    }

    grid = GridSearchCV(
        base_rf,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    best_rf = grid.best_estimator_
    print("[Scenario 3: RF tuned] Best Params:", grid.best_params_)

    y_pred = best_rf.predict(X_test)

    print("\n[Scenario 3: RF tuned] Classification Report")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(
        cm,
        class_names=label_encoder.classes_,
        title="Scenario 3: RF tuned - Confusion Matrix",
        filename="rf_s3_tuned_cm.png",
    )

    train_acc, test_acc = report_train_test_accuracy(
        best_rf, X_train, y_train, X_test, y_test, label="Scenario 3 RF tuned"
    )

    cv_scores = cross_val_score(
        best_rf, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1
    )
    print("Fold scores:", cv_scores)
    print("Mean:", cv_scores.mean(), "Std:", cv_scores.std())
    print("[Scenario 3 CV Accuracy]:", round(cv_scores.mean(), 4))

    auc_value = None
    if hasattr(best_rf, "predict_proba"):
        auc_value = plot_roc_curve(
            best_rf, X_test, y_test, label_encoder, title_suffix="_s3_tuned"
        )
        print(f"[Scenario 3 RF tuned] Test AUC (macro): {auc_value:.4f}")

    return best_rf, train_acc, test_acc, auc_value

def rf_mi_tuned(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    feature_names: list[str] | None = None,
    title_suffix: str = "_s4_mi_tuned",
):
    # Scenario 4: RF with MI + hyperparameter tuning
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)

    X_train_mi, selector = apply_mutual_information_selection(X_train_df, y_train, k=5)
    X_test_mi = selector.transform(X_test_df)

    # Indeces of selected features and mapping to the column names
    if feature_names:
        selected_idx = selector.get_support(indices=True)
        selected_feature_names = [feature_names[i] for i in selected_idx]
    else:
        selected_feature_names = []
    
    base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_grid = {
        "n_estimators": [100, 200, 400],
        "max_depth": [5, 8, 12, None],
        "min_samples_split": [5, 10, 20],
        "min_samples_leaf": [2, 5, 10, 20],
        "max_features": ["sqrt"],
    }

    grid = GridSearchCV(
        base_rf,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train_mi, y_train)

    best_rf = grid.best_estimator_
    print("[Scenario 4: RF + MI tuned] Best Params:", grid.best_params_)

    y_pred = best_rf.predict(X_test_mi)

    print("\n[Scenario 4: RF + MI tuned] Classification Report")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    cm_filename = f"rf_confusion{title_suffix}.png"
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(
        cm,
        class_names=label_encoder.classes_,
        title=f"RF + MI tuned {title_suffix}",
        filename=cm_filename,
    )

    train_acc, test_acc = report_train_test_accuracy(
        best_rf,
        X_train_mi,
        y_train,
        X_test_mi,
        y_test,
        label="Scenario 4 RF + MI tuned",
    )

    cv_scores = cross_val_score(
        best_rf, X_train_mi, y_train, cv=5, scoring="accuracy", n_jobs=-1
    )
    print("Fold scores:", cv_scores)
    print("Mean:", cv_scores.mean(), "Std:", cv_scores.std())
    print("[Scenario 4 CV Accuracy]:", round(cv_scores.mean(), 4))

    auc_value = None
    if hasattr(best_rf, "predict_proba"):
        auc_value = plot_roc_curve(
            best_rf, X_test_mi, y_test, label_encoder, title_suffix=title_suffix
        )
        print(f"[RF + MI tuned{title_suffix}] Test AUC (macro): {auc_value:.4f}")

    return best_rf, train_acc, test_acc, auc_value, selected_feature_names

# Main pipeline
def main():
    # Using the same training / testing data split as the rest of the group (80/20)
    train_path = os.path.join(PROJECT_ROOT, "data", "train_data.csv")
    test_path = os.path.join(PROJECT_ROOT, "data", "test_data.csv")

    df_train = pd.read_csv(train_path, index_col=0)
    df_test = pd.read_csv(test_path, index_col=0)

    # Shared preprocessing with the group
    X_train_df = preprocess(df_train, attr=[])
    X_test_df = preprocess(df_test, attr=[])

    y_train_series = get_labels(df_train)["Plant_Health_Status"]
    y_test_series = get_labels(df_test)["Plant_Health_Status"]

    # Outlier removal on train only, then align labels
    inlier_mask, scores, iso_model = isolation_forest_mask(X_train_df)
    X_train_clean, y_train_clean_series = remove_outliers(
        X_train_df, inlier_mask, y_train_series
    )

    print(f"Train size before IF: {len(X_train_df)}, after IF: {len(X_train_clean)}")

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_clean = label_encoder.fit_transform(y_train_clean_series.values)
    y_test = label_encoder.transform(y_test_series.values)

    X_train = X_train_clean.values
    X_test = X_test_df.values

    # Running the 4 scenarios
    results = []
    
    print("\nScenario 1) RF: Baseline")
    _, tr1, te1, au1 = rf_baseline(X_train, X_test, y_train_clean, y_test, label_encoder)
    results.append(("Scenario 1: Baseline RF", tr1, te1, au1))

    print("\nScenario 2) RF: MI only")
    _, tr2, te2, au2 = rf_mi_only(X_train, X_test, y_train_clean, y_test, label_encoder)
    results.append(("Scenario 2: RF + MI", tr2, te2, au2))

    print("\nScenario 3) RF: Tuning only")
    _, tr3, te3, au3 = rf_tuned_only(X_train, X_test, y_train_clean, y_test, label_encoder)
    results.append(("Scenario 3: RF tuned", tr3, te3, au3))

    print("\nScenario 4) RF: MI + Tuning")
    _, tr4, te4, au4, _ = rf_mi_tuned(X_train, X_test, y_train_clean, y_test, label_encoder)
    results.append(("Scenario 4: RF + MI tuned", tr4, te4, au4))


    print("\nScenario 5) RF: MI + Tuning (NO Isolated Forest outlier removal)")

    # New label encoder for full (non‑IF) train
    label_encoder_no_if = LabelEncoder()
    y_train_no_if = label_encoder_no_if.fit_transform(y_train_series.values)
    y_test_no_if = label_encoder_no_if.transform(y_test_series.values)

    X_train_no_if = X_train_df.values
    X_test_no_if = X_test_df.values

    _, tr5, te5, au5, _ = rf_mi_tuned(
        X_train_no_if,
        X_test_no_if,
        y_train_no_if,
        y_test_no_if,
        label_encoder_no_if,
        title_suffix="_s5_mi_tuned_no_if",
    )
    results.append(
        ("Scenario 5: RF + MI tuned (no Isolated Forest outlier removal)", tr5, te5, au5)
    )

    # Summary table
    print("\nSummary of RF Scenarios")
    print("{:<36}  {:>10}  {:>10}  {:>10}".format("Scenario", "Train Acc", "Test Acc", "Test AUC"))
    for name, tr, te, au in results:
        print(f"{name:<36}  {tr:>10.4f}  {te:>10.4f}  {au:>10.4f}")

    print("\nComparison of best tuned model WITH vs WITHOUT IsolationForest:")
    print(f"With IF (Scenario 4)    - Test Acc: {te4:.4f}, Test AUC: {au4:.4f}")
    print(f"Without IF (Scenario 5) - Test Acc: {te5:.4f}, Test AUC: {au5:.4f}")


if __name__ == "__main__":
    main()