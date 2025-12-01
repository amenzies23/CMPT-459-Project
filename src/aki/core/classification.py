import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# KNN without hyperameter tuning.
def classify_with_knn_without_hyperparameter(
    X: np.ndarray, 
    y: np.ndarray, 
    label_encoder: LabelEncoder,
    X_test=None,
    y_test=None
) -> KNeighborsClassifier:
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    # Initialize default KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    knn.fit(X_train, y_train)

    # Predictions
    y_pred = knn.predict(X_test)

    # Plot ROC curve
    plot_roc_curve(knn, X_test, y_test, label_encoder)

    # Classification report
    print("[KNN Classification Report]")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("./plots/classification_confusion_matrix.png", dpi=300)
    plt.show()
    
    # Cross-validation accuracy
    cv_acc = cross_val_score(knn, X, y, cv=5, scoring="accuracy")
    print("Fold scores:", cv_acc)
    print("Mean:", cv_acc.mean(), "Std:", cv_acc.std())

    print("[Cross-validation Accuracy]:", round(cv_acc.mean(), 4))

    return knn

# KNN with hyperparameter using GridSearchCV
def classify_with_knn(
    X: np.ndarray, 
    y: np.ndarray, 
    label_encoder: LabelEncoder,
    X_test=None,
    y_test=None
) -> KNeighborsClassifier:
    X_train, y_train = X, y

    knn = KNeighborsClassifier()
    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 12, 15],
        "weights": ["uniform", "distance"]
    }

    # 5-fold
    grid = GridSearchCV(knn, param_grid, cv=5, scoring="accuracy")

    grid.fit(X_train, y_train)
    best_knn = grid.best_estimator_

    print("[Classification] Best Params:", grid.best_params_)

    y_pred = best_knn.predict(X_test)
    
    # Plot ROC curve
    plot_roc_curve(best_knn, X_test, y_test, label_encoder)

    print("[Classification Report]")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("./plots/classification_confusion_matrix.png", dpi=300)
    plt.show()
    
    # Cross-validation accuracy
    cv_acc = cross_val_score(best_knn, X, y, cv=5, scoring="accuracy").mean()
    print("Fold scores:", cv_acc)
    print("Mean:", cv_acc.mean(), "Std:", cv_acc.std())
    print("[Cross-validation Accuracy]:", round(cv_acc, 4))

    return best_knn

# Plot ROC Curve
def plot_roc_curve(knn, X_test, y_test, label_encoder):
    # Binarize labels for multiclass ROC (one-vs-rest)
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)
    
    # Predict probabilities (KNN must use predict_proba)
    y_score = knn.predict_proba(X_test)

    plt.figure(figsize=(8, 6))
    
    for i, class_label in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"Class {label_encoder.classes_[i]} (AUC = {roc_auc:.2f})")

    # Random-guess line
    plt.plot([0, 1], [0, 1], "k--", lw=1)

    plt.title("ROC Curve (One-vs-Rest)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("./plots/roc_curve.png", dpi=300)
    plt.show()