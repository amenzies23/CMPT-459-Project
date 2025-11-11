import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# KNN without hyperameter tuning.
def classify_with_knn_without_hyperparameter(
    X: np.ndarray, 
    y: np.ndarray, 
    label_encoder: LabelEncoder
) -> KNeighborsClassifier:
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    # Initialize default KNN classifier (baseline)
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    knn.fit(X_train, y_train)

    # Predictions
    y_pred = knn.predict(X_test)

    # Classification report
    print("\n[Baseline KNN Classification Report]")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title("Baseline KNN - Confusion Matrix")
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
    label_encoder: LabelEncoder
) -> KNeighborsClassifier:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

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