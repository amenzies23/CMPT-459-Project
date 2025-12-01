import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PLOTS_DIR = os.path.join(PROJECT_ROOT, "src", "alex", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Helper file for plot generating functions
# All of these plots create a `/plots` directory in this current directory
def plot_confusion_matrix(cm, class_names, title, filename):
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    out_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")


def plot_roc_curve(model, X_test, y_test, label_encoder, title_suffix=""):
    # Multiclass ROC (one-vs-rest) using predict_proba. Returns macro AUC
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)
    y_score = model.predict_proba(X_test)

    plt.figure(figsize=(8, 6))
    aucs = []

    for i, class_label in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(
            fpr,
            tpr,
            lw=2,
            label=f"{label_encoder.classes_[i]} (AUC = {roc_auc:.2f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.title(f"Random Forest ROC Curve {title_suffix}".strip())
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(alpha=0.3)

    out_path = os.path.join(PLOTS_DIR, f"rf_roc{title_suffix}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")

    return float(np.mean(aucs))


def plot_feature_importances(model, feature_names, title, filename=None, top_n=None):
    if not hasattr(model, "feature_importances_"):
        print("Model does not expose feature_importances_.")
        return

    importances = model.feature_importances_
    feature_names = np.array(feature_names)
    idx = np.argsort(importances)[::-1]

    if top_n is not None:
        idx = idx[:top_n]

    plt.figure(figsize=(8, 5))
    colors = plt.cm.YlGnBu_r(np.linspace(0, 1, len(idx)))
    plt.barh(range(len(idx)), importances[idx], color=colors, align="center")
    plt.yticks(range(len(idx)), feature_names[idx])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()

    if filename is not None:
        out_path = os.path.join(PLOTS_DIR, filename)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")