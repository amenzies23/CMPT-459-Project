from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# The Local Outlier Factor (LOF) measures how isolated a point is compared to
# its local neighbourhood.    
def remove_outliers(
    X_scaled: np.ndarray, 
    y_encoded: np.ndarray, 
    contamination: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    # Using LOF Outlier Detection
    lof = LocalOutlierFactor(n_neighbors=10, contamination=contamination)
    
    # -1 = outlier, 1 = inlier
    outlier_labels = lof.fit_predict(X_scaled)

    # Build mask for non-outlier (inliers)
    mask = outlier_labels != -1
    X_clean = X_scaled[mask]
    y_clean = y_encoded[mask]

    # Only plot first 2 dimensions
    X_inliers = X_scaled[mask]
    X_outliers = X_scaled[~mask]

    plt.figure(figsize=(7, 6))
    plt.scatter(X_inliers[:, 0], X_inliers[:, 1], c='blue', s=20, label='Inliers')
    plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red', s=40, marker='x', label='Outliers')
    plt.title("Local Outlier Factor (LOF) Outlier Visualization")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.savefig("./plots/remove_outliers.png", dpi=300)
    plt.show()
    
    removed = np.sum(outlier_labels == -1)
    print(f"[Outlier Detection] Removed {removed} outliers.")

    return X_clean, y_clean
