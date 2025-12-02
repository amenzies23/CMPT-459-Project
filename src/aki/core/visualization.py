from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def visualize_ground_truth(X, y, label_encoder):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1],
        hue=label_encoder.inverse_transform(y),
        palette="viridis", s=60
    )
    plt.title("Ground Truth: Plant Health Status")
    plt.savefig("./plots/visualize_ground_truth.png", dpi=300)
    plt.show()
    
# For visualize purpose only.
def visualize_knn_decision_boundary(X, y, n_neighbors=5):
    # Reduce to 2D for visualization only
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Fit kNN on 2D projection
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
    knn.fit(X_2d, y)

    # Create meshgrid
    h = .05
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    # Predict over the grid
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', edgecolor='k')
    plt.title(f"k-NN decision regions (k={n_neighbors})")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.savefig("./plots/visualize_knn_decision_boundary.png", dpi=300)
    plt.show()
    