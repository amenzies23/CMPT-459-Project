import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
                                          
def perform_clustering(
    X_selected: np.ndarray, 
    eps: float = 1.5, 
    min_samples: int = 5
) -> None:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X_selected)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"[Clustering] DBSCAN found {n_clusters} clusters.")

    # Visualization with PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_selected)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette="tab10", s=50)
    plt.title("DBSCAN Clustering (PCA projection)")
    plt.savefig("./plots/perform_clustering.png", dpi=300)
    plt.show()
    
    if n_clusters > 1:
        sil = silhouette_score(X_selected, cluster_labels)
        print("[Clustering] Silhouette Score:", sil)