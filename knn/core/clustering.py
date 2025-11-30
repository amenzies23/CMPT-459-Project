import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def find_best_dbscan_params(
    X: np.ndarray,
    eps_values,
    min_samples_values
):
    """
    Search over eps and min_samples to find the best DBSCAN configuration 
    using silhouette score as the evaluation metric.
    
    - Iterates through all param combinations
    - Skips invalid runs (clusters < 2)
    - Tracks and prints silhouette scores
    - Generates a scatter plot of parameter performance
    - Returns: (best_eps, best_min_samples), best_score
    """
    best_score = -1
    best_params = None

    # Collect all results for plotting
    results = []

    # For each eps value and min samples...
    for eps in eps_values:
        for min_samples_value in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=min_samples_value)
            labels = db.fit_predict(X)
            n_clusters = count_clusters(labels)

            # Skip when only one or less cluster found.
            if n_clusters < 2:
                results.append((eps, min_samples_value, n_clusters, None))
                continue

            try:
                score = silhouette_score(X, labels)
            except Exception:
                results.append((eps, min_samples_value, n_clusters, None))
                continue

            # Store the result for plotting later.
            results.append((eps, min_samples_value, n_clusters, score))

            print(f"eps={eps}, min_samples={min_samples_value} -> clusters={n_clusters}, silhouette={score:.4f}")

            # Keep track of the parameters with best score.
            if score > best_score:
                best_score = score
                best_params = (eps, min_samples_value)

    if best_params:
        print(f"Best eps={best_params[0]}, min_samples={best_params[1]}, silhouette={best_score:.4f}")
    else:
        print("No valid parameter combination found.")

    # Plot the parameter
    plot_dbscan_param_scatter(results)

    return best_params, best_score

def perform_clustering(
    X_selected: np.ndarray,
    eps_values=np.linspace(0.3, 2.5, 15),
    min_samples_values=range(3, 20)
):
    """
    - Perform grid search to find best eps and min_samples.
    - Run DBSCAN using the best parameters (or fallback defaults).
    - Report number of clusters and silhouette score.
    - Visualize by reducing data to 2D with PCA.
    - Plot clusters (noise: label = -1).
    """
    best_params, best_score = find_best_dbscan_params(
        X_selected, eps_values, min_samples_values
    )

    if best_params is None:
        print("[Clustering] Could not find valid parameters. Falling back to defaults.")
        eps = 1.5
        min_samples = 5
    else:
        eps, min_samples = best_params
        print(f"[Clustering] Using best parameters: eps={eps}, min_samples={min_samples}")

    # Run DBSCAN with chosen parameters
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X_selected)
    n_clusters = count_clusters(cluster_labels)

    print(f"[Clustering] DBSCAN found {n_clusters} clusters.")
    if n_clusters > 1:
        print("[Clustering] Silhouette Score:", silhouette_score(X_selected, cluster_labels))

    # Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_selected)

    # -1 is treated as noise by DBSCAN.
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1],
        hue=cluster_labels, palette="tab10", s=50
    )
    plt.title(f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})")
    plt.savefig("./plots/perform_clustering.png", dpi=300)
    plt.show()

def plot_dbscan_param_scatter(results, out_path="./plots/dbscan_param_scatter.png"):
    """
    Scatter plot showing how DBSCAN parameters (eps, min_samples)
    affect clustering quality. Used to visually inspect good parameter regions

    - Each point: one (eps, min_samples) combination
    - Color: silhouette score (or gray if invalid)
    """
    eps = [result[0] for result in results]
    min_samples = [result[1] for result in results]
    cluster_counts = [result[2] for result in results]
    
    silhouette_vals = [
        result[3] if result[3] is not None else -1  # Invalid cases -> gray
        for result in results
    ]

    plt.figure(figsize=(9, 7))
    sc = plt.scatter(
        eps,
        min_samples,
        c=silhouette_vals,
        cmap="viridis",
        s=120,
        edgecolor="black"
    )

    plt.colorbar(sc, label="Silhouette Score")
    plt.xlabel("eps")
    plt.ylabel("min_samples")
    plt.title("DBSCAN Parameter Search\n(eps vs min_samples colored by silhouette score)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()

def count_clusters(labels):
    """
    Return number of unique clusters, 
    ignoring noise label (-1).
    """
    return len(set(labels)) - (1 if -1 in labels else 0)