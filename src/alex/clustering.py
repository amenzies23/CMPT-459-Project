import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os, sys

    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd()))
    SRC_DIR = os.path.join(PROJECT_ROOT, "src")
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)

    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    from preprocessing import preprocess, get_labels
    from extraction import feature_extraction

    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    from sklearn.metrics import davies_bouldin_score
    return (
        GaussianMixture,
        davies_bouldin_score,
        feature_extraction,
        get_labels,
        mo,
        np,
        pd,
        plt,
        preprocess,
        silhouette_score,
        sns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## EM / Gaussian Mixture clustering on Plant Health Data

    Steps:
    1. Load the dataset and preprocess features
    2. Apply wavelet-based feature extraction + PCA
    3. Fit Gaussian Mixture Models for different numbers of clusters
    4. Select the best number of clusters using Silhoutte score
    5. Visualize the final clustering in PCA space
    """
    )
    return


@app.cell
def _(pd):
    df = pd.read_csv("data/plant_health_data.csv")
    df
    return (df,)


@app.cell
def _(df, preprocess):
    cleaned = preprocess(df, attr=[])
    cleaned
    return (cleaned,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Wavelet + PCA Feature Extraction

    We expand each time series feature with stationary wavelet transforms, then reduce redundancy with PCA.
    """
    )
    return


@app.cell
def _(cleaned, feature_extraction):
    X = feature_extraction(cleaned, components=10)
    X
    return (X,)


@app.cell
def _(X, df, get_labels, plt, sns):
    labels = get_labels(df)
    points = X.copy()
    points["y"] = labels.values.ravel()

    sns.scatterplot(points, x="PC0", y="PC1", hue="y", alpha=0.7)
    plt.title("PC0 vs PC1 colored by Plant_Health_Status")
    plt.legend(bbox_to_anchor=(1.05, 1), frameon=False)
    plt.show()
    return (points,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## EM / Gaussian Mixture Modeling

    We fit Gaussian Mixture Model with different numbers of components "k" and evaluate them using Davies Bouldin and Silhouette score
    """
    )
    return


@app.cell
def _(GaussianMixture, X, davies_bouldin_score, np, silhouette_score):
    Ks = list(range(2,21))
    db_scores = []
    sil_scores = []

    for k in Ks:
        # I tried many different params, but this gave the highest Silhouette score values out of all trials
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            init_params="kmeans",
            random_state=0
        )
        gmm.fit(X)

        labels_k = gmm.predict(X)
    
        db_scores.append(davies_bouldin_score(X, labels_k))

        if len(np.unique(labels_k)) > 1:
            sil = silhouette_score(X, labels_k)
        else:
            sil = np.nan
        sil_scores.append(sil)

    db_scores, sil_scores, Ks
    return Ks, db_scores, sil_scores


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Visualization of the Davies Bouldin and Silhouette Scores

    Below we can see the visualization of the scores we got for each k value. For Davies Bouldin, a smaller value is better, and for Silhouette a higher score is better
    """
    )
    return


@app.cell
def _(Ks, db_scores, plt, sil_scores):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # Davies Bouldin (lower is better)
    ax[0].plot(Ks, db_scores, "-o")
    ax[0].set_xlabel("Number of Clusters")
    ax[0].set_ylabel("Davies Bouldin")
    ax[0].set_title("GMM Davies Bouldin vs K")
    ax[0].grid(True)

    # Silhouette (higher is better)
    ax[1].plot(Ks, sil_scores, "-o")
    ax[1].set_xlabel("Number of Clusters")
    ax[1].set_ylabel("Silhouette score")
    ax[1].set_title("GMM Silhouette vs K")
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The above plots show the Davies Bouldin, and Silhouette score values we get from k=2 to k=20 clusters. Looking at the Silhouette scores, overall we can see very low scores. This indicates 

    For Davies Bouldin, a lower score is optimal. We see a global minimum at k=15, which is also a local maximum for the Silhouette scores. So I will visualize k=15 clusters. 
    """
    )
    return


@app.cell
def _():
    best_K=15
    return (best_K,)


@app.cell
def _(GaussianMixture, X, best_K):
    final_gmm = GaussianMixture(
        n_components=best_K,
        covariance_type="full",
        random_state=0
    )
    final_gmm.fit(X)
    cluster_labels = final_gmm.predict(X)
    cluster_labels, final_gmm
    return (cluster_labels,)


@app.cell
def _(cluster_labels, plt, points, sns):
    points_em = points.copy()
    points_em["cluster"] = cluster_labels

    sns.scatterplot(
        points_em,
        x="PC0",
        y="PC1",
        hue="cluster",
        style="y", 
        palette="tab20",
        alpha=0.8,
    )
    plt.title("EM / GMM Clusters in PCA Space")
    plt.legend(bbox_to_anchor=(1.05, 1), frameon=False)
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Overall, the visualization in PCA space shows substantial overlap between the clusters, and also between the plant stress labels. This indicates that the dataset does not form clearly separable groups when projects onto the first two principal components. 

    This is likely due to the nature of the data, coming from environmental sensor readings. These sensor readings tend to vary continuously, rather than form into clusters. Since PCA is a linear method, it also may not capture the nonlinear patterns that differentiate the plant stress levels.
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
