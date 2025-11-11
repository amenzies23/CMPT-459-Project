import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd

    data = pd.read_csv("./data/plant_health_data.csv")

    # A quick overview of the dataset.
    #print(data.head())
    print(data.info())
    #print(data.describe())
    return data, pd


@app.cell
def _(data):
    # Drop and see if there is a significant change to the dataset
    data.dropna()

    # I don't see any drops in `count`, seems like there is no missing values in this dataset.
    print(data.describe())
    return


@app.cell
def _():
    # Categorical -> Numerical 
    # I don't see any categorical, so we don't need to convert them into numerical.
    # KNN only works with numerical data because it relies on distance metrics (i.e., Euclidean, Manhattan, etc.)
    return


@app.cell
def _(data):
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8,6))
    sns.scatterplot(
        data=data,
        x="Light_Intensity",
        y="Ambient_Temperature",
        hue="Plant_Health_Status",
        palette="Set2",
        s=80
    )
    plt.title("Plant Health Clusters")
    plt.xlabel("Light Intensity")
    plt.ylabel("Ambient Temperature")
    plt.legend(title="Plant Health Status")
    plt.grid(True)
    plt.show()
    return plt, sns


@app.cell
def _(data, plt, sns):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # Separate numeric columns
    numeric_cols = data.select_dtypes(include="number").columns
    X = data[numeric_cols]

    # Scale and reduce
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Store results in new columns
    data["PCA1"] = X_pca[:, 0]
    data["PCA2"] = X_pca[:, 1]

    # Plot PCA result
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=data,
        x="PCA1",
        y="PCA2",
        hue="Plant_Health_Status",
        palette="deep",
    )
    plt.title("Plant Health Clusters (PCA Projection)")
    plt.legend(title="Health Status")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # T-SNE for visualization.
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    # Store t-SNE results
    data["TSNE1"] = X_tsne[:, 0]
    data["TSNE2"] = X_tsne[:, 1]

    # Plot t-SNE result
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=data,
        x="TSNE1",
        y="TSNE2",
        hue="Plant_Health_Status",
        palette="deep",
        s=80,
        alpha=0.9
    )
    plt.title("Plant Health Clusters (t-SNE Projection)")
    plt.legend(title="Health Status")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return (X_scaled,)


@app.cell
def _(pd, plt, sns):
    import numpy as np

    data_2 = pd.read_csv("./data/plant_health_data.csv")

    # Drop 'plant_id' if it exists
    if "Plant_ID" in data_2.columns:
        data_2 = data_2.drop(columns=["Plant_ID"])

    # Select only numeric columns
    numeric_cols_2 = data_2.select_dtypes(include="number")
    corr_2 = numeric_cols_2.corr(method="spearman")

    # Mask the diagonal
    mask = np.eye(len(corr_2), dtype=bool)

    # Plot correlation heatmap with default cmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        corr_2,
        mask=mask,
        cmap="coolwarm",     # standard diverging palette
        center=0,
        annot=True,          # show correlation values
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        square=True
    )

    # Overlay light diagonal cells
    for i in range(len(corr_2)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color="#f7f7f7", lw=0))

    plt.title("Feature Correlation Cluster (Plant Health Data)", fontsize=14)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(X_scaled, data, plt, sns):
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    data["Cluster"] = clusters

    plt.figure(figsize=(8,6))
    sns.scatterplot(
        data=data,
        x="PCA1",
        y="PCA2",
        hue="Cluster",
        palette="viridis",
        s=80
    )
    sns.scatterplot(
        x=kmeans.cluster_centers_[:,0],
        y=kmeans.cluster_centers_[:,1],
        color="red",
        marker="X",
        s=120,
        label="Centroids"
    )
    plt.title("K-Means Clusters on Plant Health Data (PCA)")
    plt.legend()
    plt.show()
    return


@app.cell
def _():
    # Feature Selection
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=5)
    # X_train_pca = pca.fit_transform(X_train)
    # X_test_pca = pca.transform(X_test)
    return


@app.cell
def _():
    # Normalization
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    return


if __name__ == "__main__":
    app.run()
