import marimo

__generated_with = "0.17.8"
app = marimo.App(width="columns")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    from preprocessing import preprocess, wavelet_features, get_labels
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from scipy.cluster import hierarchy as h
    from scipy.spatial.distance import pdist,squareform
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns


@app.cell(hide_code=True)
def _():
    mo.md(f"""
    # Plot Hierarchical Clustering of Dataset
    * Load Data
    * Preprocess and Apply Wavelet transform
    * Compute Distance Matrix
    * Visualize Distance matrix clusterplot
    * Show visualization of 2D point with highest cluster score
    """)
    return


@app.cell
def _():
    df = pd.read_csv("data/plant_health_data.csv")
    cleaned = preprocess(df,attr=[])
    cleaned
    return cleaned, df


@app.cell(hide_code=True)
def _():
    mo.md(f"""
    ## Feature Extraction
    We extract extra features using wavelets.  However, the wavelets introduce redundancy therefore we perform PCA to reduce correlation between features.
    """)
    return


@app.cell
def _(cleaned):
    pca = PCA(n_components=10)
    extracted = wavelet_features(cleaned)
    features = pca.fit_transform(extracted)
    X = pd.DataFrame(data=features, index=cleaned.index,columns=[f"PC{i}" for i in range(10)])
    X
    return (X,)


@app.cell
def _(X, df):
    points = X.copy()
    points['y'] = get_labels(df)
    sns.scatterplot(points, x='PC0',y='PC1', hue= 'y')
    return (points,)


@app.cell(column=1, hide_code=True)
def _():
    mo.md(f"""
    # Agglomerative Clustering
    There does seem to be some block structure to our dataset and features.  Now we show the distances with
    clustering. We show single link, ward, and average clustering results
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(f"""
    ### Single Linkage
    """)
    return


@app.cell
def _(X):
    sns.clustermap(X, method='single',cmap='mako',col_cluster=False)
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(f"""
    ### Average Linkage
    """)
    return


@app.cell
def _(X):
    sns.clustermap(X, method='average',cmap='mako',col_cluster=False)
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(f"""
    ### Complete/Farthest Linkage
    """)
    return


@app.cell
def _(X):
    sns.clustermap(X, method='complete',cmap='mako',col_cluster=False)
    plt.show()
    return


@app.cell(column=2)
def _(X):
    d_mat = pdist(X)
    sns.heatmap(squareform(d_mat),cmap='mako')
    plt.title("Distance between Plant Samples")
    return (d_mat,)


@app.cell
def _(d_mat):
    single = h.linkage(d_mat)
    average = h.linkage(d_mat, method='average')
    complete = h.linkage(d_mat, method='complete')
    return average, complete, single


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Single Link Dendogram
    """)
    return


@app.cell
def _(single):
    plt.figure(figsize=(50,8))
    h.dendrogram(single)
    plt.title("Single Link Dendogram")
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Average Link Dendogram
    """)
    return


@app.cell
def _(average):
    plt.figure(figsize=(50,8))
    h.dendrogram(average)
    plt.title("Average Link Dendogram")
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Complete Link Dendogram
    """)
    return


@app.cell
def _(complete):
    plt.figure(figsize=(50,8))
    h.dendrogram(complete)
    plt.title("Complete Link Dendogram")
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Of the three methods, the worst looking dendogram is single.  Most clusters are formed near the leaves of the tree, implying small clusters.  Complete and average linkage were able to find much larger clusters. This implies better performance.
    """)
    return


@app.cell(column=3, hide_code=True)
def _():
    mo.md("""
    ### Average Linkage
    """)
    return


@app.cell
def _(average, d_mat):
    sns.clustermap(
        squareform(d_mat),row_linkage=average,col_linkage=average,cmap = 'mako'
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(f"""
    ### Complete Linkage
    """)
    return


@app.cell
def _(complete, d_mat):
    sns.clustermap(
        squareform(d_mat),row_linkage=complete,col_linkage=complete,cmap = 'mako'
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Neither method produces strong block structures.  Suggesting there aren't sub regions of the distance matrix that are more similar.  The complete method creates slightly strong structure.  Therefore, I'll use that method to compute the best flat clustering
    """)
    return


@app.cell(column=4)
def _(X, complete):
    scores = []
    for t in range(2,20):
        labels = h.fcluster(complete, t = t,criterion='maxclust')
        score = silhouette_score(X, labels)
        scores.append(score)
    return (scores,)


@app.cell(hide_code=True)
def _(scores):
    plt.plot(np.arange(2,20),scores, '-o', c = 'orange')
    plt.title("Silhouette Score")
    plt.xlabel("# Clusters")
    plt.ylabel("Average Score")
    plt.grid()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Clusterings from 2-5 get worse and worse. However, there is a peak at 10.  Since low number of clusters are less interesting we chose 10 as the number of clusters.
    """)
    return


@app.cell
def _(complete, points):
    points['c'] = h.fcluster(complete, t = 10, criterion="maxclust")
    sns.scatterplot(points, x = 'PC0', y = 'PC1', hue = 'c', palette='tab20',style='y',alpha = 0.8)
    plt.legend(bbox_to_anchor = (1,1),frameon = False)
    return


@app.function
def hierarchical(X:pd.DataFrame, method = 'complete',n_clusters = 10):
    '''
        Perform Hierarchical Clustering on Dataset
        X: the extracted features from the dataset
        method: what method to use out of scipy linkages
        n_clusters: number of clusters for dataset
    '''
    d = pdist(X)
    tree = h.linkage(d, method=method)
    return h.fcluster(tree, t = n_clusters, criterion='maxclust')


if __name__ == "__main__":
    app.run()
