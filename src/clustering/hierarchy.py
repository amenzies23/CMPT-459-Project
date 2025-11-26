import marimo

__generated_with = "0.16.5"
app = marimo.App(width="columns")

with app.setup:
    # Initialization code that runs before all other cells
    from preprocessing import preprocess, wavelet_features,get_labels,get_features
    import pandas as pd
    import numpy as np
    from scipy.cluster import hierarchy as h
    from scipy.spatial.distance import pdist
    from sklearn.decomposition import PCA
    import marimo as mo
    import seaborn as sns
    import matplotlib.pyplot as plt


@app.cell
def _():
    df = pd.read_csv("data/plant_health_data.csv")
    df
    return (df,)


@app.cell
def _(df):
    processed = preprocess(df,attr=[])
    X = wavelet_features(processed)
    X
    return (X,)


@app.cell(column=1)
def _(X):
    d_mat = pdist(X,metric = 'euclidean')
    d_mat
    return (d_mat,)


@app.cell
def _(d_mat):
    clustering = h.linkage(d_mat)
    return (clustering,)


@app.cell
def _(X, clustering):
    # plt.figure(figsize=(10,15))
    sns.clustermap(
        X,row_linkage=clustering, col_cluster=False
    )
    # plt.colorbar(orientation = 'horizontal',anchor = (1,1))
    plt.show()
    return


@app.cell
def _(clustering):
    _=h.dendrogram(clustering, orientation='right')
    plt.show()
    return


@app.cell
def _(X, clustering):
    f = h.fcluster(clustering,criterion='maxclust',t = 25)
    pca = PCA(n_components=2)
    x = pca.fit_transform(X)
    return f, x


@app.cell
def _(f):
    np.unique(f)
    return


@app.cell(column=2)
def _(f, x):
    plt.scatter(x[:,0],x[:,1],c=f,alpha = 0.5,cmap='tab20')
    return


if __name__ == "__main__":
    app.run()
