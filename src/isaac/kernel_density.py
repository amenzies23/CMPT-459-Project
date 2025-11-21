import marimo

__generated_with = "0.17.8"
app = marimo.App(width="columns")

with app.setup:
    # Initialization code that runs before all other cells
    from preprocessing import preprocess, get_labels
    from extraction import feature_extraction
    import pandas as pd
    import numpy as np
    from sklearn.metrics import pairwise as pw
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KernelDensity
    from sklearn.pipeline import make_pipeline
    import marimo as mo
    import seaborn as sns
    import matplotlib.pyplot as plt


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Explore the Impacts of Different Kernels
    Kernels are an affinity function, and there are many available in the [scikit-learn]("https://scikit-learn.org/stable/modules/metrics.html") package.  Below I will explore how they apply to the [plant health]("data/plant_health_dataset.csv") dataset
    """)
    return


@app.cell
def _():
    df = pd.read_csv("data/plant_health_data.csv")
    df
    return (df,)


@app.cell
def _(df):
    y = get_labels(df)
    y
    return


@app.cell
def _(df):
    processed = preprocess(df,attr=[])
    X = feature_extraction(processed)
    X
    return (X,)


@app.cell(column=1)
def _():
    options = mo.ui.dropdown(label="kernel",options=['gaussian', 'tophat', 'epanechnikov', 'exponential','linear', 'cosine'
    ],value='tophat')
    bandwidth = mo.ui.slider(0.1,1,0.1,label="Bandwidth")
    return bandwidth, options


@app.cell
def _(bandwidth, options):
    mo.md(f"""
    # Density Estimation for Outlier Detection
    {options}
    {bandwidth}
    """)
    return


@app.cell
def _(X, bandwidth, options):
    kde = make_pipeline(
        PCA(n_components=10),
        KernelDensity(kernel=options.value,bandwidth=bandwidth.value)
    )
    kde.fit(X)
    return (kde,)


@app.cell
def _(X, kde):
    scores = kde.score_samples(X)

    sns.histplot(scores)
    plt.title("Distribution of KDE")
    plt.xlabel("KDE")
    return (scores,)


@app.cell
def _(scores):
    z = abs((scores - scores.mean()) / scores.std())
    sns.histplot(abs(z))
    plt.title("Distribution of KDE")
    plt.xlabel("KDE")
    return (z,)


@app.cell
def _(scores, z):
    scores_df = pd.DataFrame(data=np.column_stack([scores,z]),columns=['Score','Standardized'])
    scores_df
    return (scores_df,)


@app.cell
def _(scores_df):
    mo.md(f"Number of outliers: {(scores_df['Standardized'] > 3).sum()}")
    return


@app.function(column=2)
def kde_outliers(X, n_comps:int = 5, kernel = 'gaussian', bandwidth:float = 0.8, t:float = 3):
    """
        Calculate Outliers using KDE
        First dimensionality is reduced using PCA
        then kernel density estimation is performed
        finally ouput is standardized to make a binary mask
        returns (mask, standardized scores, raw scores)
    """
    estimator = make_pipeline(PCA(n_components=n_comps),KernelDensity(kernel=kernel,bandwidth=bandwidth))
    estimator.fit(X)
    scores = estimator.score_samples(X)
    z = abs((scores - scores.mean()) / scores.std())
    return z > t, z, scores


@app.cell
def _(X):
    outliers, s, k = kde_outliers(X,n_comps=10)
    outliers.sum()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
