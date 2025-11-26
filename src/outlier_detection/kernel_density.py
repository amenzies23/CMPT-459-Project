import marimo

__generated_with = "0.16.5"
app = marimo.App(width="columns")

with app.setup:
    # Initialization code that runs before all other cells
    from preprocessing import preprocess, wavelet_features,get_labels,get_features
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
    mo.md(
        r"""
    # Explore the Impacts of Different Kernels
    Kernels are an affinity function, and there are many available in the [scikit-learn]("https://scikit-learn.org/stable/modules/metrics.html") package.  Below I will explore how they apply to the [plant health]("data/plant_health_dataset.csv") dataset
    """
    )
    return


@app.cell
def _():
    df = pd.read_csv("data/plant_health_data.csv")
    df
    return (df,)


@app.cell
def _(df):
    y = get_labels(df.set_index("Timestamp"))
    y
    return (y,)


@app.cell
def _():
    kernels = {
        "linear":pw.linear_kernel,
        "laplacian":pw.laplacian_kernel,
        "sigmoid":pw.sigmoid_kernel,
        "polynomial":pw.polynomial_kernel,
        "cosine":pw.cosine_similarity,
        "rbf":pw.rbf_kernel
    }
    kernel = mo.ui.dropdown(label="Choose Kernel",options=list(kernels.keys()),value='linear')
    kernel
    return kernel, kernels


@app.cell
def _(kernel, kernels):
    k = kernels[kernel.value]
    k
    return (k,)


@app.cell
def _(df):
    processed = preprocess(df,attr=[])
    X = wavelet_features(processed)
    X
    return (X,)


@app.cell(column=1)
def _():
    max_iter = mo.ui.slider(start=250 ,stop=1000)
    perp = mo.ui.slider(start = 5,stop=50)
    return max_iter, perp


@app.cell(hide_code=True)
def _(max_iter, perp):
    mo.md(
        f"""
    # Dimnensionality Reduction with T-SNE
    Max Iterations: {max_iter}\n
    Perplexity: {perp}
    """
    )
    return


@app.cell
def _():
    mo.md(f"""## Original Features""")
    return


@app.cell
def _(X, max_iter, perp):
    tsne = TSNE(init='random',n_components=2,max_iter=max_iter.value,perplexity=perp.value)
    points = tsne.fit_transform(X)
    points = pd.DataFrame(data=points,index=X.index,columns=["T1","T2"])
    points
    return (points,)


@app.cell
def _(X, k, max_iter, perp):
    a = TSNE(init='random',n_components=2,max_iter=max_iter.value,perplexity=perp.value, metric="precomputed")
    A = k(X)
    A = A.max() - A
    affinity = a.fit_transform(A)
    affinity = pd.DataFrame(data=affinity,index=X.index,columns=["T1","T2"])
    affinity
    return (affinity,)


@app.cell(column=2, hide_code=True)
def _():
    mo.md(
        f"""
    # Below is a Scatterplot of our Data Points
    we notice how the points are overlaid and hard to distinguish. We will attemp to use kernels to make this 
    data more easily seperable
    """
    )
    return


@app.cell
def _(points, y):
    points['y' ] = y
    sns.scatterplot(points,x = "T1", y="T2", hue = 'y',palette=['red','orange','green'], alpha = 0.5)
    plt.legend(bbox_to_anchor = (0.95,1.1), ncols = 3,frameon = False)
    return


@app.cell(hide_code=True)
def _(kernel):
    mo.md(f"""# T-SNE visualization using the {kernel.value} Kernel""")
    return


@app.cell
def _(affinity, y):
    affinity['y'] = y
    sns.scatterplot(affinity,x = "T1", y="T2", hue = 'y',palette=['red','orange','green'], alpha = 0.5)
    plt.legend(bbox_to_anchor = (0.95,1.1), ncols = 3,frameon = False)
    return


@app.cell(column=3)
def _():
    options = mo.ui.dropdown(label="kernel",options=['gaussian', 'tophat', 'epanechnikov', 'exponential','linear', 'cosine'
    ],value='tophat')
    bandwidth = mo.ui.slider(0,1,0.1,label="Bandwidth")
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
def _(bandwidth, df, options):
    kde = make_pipeline(
        PCA(n_components=10),
        KernelDensity(kernel=options.value,bandwidth=bandwidth.value)
    )
    kde.fit(get_features(df))
    return (kde,)


@app.cell
def _(df, kde, points):
    scores = kde.score_samples(get_features(df))
    points['os'] = scores
    sns.histplot(points, x = 'os',hue = 'y')
    return


@app.cell
def _(points):
    sns.scatterplot(
        points,x = "T1", y="T2", hue = 'y',palette=['red','orange','green']
        ,size='os',sizes=[20,200], alpha = 0.5
    )
    plt.legend(bbox_to_anchor = (1.05,1), ncols = 1,frameon = False)
    return


if __name__ == "__main__":
    app.run()
