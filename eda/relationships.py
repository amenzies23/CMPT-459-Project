import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
    from sklearn.pipeline import make_pipeline
    return (
        PCA,
        StandardScaler,
        TSNE,
        make_pipeline,
        mo,
        mutual_info_classif,
        np,
        pd,
        plt,
        sns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Read Dataset""")
    return


@app.cell
def _(pd):
    df = pd.read_csv("/home/isaac/dev/sfu/cmpt459/CMPT-459-Project/data/plant_health_data.csv")
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Extract Feature Columns""")
    return


@app.function
def get_features(df):
    return df.drop(columns = ["Timestamp", "Plant_ID","Plant_Health_Status"])


@app.cell
def _(df):
    features = get_features(df)
    features
    return (features,)


@app.cell
def _(df):
    y = df["Plant_Health_Status"]
    y.unique()
    return (y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Plot Feature Correlations""")
    return


@app.cell
def _(features, sns):

    matrix = sns.heatmap(features.corr(),annot=True,fmt=".2f")
    return (matrix,)


@app.cell
def _(matrix, mo):
    mo.md(f"""<center>{mo.as_html(matrix)}</center>""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Information Between Features and Classes""")
    return


@app.cell
def _(df, features, mutual_info_classif, pd):
    mi = mutual_info_classif(features, df["Plant_Health_Status"])
    mi = pd.Series(data=mi, index=features.columns)
    return (mi,)


@app.cell
def _(mi, plt):
    sorted = mi.sort_values(ascending = False)
    plt.plot(sorted)
    plt.xticks(sorted.index, sorted.index, rotation = 45)
    plt.ylabel("Mutual Information to Class Label")
    plt.xlabel("Feature")
    line = plt.title("Mutual Information of Features")
    return (line,)


@app.cell
def _(line, mo):
    mo.md(
        f"""
    <center>{mo.as_html(line)}</center>
    We see that temperature seems to have little to do with plant stress. However, soil moisture, PH, and Nitrogen are better predictors.  By far soil moisture is the strongest predictor.  Suggesting to hobby gardeners that good results can come just from making sure their plants are watered.  However, for more dedicated gardeners purchasing light intensity, PH and Nitrogen sensors seem to be best for monitoring plant stress.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Visualize Samples
    visualize with T-SNE and PCA to consider the non-linearity of the manifold
    """
    )
    return


@app.cell
def _(mo):
    s = mo.ui.slider(start=250,stop=500)
    s
    return (s,)


@app.cell
def _(PCA, StandardScaler, TSNE, make_pipeline, s):
    pca = make_pipeline(StandardScaler(), PCA(n_components = 2))
    tsne = make_pipeline(StandardScaler(),TSNE(n_components=2,max_iter=s.value))
    return pca, tsne


@app.cell
def _(features, pca):
    pcs = pca.fit_transform(features)
    return (pcs,)


@app.cell
def _(features, tsne):
    ts_pts = tsne.fit_transform(features)
    return (ts_pts,)


@app.cell
def _(np, pcs, pd, ts_pts, y):
    labelled_pcs = pd.DataFrame(data=np.column_stack([pcs,y]),columns=["PC1","PC2", "Stress Level"])
    labelled_ts = pd.DataFrame(data=np.column_stack([ts_pts,y]),columns=["T1","T2", "Stress Level"])
    return labelled_pcs, labelled_ts


@app.cell
def _(labelled_pcs, plt, sns):
    pca_plot = sns.scatterplot(labelled_pcs,x="PC1",y="PC2",hue="Stress Level")
    pca_plot = plt.title("PCA Visualization of Features")
    return (pca_plot,)


@app.cell
def _(labelled_ts, plt, sns):
    sns.scatterplot(labelled_ts,x="T1",y="T2",hue="Stress Level")
    tsne_plot = plt.title("T-SNE Visualization of Features")
    return (tsne_plot,)


@app.cell
def _(mo, pca_plot, tsne_plot):
    mo.md(
        f"""
    We visualize our features with both PCA and T-SNE to see how our features related to our class labels.  We see below that our classes do not seem to be easily seperated with linear or non-linear methods.  Possibly due to our many uninformative features.

    <center> {mo.as_html(pca_plot)} {mo.as_html(tsne_plot)} </center>
    """
    )
    return


if __name__ == "__main__":
    app.run()
