import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    app = mo.App(width="columns")

    import os, sys

    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd()))
    SRC_DIR = os.path.join(PROJECT_ROOT, "src")
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.ensemble import IsolationForest
    from sklearn.decomposition import PCA

    from preprocessing import preprocess
    from extraction import feature_extraction
    return (
        IsolationForest,
        PCA,
        feature_extraction,
        mo,
        pd,
        plt,
        preprocess,
        sns,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    # Isolation Forest Outlier Detection
    In this section I will apply **Isolation Forest** outlier detection to the plan health dataset. 
    ## Steps:
    1. Load the raw dataset
    2. Apply the same pre-procesising pipeline as the clustering stage (feature selection, normalization, and wavelet-based feature extraction)
    3. Fit an isolation forest model on the resulting PCA features
    4. Visualize the outliers in a 2D projection of the feature space
    """
    )
    return


@app.cell
def _(pd):
    df = pd.read_csv("../../data/plant_health_data.csv")
    df
    return (df,)


@app.cell
def _(df, preprocess):
    cleaned = preprocess(df, attr=[]) 
    cleaned
    return (cleaned,)


@app.cell
def _(cleaned, feature_extraction):
    X = feature_extraction(cleaned, components=10)
    X
    return (X,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Isolation Forest setup
    Isolation Forest is an ensemble of random trees trained to isolate individual points. This is used for outlier detection by recognizing points that can be isolated very quickly, and receive low anomoly scores. To accomplish this, I will:
    - Train the model on the PCA features extracted from the sensor data
    - Use a fixed random seed for reproducability
    - Let the model infer the proportion of outliers
    """
    )
    return


@app.cell
def _(IsolationForest, X):
    # Training the model
    iso = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42
    )
    iso.fit(X)

    # -1 = outlier, 1 = inlier
    preds = iso.predict(X)
    scores = iso.decision_function(X)
    return (preds,)


@app.cell
def _(X, preds):
    # Viewing the output
    X_out = X.copy()
    X_out["is_outlier"] = (preds == -1).astype(int)
    X_out
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Visualizing outliers in PCA space
    The scatter plot below shows the first two principal components of the feature space. The blue points correspond to inliers (normal sensor behavior), while the red points correpsond to the samples flagged as outliers by the Isolation Forest.
    """
    )
    return


@app.cell
def _(PCA, X, pd, plt, preds, sns):
    pca2 = PCA(n_components=2)
    pcs2 = pca2.fit_transform(X)

    df_plot = pd.DataFrame({
        "PC1": pcs2[:,0],
        "PC2": pcs2[:,1],
        "outlier": preds
    })

    plt.figure(figsize=(8,6))

    sns.scatterplot(
        data=df_plot,
        x="PC1", y="PC2",
        hue="outlier",
        palette={1:"blue", -1:"red"},
        alpha=0.6
    )

    plt.title("Isolation Forest Outliers in PCA Space")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""As we can see from the plot above, the outliers tend to lie near the outer edges of the main cluster of points, or in relatively sparse regions. The outliers do not form a single cluster, they are scattered around the outside which suggests they are ikely due to noisy / unreliable measurements. These points appear to be sparse and randomly spread throughout the plot. This suggests it may be ideal to remove the outliers for further classification.""")
    return


@app.cell
def _(mo):
    mo.md(r"""I will provide functions to be exposed here for running an isolation forest, and removing the outliers to be used in further classification tasks.""")
    return


@app.cell
def _(IsolationForest, pd):
    def isolation_forest_mask(
        X: pd.DataFrame,
        n_estimators: int = 200,
        contamination: str | float = "auto",
        random_state: int = 42,
    ):
        iso = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        iso.fit(X)

        # sklearn returns 1 for inliers, -1 for outliers
        preds = iso.predict(X)
        inlier_mask = (preds == 1)

        scores = iso.decision_function(X)
        return inlier_mask, scores, iso
    return


@app.cell
def _(pd):
    def remove_outliers(
        X: pd.DataFrame,
        inlier_mask,
        y: pd.Series | None = None,
    ):
        X_clean = X.loc[inlier_mask].copy()
        if y is None:
            return X_clean, None

        # Align labels by index so rows stay matched
        y_clean = y.loc[X_clean.index].copy()
        return X_clean, y_clean
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
