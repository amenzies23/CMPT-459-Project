import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.svm import SVC
    from preprocessing import preprocess, get_labels
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import marimo as mo
    return get_labels, mo, mutual_info_classif, pd, plt, preprocess, sns


@app.cell
def _(get_labels, pd, preprocess):
    df = pd.read_csv("data/train_data.csv", index_col=0)
    X = preprocess(df, attr=[])
    y = get_labels(df)
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Feature Selection
    We want to discover which features have relationships with the ground truth.  We use mutual information to do so.  Mutual information quantifies how much knowing the value of one columns lets you know the value of another.  In this case higher is better.
    """
    )
    return


@app.cell(hide_code=True)
def _(X, mutual_info_classif, pd, plt, sns, y):
    scores = mutual_info_classif(X,y.values.flatten())
    scores = pd.Series(index=X.columns, data=scores)
    sns.lineplot(
        scores.sort_values(ascending=False),
    )
    sns.despine()
    plt.grid(True)
    plt.xticks(rotation = 90)
    plt.ylabel("Mutual Information")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Soil Moisture has a high mutual information with the class labels.  But almost none of the other features correspond with the class labels.  Since We only have 10 features we choose to just keep them all.  """)
    return


if __name__ == "__main__":
    app.run()
