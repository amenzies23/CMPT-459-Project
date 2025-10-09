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
    return mo, mutual_info_classif, pd, plt, sns


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
    mo.md(
        f"""
            # Feature Correlations
            {mo.as_html(matrix)}
        """
    )
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
        {mo.as_html(line)}
        We see that temperature seems to have little to do with plant stress where soil moisture, PH, and Nitrogen are better predictors.  By far soil moisture is the strongest predictor.  Suggesting to hobby gardeners that good results can come just from making sure their plants are watered.  However, for more dedicated gardeners purchasing light intensity, PH and Nitrogen sensors seem to be best for lowering plant stress.  
        """
    )
    return


if __name__ == "__main__":
    app.run()
