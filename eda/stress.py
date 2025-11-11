import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    import numpy as np
    import pandas as pd
    import pywt as pwt
    from relationships import get_features
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import seaborn as sns
    import matplotlib.pyplot as plt


@app.cell
def _():
    df = pd.read_csv("/home/isaac/dev/sfu/cmpt459/CMPT-459-Project/data/plant_health_data.csv")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index("Timestamp")
    return (df,)


@app.function
def standardize(df, index = None):
    x = StandardScaler().fit_transform(df)
    return pd.DataFrame(data=x, columns=df.columns, index=index)


@app.function
def extract_features(df):
    return (df.pipe(get_features).pipe(standardize, df['Timestamp']))


@app.cell
def _(df):
    X = extract_features(df.reset_index())
    X
    return


@app.cell
def _(df):
    y = df['Plant_Health_Status'].reset_index()
    y["Timestamp"] = y['Timestamp'].dt.floor("h")
    y
    return (y,)


@app.cell
def _(df):
    df['Plant_ID'].value_counts()
    return


@app.function
def proportion(group):
    group = group.drop(columns = 'Timestamp')
    return group.value_counts(normalize = True)


@app.cell
def _(y):
    stress_levels = y.groupby("Timestamp").apply(proportion).reset_index()
    stress_levels = stress_levels

    return (stress_levels,)


@app.cell
def _(stress_levels):
    pivoted = stress_levels.pivot(
        index='Timestamp',
        columns='Plant_Health_Status',
        values='proportion'
    ).fillna(0)
    order = ["Healthy","Moderate Stress", "High Stress"]
    daily = pivoted.resample("D").mean()[order]
    weekly = pivoted.resample("W").mean()[order]

    return daily, weekly


@app.cell
def _(daily):
    daily.plot(kind='bar', stacked=True, colormap='summer', figsize=(10,6))
    return


@app.cell
def _(weekly):
    weekly.plot(kind='bar', stacked=True, colormap='summer', figsize=(10,6))
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
