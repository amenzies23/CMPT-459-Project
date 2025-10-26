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


@app.cell(hide_code=True)
def _():
    mo.md(
        f"""
    # Extract Features
    * Drop Timestamp (too informative), Plant Id (too informative) and  Plant Health Status (predictor)
    * Standardize Data
    * Add Index and Column names again
    """
    )
    return


@app.function
def standardize(df, index = None):
    x = StandardScaler().fit_transform(df)
    return pd.DataFrame(data=x, columns=df.columns, index=index)


@app.function
def extract_features(df):
    return (df.pipe(get_features).pipe(standardize, df['Timestamp']))


@app.cell(hide_code=True)
def _():
    mo.md(
        f"""
    # Extracted Features
    * Index has been set to timestamp
    """
    )
    return


@app.cell
def _(df):
    X = extract_features(df.reset_index())
    X
    return (X,)


@app.cell(hide_code=True)
def _():
    levels = mo.ui.dropdown(options=list(range(10)), value=3)
    mo.md(f'''
        # Wavelet Transform
        Helps to view different frequencies of signal.  Adds one column per frequency level\n
        Number of Frequency Levels to view {levels}
    ''')
    return (levels,)


@app.function
def wavelet(col, level = 3, method = 'db2'):
    '''
        Compute Stationary Wavelets for A Given Column
        Params: 
        * col: pd.Series column to perform DWT on
        * level: int number of resolution to look at
        * method: str method to use
    '''
    waves = pwt.swt(col, wavelet = method, level = level)
    cols = pd.DataFrame(index=col.index)
    for l in range(level):
        cols[f'{col.name}_a_{l}'] = waves[l][0]
        cols[f'{col.name}_d_{l}'] = waves[l][0]
    return cols


@app.function
def wavelet_features(df, levels = 3, method = 'db2'):
    '''
        Applies wavelet function column wise.  Output is axpanded into a larger dataframe
        * Pandas Dataframe with all numeric columns
        * level: int number of resolution to look at
        * method: str method to use
    '''
    wave_dfs = [wavelet(df[col],level=levels, method=method) for col in df.columns]
    return pd.concat(wave_dfs, axis=1)


@app.cell
def _(X, levels):

    w = wavelet_features(X, levels= levels.value)
    w
    return (w,)


@app.cell(hide_code=True)
def _():
    mo.md(
        f"""
    # Visualization
    View line plot per-plant and per-feature.  A moving average has been applied to show longer term trends.  you can choose which plant to plot from the dataset.  For each plant a specific approximation level is plotted
    """
    )
    return


@app.cell
def _(levels):
    window_size = mo.ui.slider(start=1, stop=100)
    view_plant = mo.ui.dropdown(options=list(range(1,11)), value=1)
    level2view = mo.ui.dropdown(options=list(range(levels.value)), value=0)
    mo.md(f'''
    Moving Average Window Size {window_size} \n
    Plant to Plot: {view_plant} \n
    Frequency Level (0-{levels.value}): {level2view}
    ''')
    return level2view, view_plant, window_size


@app.cell
def _(df, level2view, view_plant, w):
    coi = w.filter(regex=f"a_{level2view.value}")
    coi = coi.loc[df['Plant_ID'] == view_plant.value]
    return (coi,)


@app.cell
def _():
    sns.set_theme(
        context='notebook',
        style='whitegrid',
        palette='muted'
    )
    return


@app.cell
def _(coi, window_size):
    low = coi.rolling(window = window_size.value).mean()
    return (low,)


@app.cell(hide_code=True)
def _(low, view_plant, window_size):
    sns.lineplot(low,dashes=False)
    plt.xticks(rotation = 45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.title(f"Feature Approximation Wavelets\n (Moving Average Window = {window_size.value}, Plant = {view_plant.value})")
    return


@app.cell
def _(coi, level2view):
    plt.figure(figsize=(8,8))
    sns.heatmap(coi.corr(),annot = True, fmt=".2f")
    plt.title(f"Correlation of features for Level {level2view.value}")
    return


@app.cell
def _():
    mo.md(f"""
    # Reduce Dimensionality
    use PCA and T-SNE to visualize the relationships between each example for the given level
    """)
    return


@app.cell
def _(coi, df):
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2)
    pcs = pd.DataFrame(data=pca.fit_transform(coi), index=coi.index, columns=['PC1','PC2'])
    pcs['y'] = df['Plant_Health_Status']
    tss = pd.DataFrame(tsne.fit_transform(coi), index=coi.index, columns=['T1', 'T2'])
    tss['y'] = df['Plant_Health_Status']
    return pcs, tss


@app.cell
def _():
    return


@app.cell
def _(level2view, pcs):
    sns.scatterplot(pcs,x = 'PC1', y = 'PC2',hue='y')
    plt.title(f"PCA Plot of Features for Level {level2view.value}")
    return


@app.cell
def _(level2view, tss):
    sns.scatterplot(tss,x='T1',y='T2',hue='y')
    plt.title(f"TSNE Plot of Features for Level {level2view.value}")
    return


if __name__ == "__main__":
    app.run()
