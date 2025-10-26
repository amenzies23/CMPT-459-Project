import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
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


@app.cell
def _():
    df = pd.read_csv("/home/isaac/dev/sfu/cmpt459/CMPT-459-Project/data/plant_health_data.csv")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df
    return (df,)


@app.cell
def _(df):
    y = df['Plant_Health_Status']
    X=  df.drop(columns='Plant_Health_Status')
    return X, y


@app.cell
def _(X, y):
    ids = X['Plant_ID'].unique()
    fig_cors = []
    for id in ids:
        x= X.loc[X['Plant_ID'] == id, :].drop(columns = "Plant_ID")
        fig_cors.append(sns.heatmap(x.corr(),annot=True,fmt='.2f'))
        plt.title(f"Feature Correlations for Plant {id}")
        plt.figure()
    plt.title(f"Mutual Information for Plant {id}")
    plant_mi = pd.DataFrame()
    for id in ids:
        mi = mutual_info_classif(x.drop(columns="Timestamp"),y.loc[X['Plant_ID'] == id])
        plant_mi[f'Plant_{id}'] = pd.Series(data=mi, index=x.drop(columns="Timestamp").columns)
    
    plant_mi
    return fig_cors, plant_mi


@app.cell
def _(plant_mi):
    sns.lineplot(plant_mi)
    plt.xticks(plant_mi.index,labels=plant_mi.index, rotation = 90)
    plt.title('Mutual Information with Features')
    return


@app.cell
def _(plant_mi):
    sns.lineplot(np.sqrt(plant_mi))
    plt.xticks(plant_mi.index,labels=plant_mi.index, rotation = 90)
    plt.title('SQRT(Mutual Information) with Features')
    return


@app.cell
def _(fig_cors):
    fig_cors
    return


if __name__ == "__main__":
    app.run()
