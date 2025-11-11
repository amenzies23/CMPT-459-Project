# Isaac von Riedemann
# 301423851
# Each function should operate on a dataframe and any params
# All functions should be chained together using pipe
import marimo as mo
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pywt as pwt
'''
    Get Class Labels from Dataset (y)
'''
def get_labels(df):
    return df.loc[:,["Plant_Health_Status"]]

'''
    Get All Features from the dataset (X)
'''
def get_features(df):
    return df.drop(columns = ["Timestamp", "Plant_ID","Plant_Health_Status"])

'''
    Normalize all features for preparing for model
'''
def normalize(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(data=scaled,index=df.index, columns=df.columns)


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


def wavelet_features(df, levels = 3, method = 'db2'):
    '''
        Applies wavelet function column wise.  Output is axpanded into a larger dataframe
        * Pandas Dataframe with all numeric columns
        * level: int number of resolution to look at
        * method: str method to use
    '''
    wave_dfs = [wavelet(df[col],level=levels, method=method) for col in df.columns]
    return pd.concat(wave_dfs, axis=1)


'''
    Extract Informative features for Model
'''
def extract_features(df, attr:list = []):
    assert len(attr) > 0
    return df.loc[:, attr]

'''
    Preprocessing recipe to use
'''
def preprocess(df, attr:list = ["Soil_Moisture","Nitrogen_Level","Soil_pH"]):
    return (
        df
        .pipe(get_features)
        .pipe(extract_features,attr = attr)
        .pipe(normalize)
    )


if __name__ == "__main__":
    df = pd.read_csv("../data/plant_health_data.csv")
    model_ready = preprocess(df)
    print(model_ready.head())