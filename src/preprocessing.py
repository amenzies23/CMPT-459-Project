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
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df.set_index('Timestamp').loc[:,["Plant_Health_Status"]]

'''
    Get All Features from the dataset (X)
'''
def get_features(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index("Timestamp")
    return df.drop(columns = [ "Plant_ID","Plant_Health_Status"])

'''
    Normalize all features for preparing for model
'''
def normalize(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(data=scaled,index=df.index, columns=df.columns)


'''
    Extract Informative features for Model
'''
def extract_features(df, attr:list = []):
    if len(attr) == 0: return df
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
