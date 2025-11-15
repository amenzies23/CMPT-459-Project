import pywt as pwt
from sklearn.decomposition import PCA
import pandas as pd
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


def feature_extraction(cleaned:pd.DataFrame,levels:int =3, method:str = 'db2',components:int = 10):
    pca = PCA(n_components = components)
    pcs = cleaned.pipe(wavelet_features).pipe(pca.fit_transform)
    return pd.DataFrame(data = pcs, index = cleaned.index, columns = [f"PC{i}" for i in range(components)])
