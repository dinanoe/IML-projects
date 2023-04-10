import numpy as np
import pandas as pd
from constants import SEASONS

def fill_with_avrege(df: pd.DataFrame):
    """
        Fill all the row that have NaN with the avrege of with the seasonal avrage
    """
    for season in SEASONS:
        filt = (df["season"] == season)
        df_season = df[filt].copy()
        for col in df_season:
            if df[col].dtype == np.float64:
                avrege = df_season[col].median()
                #print(avrege)
                #print(f"avrege on {season} for price in {col[6:]}: {avrege}")
                df_season[col].fillna(avrege, inplace=True)
        df.update(df_season) 
    return df

def drop_empty(df: pd.DataFrame):
    """
        Remove all row that have a missing value
    """
    return df.dropna()

def convert_season(season):
    """
        Convert season to the respective index
    """
    return SEASONS.index(season) 
