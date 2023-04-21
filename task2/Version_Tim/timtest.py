import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

train_df = pd.read_csv("data/train.csv")

test_df = pd.read_csv("data/test.csv")

train_df = train_df.dropna(subset='price_CHF')

le = LabelEncoder()
label = le.fit_transform(train_df['season'])
train_df['season'] = label

for item in train_df.columns:
    train_df[item] = train_df.groupby('season')[item].transform(lambda x: x.fillna(x.median()))

train_df.to_csv('data/new_train.csv')
