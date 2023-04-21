import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PowerTransformer
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic

from util import convert_season
from plots import Plot
from data import Data
from constants import PRICES
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import numpy as np

import random
random.seed(1234)

def data_imputation(train_df, test_df):
    """
    """
    le = LabelEncoder()
    label = le.fit_transform(train_df['season'])
    label_test = le.fit_transform(test_df['season'])

    train_df['season'] = label
    test_df['season'] = label_test

    for item in train_df.columns:
        train_df[item] = train_df.groupby('season')[item].transform(lambda x: x.fillna(x.median()))
    for item in test_df.columns:
        test_df[item] = test_df.groupby('season')[item].transform(lambda x: x.fillna(x.median()))

    return train_df, test_df

def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("data/train.csv")
        
    # Load test data
    test_df = pd.read_csv("data/test.csv")

    # Drop the values that don't have CHF
    base = "price_CHF" #"price_CHF"
    train_df = train_df.dropna(subset=[base])

    # Convert season to number for the model
    train_df["season"] = train_df["season"].apply(convert_season)
    test_df["season"] = test_df["season"].apply(convert_season)

    data = Data(noise=0, C=1, gamma=1)
    train_df = data.fill_na_train(train_df)
    #test_df = data.fill_na_test(test_df)

    _, test_df = data_imputation(train_df, test_df)

    y_train = train_df['price_CHF']
    X_train = train_df.drop(['price_CHF'], axis=1)
    X_test = test_df

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test

def modeling_and_prediction(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame):
    """
    This function defines the model, fits training data and then does the prediction with the test data 

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """

    models = GaussianProcessRegressor()

    param_grid = [{
        "alpha": [1e-2, 1e-3],
        "kernel": [RBF(l) for l in np.logspace(-1, 1, 2)]
    }, {
        "alpha": [1e-2, 1e-3],
        "kernel": [DotProduct(sigma_0) for sigma_0 in np.logspace(-1, 1, 2)]
    }, {
        "alpha": [1e-2, 1e-3],
        "kernel": [Matern(length_scale=1.0, nu=1.5)]
    }]

    clf = GridSearchCV(models, param_grid)
    clf.fit(X_train, y_train)

    alpha = clf.best_params_['alpha']
    kernel = clf.best_params_['kernel']

    gpr = GaussianProcessRegressor(alpha=alpha, kernel=kernel)
    gpr.fit(X_train, y_train)

    y_pred = gpr.predict(X_test)
    
    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred=modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("Results file successfully generated!")
