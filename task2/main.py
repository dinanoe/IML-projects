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

import random
random.seed(1234)

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
    train_df = train_df.dropna(subset=['price_CHF'])

    # Convert season to number for the model
    train_df["season"] = train_df["season"].apply(convert_season)
    test_df["season"] = test_df["season"].apply(convert_season)

    #test_df = fill_linear(test_df)   

    data = Data(noise=0.4)
    #train_df = data.fill_na_train(train_df)
    #test_df = data.fill_na_test(test_df)

    plt = Plot()
    # target = "price_SVK"
    # was_na_svw = test_df[target].isna().copy()
    for price in PRICES:
        plt.add(test_df["price_AUS"], test_df[price], label=f"AUS/{price[-3:]}") #color=(1.0, 0.8, 0.0)
    # plt.add(test_df.loc[was_na_svw, [target]][target], test_df.loc[was_na_svw, ["price_CHF"]]["price_CHF"], label="Missing", color="red")
    # plt.save()
    plt.show() 

    exit()

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
    #         "feature_names_in_": ["season", "price_AUS", "price_CZE", "price_GER", "price_ESP", "price_FRA", "price_UK", "price_ITA", "price_POL", "price_SVK"]

    plt = Plot()
    plt.add(X_train["price_GER"], y_train)
    #plt.add(X_train["price_CZE"], y_train)
    #plt.add(X_train["price_ESP"], y_train)
    #plt.add(X_train["price_FRA"], y_train)
    #plt.add(X_train["price_UK"], y_train)
    #plt.add(X_train["price_ITA"], y_train)
    #plt.add(X_train["price_POL"], y_train)
    #plt.add(X_train["price_SVK"], y_train)
    plt.show()

    param_grid = {
        'alpha': [10],
        'kernel': [DotProduct()],  #, RBF(), Matern(, ), RationalQuadratic()],
        "n_restarts_optimizer": [0], 
    }

    model = GaussianProcessRegressor()
    grid_search = GridSearchCV(model, param_grid, cv=8, scoring="r2")

    pt = PowerTransformer(method='yeo-johnson', standardize=False)


    X_train_trasormed = pt.fit_transform(X_train)
    # Remove feature names from X_train_transformed

    grid_search.fit(X_train_trasormed, y_train.values)

    print("Best parameters: ", grid_search.best_params_)
    print("Best estimator: ", grid_search.best_estimator_)
    print("Best score: ", grid_search.best_score_)

    best_model = grid_search.best_estimator_

    # pt = PowerTransformer(method='yeo-johnson', standardize=False)
    # X_test_trasormed = pt.fit_transform(X_test)
    y_pred = best_model.predict(X_test)

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
