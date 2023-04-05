# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic

from util import fill_with_avrege, convert_season

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

    # Fill the empty with the avrege
    train_df= fill_with_avrege(train_df)
    test_df = fill_with_avrege(test_df)

    # Convert season to number for the model
    # TODO: Do somethins with the seasons since it will ifluence it for sure
    train_df["season"] = train_df["season"].apply(convert_season)
    test_df["season"] = test_df["season"].apply(convert_season)

    y_train = train_df['price_CHF']
    X_train = train_df.drop(['price_CHF'], axis=1)
    X_test = test_df

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test

def modeling_and_prediction(X_train, y_train, X_test):
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
    #y_pred=np.zeros(X_test.shape[0])

    model = LinearRegression()
    model.fit(X_train, y_train)
    # Test the model on the fold

    kf = KFold(n_splits=10)
    for train, test in kf.split(X_train, y_train):  
        model.fit(X_train.iloc[train], y_train.iloc[train])
        y_pred = model.predict(X_train.iloc[test])
        print(f"R2 score for test set linear: {r2_score(y_train.iloc[test], y_pred)}")

    #gpr = GaussianProcessRegressor(kernel=DotProduct())
    #gpr.max_iter = 100
    #kf = KFold(n_splits=10)
    #for train, test in kf.split(X_train, y_train):  
    #    gpr.fit(X_train.iloc[train], y_train.iloc[train])
    #    y_pred = model.predict(X_train.iloc[test])
    #    print(f"R2 score for test set gpr: {r2_score(y_train.iloc[test], y_pred)}")

    y_pred = model.predict(X_test)
    
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
