# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic

def data_imputation(train_df, test_df):

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
    """

    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    train_df = imputer.fit_transform(train_df)
    test_df = imputer.fit_transform(test_df)

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
    
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')
    
    # Load test data
    test_df = pd.read_csv("data/test.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))

    train_df = train_df.dropna(subset=['price_CHF'])

    train_df_imp, test_df_imp = data_imputation(train_df, test_df)

    X_train = train_df_imp.drop(['price_CHF'], axis = 1)
    y_train = train_df_imp['price_CHF']
    X_test = test_df_imp

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
    }, {
        "alpha": [1e-2, 1e-3],
        "kernel": [Matern(length_scale=1.0, nu=1.5)]
    }, ]

    gpr = GaussianProcessRegressor(kernel=Matern(length_scale=1.0, nu=1.5), alpha=0.01)
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
    print("\nResults file successfully generated!")

