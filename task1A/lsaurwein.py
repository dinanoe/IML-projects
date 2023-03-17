# This serves as a template which will guide you through the implementation of this task. It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


def fit(X: np.ndarray, y: np.ndarray, λ: float) -> np.ndarray:
    """
    This function receives training data points, then fits the ridge regression on this data
    with regularization hyperparameter lambda. The weights w of the fitted ridge regression
    are returned. 

    Parameters
    ----------
    X: matrix of floats, dim = (135,13), inputs with 13 features
    y: array of floats, dim = (135,), input labels)
    lam: float. lambda parameter, used in regularization term

    Returns
    ----------
    w: array of floats: dim = (13,), optimal parameters of ridge regression
    """
    I = np.eye(X.shape[1])
    w = np.linalg.inv(X.T.dot(X) + λ*I).dot(X.T.dot(y))
    assert w.shape == (13,)
    return w


def calculate_RMSE(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """This function takes test data points (X and y), and computes the empirical RMSE of 
    predicting y from X using a linear model with weights w. 

    Parameters
    ----------
    w: array of floats: dim = (13,), optimal parameters of ridge regression 
    X: matrix of floats, dim = (15,13), inputs with 13 features
    y: array of floats, dim = (15,), input labels

    Returns
    ----------
    RMSE: float: dim = 1, RMSE value
    """
    RMSE = 0
    y_i = X.dot(w) # This sould be X.T for the linear model but the dimensions don't match
    print(y_i)
    n = np.shape(y)[0]
    for i in range(n):
        RMSE += pow(y[i] - y_i[i], 2)
    RMSE = pow((1 / n) * RMSE, 1/2)
    assert np.isscalar(RMSE)
    return RMSE


def average_LR_RMSE(X: np.ndarray, y: np.ndarray, lambdas: list[float], n_folds: int):
    """
    Main cross-validation loop, implementing 10-fold CV. In every iteration (for every train-test split), the RMSE for every lambda is calculated, 
    and then averaged over iterations.
    
    Parameters
    ---------- 
    X: matrix of floats, dim = (150, 13), inputs with 13 features
    y: array of floats, dim = (150, ), input labels
    lambdas: list of floats, len = 5, values of lambda for which ridge regression is fitted and RMSE estimated
    n_folds: int, number of folds (pieces in which we split the dataset), parameter K in KFold CV
    
    Returns
    ----------
    avg_RMSE: array of floats: dim = (5,), average RMSE value for every lambda
    """
    RMSE_mat = np.zeros((n_folds, len(lambdas)))
   
    for i, λ in enumerate(lambdas):
        # Split the dataset
        kf = KFold(n_splits=n_folds)
        k = 0
        for train, test in kf.split(X):
            w = fit(X[train], y[train], λ)
            RMSE_mat[k][i] = calculate_RMSE(w, X[test], y[test])
            k += 1
    avg_RMSE = np.mean(RMSE_mat, axis=0)
    assert avg_RMSE.shape == (5,)
    return avg_RMSE


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns="y")
    # print a few data samples
    print(data.head())

    X = data.to_numpy()
    # The function calculating the average RMSE
    lambdas = [0.1, 1, 10, 100, 200]
    n_folds = 10
    avg_RMSE = average_LR_RMSE(X, y, lambdas, n_folds)
    # Save results in the required format
    np.savetxt("./results.csv", avg_RMSE, fmt="%.12f")
