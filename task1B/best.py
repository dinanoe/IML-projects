"""
    Uses default methods, test all of them to get the best performing one
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso

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
    y_i = X.dot(w)
    n = np.shape(y)[0]
    for i in range(n):
        RMSE += pow(y[i] - y_i[i], 2)
    RMSE = pow((1 / n) * RMSE, 1/2)
    assert np.isscalar(RMSE)
    return RMSE

def transform_data(X: np.ndarray) -> np.ndarray:
    """
    This function transforms the 5 input features of matrix X (x_i denoting the i-th component of X) 
    into 21 new features phi(X) in the following manner:
    5 linear features: phi_1(X) = x_1, phi_2(X) = x_2, phi_3(X) = x_3, phi_4(X) = x_4, phi_5(X) = x_5
    5 quadratic features: phi_6(X) = x_1^2, phi_7(X) = x_2^2, phi_8(X) = x_3^2, phi_9(X) = x_4^2, phi_10(X) = x_5^2
    5 exponential features: phi_11(X) = exp(x_1), phi_12(X) = exp(x_2), phi_13(X) = exp(x_3), phi_14(X) = exp(x_4), phi_15(X) = exp(x_5)
    5 cosine features: phi_16(X) = cos(x_1), phi_17(X) = cos(x_2), phi_18(X) = cos(x_3), phi_19(X) = cos(x_4), phi_20(X) = cos(x_5)
    1 constant features: phi_21(X)=1

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features

    Returns
    ----------
    X_transformed: array of floats: dim = (700,21), transformed input with 21 features
    """
    X_transformed = np.zeros((700, 21))
    # Linear
    for i in range(5):
        X_transformed[:, i] = X[:, i]
    # Quadratic
    for i in range(5):
        X_transformed[:, i+5] = X[:, i]**2
    # Exponential
    for i in range(5):
        X_transformed[:, i+10] = np.exp(X[:, i])
    # Cosine
    for i in range(5):
        X_transformed[:, i+15] = np.cos(X[:, i])
    X_transformed[:, 20] = np.ones((700, ))
    assert X_transformed.shape == (700, 21)
    return X_transformed


def fit_with_function(X, y, Type, fold, λ = None) -> np.ndarray:
    best_rmse = np.inf
    best_w = None

    def fit_with_lambda(λ, train, test):
        nonlocal best_rmse, best_w
        if λ:
            model = Type(alpha=λ, max_iter=100000)
        else:
            model = Type()
        model.fit(X[train], y[train])
        rmse = calculate_RMSE(model.coef_, X[test], y[test])
        if best_rmse > rmse:
            best_rmse = rmse
            best_w = model.coef_

    def fit_with_fold(fold: int):
        nonlocal best_rmse, best_w
        kf = KFold(n_splits=fold)
        for train, test in kf.split(X):   
            if type(λ) == list:
                for i in np.arange(λ[0], λ[1], 0.001):   
                    fit_with_lambda(i, train, test)  
            else:
                fit_with_lambda(λ, train, test)  

    if type(fold) == list:
        for f in range(fold[0], fold[1]):
            fit_with_fold(f)
    else:
        fit_with_fold(fold)
    return best_w, best_rmse
        


def fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    This function receives training data points, transform them, and then fits the linear regression on this 
    transformed data. Finally, it outputs the weights of the fitted linear regression. 

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features
    y: array of floats, dim = (700,), input labels)

    Returns
    ----------
    w: array of floats: dim = (21,), optimal parameters of linear regression
    """
    X = transform_data(X)
    #w_linear, rmse_linear = fit_with_function(X, y, LinearRegression, [2, 50])
    #print("linear", rmse_linear)
    #w_ridge, rmse_ridge = fit_with_function(X, y, Ridge, 13, [10, 300])
    #print(w_ridge)
    #print("ridge", rmse_ridge)
    w_lasso, rmse_lasso = fit_with_function(X, y, Lasso, 11, 100)
    print("lasso", rmse_lasso)
    w_lasso, rmse_lasso = fit_with_function(X, y, Lasso, 11, 0.0001)
    print("lasso", rmse_lasso)
    w_lasso, rmse_lasso = fit_with_function(X, y, Lasso, 11, 0.00001)
    print("lasso", rmse_lasso)

    #exit()
    # TODO: Enter your code here
    assert w_lasso.shape == (21,)
    return w_lasso


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    # print a few data samples
    #print(data.head())

    X = data.to_numpy()
    # The function retrieving optimal LR parameters
    w = fit(X, y)
    # Save results in the required format
    np.savetxt("./results.csv", w, fmt="%.12f")