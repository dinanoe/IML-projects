# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

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
    ln = LinearRegression(X, y)
    w = ln.fit_linear()
    ridge = ln.fit_ridge([100, 10000], [5, 15])

    #exit()
    # TODO: Enter your code here
    assert w.shape == (21,)
    return ridge

class LinearRegression:
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y

    def fit_linear(self) -> np.ndarray:
        """
            Fit the model with the basic linear regression and return the weights
        """
        return np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T.dot(self.y))

    def fit_ridge(self, lambdas: list[float], n_folds: list[int]) -> np.ndarray:
        """
            Fit the model using the ridge of task 1
        """
        fit = lambda X, y, λ : np.linalg.inv(X.T.dot(X) + λ*np.eye(X.shape[1])).dot(X.T.dot(y))

        best_rmse = 100
        best_w = None 
        best_λ = None
        best_fold = None
        for λ in np.arange(lambdas[0], lambdas[1], 10):
            print(f"lambda: {λ}", end="\r")
            for fold in range(n_folds[0], n_folds[1]):
                kf = KFold(n_splits=fold)
                for train, test in kf.split(X):
                    w = fit(self.X[train], self.y[train], λ)
                    rmse = calculate_RMSE(w, self.X[test], self.y[test])
                    if (best_rmse > rmse):
                        best_rmse = rmse
                        best_w = w
                        best_λ = λ
                        best_fold = fold
        print(best_rmse)
        print(best_λ)
        print(best_fold)
        return best_w

    def fit_lasso(self, interations: int, learning_rate, l1_penality) -> np.ndarray:
        """
            Return the wrigts using lasso
            https://www.geeksforgeeks.org/implementation-of-lasso-regression-from-scratch-using-python/
        """
        m, n = self.X.shape

        def update_weights(w, b) :
            y_pred = self.X.dot(w) + b
            # calculate gradients  
            dW = np.zeros(n)
            for j in range(n) :
                if w[j] > 0 :
                    dW[j] = ( - ( 2 * ( self.X[:, j] ).dot( self.y - y_pred ) )  + l1_penality ) / m
                else :
                    dW[j] = ( - ( 2 * ( self.X[:, j] ).dot( self.y - y_pred ) ) - l1_penality ) / m
            db = - 2 * np.sum(y - y_pred ) / m 
            w = w - learning_rate * dW
            b = b - learning_rate * db
            return w, b
        
        w = np.zeros(n)
        b = 0

        for i in range(interations) :    
            w, b = update_weights(w , b)

        return w

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    # print a few data samples
    print(data.head())

    X = data.to_numpy()
    # The function retrieving optimal LR parameters
    w = fit(X, y)
    # Save results in the required format
    np.savetxt("./results.csv", w, fmt="%.12f")
