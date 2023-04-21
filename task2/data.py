import numpy as np
import pandas as pd
from sklearn.svm import SVR
from constants import PRICES

class Data:
    def __init__(self, gamma=0.1, C=1, noise=0) -> None:
        self.gamma = gamma
        self.C = C
        self.noise = noise
        self.models = {}

    def fill_na_train(self, df: pd.DataFrame, target="price_CHF") -> pd.DataFrame:
        for price in PRICES: 
            if price == target:
                continue
            svr = SVR(kernel="rbf", gamma=self.gamma, C=self.C)
            missing_price = df[price].isnull()

            # Fit the model on the cuttent price
            df_model = df.loc[~missing_price, [price, target]].copy()
            X = df_model[target].copy().T.values.reshape(-1, 1)
            y = df_model[price].T.values.reshape(-1, 1)
            svr.fit(X, y.ravel())
            self.models[price] = svr

            # Update the values with the prediction of the model and the noise
            df_missing = df.loc[missing_price, [price, target]]
            predicted_values = svr.predict(df_missing[target].T.values.reshape(-1, 1))
            random_noise = np.random.uniform(-self.noise, self.noise, size=len(predicted_values))
            df_missing[price] = predicted_values + random_noise
            
            df.update(df_missing[price])
        return df

    def fill_na_test(self, df: pd.DataFrame) -> pd.DataFrame:
        for price in PRICES: 
            missing_price = df[price].isnull()
            # Update the values with the prediction of the model and the noise
            df_missing = df.loc[missing_price, [price]]
            predicted_values = self.models[price].predict(df_missing["price_CHF"].T.values.reshape(-1, 1))
            random_noise = np.random.uniform(-self.noise, self.noise, size=len(predicted_values))
            df_missing[price] = predicted_values +random_noise
            df.update(df_missing[price])
        return df