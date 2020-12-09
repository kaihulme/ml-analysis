import numpy as np
import pandas as pd
import pymc3 as pm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score


class BayesianLinearRegression(BaseEstimator, RegressorMixin):

    def __init__(self, features=None, target=None, is_frame=True):
        """
        Arguments:
            - features: list of string type feature names
                        aligning with columns in X.
            - target: the string type target feature
                      aligning with the single column in y.
            - is_frame: indicates if fit will use Pandas
                        DataFrame, if false it must be
                        Numpy array and features and target
                        must be specified.
        """
        if not is_frame and not features.any() and not target:
            "ERROR: please specifiy feature and target names if using Numpy type array!"
            raise ValueError()
        self.features = features
        self.target = target
        self.is_frame = is_frame
        self.model = []
        self.trace = []
        self.mean_coeffs = []
        self.mse = 0
        self.rmse = 0
        self.r2 = 0

    # 'fit' to training data by sampling from mdoel with MCMC
    def fit(self, X, y, n_samples=5000, n_tune=2000, n_cores=8):
        # generate formula
        if (self.is_frame):
            formula = y.columns[0] + " ~ "
            for name in X.columns:
                formula += name + " + "
            formula = formula[:-3]
            data = pd.concat((y, X), axis=1)
        # if array, generate dataframe and formula from target and features
        else:
            columns = [self.target]
            formula = self.target + " ~ "
            for name in self.features:
                columns.append(name)
                formula += name + " + "
            formula = formula[:-3]
            data = np.column_stack((y, X))
            data = pd.DataFrame(data, columns=columns)
        with pm.Model() as self.model:
            # Normal prior
            family = pm.glm.families.Normal()
            # Create model from formula
            pm.GLM.from_formula(formula, data=data, family=family)
            self.trace = pm.sample(draws=n_samples, tune=n_tune,
                                   progressbar=True, cores=n_cores)
            return self

    # predict values for X
    def predict(self, X):
        # get mean of coefficients
        all_coeffs = np.asarray([self.trace[name] for name in self.trace.varnames])
        self.mean_coeffs = all_coeffs.mean(axis=1)
        # set intercept and variable coefficients
        n_coeffs = X.shape[1]
        coeffs = self.mean_coeffs[:n_coeffs+1]
        # get predictions
        X = np.column_stack((np.ones((X.shape[0])), X))
        preds = np.dot(coeffs, X.T)
        return preds

    # calculate mae and rmse for model on X, y
    def score(self, X, y):
        preds = self.predict(X)
        # calculate mse, rmse, r2
        self.mse = mean_squared_error(y, preds)
        self.rmse = mean_squared_error(y, preds, squared=False)
        self.r2 = r2_score(y, preds)
        return self.r2