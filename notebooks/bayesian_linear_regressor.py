import numpy as np
import pandas as pd
import pymc3 as pm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score


class BayesianLinearRegression(BaseEstimator, RegressorMixin):

    def __init__(self):
        self.model = []
        self.trace_ = []
        self.mean_coeffs_ = []
        self.mse_ = 0
        self.rmse_ = 0
        self.r2_ = 0

    # 'fit' to training data by sampling from mdoel with MCMC
    def fit(self, X, y, formula="null", n_samples=1000, n_tune=500, n_cores=2):
        if (formula == "null"):
            formula = y.columns[0] + " ~ "
            for name in X.columns:
                formula += name + " + "
            formula = formula[:-3]
        print(f"Sampling with formula:\n{formula}\n")
        data = pd.concat((y, X), axis=1)
        with pm.Model() as self.model_:
            # Normal prior
            family = pm.glm.families.Normal()
            # Create model from formula
            pm.GLM.from_formula(formula, data=data, family=family)
            self.trace_ = pm.sample(draws=n_samples, tune=n_tune,
                                    progressbar=True, cores=n_cores)
            return self

    # predict values for X
    def predict(self, X):
        # get mean of coefficients
        all_coeffs = np.asarray([self.trace_[name] for name in self.trace_.varnames])
        self.mean_coeffs_ = all_coeffs.mean(axis=1)
        # set intercept and variable coefficients
        n_coeffs = X.shape[1]
        coeffs = self.mean_coeffs_[:n_coeffs+1]
        # get predictions
        X = np.column_stack((np.ones((X.shape[0])), X))
        preds = np.dot(coeffs, X.T)
        return preds

    # calculate mae and rmse for model on X, y
    def score(self, X, y):
        preds = self.predict(X)
        # calculate mse, rmse, r2
        self.mse_ = mean_squared_error(y, preds)
        self.rmse_ = mean_squared_error(y, preds, squared=False)
        self.r2_ = r2_score(y, preds)
#         labels = np.asarray(y).reshape(-1)
#         errors = preds - labels
#         self.mae_ = np.mean(abs(errors))
#         self.rmse_ = np.sqrt(np.mean(errors**2))
        return self.r2_