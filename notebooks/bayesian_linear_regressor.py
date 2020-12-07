import numpy as np
import pandas as pd
import pymc3 as pm
from sklearn.base import BaseEstimator, RegressorMixin


class BayesianLinearRegression(BaseEstimator, RegressorMixin):

    def __init__(self):
        self.model = []
        self.trace_ = []
        self.mean_coeffs_ = []
        self.mae_ = 0
        self.rmse_ = 0

    # 'fit' to training data by sampling from mdoel with MCMC
    def fit(self, X, y, n_samples=1000, n_tune=500, n_cores=2):
        formula = "MedHouseVal ~ MedInc + AveOccup + AveBedrmsPerRoom +\
                   AveAddRooms + EstHouses + DistToCity"
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
        # calculate mae, rmse
        labels = np.asarray(y).reshape(-1)
        errors = preds - labels
        self.mae_ = np.mean(abs(errors))
        self.rmse_ = np.sqrt(np.mean(errors**2))
        return self.rmse_