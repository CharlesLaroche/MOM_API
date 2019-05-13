# Author : Corentin Jaumin
# Last update : 03/04/2019

import numpy as np
from sklearn.preprocessing import normalize
from ..procedure.procedure_MOM import *


class mom_elasticnet():

    def __init__(self, rho=1., Lambda=0, K=1, random_state=True, max_iter=1000, tol=10e-5):
        self.rho = rho
        self.Lambda = Lambda
        self.K = K
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.intercept = None
        self.Loss = None

    def fit(self, X, y, bias=True, coef_init=None):

        self.bias = bias
        self.hist = []
        n_samples, n_features = X.shape

        if coef_init == None:
            w = np.ones((n_features))

        elif w.shape[0] != n_features:
            raise Error(
                'Shape of coef array does not match number of features')

        else:
            w = coef_init

        block_size = n_samples // self.K

        X_corrected = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        Residuals = y - X_corrected @ w

        Loss = np.empty(0)
        var_w = self.tol + 1
        n_iter = 0

        while n_iter < self.max_iter and var_w > self.tol:

            median_indexes = MOM(np.square(Residuals), self.K)[1]
            self.hist += median_indexes.tolist()

            X_mean = np.mean(X[median_indexes], axis=0)
            X_std = np.std(X[median_indexes], axis=0)
            X_corrected = (X - X_mean)/X_std
            Residuals = y - X_corrected @ w

            X_train = X_corrected[median_indexes]
            R_train = Residuals[median_indexes]
            Loss = np.append(Loss, 0.5 * np.mean(np.square(R_train)))

            var_w = 0

            for j in range(n_features):
                w_j = w[j]
                R_train = R_train + w_j * X_train[:, j]
                w[j] = scalar_soft_thresholding(
                    self.rho * self.Lambda, np.mean(R_train * X_train[:, j]))/(1 + self.Lambda * (1 - self.rho))
                var_w += (w_j - w[j])**2

            n_iter += 1

        y_mean = np.mean(y[median_indexes])
        self.Loss = Loss
        self.w = w / X_std
        self.intercept = y_mean - np.sum(self.w * X_mean)

    def predict(self, X):

        return(self.intercept + np.dot(X, self.w))

    def score(self, X, y):
        """
        R2 computation
        """
        y_predict = self.intercept + X @ self.w
        y_mean = np.mean(y)
        return(1-(np.sum(np.square(y-y_predict)))/(np.sum(np.square(y-y_mean))))
