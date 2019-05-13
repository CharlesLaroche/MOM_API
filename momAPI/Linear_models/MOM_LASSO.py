# Author : Charles Laroche
# Last update : 03/04/2019

from ..procedure.procedure_MOM import *
import numpy as np
from math import *
import time


class mom_lasso():

    """
    Computation of the MOM version of the LASSO estimator given lamb K and.

    """

    def __init__(self, K, lamb=1, iter_max=200):
        """
        K : Number of blocks in which we split our database 
        lamb : penalization parameter in the lasso
        iter_max : number of iteration in the gradient descent
        hist : frequency of occurence in the median block in the n_hist last iteration
        """
        self.hist = []
        self.params = {'K': K, 'iter_max': iter_max, 'lamb': lamb}

    def set_params(**params):
        """
        This function allows the user to change K , lamb , iter_max of the estimator

        """
        for key, item in params.item():
            try:
                self.params[key] = item
            except:
                raise Exception('{} not in params list'.format(key))

    def fit(self, X, Y, method="ADMM", step_size=0.0001, initialize="zero", n_hist=50):
        """
        Training phase.

        method : method to fit the estimator (ADMM, SUBGRAD : subgradient , ISTA and FISTA)

        step_size : step_size of the gradient descent (not used in every method)

        initialize : initialisation of the coefficients (zero for a beta equal to 0 , ones for a beta with all coefficients
        equal to 1 or random for random coefficients between (0,1))

        n_hist : the number of step we want to count the frequency of occurence in the median block

        """
        self.hist = []
        n, p = np.shape(X)
        j = n // self.params['K']

        if initialize == "zero":
            t = np.zeros(p)

        if initialize == "random":
            t = np.random.rand(p)

        if initialize == "ones":
            t = np.ones(p)

        if method == "ADMM":
            if initialize == "zero":
                z = np.zeros(p)
                u = np.zeros(p)

            if initialize == "random":
                z = np.random.rand(p)
                u = np.random.rand(p)

            if initialize == "ones":
                z = np.ones(p)
                u = np.ones(p)

            rhoM = 5 * np.identity(p)

            for l in range(self.params['iter_max']):

                k = MOM(P_quadra(X, Y, t), self.params['K'])[1]

                if l > self.params['iter_max']-n_hist:
                    self.hist += k.tolist()

                Xk = X[k]
                Yk = Y[k]

                t = np.linalg.solve((Xk.T) @ Xk + rhoM,
                                    (Xk.T) @ Yk + 5 * z - u)
                z = soft_thresholding(self.params["lamb"] / 5, t + u / 5)
                u = u + 5 * (t - z)

            self.t = t

        if method == "ISTA":
            mu = 0.9
            for l in range(self.params['iter_max']):

                k = MOM(P_quadra(X, Y, t), self.params['K'])[1]
                if l > self.params['iter_max']-n_hist:
                    self.hist += k.tolist()

                Xk = X[k]
                Yk = Y[k]

                # Beginning of backtracking with c = 1/2
                gamma = 1
                t_prev = t
                F = quadra_loss(Xk, Yk, t_prev)

                t = soft_thresholding(
                    self.params["lamb"] * gamma, t - gamma * grad(Xk, Yk, t))
                delta = quadra_loss(Xk, Yk, t) - F - grad(Xk, Yk, t_prev).T * (
                    t - t_prev) - (1 / (2 * gamma)) * np.linalg.norm(t - t_prev) ** 2

                while delta > 1e-3:
                    gamma *= mu
                    t = soft_thresholding(
                        self.params["lamb"] * gamma, t_prev - gamma * grad(Xk, Yk, t_prev))
                    delta = quadra_loss(Xk, Yk, t) - F - grad(Xk, Yk, t_prev).T * (
                        t - t_prev) - (1 / (2 * gamma))*np.linalg.norm(t - t_prev) ** 2

            self.t = t

        if method == "FISTA":
            mu = 0.9
            for l in range(self.params['iter_max']):

                k = MOM(P_quadra(X, Y, t), self.params['K'])[1]
                if l > self.params['iter_max']-n_hist:
                    self.hist += k.tolist()

                Xk = X[k]
                Yk = Y[k]

                # Beginning of backtracking with c = 1/2
                gamma = 1
                t_prev = t
                F = quadra_loss(Xk, Yk, t_prev)

                t = soft_thresholding(
                    self.params["lamb"] * gamma, z - gamma * grad(Xk, Yk, z))
                delta = quadra_loss(Xk, Yk, t) - F - grad(Xk, Yk, t_prev).T * (
                    t - t_prev) - (1 / (2 * gamma)) * np.linalg.norm(t - t_prev) ** 2

                while delta > 1e-3:
                    gamma *= mu
                    t = soft_thresholding(
                        self.params["lamb"] * gamma, z - gamma * grad(Xk, Yk, z))
                    delta = quadra_loss(Xk, Yk, t) - F - grad(Xk, Yk, t_prev).T * (
                        t - t_prev) - (1 / (2 * gamma)) * np.linalg.norm(t - t_prev) ** 2

                z = t + (l / (l + 3)) * (t - t_prev)

            self.t = t

        if method == "SUBGRAD":

            for l in range(self.params['iter_max']):
                k = MOM(P_quadra(X, Y, t), self.params['K'])[1]
                if l > self.params['iter_max']-n_hist:
                    self.hist += k.tolist()

                Xk = X[k]
                Yk = Y[k]

                t = t - step_size * \
                    subgrad(Xk, Yk, t, self.params["lamb"]) / np.sqrt(l + 1)

            self.t = t

    def predict(self, X):
        return X @ self.t

    def score(self, X, Y):
        """
        mean squared error
        """
        return quadra_loss(X, Y, self.t)

    def get_params(deep=False):
        return self.params

    def coefs(self):
        return list(np.array(self.t))
