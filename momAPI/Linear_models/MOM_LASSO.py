# Author : Charles Laroche
# Last update : 27/06/2019

from ..procedure.procedure_MOM import *
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator


class MomLasso(BaseEstimator):
    """
    Computation of the MOM version of the LASSO estimator given lamb k and.

    Parameters : - k : Number of blocks in which we split our database
                - lamb : penalization parameter in the lasso
                - iter_max : number of iteration in the gradient descent
                - hist : frequency of occurence in the median block in the n_hist last iteration

    """

    def __init__(self, k, lamb=1, iter_max=200):
        super(MomLasso, self).__init__()
        self.k = k
        self.iter_max = iter_max
        self.lamb = lamb
        self.hist = []
        self.t = None

    def fit(self, x, y, method="ADMM", step_size=0.0001, initialize="zero", n_hist=50):
        """
        Training phase.

        Parameters : - method : method to fit the estimator (ADMM, SUBGRAD : subgradient , ISTA and fISTA)
er(MomLasso, self).__init__()
        self.hist = []
                     - step_size : step_size of the gradient descent (not used in every method)

                     - initialize : initialisation of the coefficients (zero for a beta equal to 0 , ones for a beta
                      with all coefficients equal to 1 or random for random coefficients between (0,1))

                     - n_hist : the number of step we want to count the frequency of occurrence in the median block

        """

        self.hist = []
        n, p = np.shape(x)
        t = np.zeros(p)

        if initialize == "random":
            t = np.random.rand(p)

        if initialize == "ones":
            t = np.ones(p)

        if method == "ADMM":
            z = np.zeros(p)
            u = np.zeros(p)

            if initialize == "random":
                z = np.random.rand(p)
                u = np.random.rand(p)

            if initialize == "ones":
                z = np.ones(p)
                u = np.ones(p)

            rho = 5 * np.identity(p)
            for l in range(self.iter_max):
                k = mom(p_quadra(x, y, t), self.k)[1]
                if l > self.iter_max-n_hist:
                    self.hist += k.tolist()

                xk = x[k]
                yk = y[k]

                t = np.linalg.solve(xk.T @ xk + rho, xk.T @ yk + 5 * z - u)
                z = soft_thresholding(self.lamb / 5, t + u / 5)
                u = u + 5 * (t - z)

            self.t = t

        if method == "ISTA":
            mu = 0.9
            for l in range(self.iter_max):

                k = mom(p_quadra(x, y, t), self.k)[1]
                if l > self.iter_max-n_hist:
                    self.hist += k.tolist()

                xk = x[k]
                yk = y[k]

                # Beginning of backtracking with c = 1/2
                gamma_ = 1
                t_prev = t
                f = quadra_loss(xk, yk, t_prev)

                t = soft_thresholding(
                    self.lamb * gamma_, t - gamma_ * grad(xk, yk, t))
                delta = quadra_loss(xk, yk, t) - f - grad(xk, yk, t_prev).T * (
                    t - t_prev) - (1 / (2 * gamma_)) * np.linalg.norm(t - t_prev) ** 2

                while delta > 1e-3:
                    gamma_ *= mu
                    t = soft_thresholding(
                        self.lamb * gamma_, t_prev - gamma_ * grad(xk, yk, t_prev))
                    delta = quadra_loss(xk, yk, t) - f - grad(xk, yk, t_prev).T * (
                        t - t_prev) - (1 / (2 * gamma_))*np.linalg.norm(t - t_prev) ** 2

            self.t = t

        if method == "FISTA":
            z = t
            mu = 0.9
            for l in range(self.iter_max):
                k = mom(p_quadra(x, y, t), self.k)[1]
                if l > self.iter_max-n_hist:
                    self.hist += k.tolist()

                xk = x[k]
                yk = y[k]

                # Beginning of backtracking with c = 1/2
                gamma_ = 1
                t_prev = t
                f = quadra_loss(xk, yk, t_prev)

                t = soft_thresholding(self.lamb * gamma_, z - gamma_ * grad(xk, yk, z))
                delta = quadra_loss(xk, yk, t) - f - grad(xk, yk, t_prev).T * (
                    t - t_prev) - (1 / (2 * gamma_)) * np.linalg.norm(t - t_prev) ** 2

                while delta > 1e-3:
                    gamma_ *= mu
                    t = soft_thresholding(
                        self.lamb * gamma_, z - gamma_ * grad(xk, yk, z))
                    delta = quadra_loss(xk, yk, t) - f - grad(xk, yk, t_prev).T * (
                        t - t_prev) - (1 / (2 * gamma_)) * np.linalg.norm(t - t_prev) ** 2

                z = t + (l / (l + 3)) * (t - t_prev)

            self.t = t

        if method == "SUBGRAD":

            for l in range(self.iter_max):
                k = mom(p_quadra(x, y, t), self.k)[1]
                if l > self.iter_max-n_hist:
                    self.hist += k.tolist()

                xk = x[k]
                yk = y[k]

                t = t - step_size * \
                    subgrad(xk, yk, t, self.lamb) / np.sqrt(l + 1)

            self.t = t

    def predict(self, x):
        return x @ self.t

    def score(self, x, y):
        """ MSE """
        return mean_squared_error(self.predict(x), y)

    def coefs_(self):
        return list(np.array(self.t))
