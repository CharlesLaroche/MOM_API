# Author : Charles Laroche
# Last update : 03/04/2019

import numpy as np
from ..procedure.procedure_MOM import mom, grad
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator


class MomMatchingPursuit(BaseEstimator):
    """ Implementation of MOM adaptation of matching pursuit.

        Parameters : - k : number of blocks for the MOM estimator

    """

    def __init__(self, k):
        super(MomMatchingPursuit, self).__init__()
        self.k = k
        self.var = None
        self.beta = None
        self.coefs = None

    def fit(self, x, y, m=-1, step_size=0.01, iter_max=200):
        """ Gradient descent for matching pursuit.

            Parameters : - m : Number of features to use (default = -1 = all)
                         - step_size : step size of the gradient descent
                         - iter_max : number of iterations in the gradient descent """

        n, p = np.shape(x)
        if m == -1:
            m = p

        beta = np.zeros((m + 1, p))
        a = []
        a_c = list(range(p))
        r = y

        for l in range(m):
            # Block selection
            k = mom(np.square(r), self.k)[1]
            xk = x[k]
            rk = r[k]

            c = xk.T @ rk

            # Selection of the variable most correlated with the residual
            j = a_c[np.argmax(abs(c[a_c]))]
            j = a_c[j]
            a.append(j)
            a_c.remove(j)

            # Gradient descent
            beta_l = np.zeros(l + 1)

            for i in range(iter_max):

                # Block selection
                k = mom(np.square(r), self.k)[1]
                xk = x[k]
                yk = y[k]

                beta_l -= (step_size / np.sqrt(i + 1)) * grad(xk[:, a], yk, beta_l)

                r = y - x[:, a] @ beta_l

            beta[l + 1][a] = beta_l.reshape((1, l + 1))

        self.var = a
        self.beta = beta

    def predict(self, x):
        return x @ self.coefs

    def score(self, x, y):
        return mean_squared_error(self.predict(x), y)

    def coefs_(self):
        return list(self.beta[-1])
