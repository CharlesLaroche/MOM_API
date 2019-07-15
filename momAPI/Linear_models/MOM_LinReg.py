# Author : Charles Laroche
# Last update : 15/07/2019

from ..procedure.procedure_MOM import *
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator


class MomLinReg(BaseEstimator):
    """
    Computation of the MOM version of the LASSO estimator given lamb k and.

    Parameters : - k : Number of blocks in which we split our database
                - lamb : penalization parameter in the lasso
                - iter_max : number of iteration in the gradient descent
                - hist : frequency of occurence in the median block in the n_hist last iteration

    """

    def __init__(self, k, iter_max=200):
        super(MomLinReg, self).__init__()
        self.k = k
        self.iter_max = iter_max
        self.t = None
        self.hist = []

    def fit(self, x, y, step_size=0.0001, initialize="zero", n_hist=50):
        """
        Training phase.

        Parameters : - step_size : step_size of the gradient descent (not used in every method)

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

        for l in range(self.iter_max):
            k = mom(p_quadra(x, y, t), self.k)[1]
            if l > self.iter_max - n_hist:
                self.hist += k.tolist()

            xk = x[k]
            yk = y[k]

            t += (step_size / np.sqrt(l + 1)) * xk.T @ (yk - xk @ t)

        self.t = t

    def predict(self, x):
        return x @ self.t

    def score(self, x, y):
        """ MSE """
        return mean_squared_error(self.predict(x), y)

    def coefs_(self):
        return list(np.array(self.t))
