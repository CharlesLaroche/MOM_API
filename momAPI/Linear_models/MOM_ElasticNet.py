# Author : Corentin Jaumin
# Last update : 27/06/2019

from ..procedure.procedure_MOM import *
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator


class MomElasticNet(BaseEstimator):
    """ Class for MOM adaptation of ElasticNet estimator.

        Parameters : - rho/lamb : penalization term (lamb * [rho * |β|l1 + (1 - rho) * |β|l2²])
                     - k : number of blocks for the MOM estimator
                     - max_iter : number of iteration in the gradient descent
                     - tol : tolerance parameter
    """
    
    def __init__(self, rho=1.0, lamb=0, k=1, max_iter=50, tol=10e-5):
        super(MomElasticNet, self).__init__()
        self.rho = rho
        self.lamb = lamb
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.hist = []
        self.loss = None
        self.w = None
        self.intercept = None
    
    def fit(self, x, y, coef_init=None):
        """" Coordinate gradient descent for MOM ElasticNet. """

        self.hist = []
        n_samples, n_features = x.shape
        
        if not coef_init:
            w = np.ones(n_features)

        else:
            w = coef_init 
        
        x_corrected = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
        y_mean = np.mean(y)
        
        loss = np.empty(0)
        var_w = self.tol + 1
        n_iter = 0
        x_mean = 0
        x_std = 1

        while n_iter < self.max_iter and var_w > self.tol:
            
            res = y - y_mean - x_corrected @ w
            median_indexes = mom(np.square(res), self.k)[1]

            if n_iter >= 10:
                self.hist += median_indexes.tolist()
        
            x_mean = np.mean(x[median_indexes], axis=0)
            x_std = np.std(x[median_indexes], axis=0)
            x_corrected = (x - x_mean) / x_std
            y_mean = np.mean(y[median_indexes])
            res = y - y_mean - x_corrected @ w

            x_train = x_corrected[median_indexes]
            r_train = res[median_indexes]
            loss = np.append(loss, np.mean(np.square(r_train)))
            r_train_adj = 0
        
            var_w = 0
        
            for j in range(n_features):
                w_j = w[j]
                if j == 0:
                    r_train_adj = r_train + w_j * x_train[:, j]

                else:
                    r_train_adj = r_train_adj + w_j * x_train[:, j] - w[j-1] * x_train[:, (j - 1)]
                w[j] = scalar_soft_thresholding(self.rho * self.lamb, np.mean(r_train_adj * x_train[:, j])) / \
                    (1 + self.lamb * (1 - self.rho))
                var_w += (w_j - w[j]) ** 2
                        
            n_iter += 1
        
        self.loss = loss
        self.w = w / x_std
        self.intercept = y_mean - np.sum(self.w * x_mean)

    def predict(self, x):
        return self.intercept + x @ self.w

    def score(self, x, y):
        """ MSE """
        return mean_squared_error(self.predict(x), y)
    
    def coefs_(self):
        return list(self.w)


class MomElasticNetCV:
    """ Cross validation for MOM ElasticNet.

        Parameters : - rhos : list of rhos to test
                     - lambs : list of lamb to test
                     - k_list : list of K to test
                     - v : V in cross validation V fold

                     """

    def __init__(self, rhos=1., lambs=0, k_list=1, v=5, max_iter=20, tol=10e-5):
        self.V = v
        self.max_iter = max_iter
        self.tol = tol
        self.rho_list = list(rhos)
        self.lamb_list = list(lambs)
        self.k_list = list(k_list)
        self.loss = None
        self.hyper_params = None
        self.best_k = None
        self.best_lamb = None
        self.best_rho = None
        self.w = None
        self.intercept = None

    def fit(self, x, y):
        
        loss = []
        hyper_params = []
        
        if self.V == 1:
            raise Exception('Cross-Validation cannot be done with V = 1 fold')
        
        for k in self.k_list:
            for rho in self.rho_list:
                for lamb in self.lamb_list:

                    loss = cross_validation_v_fold(MomElasticNet(rho, lamb, k, max_iter=self.max_iter,
                                                                 tol=self.tol), x, y, self.V, k, True)
                    loss.append(np.mean(loss))
                    hyper_params.append([k, lamb, rho])
                    
        index = np.argmin(loss)[0]
                
        self.best_k, self.best_lamb, self.best_rho = hyper_params[index]
        self.loss, self.hyper_params = loss, hyper_params
        
        model = MomElasticNet(self.best_rho, self.best_lamb, self.best_k, max_iter=self.max_iter, tol=self.tol)
        model.fit(x, y)
        self.w, self.intercept = model.w, model.intercept

    def coefs_(self):
        return list(self.w)


def cross_validation_v_fold(model, x, y, v, k, random=False):

    n = len(y)
    
    if random:
        idx = alea.permutation(n)
        
    else:
        idx = np.arange(n)
    
    score = []

    for i in range(v):
        
        x_train = np.concatenate((x[idx[: i * (n // v)]], x[idx[(i + 1) * (n // v):]]))
        y_train = np.concatenate((y[idx[: i * (n // v)]], y[idx[(i + 1) * (n // v):]]))
        x_test = x[i * (n // v): (i + 1) * (n // v)]
        y_test = y[i * (n // v): (i + 1) * (n // v)]
        
        model.fit(x_train, y_train)
        
        err = np.zeros(len(y_test))
        
        for p in range(len(y_test)):
            err[p] = model.score([x_test[p]], [y_test[p]])
        
        score.append(mom(err, k)[0])
        
    return score
