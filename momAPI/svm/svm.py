import numpy as np
from sklearn.base import BaseEstimator
from .svmSGD import mom_svm_sgd


class MomSvm(BaseEstimator):

    def __init__(self, x, y, k, tol, max_iter, use_kernel=None, pen=0, kerparam=1, stepsize=0.1):
        super(MomSvm, self).__init__()
        self.kerparam = kerparam
        self.use_kernel = use_kernel
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.pen = pen
        self.x = x
        self.y = y
        self.stepsize = stepsize
        self.hist = []
        self.mom_loss = None
        self.loss = None
        self.kernel = None
        self.intercept = None
        self.w = None

    def fit(self):
        dic = mom_svm_sgd(self.x, self.y, self.k, self.use_kernel, self.kerparam,
                          self.max_iter, self.tol, self.stepsize, self.pen)

        self.intercept, self.w = dic['parameters']
        self.kernel = dic['kernel']
        self.loss = dic['loss']
        self.mom_loss = dic['mom_loss']
        self.hist = dic['mom_hist']

    def predict(self, x_new):
        if not self.use_kernel:
            return np.sign(x_new @ self.w + self.intercept)

        else:
            alpha = np.concatenate((self.intercept.reshape(1), self.w))
            y_new = np.zeros(len(x_new))
            for i in range(len(x_new)):
                y_new[i] = np.sum([alpha[j] * self.use_kernel(x_new[i], self.x[j], self.kerparam)
                                   for j in range(len(self.x))])
            return np.sign(y_new)

    def score(self, x_test, y_test):
        """ Computation of accuracy """
        y_pred = self.predict(x_test)
        return 1 - np.sum(np.abs(y_pred - y_test)) / (2 * len(y_test))
