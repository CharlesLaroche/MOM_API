# Author : Charles Laroche
# Last update : 03/04/2019

import numpy as np
from ..procedure.procedure_MOM import MOM, grad


class mom_matching_pursuit():

    def __init__(self, K):

        self.params = {'K': K}

    def fit(self, X, Y, m=-1, step_size=0.01, iter_max=200):

        n, p = np.shape(X)

        if m == -1:
            m = p

        mu = np.zeros((n, 1))
        beta = np.zeros((m + 1, p))
        A = []
        A_C = list(range(p))
        R = Y

        for l in range(m):

            beta_l = beta[l]

            # Block selection
            k = MOM(np.square(R), self.params['K'])[1]
            Xk = X[k]
            Rk = R[k]

            c = Xk.T @ Rk

            # Selection of the variable most correlated with the residual
            j = np.argmax(abs(c[A_C]))
            j = A_C[j]
            A.append(j)
            A_C.remove(j)

            # Gradient descent
            beta_l = np.zeros(l + 1)

            for i in range(iter_max):

                # Block selection
                k = MOM(np.square(R), self.params['K'])[1]
                Xk = X[k]
                Yk = Y[k]

                beta_l = beta_l - (step_size / np.sqrt(i + 1)
                                   ) * grad(Xk[:, A], Yk, beta_l)

                R = Y - X[:, A] @ beta_l

            beta[l + 1][A] = beta_l.reshape((1, l + 1))

        self.var = A
        self.beta = beta
        self.coefs = beta[-1]

    def score(self, X, Y):
        return quadra_loss(X, Y, self.beta)

    def set_params(self):
        pass
