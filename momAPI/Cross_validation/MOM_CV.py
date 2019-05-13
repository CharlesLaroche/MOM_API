# Author : Charles Laroche
# Last update : 03/04/2019

import numpy as np
import numpy.random as alea
import sklearn.metrics
from ..procedure.procedure_MOM import MOM

"""
Computation of the MOM version of cross_validation_V_fold giving a model with fit and a loss function

Instead of computing the whole error in each train/test set we compute the MOM of the error

model : model with fit and score function
X : the database
Y : the target
V : the number of train/test set to compute
K : the number of blocks in which we split our database to compute the error
random : true if we shuffle the data by default to false
"""


def cross_validation_V_fold(model, X, Y, V, K,
                            loss = sklearn.metrics.mean_squared_error,
                            random=False):

    n = len(Y)
    if random == True:
        idx = alea.permutation(n)
    else:
        idx = np.arange(n)
        
    score = []
    for i in range(V):
        X_train = np.concatenate(
            (X[idx[: i * (n // V)]], X[idx[(i + 1) * (n // V):]]))
        Y_train = np.concatenate(
            (Y[idx[: i * (n // V)]], Y[idx[(i + 1) * (n // V):]]))
        X_test = X[i * (n // V): (i + 1) * (n // V)]
        Y_test = Y[i * (n // V): (i + 1) * (n // V)]

        model.fit(X_train, Y_train)

        err = np.zeros(len(Y_test))
        Y_mod = model.predict(X_test)
        
        for p in range(len(Y_test)):
        
            err[p] = loss([Y_mod[p]] , [Y_test[p]])

        score.append(MOM(err, K)[0])

    return score
