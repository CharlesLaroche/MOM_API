# Author : Charles Laroche
# # Last update : 27/06/2019

import numpy as np
import numpy.random as alea
from sklearn.metrics import mean_squared_error
from ..procedure.procedure_MOM import mom


def cross_validation_v_fold(model, x, y, v, k, loss=mean_squared_error, random=False):
    """
    Computation of the MOM version of cross_validation_V_fold giving a model with fit and a loss function

    Instead of computing the whole error in each train/test set we compute the MOM of the error

    Parameters : - model : model with fit and score function
                 - x : the database
                 - y : the target
                 - v : the number of train/test set to compute
                 - k : the number of blocks in which we split our database to compute the error
                 - random : true if we shuffle the data by default to false

    """

    n = len(y)
    if random:
        idx = alea.permutation(n)
    else:
        idx = np.arange(n)
        
    score = []
    for i in range(v):
        x_train = np.concatenate(
            (x[idx[: i * (n // v)]], x[idx[(i + 1) * (n // v):]]))
        y_train = np.concatenate((y[idx[: i * (n // v)]], y[idx[(i + 1) * (n // v):]]))
        x_test = x[i * (n // v): (i + 1) * (n // v)]
        y_test = y[i * (n // v): (i + 1) * (n // v)]

        model.fit(x_train, y_train)

        y_mod = model.predict(x_test)
        err = loss(y_mod, y_test, multioutput='raw_values')

        score.append(mom(err, k)[0])

    return score
