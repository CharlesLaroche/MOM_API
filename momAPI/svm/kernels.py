import numpy as np


def polynomial(x, y, kerparam):
    return (1 + x @ y) ** kerparam


def gaussian(x, y, kerparam):
    return np.exp(-np.sum(np.square(y - x)) / (2 * kerparam))
