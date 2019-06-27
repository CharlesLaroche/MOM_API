# Author : Charles Laroche
# Last update : 27/06/2019

import numpy as np
import numpy.random as alea
from math import *


def mom(x, k):
    """
    x: np.array
    k: int
    return: Median of means of the k blocks with the indexes of the values inside the median block ((float,int list))

    """
    n = len(x)
    j = n // k
    idx = alea.permutation(n)
    means_blocks = np.zeros(k)

    if k == 1:
        return np.mean(x), idx

    for i in range(k):
        means_blocks[i] = np.mean(x[idx[j * i: j * (i + 1)]])

    indices = np.argsort(means_blocks)[int(np.ceil(len(means_blocks) / 2))]

    return np.median(means_blocks[indices]), idx[j * indices: j * (indices + 1)]


def lt_lt_prime(x, y, t, t_prime):
    """
    x: predictor np.matrix
    y: target np.array
    t,t_prime: coefs (np.array)
    return: (y,xt)**2-(y,xt_prime)**2 (np.array)

    """
    n, _ = np.shape(y)

    nt = y - x @ t
    nt_prime = y - x @ t_prime
    tabl = np.array((np.square(nt, nt) - np.square(nt_prime, nt_prime))).flatten()
    return tabl


def subgrad(x, y, t, lamb):
    """
    x: data base (matrix)
    y: results (array)
    t: linear approximation of y with x (array)
    lamb: int
    return: subgradient of the lagrangian of the lasso evaluated in t (array)

    """
    return -2 * x.T @ (y - x @ t) + lamb * np.sign(t)


def grad(x, y, t):
    """
    x: data base (matrix)
    y: results (array)
    t: linear approximation of y with x (array)
    return: gradient of the least square regression evaluated in t (array)

    """
    return -2 * x.T @ (y - x @ t)


def norm1(x):
    return np.sum(np.abs(x))


def lagrangian_lasso(x, y, t, lamb):
    """
    x: data base (matrix)
    y: results (array)
    t: linear approximation of y with x (array)
    lamb: int
    return: lagrangian of the lasso

    """
    return (y - x @ t).T @ (y - x @ t) + lamb * norm1(t)


def som(t, x, deb, fin):
    """
    t: (array)
    x: data base (matrix)
    deb: index of the first element of the sum
    fin: index of the last element of the sum
    return: (array)

    """
    c = 0
    for i in range(deb, fin):
        c += int(t[i][0]) * x[:, i]

    return c


def p_quadra(x, y, t):
    """
    x: data base (matrix)
    y: results (array)
    t: linear approximation of y with x (array)
    return: quadratic error between yi and (xt)i (float list)

    """
    nt = y - x @ t

    return np.square(nt)


def scalar_soft_thresholding(lamb, t):
    """
    lamb: (float)
    t: (array)
    return soft thresholding of t-lamb and 0

    """
    return np.sign(t) * max(abs(t) - lamb, 0)


def soft_thresholding(lamb, t):
    """
    lamb: (float)
    t: (array)
    return soft thresholding of t-lamb and 0

    """
    p = np.zeros(len(t))

    for i in range(len(t)):
        p[i] = np.sign(t[i]) * max(abs(t[i]) - lamb, 0)

    return p


def quadra_loss(x, y, t):
    """
    x: data base (matrix)
    y: results (array)
    t: linear approximation of y with x (array)
    return: quadratic error between y and xt (float)

    """
    return np.linalg.norm(y - x @ t) ** 2


def min_pos(tabl):
    """
    tabl: float array
    return: min of the positive elements in tabl (float)

    """
    mini = inf
    ind = -1
    n, p = np.shape(tabl)

    for i in range(p):
        if 1e-10 < tabl[0, i] < mini:
            mini = tabl[0, i]
            ind = i

    return mini, ind
