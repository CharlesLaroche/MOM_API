import numpy as np
import numpy.random as alea
import random as rd
from math import *


def MOM(X, K):

    #X: np.array
    #K: int
    # return: Median of means of the K blocks with the indexes of the values inside the median block ((float,int list))
    n = len(X)
    j = n // K
    idx = alea.permutation(n)
    means_blocks = np.zeros(K)

    if K == 1:

        return (np.mean(X), idx)

    for i in range(K):

        means_blocks[i] = np.mean(X[idx[j * i: j * (i + 1)]])

    indices = np.argsort(means_blocks)[int(np.ceil(len(means_blocks) / 2))]

    return (np.median(means_blocks[indices]), idx[j * indices: j * (indices + 1)])


def lt_lt_prime(X, Y, t, t_prime):

    # X: predictor np.matrix
    # Y: target np.array
    # t,t_prime: coefs (np.array)
    # return: (Y,Xt)**2-(Y,Xt_prime)**2 (np.array)

    n, _ = np.shape(Y)

    Nt = Y - X @ t
    Nt_prime = Y - X @ t_prime
    TABL = np.array(
        (np.square(Nt, Nt) - np.square(Nt_prime, Nt_prime))).flatten()

    return TABL


def subgrad(X, Y, t, lamb):

    # X: data base (matrix)
    #Y: results (array)
    # t: linear approximation of Y with X (array)
    #lamb: int
    # return: subgradient of the lagrangian of the lasso evaluated in t (array)

    return -2 * (X.T) @ (Y - X @ t) + lamb * np.sign(t)


def grad(X, Y, t):

    # X: data base (matrix)
    #Y: results (array)
    # t: linear approximation of Y with X (array)
    # return: gradient of the least square regression evaluated in t (array)

    return -2 * (X.T) @ (Y - X @ t)


def norm1(X):

    return np.sum(np.abs(X))


def F(X, Y, t, lamb):

    # X: data base (matrix)
    #Y: results (array)
    # t: linear approximation of Y with X (array)
    #lamb: int
    # return: lagrangian of the lasso

    return ((Y - X @ t).T) @ (Y - X @ t) + lamb * norm1(t)


def som(t, X, deb, fin):

    #t= (array)
    # X: data base (matrix)
    # deb: index of the first element of the sum
    # fin: index of the last element of the sum
    # return: (array)

    c = 0

    for i in range(deb, fin):

        c += int(t[i][0]) * X[:, i]

    return c


def part_pos(x):

    return max(0, x)


def P_quadra(X, Y, t):

    # X: data base (matrix)
    #Y: results (array)
    # t: linear approximation of Y with X (array)
    # return: quadratic error between Yi and (Xt)i (float list)

    Nt = Y - X @ t

    return np.square(Nt)


def scalar_soft_thresholding(lamb, t):

    #lamb: (float)
    #t: (array)
    # return soft thresholding of t-lamb and 0

    return np.sign(t) * max(abs(t) - lamb, 0)


def soft_thresholding(lamb, t):

    #lamb: (float)
    #t: (array)
    # return soft thresholding of t-lamb and 0

    P = np.zeros(len(t))

    for i in range(len(t)):

        P[i] = np.sign(t[i]) * max(abs(t[i]) - lamb, 0)

    return P


def quadra_loss(X, Y, t):

    # X: data base (matrix)
    #Y: results (array)
    # t: linear approximation of Y with X (array)
    # return: quadratic error between Y and Xt (float)

    return np.linalg.norm(Y - X @ t) ** 2


def min_pos(tabl):

    # tabl: float array
    # return: min of the positive elements in tabl (float)

    mini = inf
    ind = -1
    n, p = np.shape(tabl)

    for i in range(p):

        if tabl[0, i] < mini and tabl[0, i] > 1e-10:

            mini = tabl[0, i]
            ind = i

    return mini, ind
