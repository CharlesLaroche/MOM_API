import numpy as np
import numpy.random as alea
import random as rd


def create_t_0(n, sparsity):

    idx = np.arange(n)
    beta = (n / 10) * (-1) ** (abs(idx - 1)) * np.exp(-idx / 10.)
    sel = alea.permutation(n)
    sel1 = sel[0: int(sparsity / 4)]
    beta[sel1] = 10
    sel11 = sel[int(sparsity / 4): int(sparsity / 2)]
    beta[sel11] = -10
    sel0 = sel[sparsity:]
    beta[sel0] = 0.

    return beta


def data1(n_samples, t, sigma):

    n_features = np.shape(t)[0]
    cov = np.identity(n_features)

    X = alea.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    Y = X @ t + sigma * alea.randn(n_samples)

    return Y, X


def data2(n_outliers, n_features, type_outliers=1, beta=1, rho=1):

    if type_outliers == 1:

        Y = np.ones(n_outliers)
        X = np.ones((n_outliers, n_features))

    elif type_outliers == 2:

        Y = 10000 * np.ones(n_outliers)
        X = np.ones((n_outliers, n_features))

    elif type_outliers == 3:

        Y = np.random.randint(2, size=n_outliers)
        X = np.random.rand(n_outliers, n_features)

    else:

        cov = np.identity(n_features)

        X = feature_mat(n_features, n_outliers, rho)
        Y = X.dot(beta) + sigma * alea.randn(n_samples)

    return Y, X


def data3(n_heavy_tail, beta, deg=2):

    n_features = beta.size
    cov = np.identity(n_features)

    X = alea.multivariate_normal(np.zeros(n_features), cov, size=n_heavy_tail)
    Y = X.dot(beta) + np.random.standard_t(deg, size=n_heavy_tail)

    return Y, X


def data_merge(Y1, X1, Y2, X2):

    Y = np.concatenate((Y1, Y2), axis=0)
    X = np.concatenate((X1, X2), axis=0)
    Y = np.reshape(Y, (Y.shape[0], 1))
    R = np.concatenate((Y, X), axis=1)
    alea.shuffle(R)

    Y = R[:, 0]
    X = R[:, 1:]

    return Y, X
