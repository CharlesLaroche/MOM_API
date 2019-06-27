import numpy as np
import numpy.random as alea


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

    x = alea.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    y = x @ t + sigma * alea.randn(n_samples)

    return y, x


def data2(n_outliers, n_features, type_outliers=1):

    if type_outliers == 1:
        y = np.ones(n_outliers)
        x = np.ones((n_outliers, n_features))

    elif type_outliers == 2:
        y = 10000 * np.ones(n_outliers)
        x = np.ones((n_outliers, n_features))

    elif type_outliers == 3:
        y = np.random.randint(2, size=n_outliers)
        x = np.random.rand(n_outliers, n_features)

    else:
        raise AttributeError('type_outliers must be 1,2 or 3')

    return y, x


def data3(n_heavy_tail, beta, deg=2):

    n_features = beta.size
    cov = np.identity(n_features)

    x = alea.multivariate_normal(np.zeros(n_features), cov, size=n_heavy_tail)
    y = x.dot(beta) + np.random.standard_t(deg, size=n_heavy_tail)

    return y, x


def data_merge(y1, x1, y2, x2):

    y = np.concatenate((y1, y2), axis=0)
    x = np.concatenate((x1, x2), axis=0)
    y = np.reshape(y, (y.shape[0], 1))
    r = np.concatenate((y, x), axis=1)
    alea.shuffle(r)

    y = r[:, 0]
    x = r[:, 1:]

    return y, x
