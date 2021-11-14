import numpy as np

def get_random_psd(n):
    x = np.random.normal(0, 1, size=(n, n))
    return np.dot(x, x.transpose())

def get_synthetic_data(n, d):
    X1 = np.random.multivariate_normal(np.random.rand(1, d)[0], get_random_psd(d), n)
    X2 = np.random.multivariate_normal(np.random.rand(1, d)[0], get_random_psd(d), n)
    labels = np.concatenate((np.zeros((n, 1)), np.ones((n, 1))))
    X3 = np.concatenate((X1, X2))
    X3 = np.concatenate((labels, X3), axis=1)
    np.random.shuffle(X3)
    labels = np.reshape(X3[:, 0], (2 * n, 1))
    X3 = X3[:, 1:d + 1]
    return X3, np.squeeze(labels)