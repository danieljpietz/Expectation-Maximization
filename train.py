import numpy as np
from scipy.special import logsumexp
from scipy import stats

stop_cond = 10 ** -4


def expectation_step(X, params):
    log_likelihood = np.log([1 - np.squeeze(params[0]), np.squeeze(params[0])])[np.newaxis, ...] + \
           np.log([stats.multivariate_normal(np.squeeze(params[1]), np.squeeze(params[3])).pdf(X),
                   stats.multivariate_normal(np.squeeze(params[2]), np.squeeze(params[4])).pdf(X)]).T
    l_l_normalized = logsumexp(log_likelihood, axis=1, keepdims=True)
    return np.squeeze(l_l_normalized), np.squeeze(np.exp(log_likelihood - l_l_normalized))


def maximization_step(X, params):
    _, heuristics = expectation_step(X, params)
    heuristic0_sum = np.sum(heuristics[:, 0])
    heuristic1_sum = np.sum(heuristics[:, 1])
    phi = np.mean(heuristic1_sum/heuristics.shape[0])
    mean0 = np.dot(heuristics[:, 0][np.newaxis, ...], X) / heuristic0_sum
    mean1 = np.dot(heuristics[:, 1][np.newaxis, ...], X) / heuristic1_sum
    diff0 = X - mean0
    var0 = diff0.T @ (diff0 * heuristics[:, 0][..., np.newaxis]) / heuristic0_sum
    diff1 = X - mean1
    var1 = diff1.T @ (diff1 * heuristics[:, 1][..., np.newaxis]) / heuristic1_sum
    params = [phi, mean0, mean1, var0, var1]
    return params


def EM(X, params):
    avg_likely = []
    current = 0
    stop_filter = 0.95
    while True:
        likelihood, _ = expectation_step(X, params)
        avg_likelihood = np.mean(likelihood)
        avg_likely.append(avg_likelihood)
        params = maximization_step(X, params)
        if len(avg_likely) == 1:
            current = avg_likely[-1]
        else:
            last = current
            current = (stop_filter * current) + (1 - stop_filter) * avg_likely[-1]
            if abs(current - last) < stop_cond:
                break
        params = maximization_step(X, params)
    _, posterior = expectation_step(X, params)
    labels_predicted = np.argmax(posterior, axis=1)
    return labels_predicted, avg_likely, params
    pass
