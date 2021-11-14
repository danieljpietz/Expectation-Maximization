import numpy as np
from scipy.special import logsumexp
from scipy import stats

def expectation_step(x, params):
    log_p_y_x = np.log([1-np.squeeze(params[0]), np.squeeze(params[0])])[np.newaxis, ...] + \
                np.log([stats.multivariate_normal(np.squeeze(params[1]), np.squeeze(params[3])).pdf(x),
            stats.multivariate_normal(np.squeeze(params[2]), np.squeeze(params[4])).pdf(x)]).T
    log_p_y_x_norm = logsumexp(log_p_y_x, axis=1, keepdims=True)
    return np.squeeze(log_p_y_x_norm), np.squeeze(np.exp(log_p_y_x - log_p_y_x_norm))

def maximization_step(x, params):
    total_count = x.shape[0]
    _, heuristics = expectation_step(x, params)
    heuristic0 = heuristics[:, 0]
    heuristic1 = heuristics[:, 1]
    sum_heuristic1 = np.sum(heuristic1)
    sum_heuristic0 = np.sum(heuristic0)
    phi = (sum_heuristic1/total_count)
    mu0 = (heuristic0[..., np.newaxis].T.dot(x)/sum_heuristic0).flatten()
    mu1 = (heuristic1[..., np.newaxis].T.dot(x)/sum_heuristic1).flatten()
    diff0 = x - mu0
    sigma0 = diff0.T.dot(diff0 * heuristic0[..., np.newaxis]) / sum_heuristic0
    diff1 = x - mu1
    sigma1 = diff1.T.dot(diff1 * heuristic1[..., np.newaxis]) / sum_heuristic1
    params = [phi,  mu0, mu1, sigma0, sigma1]
    return params

def get_avg_log_likelihood(x, params):
    loglikelihood, _ = expectation_step(x, params)
    return np.mean(loglikelihood)

def EM(x, params):
    avg_loglikelihoods = []
    while True:
        avg_loglikelihood = get_avg_log_likelihood(x, params)
        avg_loglikelihoods.append(avg_loglikelihood)
        if len(avg_loglikelihoods) > 2 and abs(avg_loglikelihoods[-1] - avg_loglikelihoods[-2]) < 0.0001:
            break
        params = maximization_step(x, params)
    _, posterior = expectation_step(x, params)
    forecasts = np.argmax(posterior, axis=1)
    return forecasts, posterior, avg_loglikelihoods, params
    pass
