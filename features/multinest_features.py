import math
import time
import pandas as pd
import numpy as np
import feather
from functools import partial
from pymultinest.solve import solve
from pymultinest.analyse import Analyzer
from numba import jit



def model_newling(params, t):
    A = math.exp(params[0])
    phi = params[1]
    sigma = math.exp(params[2])
    k = math.exp(params[3])

    tau = (t > phi) * (t - phi) / sigma

    F = A * (tau ** k) * np.exp(-tau) * (k ** -k) * (math.e ** k)
    return F


def prior_newling(params, data):
    dt = data['mjd'].max() - data['mjd'].min()
    n = data.flux.reset_index()['flux'].idxmax()
    tmax = data['mjd'].iloc[n]
    fmax = data['flux'].max()
    math.log(fmax)
    # TABLE.2 from https://arxiv.org/pdf/1603.00882.pdf
    return np.array([2 * params[0] + math.log(fmax) - 1,
                     20 * params[1] + tmax - 10,
                     7 * params[2] - 3,
                     8 * params[3] - 4])


def loglike(params, data):
    y_predicted = model_newling(params, data['mjd'])
    y_actual = data['flux']

    return -(((y_predicted - y_actual) / data['flux_err']) ** 2).sum()


def opt(data, debug=False, logname="chains/1"):
    parameters = ["A", "phi", "sigma", "k"]
    prior = partial(prior_newling, data=data)
    log = partial(loglike, data=data)
    print('start')
    result = solve(LogLikelihood=log, Prior=prior, sampling_efficiency=0.9,
                   n_live_points=100, evidence_tolerance=3, n_dims=len(parameters), verbose=debug,
                   outputfiles_basename=logname)

    print('finished.')
    print('result:')
    print(result)
    a = Analyzer(n_params=len(parameters), outputfiles_basename=logname)
    best = a.get_best_fit()

    return best['log_likelihood'], best['parameters']


df = feather.read_dataframe('../input/all_0.f')
df = df[df.detected == 1]
df.set_index('object_id', inplace=True)

s = time.time()
loglike, params = opt(df.loc[1920].query('passband == 2'),debug=True, logname='chains/{}_{}'.format(1920, 2))

print('log-likelihood: {}'.format(loglike))
print('params: {}'.format(params))
print('elapsed time: {}'.format(time.time() - s))




