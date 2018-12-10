import math
import time
import traceback
import sys
import pandas as pd
import numpy as np
import feather
import itertools
from functools import partial
from pymultinest.run import run
from pymultinest.analyse import Analyzer
from numba import jit
from joblib import Parallel, delayed


def solve(LogLikelihood, Prior, n_dims, **kwargs):
    kwargs['n_dims'] = n_dims

    outputfiles_basename = kwargs['outputfiles_basename']

    def SafePrior(cube, ndim, nparams):
        try:
            a = np.array([cube[i] for i in range(n_dims)])
            b = Prior(a)
            for i in range(n_dims):
                cube[i] = b[i]
        except Exception as e:
            import sys
            sys.stderr.write('ERROR in prior: %s\n' % e)
            raise

    def SafeLoglikelihood(cube, ndim, nparams, lnew):
        a = np.array([cube[i] for i in range(n_dims)])
        l = float(LogLikelihood(a))
        if not np.isfinite(l):
            import sys
            sys.stderr.write('WARNING: loglikelihood not finite: %f\n' % (l))
            sys.stderr.write('         for parameters: %s\n' % a)
            sys.stderr.write('         returned very low value instead\n')
            return -1e100
        return l

    kwargs['LogLikelihood'] = SafeLoglikelihood
    kwargs['Prior'] = SafePrior
    run(**kwargs)

    analyzer = Analyzer(n_dims, outputfiles_basename=outputfiles_basename)
    best = analyzer.get_best_fit()

    return [best['log_likelihood']]+best['parameters']


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
    try:
        y_predicted = model_newling(params, data['mjd'])
        y_actual = data['flux']

        return -(((y_predicted - y_actual) / data['flux_err']) ** 2).sum()
    except Exception as e:
        print('!!!!! exception: {}'.format(e))
        #print(traceback.format_exc())
        return -1e30

def opt(data, debug=False, logname="chains/1"):
    if len(data) < 5:
        return [np.nan]*5
    parameters = ["A", "phi", "sigma", "k"]
    prior = partial(prior_newling, data=data)
    log = partial(loglike, data=data)
    result = solve(LogLikelihood=log, Prior=prior, sampling_efficiency=0.9,
                   n_live_points=100, evidence_tolerance=3, n_dims=len(parameters), verbose=debug,
                   outputfiles_basename=logname, max_iter=8000)

    print('finished.')
    print('result:')
    print(result)
    a = Analyzer(n_params=len(parameters), outputfiles_basename=logname)
    best = a.get_best_fit()

    return [best['log_likelihood']]+best['parameters']

def opt_(df, id, passband):
    try:
        return [id,passband]+opt(df.loc[id].query('passband == {}'.format(passband)), logname='chains/{}_{}'.format(id, passband))
    except Exception as e:
        print('##### exception: {}'.format(e))
        #print(traceback.format_exc())
        return [id,passband]+[np.nan]*5


if __name__ == "__main__":
    passbands = [1, 2, 3, 4]
    df = pd.read_csv('../input/training_set.csv')
    meta = feather.read_dataframe('../input/meta.f')
    meta = meta[meta.hostgal_photoz > 0]

    chunk = 30
    n_skip = 0

    df = df[df.detected == 1]
    df = df[df.object_id.isin(meta.object_id)] # only for extragalactic
    df.set_index('object_id', inplace=True)

    object_ids = df.index.unique()

    if len(sys.argv) > 1:
        if sys.argv[1] == '--debug':
            object_ids = object_ids[:100]
        else:
            n_skip = int(sys.argv[1])

    print('chunk: {}, n_skip: {}'.format(chunk, n_skip))
    print('total {} objects'.format(len(object_ids)))


    s = time.time()
    n_chunks = int(math.ceil(len(object_ids) / chunk))

    #for i,p in itertools.product(object_ids,passbands):
    #    r = opt_(i,p)

    for c in range(n_chunks):
        offset = c * chunk
        if offset < n_skip:
            continue
        try:
            r = Parallel(n_jobs=-1)([delayed(opt_)(df,i,p) for i,p in itertools.product(object_ids[offset:offset+chunk],passbands)])
            ret = pd.DataFrame(np.array(r), columns=['object_id','passband']+["newling_loglike","newling_A", "newling_phi", "newling_sigma", "newling_k"])
            ret.to_feather('newling_{}_{}.f'.format(offset, offset+chunk-1))
        except:
            print(traceback.format_exc())
            pass



