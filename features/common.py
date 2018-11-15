import pandas as pd
import time
from termcolor import colored
import traceback
import gc
import numpy as np


class Input:
    def __init__(self, meta, passband, lc, lombscargle):
        self.meta = meta
        self.passband = passband
        self.lc = lc
        self.ls = lombscargle



def unstack(aggs):
    if 'passband' in aggs and 'object_id' in aggs:
        aggs.set_index(['object_id','passband'], inplace=True)

    aggs = aggs.unstack()
    aggs.columns = [e[0] + '_ch' + str(e[1]) for e in aggs.columns]
    aggs.reset_index(inplace=True)
    return aggs


def aggregate_by_id_passbands(lc, col, agg, columns=None, prefix=''):
    aggs = lc.groupby(['object_id', 'passband'])[col].agg(agg)

    if columns is not None:
        aggs.columns = columns
    else:
        aggs.columns = [prefix + e + '(' + col + ')' for e in aggs.columns]
    return unstack(aggs)


def aggregate_by_id(lc, col, agg):
    aggs = lc.groupby(['object_id'])[col].agg(agg)
    aggs.columns = [e + '(' + col + ')' for e in aggs.columns]
    aggs.reset_index(inplace=True)
    return aggs


def diff_among_ch(meta: pd.DataFrame, agg, target='flux', skip=1, prefix=''):
    cols = []
    for c in range(6 - skip):
        n = prefix+'{}({})_ch{}'.format(agg, target, c + skip)
        p = prefix+'{}({})_ch{}'.format(agg, target, c)
        dst = prefix+'diff({}({}))_{}_{}'.format(agg, target, c, c + skip)
        meta[dst] = meta[n] - meta[p]
        cols.append(dst)

    return meta[['object_id'] + cols]


def requires_one(meta, feature_name, src_file, target_dir, on='object_id', debug=False):
    if feature_name in meta:
        if debug:
            print('{} already found. skipped'.format(feature_name))
        return meta

    if debug:
        print('load {} and merge to get {}'.format(src_file, feature_name))

    d = pd.read_feather(target_dir + '/' + src_file)

    len_before = len(meta)
    meta = pd.merge(meta, d, on=on, how='left')
    assert len(meta) == len_before

    return meta


def requires(meta, feature_name, src_file, target_dir, on='object_id', debug=False):
    if isinstance(feature_name, list):
        for i, f in enumerate(feature_name):
            meta = requires_one(meta, f, src_file[i], target_dir, on, debug)
        return meta
    else:
        return requires_one(meta, feature_name, src_file, target_dir, on, debug)


def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_{}'.format(n)
    return percentile_

def _top(f):
    if isinstance(f, list):
        return f[0]
    else:
        return f

def feature(name, required_feature=None, required_feature_in=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                print('start : {}'.format(func.__name__))
                s = time.time()

                meta = kwargs['input'].meta
                if required_feature is not None and _top(required_feature) not in meta:
                    kwargs['input'].meta = requires(meta, required_feature, required_feature_in, kwargs['target_dir'], debug=kwargs['debug'])

                ret = func(*args, **kwargs)

                if kwargs['debug']:
                    ret.head(1000).to_csv(kwargs['target_dir'] + '/{}.csv'.format(name))

                ret.reset_index(drop=True).to_feather(kwargs['target_dir'] + '/{}.f'.format(name))

                print('end : {} (time: {})'.format(func.__name__, time.time() - s))
                gc.collect()
                return ret
            except Exception as e:
                print(colored('error on function {}'.format(func.__name__), 'red'))
                print(type(e))
                traceback.print_exc()

        return wrapper

    return decorator