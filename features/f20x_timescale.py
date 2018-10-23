import pandas as pd
from .common import *
import numpy as np
import gc


def timescale(passband: pd.DataFrame, lc: pd.DataFrame, threshold: float, time_agg: str):
    lc_ = pd.merge(lc, passband[['object_id','passband','max(flux)']], on=['object_id','passband'], how='left')

    lc_ = lc_[np.abs(lc_['flux']) > lc_['max(flux)'] * threshold].copy()

    last_time = lc_.groupby(['object_id', 'passband'])['mjd'].agg([time_agg]).reset_index()
    del lc_
    gc.collect()

    last_time.columns = ['object_id', 'passband', 'ref_mjd']

    dst = pd.merge(last_time, passband, on=['object_id', 'passband'], how='left')

    dst_col = 'timescale_th{}_{}'.format(threshold, time_agg)
    dst[dst_col] = dst['ref_mjd'] - dst['time(max(flux))']

    dst = dst[['object_id', 'passband', dst_col]]
    print(dst.head())
    return unstack(dst)


@feature('f200')
def f200_timescale_th015_max(input: Input, debug=True, target_dir='.'):
    dst = timescale(input.passband, input.lc, 0.15, 'max')
    return dst


@feature('f201')
def f201_timescale_th015_min(input: Input, debug=True, target_dir='.'):

    dst = timescale(input.passband, input.lc, 0.15, 'min')
    return dst


@feature('f202')
def f202_timescale_th035_max(input: Input, debug=True, target_dir='.'):

    dst = timescale(input.passband, input.lc, 0.35, 'max')
    return dst


@feature('f203')
def f203_timescale_th035_min(input: Input, debug=True, target_dir='.'):

    dst = timescale(input.passband, input.lc, 0.35, 'min')
    return dst


@feature('f204')
def f204_timescale_th035_max(input: Input,  debug=True, target_dir='.'):

    dst = timescale(input.passband, input.lc, 0.5, 'max')
    return dst


@feature('f205')
def f205_timescale_th035_min(input: Input, debug=True, target_dir='.'):

    dst = timescale(input.passband, input.lc, 0.5, 'min')
    return dst