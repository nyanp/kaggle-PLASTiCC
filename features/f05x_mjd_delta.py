import pandas as pd
from .common import *
import numpy as np


# https://www.kaggle.com/scirpus/plasticc-pipeline-starter
@feature('f050')
def f050_mjddelta(input: Input, debug=True, target_dir='.'):
    mjddelta = input.lc[input.lc['detected'] == 1]
    mjddelta = mjddelta.groupby('object_id').agg({'mjd': ['min', 'max']})
    mjddelta['delta'] = mjddelta[mjddelta.columns[1]] - mjddelta[mjddelta.columns[0]]
    mjddelta = mjddelta['delta'].reset_index(drop=False)

    return mjddelta


@feature('f051')
def f051_mjddelta_per_ch(input: Input, debug=True, target_dir='.'):
    mjddelta = input.lc[input.lc['detected'] == 1]
    mjddelta = mjddelta.groupby(['object_id', 'passband']).agg({'mjd': ['min', 'max']})
    mjddelta['delta'] = mjddelta[mjddelta.columns[1]] - mjddelta[mjddelta.columns[0]]
    mjddelta = mjddelta['delta'].reset_index(drop=False)

    return unstack(mjddelta)


@feature('f052')
def f052_max_to_detected_time(input: Input, debug=True, target_dir='.'):
    mjddelta = input.lc[input.lc['detected'] == 1]
    mjddelta = mjddelta.groupby(['object_id']).agg({'mjd': ['min', 'max']})
    mjddelta['delta'] = mjddelta[mjddelta.columns[1]] - mjddelta[mjddelta.columns[0]]
    mjddelta = mjddelta.reset_index(drop=False)
    mjddelta.columns = ['object_id', 'time(first(detected))', 'time(last(detected))', 'mjddelta']
    mjddelta = pd.merge(mjddelta, input.passband.groupby('object_id')['time(max(flux))'].mean().reset_index(),
                        on=['object_id'], how='left')
    mjddelta['delta(max(flux), last(detected))'] = mjddelta['time(last(detected))'] - mjddelta['time(max(flux))']
    mjddelta['delta(first(detected), max(flux))'] = mjddelta['time(max(flux))'] - mjddelta['time(first(detected))']

    return mjddelta[['object_id', 'delta(max(flux), last(detected))', 'delta(first(detected), max(flux))']]


@feature('f053')
def f053_max_to_detected_time_ch(input: Input, debug=True, target_dir='.'):
    mjddelta = input.lc[input.lc['detected'] == 1]
    mjddelta = mjddelta.groupby(['object_id', 'passband']).agg({'mjd': ['min', 'max']})
    mjddelta['delta'] = mjddelta[mjddelta.columns[1]] - mjddelta[mjddelta.columns[0]]
    mjddelta = mjddelta.reset_index(drop=False)
    mjddelta.columns = ['object_id', 'passband', 'time(first(detected))', 'time(last(detected))', 'mjddelta']
    mjddelta = pd.merge(mjddelta, input.passband, on=['object_id', 'passband'], how='left')
    mjddelta['delta(max(flux), last(detected))'] = mjddelta['time(last(detected))'] - mjddelta['time(max(flux))']
    mjddelta['delta(first(detected), max(flux))'] = mjddelta['time(max(flux))'] - mjddelta['time(first(detected))']

    return unstack(
        mjddelta[['object_id', 'passband', 'delta(max(flux), last(detected))', 'delta(first(detected), max(flux))']])


@feature('f054')
def f054_mjddelta_per_ch(input: Input, debug=True, target_dir='.'):
    mjddelta = input.lc[input.lc['detected'] == 1]
    mjddelta = mjddelta.groupby(['object_id', 'passband']).agg({'mjd': ['min', 'max']})
    mjddelta['delta'] = mjddelta[mjddelta.columns[1]] - mjddelta[mjddelta.columns[0]]
    mjddelta = mjddelta['delta'].reset_index(drop=False)

    mjddelta = unstack(mjddelta)

    cols = []
    for i in range(5):
        col = 'deltadiff_ch{}_{}'.format(i, i + 1)
        mjddelta[col] = mjddelta['delta_ch{}'.format(i)] - mjddelta['delta_ch{}'.format(i + 1)]
        cols.append(col)

    return mjddelta[['object_id'] + cols]
