import pandas as pd
from .common import *
import numpy as np

def _prep(input: Input):
    if 'flux_prev' not in input.lc:
        input.lc['flux_prev'] = input.lc.groupby(['object_id', 'passband'])['flux'].shift(1)
        input.lc['mjd_prev'] = input.lc.groupby(['object_id', 'passband'])['mjd'].shift(1)
        input.lc['flux_err_prev'] = input.lc.groupby(['object_id', 'passband'])['flux_err'].shift(1)
        input.lc['flux_diff'] = input.lc['flux'] - input.lc['flux_prev']
        input.lc['mjd_diff'] = input.lc['mjd'] - input.lc['mjd_prev']
        input.lc['flux_err_diff'] = input.lc['flux_err'] - input.lc['flux_err_prev']
        input.lc['flux_slope'] = input.lc['flux_diff'] / input.lc['mjd_diff']


@feature('f110')
def f110_flux_slope_minmax(input: Input, debug=True, target_dir='.'):
    _prep(input)

    aggs = aggregate_by_id_passbands(input.lc, 'flux_slope', ['max', 'min'])

    return aggs


@feature('f111')
def f111_flux_slope_percentile05(input: Input, debug=True, target_dir='.'):
    _prep(input)

    aggs = aggregate_by_id_passbands(input.lc, 'flux_slope', [percentile(95), percentile(5)])

    return aggs


@feature('f112', required_feature='max(flux)_ch1', required_feature_in='f110.f')
def f112_flux_slope_percentile05_filter(input: Input, debug=True, target_dir='.'):
    _prep(input)

    input.lc['flux_slope_filtered'] = (np.abs(input.lc['flux_slope']) > input.lc['flux_err']).astype(np.int32) * input.lc['flux_slope']

    aggs = aggregate_by_id_passbands(input.lc, 'flux_slope_filtered', [percentile(95), percentile(5)])

    return aggs



