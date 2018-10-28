import pandas as pd
import numpy as np
from .common import *


def _mid(v, ch):
    return 'flux_percentile_ratio_mid{}_ch{}'.format(v, ch)


def _perc(v, ch):
    return 'percentile_{}(flux)_ch{}'.format(v, ch)


def ratio(meta, v1, v2):
    cols = []
    for i in range(6):
        c = _mid(v2 - v1, i)
        cols.append(c)
        meta[c] = (meta[_perc(v2, i)] - meta[_perc(v1, i)]) / (meta[_perc(95, i)] - meta[_perc(5, i)])
    return meta[['object_id']+cols]

# https://arxiv.org/pdf/1709.06257.pdf

@feature('f120',
         required_feature=['percentile_60(flux)_ch1',
                           'percentile_40(flux)_ch1',
                           'percentile_95(flux)_ch1',
                           'percentile_5(flux)_ch1'],
         required_feature_in=['f022.f',
                              'f023.f',
                              'f020.f',
                              'f021.f'])
def f120_flux_quantile_ratio_mid20(input: Input, debug=True, target_dir='.'):
    return ratio(input.meta, 40, 60)


@feature('f121',
         required_feature=['percentile_675(flux)_ch1',
                           'percentile_325(flux)_ch1',
                           'percentile_95(flux)_ch1',
                           'percentile_5(flux)_ch1'],
         required_feature_in=['f024.f',
                              'f025.f',
                              'f020.f',
                              'f021.f'])
def f121_flux_quantile_ratio_mid35(input: Input, debug=True, target_dir='.'):
    return ratio(input.meta, 32.5, 67.5)


@feature('f122',
         required_feature=['percentile_75(flux)_ch1',
                           'percentile_25(flux)_ch1',
                           'percentile_95(flux)_ch1',
                           'percentile_5(flux)_ch1'],
         required_feature_in=['f026.f',
                              'f027.f',
                              'f020.f',
                              'f021.f'])
def f122_flux_quantile_ratio_mid50(input: Input, debug=True, target_dir='.'):
    return ratio(input.meta, 25, 75)


@feature('f123',
         required_feature=['percentile_825(flux)_ch1',
                           'percentile_175(flux)_ch1',
                           'percentile_95(flux)_ch1',
                           'percentile_5(flux)_ch1'],
         required_feature_in=['f028.f',
                              'f029.f',
                              'f020.f',
                              'f021.f'])
def f123_flux_quantile_ratio_mid65(input: Input, debug=True, target_dir='.'):
    return ratio(input.meta, 17.5, 82.5)
