import pandas as pd
from .common import *


@feature('f060')
def f060_max_ch_flux_detected(input: Input, debug=True, target_dir='.'):
    aggs = aggregate_by_id_passbands(input.lc[input.lc.detected == 1], 'flux', ['max'], prefix='detected_')

    return aggs


@feature('f061')
def f061_median_ch_flux_detected(input: Input, debug=True, target_dir='.'):
    aggs = aggregate_by_id_passbands(input.lc[input.lc.detected == 1], 'flux', ['median'], prefix='detected_')

    return aggs


@feature('f062', required_feature='detected_max(flux)_ch1', required_feature_in='f060.f')
def f062_max_flux_diff1_ch_detected(input: Input, debug=True, target_dir='.'):
    dst = diff_among_ch(input.meta, 'max', 'flux', 1, prefix='detected_')
    return dst


@feature('f063', required_feature='detected_median(flux)_ch1', required_feature_in='f061.f')
def f063_median_flux_diff1_ch_detected(input: Input, debug=True, target_dir='.'):
    dst = diff_among_ch(input.meta, 'median', 'flux', 1, prefix='detected_')
    return dst

