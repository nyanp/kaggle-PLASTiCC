import pandas as pd
from .common import *

# max(flux)_ch1 - max(flux)_ch2

@feature('f100', required_feature='max(flux)_ch1', required_feature_in='f000.f')
def f100_max_flux_diff1_ch(input: Input, debug=True, target_dir='.'):
    dst = diff_among_ch(input.meta, 'max', 'flux', 1)
    return dst

@feature('f101', required_feature='mean(flux)_ch1', required_feature_in='f000.f')
def f101_mean_flux_diff1_ch(input: Input, debug=True, target_dir='.'):
    dst = diff_among_ch(input.meta, 'mean', 'flux', 1)
    return dst

@feature('f102', required_feature='median(flux)_ch1', required_feature_in='f000.f')
def f102_median_flux_diff1_ch(input: Input, debug=True, target_dir='.'):
    dst = diff_among_ch(input.meta, 'median', 'flux', 1)
    return dst


@feature('f103', required_feature='max(flux)_ch1', required_feature_in='f000.f')
def f103_max_flux_diff2_ch(input: Input, debug=True, target_dir='.'):
    dst = diff_among_ch(input.meta, 'max', 'flux', 2)
    return dst


@feature('f104', required_feature='max(flux)_ch1', required_feature_in='f000.f')
def f104_max_flux_diff3_ch(input: Input, debug=True, target_dir='.'):
    dst = diff_among_ch(input.meta, 'max', 'flux', 3)
    return dst

@feature('f105', required_feature='percentile_95(flux)_ch1', required_feature_in='f020.f')
def f105_percentile95_flux_diff1_ch(input: Input, debug=True, target_dir='.'):
    dst = diff_among_ch(input.meta, 'percentile_95', 'flux', 1)
    return dst
