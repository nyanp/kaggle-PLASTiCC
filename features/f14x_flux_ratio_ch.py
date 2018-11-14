import pandas as pd
from .common import *

# max(flux)_ch1 - max(flux)_ch2


def amp_ratio_among_ch(meta: pd.DataFrame, skip=1):
    cols = []

    for i in range(6):
        meta['amp(flux)_ch{}'.format(i)] = meta['max(flux)_ch{}'.format(i)] - meta['min(flux)_ch{}'.format(i)]

    for c in range(6 - skip):
        n = 'amp(flux)_ch{}'.format(c + skip)
        p = 'amp(flux)_ch{}'.format(c)
        dst = 'amp(flux)_ch{}/ch{}'.format(c, c + skip)
        meta[dst] = meta[p] / meta[n]
        cols.append(dst)

    return meta[['object_id'] + cols]


@feature('f140', required_feature='max(flux)_ch1', required_feature_in='f000.f')
def f140_flux_amplitude_ratio1_ch(input: Input, debug=True, target_dir='.'):
    return amp_ratio_among_ch(input.meta, 1)


@feature('f141', required_feature='max(flux)_ch1', required_feature_in='f000.f')
def f141_flux_amplitude_ratio2_ch(input: Input, debug=True, target_dir='.'):
    return amp_ratio_among_ch(input.meta, 2)


@feature('f142', required_feature='max(flux)_ch1', required_feature_in='f000.f')
def f142_flux_amplitude_ratio3_ch(input: Input, debug=True, target_dir='.'):
    return amp_ratio_among_ch(input.meta, 3)


@feature('f143', required_feature='max(flux)_ch1', required_feature_in='f000.f')
def f143_flux_amplitude_ratio4_ch(input: Input, debug=True, target_dir='.'):
    return amp_ratio_among_ch(input.meta, 4)


@feature('f144', required_feature='max(flux)_ch1', required_feature_in='f000.f')
def f144_flux_amplitude_ratio5_ch(input: Input, debug=True, target_dir='.'):
    return amp_ratio_among_ch(input.meta, 5)
