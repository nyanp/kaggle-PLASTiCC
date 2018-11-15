from .common import *
from astropy.cosmology import default_cosmology
cosmo = default_cosmology.get()

def z2pc(z):
    return cosmo.luminosity_distance(z).value

@feature('f150', required_feature='max(flux)_ch1', required_feature_in='f000.f')
def f150_luminosity_minmax_ch(input: Input, debug=True, target_dir='.'):
    meta = input.meta
    meta['Mpc'] = meta['hostgal_photoz'].apply(z2pc)
    meta['Gpc'] = meta['Mpc'] / 1000.0

    features = []
    for i in range(6):
        ch = i
        meta['flux_diff_ch{}'.format(ch)] = meta['max(flux)_ch{}'.format(ch)] - meta['min(flux)_ch{}'.format(ch)]
        meta['luminosity_diff_ch{}'.format(ch)] = meta['flux_diff_ch{}'.format(ch)] * meta['Gpc'] * meta['Gpc']
        features.append('luminosity_diff_ch{}'.format(ch))

    return meta[['object_id']+features]


@feature('f151', required_feature='luminosity_diff_ch1', required_feature_in='f150.f')
def f151_luminosity_minmax_ch_ratio(input: Input, debug=True, target_dir='.'):
    features = []
    meta = input.meta

    for i in range(5):
        c = '_ch{}'.format(i)
        n = '_ch{}'.format(i + 1)

        meta['luminosity_ratio' + c + n] = meta['luminosity_diff' + c] / meta['luminosity_diff' + n]
        features.append('luminosity_ratio' + c + n)

    return meta[['object_id']+features]


@feature('f152', required_feature='luminosity_diff_ch1', required_feature_in='f150.f')
def f152_luminosity_minmax_ch(input: Input, debug=True, target_dir='.'):
    features = []
    meta = input.meta

    for i in range(5):
        c = '_ch{}'.format(i)
        n = '_ch{}'.format(i + 1)

        meta['luminosity_diff' + c + n] = meta['luminosity_diff' + c] - meta['luminosity_diff' + n]
        features.append('luminosity_diff' + c + n)

    return meta[['object_id']+features]


@feature('f153', required_feature='max(flux)_ch1', required_feature_in='f000.f')
def f153_luminosity_minmax_ch(input: Input, debug=True, target_dir='.'):
    meta = input.meta
    meta['Mpc'] = meta['hostgal_photoz'].apply(z2pc)
    meta['Gpc'] = meta['Mpc'] / 1000.0

    features = []
    for i in range(6):
        ch = i
        meta['max(luminosity)_ch{}'.format(ch)] = meta['max(flux)_ch{}'.format(ch)] * meta['Gpc'] * meta['Gpc']
        features.append('max(luminosity)_ch{}'.format(ch))

    return meta[['object_id']+features]

