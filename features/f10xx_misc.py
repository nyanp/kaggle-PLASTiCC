import pandas as pd
from astropy.cosmology import default_cosmology
from tsfresh.feature_extraction import extract_features
from tqdm import tqdm

import common
import config
import gc
from .common import *

cosmo = default_cosmology.get()


@feature('f1000')
def f1000_salt2_normalized_chisq(input: Input, **kw):
    salt2 = common.load_feature("f500")
    meta_ = pd.merge(input.meta, salt2, on='object_id', how='left')

    count = input.lc.groupby('object_id')['mjd'].count().reset_index()
    count.columns = ['object_id', 'n_observed']

    meta_ = pd.merge(meta_, count, on='object_id', how='left')
    meta_['sn_salt2_chisq_norm'] = meta_['sn_salt2_chisq'] / meta_['n_observed']

    return meta_[['object_id', 'sn_salt2_chisq_norm']]


@feature('f1001')
def f1001_detected_to_risetime_ratio(input: Input, **kw):
    delta = common.load_feature("f050")
    risetime = common.load_feature("f053")

    meta_ = pd.merge(input.meta, delta, on='object_id', how='left')
    meta_ = pd.merge(meta_, risetime, on='object_id', how='left')

    features = []
    for i in range(6):
        f = 'delta(first(detected), max(flux))_ch{}_to_delta_ratio'.format(i)
        meta_[f] = meta_['delta(first(detected), max(flux))_ch{}'.format(i)] / meta_['delta']
        features.append(f)

    return meta_[['object_id']+features]


@feature('f1002')
def f1002_detected_to_falltime_ratio(input: Input, **kw):
    delta = common.load_feature("f050")
    falltime = common.load_feature("f053")

    meta_ = pd.merge(input.meta, delta, on='object_id', how='left')
    meta_ = pd.merge(meta_, falltime, on='object_id', how='left')

    features = []
    for i in range(6):
        f = 'delta(max(flux), last(detected))_ch{}_to_delta_ratio'.format(i)
        meta_[f] = meta_['delta(max(flux), last(detected))_ch{}'.format(i)] / meta_['delta']
        features.append(f)

    return meta_[['object_id']+features]


def z2pc(z):
    return cosmo.luminosity_distance(z).value


@feature('f1003')
def f1003_luminosity_by_estimated_redshift(input: Input, **kw):
    aggregate = common.load_feature("f000")
    redshift = common.load_feature("f600")
    meta_ = pd.merge(input.meta, redshift, on='object_id', how='left')
    meta_ = pd.merge(meta_, aggregate, on='object_id', how='left')

    meta_['Mpc'] = meta_['hostgal_z_predicted'].apply(z2pc)
    meta_['Gpc'] = meta_['Mpc'] / 1000.0

    features = []
    for i in range(6):
        ch = i
        meta_['flux_diff_ch{}'.format(ch)] = meta_['max(flux)_ch{}'.format(ch)] - meta_['min(flux)_ch{}'.format(ch)]
        meta_['luminosity_diff_ch{}_estimated'.format(ch)] = meta_['flux_diff_ch{}'.format(ch)] * meta_['Gpc'] * meta_['Gpc']
        features.append('luminosity_diff_ch{}_estimated'.format(ch))

    features_renamed = ['luminosity_est_diff_ch{}'.format(i) for i in range(6)]

    rename = {features[i]: features_renamed[i] for i in range(6)}

    meta_.rename(columns=rename, inplace=True)

    return meta_[['object_id']+features_renamed]


def extract_features_postproc(df: pd.DataFrame):
    df.sort_values(by='id', inplace=True)
    df.rename(columns={'id': 'object_id'}, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


@feature('f1004')
def f1004_tsfresh_flux(input: Input, **kw):
    fcp = {
        'longest_strike_above_mean': None,
        'longest_strike_below_mean': None,
        'mean_change': None,
        'mean_abs_change': None,
        'length': None,
    }

    dfs = []
    for i in tqdm(range(30)):
        lc = common.load_partial_lightcurve(i)
        partial = extract_features(lc,
                                   column_id='object_id',
                                   column_value='flux',
                                   default_fc_parameters=fcp, n_jobs=0)
        dfs.append(partial.reset_index())
        gc.collect()

    return extract_features_postproc(pd.concat(dfs))


@feature('f1005')
def f1005_tsfresh_flux_per_passband(input: Input, **kw):
    fcp = {
        'fft_coefficient': [
                {'coeff': 0, 'attr': 'abs'},
                {'coeff': 1, 'attr': 'abs'}
            ],
        'kurtosis': None,
        'skewness': None,
    }

    dfs = []
    for i in tqdm(range(30)):
        lc = common.load_partial_lightcurve(i)
        partial = extract_features(lc,
                                   column_id='object_id',
                                   column_sort='mjd',
                                   column_kind='passband',
                                   column_value='flux',
                                   default_fc_parameters=fcp, n_jobs=0)
        dfs.append(partial.reset_index())
        gc.collect()

    return extract_features_postproc(pd.concat(dfs))


@feature('f1006')
def f1006_tsfresh_mjd(input: Input, **kw):
    fcp = {
        'mean_change': None,
        'mean_abs_change': None,
    }
    lc = input.lc
    df_det = lc[lc['detected'] == 1].copy()

    dfs = []
    for i in tqdm(range(30)):
        lc = common.load_partial_lightcurve(i)
        partial = extract_features(df_det,
                                   column_id='object_id',
                                   column_value='mjd',
                                   default_fc_parameters=fcp, n_jobs=0)
        dfs.append(partial.reset_index())
        gc.collect()

    return extract_features_postproc(pd.concat(dfs))


def filter_lc_by_snr(lc: pd.DataFrame, snr: int):
    lc['SNR'] = lc['flux'] / lc['flux_err']
    lc['absSNR'] = np.abs(lc['SNR'])
    lc['flag'] = (lc['absSNR'] > snr).astype(np.uint8)
    return lc[lc['flag'] == 1]


@feature('f1080')
def f1080_snr3_minmax_diff(input: Input, **kw):
    lc_detected = filter_lc_by_snr(input.lc, snr=3)
    mjddelta = lc_detected.groupby('object_id').agg({'mjd': ['min', 'max']})
    mjddelta['delta'] = mjddelta[mjddelta.columns[1]] - mjddelta[mjddelta.columns[0]]
    mjddelta = mjddelta['delta'].reset_index(drop=False)

    return mjddelta


@feature('f1081')
def f1081_first_is_detected(input: Input, **kw):
    first = input.lc[['object_id', 'mjd', 'detected']].groupby('object_id').first()
    first.columns = ['mjd', 'detected_on_first_observation']

    return first.drop('mjd', axis=1).reset_index()


@feature('f1082')
def f1082_last_is_detected(input: Input, **kw):
    last = input.lc[['object_id', 'mjd', 'detected']].groupby('object_id').last()
    last.columns = ['mjd', 'detected_on_last_observation']

    return last.drop('mjd', axis=1).reset_index()


@feature('f1083')
def f1083_max_flux_within_snr3(input: Input, **kw):
    lc_detected = filter_lc_by_snr(input.lc, snr=3)
    return aggregate_by_id_passbands(lc_detected, 'flux', ['max'], prefix='snr3_')


@feature('f1084')
def f1084_min_flux_within_snr3(input: Input, **kw):
    lc_detected = filter_lc_by_snr(input.lc, snr=3)
    return aggregate_by_id_passbands(lc_detected, 'flux', ['min'], prefix='snr3_')


@feature('f1085')
def f1085_luminosity_diff_within_snr3(input: Input, **kw):
    redshift = common.load_feature('f600')
    max_flux = common.load_feature('f1083')
    min_flux = common.load_feature('f1084')

    meta = pd.merge(input.meta, redshift, on='object_id', how='left')
    meta = pd.merge(meta, max_flux, on='object_id', how='left')
    meta = pd.merge(meta, min_flux, on='object_id', how='left')

    meta['Mpc'] = meta['hostgal_z_predicted'].apply(z2pc)
    meta['Gpc'] = meta['Mpc'] / 1000.0

    features = []
    for i in range(6):
        ch = i
        meta['snr3_flux_diff_ch{}'.format(ch)] = meta['snr3_max(flux)_ch{}'.format(ch)] - meta['snr3_min(flux)_ch{}'.format(ch)]
        meta['snr3_luminosity_diff_ch{}'.format(ch)] = meta['snr3_flux_diff_ch{}'.format(ch)] * meta['Gpc'] * meta['Gpc']
        features.append('snr3_luminosity_diff_ch{}'.format(ch))

    return meta[['object_id']+features]


@feature('f1086')
def f1086_first_detected_to_prev_mjd_diff(input: Input, **kw):
    lc = input.lc
    lc['mjd_prev'] = lc[['object_id', 'mjd']].groupby('object_id').shift(1)
    lc_detected = lc[lc.detected == 1]

    first_obs = lc_detected.groupby('object_id').first()
    first_obs['first_detected_to_prev_mjd_diff'] = first_obs['mjd'] - first_obs['mjd_prev']

    return first_obs[['first_detected_to_prev_mjd_diff']].reset_index()


@feature('f1087')
def f1087_last_detected_to_next_mjd_diff(input: Input, **kw):
    lc = input.lc
    lc['mjd_next'] = lc[['object_id', 'mjd']].groupby('object_id').shift(-1)
    lc_detected = lc[lc.detected == 1]

    last_obs = lc_detected.groupby('object_id').last()
    last_obs['last_detected_to_next_mjd_diff'] = last_obs['mjd_next'] - last_obs['mjd']

    return last_obs[['last_detected_to_next_mjd_diff']].reset_index()


@feature('f1088')
def f1088_first_detected_to_prev_mjd_diff_perch(input: Input, **kw):
    lc = input.lc
    lc['mjd_prev'] = lc[['object_id', 'mjd']].groupby('object_id').shift(1)
    lc_detected = lc[lc.detected == 1]

    first_obs_pb = lc_detected.groupby(['object_id', 'passband']).first()
    first_obs_pb['first_detected_to_prev_mjd_diff_perch'] = first_obs_pb['mjd'] - first_obs_pb['mjd_prev']

    return unstack(first_obs_pb[['first_detected_to_prev_mjd_diff_perch']])


@feature('f1089')
def f1089_last_detected_to_next_mjd_diff_perch(input: Input, **kw):
    lc = input.lc
    lc['mjd_next'] = lc[['object_id', 'mjd']].groupby('object_id').shift(-1)
    lc_detected = lc[lc.detected == 1]

    last_obs_pb = lc_detected.groupby(['object_id', 'passband']).last()
    last_obs_pb['last_detected_to_next_mjd_diff_perch'] = last_obs_pb['mjd_next'] - last_obs_pb['mjd']

    return unstack(last_obs_pb[['last_detected_to_next_mjd_diff_perch']])
