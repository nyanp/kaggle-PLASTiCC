import pandas as pd
import tsfresh
import numpy as np

fcp = {
    'flux': {
        'longest_strike_above_mean': None,
        'longest_strike_below_mean': None,
        'mean_change': None,
        'mean_abs_change': None,
        'length': None,
    },

    'flux_by_flux_ratio_sq': {
        'longest_strike_above_mean': None,
        'longest_strike_below_mean': None,
    },

    'flux_passband': {
        'fft_coefficient': [
                {'coeff': 0, 'attr': 'abs'},
                {'coeff': 1, 'attr': 'abs'}
            ],
        'kurtosis' : None,
        'skewness' : None,
    },

    'mjd': {
        'maximum': None,
        'minimum': None,
        'mean_change': None,
        'mean_abs_change': None,
    },
}

from tsfresh.feature_extraction import extract_features


for i in range(30):
    lc = pd.read_feather('../input/all_{}.f'.format(i))
    print('flux-passband')
    agg_df_ts_flux_passband = extract_features(lc,
                                               column_id='object_id',
                                               column_sort='mjd',
                                               column_kind='passband',
                                               column_value='flux',
                                               default_fc_parameters=fcp['flux_passband'], n_jobs=12)
    agg_df_ts_flux_passband.reset_index().to_feather('../features/ts_flux_passband_{}.f'.format(i))

for i in range(30):
    print('flux')
    lc = pd.read_feather('../input/all_{}.f'.format(i))
    agg_df_ts_flux = extract_features(lc,
                                      column_id='object_id',
                                      column_value='flux',
                                      default_fc_parameters=fcp['flux'], n_jobs=12)
    agg_df_ts_flux.reset_index().to_feather('../features/ts_flux_{}.f'.format(i))

for i in range(30):
    print('mjd')
    lc = pd.read_feather('../input/all_{}.f'.format(i))
    df_det = lc[lc['detected']==1].copy()
    agg_df_mjd = extract_features(df_det,
                                  column_id='object_id',
                                  column_value='mjd',
                                  default_fc_parameters=fcp['mjd'], n_jobs=12)
    agg_df_mjd.reset_index().to_feather('../features/ts_mjd_{}.f'.format(i))
