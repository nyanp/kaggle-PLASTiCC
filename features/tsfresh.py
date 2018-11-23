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

lc = pd.read_feather('../input/all.f')
meta = pd.read_feather('../input/meta.f')

from tsfresh.feature_extraction import extract_features

print('flux-passband')
agg_df_ts_flux_passband = extract_features(lc,
                                           column_id='object_id',
                                           column_sort='mjd',
                                           column_kind='passband',
                                           column_value='flux',
                                           default_fc_parameters=fcp['flux_passband'], n_jobs=12)
agg_df_ts_flux_passband.reset_index().to_feather('../features/ts_flux_passband.f')

print('flux')
agg_df_ts_flux = extract_features(lc,
                                  column_id='object_id',
                                  column_value='flux',
                                  default_fc_parameters=fcp['flux'], n_jobs=12)
agg_df_ts_flux.reset_index().to_feather('../features/ts_flux.f')

print('mjd')
df_det = lc[lc['detected']==1].copy()
agg_df_mjd = extract_features(df_det,
                              column_id='object_id',
                              column_value='mjd',
                              default_fc_parameters=fcp['mjd'], n_jobs=12)


agg_df_mjd.reset_index().to_feather('../features/ts_mjd.f')
