import tsfresh
from tsfresh import extract_features
from .common import *
from tsfresh.feature_extraction.feature_calculators import number_peaks
from itertools import product
import numpy as np


def postproc(ext):
    ext.reset_index(inplace=True)

    ext = pd.concat([ext, ext['id'].str.split('_', expand=True)], axis=1)
    ext.rename(columns={0: 'object_id', 1: 'passband'}, inplace=True)
    ext['object_id'] = ext['object_id'].astype(np.int64)
    ext['passband'] = ext['passband'].astype(np.int64)
    ext.drop('id', axis=1, inplace=True)

    return unstack(ext)


@feature('f300')
def f300_num_peaks(input: Input, debug=True, target_dir='.'):
    ext = extract_features(input.lc[['id_passband', 'flux']], n_jobs=0, column_id='id_passband',
                           default_fc_parameters={"number_peaks": [{"n": 1}, {"n": 5}]})
    return postproc(ext)


@feature('f301')
def f301_quantile2(input: Input, debug=True, target_dir='.'):
    ext = extract_features(input.lc[['id_passband', 'flux']], n_jobs=0, column_id='id_passband',
                           default_fc_parameters={"quantile": [{"q": .2}]})
    return postproc(ext)


@feature('f302')
def f302_quantile8(input: Input, debug=True, target_dir='.'):
    ext = extract_features(input.lc[['id_passband', 'flux']], n_jobs=0, column_id='id_passband',
                           default_fc_parameters={"quantile": [{"q": .8}]})
    return postproc(ext)


@feature('f303')
def f303_c3(input: Input, debug=True, target_dir='.'):
    ext = extract_features(input.lc[['id_passband', 'flux']], n_jobs=0, column_id='id_passband',
                           default_fc_parameters={"c3": [{"lag": 1}]})
    return postproc(ext)


@feature('f304')
def f304_autocorr1(input: Input, debug=True, target_dir='.'):
    ext = extract_features(input.lc[['id_passband', 'flux']], n_jobs=0, column_id='id_passband',
                           default_fc_parameters={"autocorrelation": [{"lag": 1}]})
    return postproc(ext)


@feature('f305')
def f305_autocorr2(input: Input, debug=True, target_dir='.'):
    ext = extract_features(input.lc[['id_passband', 'flux']], n_jobs=0, column_id='id_passband',
                           default_fc_parameters={"autocorrelation": [{"lag": 2}]})
    return postproc(ext)


@feature('f306')
def f306_autocorr3(input: Input, debug=True, target_dir='.'):
    ext = extract_features(input.lc[['id_passband', 'flux']], n_jobs=0, column_id='id_passband',
                           default_fc_parameters={"autocorrelation": [{"lag": 3}]})
    return postproc(ext)


@feature('f307')
def f307_autocorr4(input: Input, debug=True, target_dir='.'):
    ext = extract_features(input.lc[['id_passband', 'flux']], n_jobs=0, column_id='id_passband',
                           default_fc_parameters={"autocorrelation": [{"lag": 4}]})
    return postproc(ext)


@feature('f308')
def f308_autocorr5(input: Input, debug=True, target_dir='.'):
    ext = extract_features(input.lc[['id_passband', 'flux']], n_jobs=0, column_id='id_passband',
                           default_fc_parameters={"autocorrelation": [{"lag": 5}]})
    return postproc(ext)


@feature('f309')
def f309_autocorr_mean(input: Input, debug=True, target_dir='.'):
    ext = extract_features(input.lc[['id_passband', 'flux']], n_jobs=0, column_id='id_passband',
                           default_fc_parameters={"agg_autocorrelation": [{"f_agg": "mean"}]})
    return postproc(ext)


@feature('f310')
def f310_autocorr_median(input: Input, debug=True, target_dir='.'):
    ext = extract_features(input.lc[['id_passband', 'flux']], n_jobs=0, column_id='id_passband',
                           default_fc_parameters={"agg_autocorrelation": [{"f_agg": "median"}]})
    return postproc(ext)


@feature('f311')
def f311_autocorr_var(input: Input, debug=True, target_dir='.'):
    ext = extract_features(input.lc[['id_passband', 'flux']], n_jobs=0, column_id='id_passband',
                           default_fc_parameters={"agg_autocorrelation": [{"f_agg": "var"}]})
    return postproc(ext)


@feature('f321')
def f321_partial_autocorr_lag10(input: Input, debug=True, target_dir='.'):
    ext = extract_features(input.lc[['id_passband', 'flux']], n_jobs=0, column_id='id_passband',
                           default_fc_parameters={"partial_autocorrelation": [{"lag": lag} for lag in range(10)]})
    return postproc(ext)


@feature('f330')
def f330_number_cwt_peaks(input: Input, debug=True, target_dir='.'):
    ext = extract_features(input.lc[['id_passband', 'flux']], n_jobs=0, column_id='id_passband',
                           default_fc_parameters={"number_cwt_peaks": [{"n": n} for n in [1, 5]]})
    return postproc(ext)


@feature('f340')
def f340_number_crossing_m(input: Input, debug=True, target_dir='.'):
    ext = extract_features(input.lc[['id_passband', 'flux']], n_jobs=0, column_id='id_passband',
                           default_fc_parameters={"number_crossing_m": [{"m": 0}]})
    return postproc(ext)


@feature('f350')
def f350_linear_trend(input: Input, debug=True, target_dir='.'):
    ext = extract_features(input.lc[['id_passband', 'flux']], n_jobs=0, column_id='id_passband', default_fc_parameters={
        "linear_trend": [{"attr": "pvalue"}, {"attr": "rvalue"}, {"attr": "intercept"},
                         {"attr": "slope"}, {"attr": "stderr"}]})
    return postproc(ext)


@feature('f360')
def f360_fft_coefficient(input: Input, debug=True, target_dir='.'):
    ext = extract_features(input.lc[['id_passband', 'flux']], n_jobs=0, column_id='id_passband',
                           default_fc_parameters={"fft_coefficient": [{"coeff": k, "attr": a} for a, k in
                                                                      product(["real", "imag", "abs", "angle"],
                                                                              range(10))]})
    return postproc(ext)


@feature('f370')
def f370_fft_aggregated(input: Input, debug=True, target_dir='.'):
    ext = extract_features(input.lc[['id_passband', 'flux']], n_jobs=0, column_id='id_passband',
                           default_fc_parameters={"fft_aggregated": [{"aggtype": s} for s in
                                                                     ["centroid", "variance", "skew", "kurtosis"]]})
    return postproc(ext)


@feature('f361')
def f361_fft_coefficient(input: Input, debug=True, target_dir='.'):
    fcp = {'fft_coefficient': [{'coeff': 0, 'attr': 'abs'}, {'coeff': 1, 'attr': 'abs'}], 'kurtosis': None,
           'skewness': None}
    agg_df_ts = tsfresh.extract_features(input.lc, column_id='object_id', column_sort='mjd', column_kind='passband',
                                         column_value='flux', default_fc_parameters=fcp, n_jobs=0)
    agg_df_ts.index.rename('object_id', inplace=True)
    return agg_df_ts
