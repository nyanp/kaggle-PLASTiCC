import pandas as pd
import numpy as np
from tqdm import tqdm

import common


def save_v1():
    features = ['f000', 'f001', 'f002', 'f010', 'f026', 'f050', 'f051', 'f052', 'f053', 'f054',
                'f061', 'f063', 'f100', 'f1000', 'f1001', 'f1002', 'f1003', 'f1004', 'f1005',
                'f1006', 'f101', 'f1010', 'f102', 'f103', 'f104', 'f106', 'f107', 'f108', 'f1080',
                'f1081', 'f1082', 'f1083', 'f1085', 'f1086', 'f1087', 'f1088', 'f1089',
                'f109', 'f110', 'f140', 'f141', 'f142', 'f143', 'f144', 'f150', 'f151', 'f152',
                'f153', 'f200', 'f201', 'f202', 'f203', 'f204', 'f205', 'f300', 'f301', 'f302',
                'f303', 'f304', 'f305', 'f306', 'f307', 'f308', 'f309', 'f310', 'f311', 'f330',
                'f340', 'f350', 'f370', 'f400', 'f500', 'f505', 'f506', 'f507', 'f600', 'f701']

    best_subset_v1 = ['object_id', 'hostgal_photoz_err', 'distmod', 'mwebv', 'mean(flux)_ch0', 'mean(flux)_ch1',
               'mean(flux)_ch2', 'mean(flux)_ch3', 'mean(flux)_ch4', 'mean(flux)_ch5', 'max(flux)_ch0',
               'max(flux)_ch1', 'min(flux)_ch0', 'min(flux)_ch1', 'min(flux)_ch2', 'min(flux)_ch3', 'min(flux)_ch4',
               'min(flux)_ch5', 'median(flux)_ch0', 'median(flux)_ch1', 'median(flux)_ch2', 'median(flux)_ch3',
               'median(flux)_ch4', 'median(flux)_ch5', 'std(flux)_ch0', 'std(flux)_ch1', 'std(flux)_ch4',
               'std(flux)_ch5', 'timescale_th0.35_max_ch0', 'timescale_th0.35_max_ch1', 'timescale_th0.35_max_ch2',
               'timescale_th0.35_max_ch3', 'timescale_th0.35_max_ch4', 'timescale_th0.35_max_ch5',
               'diff(max(flux))_0_1', 'diff(max(flux))_1_2', 'diff(max(flux))_2_3', 'diff(max(flux))_3_4',
               'diff(max(flux))_4_5', 'mean(detected)_ch1', 'mean(detected)_ch2', 'mean(detected)_ch3',
               'mean(detected)_ch5', 'std(detected)_ch3', 'std(detected)_ch5', 'diff(max(flux))_0_3',
               'diff(max(flux))_1_4', 'diff(max(flux))_2_5', 'timescale_th0.5_min_ch0', 'timescale_th0.5_min_ch1',
               'timescale_th0.5_min_ch2', 'timescale_th0.5_min_ch4', 'timescale_th0.5_min_ch5', 'mean(flux)',
               'max(flux)', 'min(flux)', 'timescale_th0.35_min_ch0', 'timescale_th0.35_min_ch1',
               'timescale_th0.35_min_ch2', 'timescale_th0.35_min_ch4', 'timescale_th0.35_min_ch5',
               'timescale_th0.15_max_ch0', 'timescale_th0.15_max_ch1', 'timescale_th0.15_max_ch2',
               'timescale_th0.15_max_ch3', 'timescale_th0.15_max_ch4', 'timescale_th0.15_max_ch5',
               'max(flux_slope)_ch0', 'max(flux_slope)_ch1', 'max(flux_slope)_ch2', 'max(flux_slope)_ch3',
               'max(flux_slope)_ch4', 'max(flux_slope)_ch5', 'min(flux_slope)_ch0', 'min(flux_slope)_ch1',
               'min(flux_slope)_ch2', 'min(flux_slope)_ch3', 'min(flux_slope)_ch4', 'min(flux_slope)_ch5',
               'flux__c3__lag_1_ch0', 'flux__c3__lag_1_ch1', 'flux__c3__lag_1_ch2', 'flux__c3__lag_1_ch3',
               'flux__c3__lag_1_ch4', 'flux__c3__lag_1_ch5', 'flux__autocorrelation__lag_1_ch0',
               'flux__autocorrelation__lag_1_ch1', 'flux__autocorrelation__lag_1_ch2',
               'flux__autocorrelation__lag_1_ch3', 'flux__autocorrelation__lag_1_ch4',
               'flux__autocorrelation__lag_1_ch5', 'delta', 'max(astropy.lombscargle.power)_ch0',
               'max(astropy.lombscargle.power)_ch1', 'max(astropy.lombscargle.power)_ch2',
               'max(astropy.lombscargle.power)_ch3', 'max(astropy.lombscargle.power)_ch4',
               'max(astropy.lombscargle.power)_ch5', 'astropy.lombscargle.timescale_ch0',
               'astropy.lombscargle.timescale_ch1', 'astropy.lombscargle.timescale_ch2',
               'astropy.lombscargle.timescale_ch3', 'astropy.lombscargle.timescale_ch4',
               'astropy.lombscargle.timescale_ch5', 'diff(max(flux))_0_4', 'diff(max(flux))_1_5',
               'diff(max(flux))_0_5', 'diff(min(flux))_0_1', 'diff(min(flux))_2_3', 'diff(min(flux))_4_5',
               'amp(flux)_ch0/ch1', 'amp(flux)_ch1/ch2', 'amp(flux)_ch2/ch3', 'amp(flux)_ch3/ch4', 'amp(flux)_ch4/ch5',
               'amp(flux)_ch0/ch2', 'amp(flux)_ch1/ch3', 'amp(flux)_ch2/ch4', 'amp(flux)_ch3/ch5', 'amp(flux)_ch0/ch3',
               'amp(flux)_ch1/ch4', 'amp(flux)_ch2/ch5', 'amp(flux)_ch0/ch4', 'amp(flux)_ch1/ch5', 'amp(flux)_ch0/ch5',
               'delta(max(flux), last(detected))', 'delta(first(detected), max(flux))',
               'delta(max(flux), last(detected))_ch1', 'delta(max(flux), last(detected))_ch2',
               'delta(max(flux), last(detected))_ch3', 'delta(max(flux), last(detected))_ch4',
               'delta(max(flux), last(detected))_ch5', 'delta(first(detected), max(flux))_ch1',
               'delta(first(detected), max(flux))_ch2', 'delta(first(detected), max(flux))_ch3',
               'delta(first(detected), max(flux))_ch4', 'delta(first(detected), max(flux))_ch5',
               'detected_median(flux)_ch1', 'detected_median(flux)_ch2', 'detected_median(flux)_ch3',
               'detected_median(flux)_ch4', 'detected_median(flux)_ch5', 'detected_diff(median(flux))_0_1',
               'detected_diff(median(flux))_1_2', 'detected_diff(median(flux))_2_3', 'detected_diff(median(flux))_3_4',
               'detected_diff(median(flux))_4_5', '0__fft_coefficient__coeff_0__attr_"abs"',
               '0__fft_coefficient__coeff_1__attr_"abs"', '0__kurtosis', '0__skewness',
               '1__fft_coefficient__coeff_0__attr_"abs"', '1__fft_coefficient__coeff_1__attr_"abs"', '1__kurtosis',
               '1__skewness', '2__fft_coefficient__coeff_1__attr_"abs"', '2__kurtosis', '2__skewness',
               '3__fft_coefficient__coeff_0__attr_"abs"', '3__kurtosis', '3__skewness',
               '4__fft_coefficient__coeff_0__attr_"abs"', '4__fft_coefficient__coeff_1__attr_"abs"', '4__kurtosis',
               '4__skewness', '5__fft_coefficient__coeff_0__attr_"abs"', '5__fft_coefficient__coeff_1__attr_"abs"',
               '5__kurtosis', '5__skewness', 'hostgal_z_predicted', 'sn_salt2_chisq', 'sn_salt2_z', 'sn_salt2_t0',
               'sn_salt2_x0', 'sn_salt2_x1', 'sn_salt2_c', 'sn_salt2_z_err', 'sn_salt2_t0_err', 'sn_salt2_x0_err',
               'sn_salt2_x1_err', 'sn_salt2_c_err', 'luminosity_est_diff_ch0', 'luminosity_est_diff_ch1',
               'luminosity_est_diff_ch2', 'luminosity_est_diff_ch3', 'luminosity_est_diff_ch4',
               'luminosity_est_diff_ch5']

    best16_v1 = ['object_id', 'sn_salt2_c', 'delta', 'sn_salt2_x1', 'distmod',
                 'luminosity_est_diff_ch4', 'luminosity_est_diff_ch5',
                 'luminosity_est_diff_ch3', 'luminosity_est_diff_ch2',
                 'hostgal_photoz_err', 'hostgal_z_predicted', 'luminosity_est_diff_ch0',
                 'luminosity_est_diff_ch1', 'sn_salt2_chisq', 'sn_salt2_z',
                 'amp(flux)_ch3/ch5', '0__skewness']

    base = common.load_metadata()[['object_id', 'distmod', 'hostgal_photoz_err', 'mwebv', 'target']]

    for f in tqdm(features):
        tmp = common.load_feature(f)
        if f == 'f1080':
            tmp.columns = ['object_id', 'delta_SNR3']
        if f == 'f1010':
            tmp.columns = ['object_id'] + [c + '_estimated' for c in tmp.columns.tolist()[1:]]

        for c in tmp:
            if c == 'object_id':
                continue

            if c in base:
                print('{} is already in base(f): {}, {}'.format(c, f, base.columns.tolist()))
            assert c not in base
        tmp['object_id'] = tmp['object_id'].astype(np.int32)
        base = pd.merge(base, tmp, on='object_id', how='left')

    # -> yuval
    _save(base[best_subset_v1+['target']], 'nyanp_feat_v1_{}')
    _save(base[best16_v1+['target']], 'nyanp_feat_v1_{}_top16')

    # add prefix to oof features
    xlist = [
        'hostgal_z_predicted',
        'hostgal_photoz_predicted_diff',
        'luminosity_est_diff_ch0',
        'luminosity_est_diff_ch1',
        'luminosity_est_diff_ch2',
        'luminosity_est_diff_ch3',
        'luminosity_est_diff_ch4',
        'luminosity_est_diff_ch5',
        'luminosity_diff_ch0_estimated',
        'luminosity_diff_ch1_estimated',
        'luminosity_diff_ch2_estimated',
        'luminosity_diff_ch3_estimated',
        'luminosity_diff_ch4_estimated',
        'luminosity_diff_ch5_estimated'
    ]

    renames = {x: 'xxx_' + x for x in xlist}
    base.rename(columns=renames, inplace=True)

    # -> mamas
    _save(base.drop(['distmod', 'hostgal_photoz_err', 'mwebv'], axis=1), 'features_nyanp_all_v1_{}')


# -> mamas
def save_v2():
    base = common.load_metadata()[['object_id', 'target']]
    base = pd.merge(base, common.load_feature('f509'), on='object_id', how='inner')
    _save(base, 'features_nyanp_all_v2_{}')


# -> mamas
def save_v3():
    features = ['f510', 'f511', 'f512']
    base = common.load_metadata()[['object_id', 'target']]

    for f in features:
        tmp = common.load_feature(f)
        base = pd.merge(base, tmp, on='object_id', how='left')
    _save(base, 'features_nyanp_all_v3_{}')


# -> mamas
def save_v4():
    features = ['f513', 'f515', 'f517']
    base = common.load_metadata()[['object_id', 'target']]

    for f in features:
        tmp = common.load_feature(f)
        base = pd.merge(base, tmp, on='object_id', how='left')
    _save(base, 'features_nyanp_all_v4_{}')


def _save(base, postfix):
    base_train = base[~base.target.isnull()].drop('target', axis=1).reset_index(drop=True)
    base_test = base[base.target.isnull()].drop('target', axis=1).reset_index(drop=True)

    print(base_test.shape)
    print(base_train.shape)
    common.save_shared_file(base_train, postfix.format('train')+'.f')
    common.save_shared_file(base_test, postfix.format('test')+'.f')


print('save_v1')
save_v1()
print('save_v2')
save_v2()
print('save_v3')
save_v3()
print('save_v4')
save_v4()
