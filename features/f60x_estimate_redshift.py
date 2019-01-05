from typing import List

import lightgbm as lgb
import pandas as pd
import numpy as np

from astropy.cosmology import default_cosmology
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm

import common
import config

cosmo = default_cosmology.get()


def z2pc(z):
    return cosmo.luminosity_distance(z).value


def f600_estimate_redshift():
    params = {
        'boosting_type': 'gbdt',
        'objective': 'mse',
        'learning_rate': 0.05,
        'subsample': .9,
        'colsample_bytree': .7,
        'reg_alpha': 1.0e-2,
        'reg_lambda': 1.0e-2,
        'verbose': -1,
        'max_depth': 5,
        'importance_type': 'gain',
        'n_jobs': -1
    }
    features = ['f000', 'f202', 'f100', 'f002', 'f104', 'f205', 'f010', 'f203', 'f200', 'f110',
         'f303', 'f304', 'f050', 'f400', 'f106', 'f107', 'f108', 'f140', 'f141', 'f142',
         'f143', 'f144', 'f150', 'f052', 'f053', 'f061', 'f063', 'f361']

    # NOTE: remove_galactic_test_data=False is meaningless. It is enabled just to reproduce original feature value.
    _make_redshift_feature(params, features, "f600", nfolds=5, remove_galactic_test_data=False)


def f601_estimate_redshift():
    params = {
        'boosting_type': 'gbdt',
        'objective': 'mse',
        'learning_rate': 0.02,
        'subsample': .9,
        'colsample_bytree': .7,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0e-2,
        'verbose': -1,
        'max_depth': 7,
        'importance_type': 'gain',
        'n_jobs': -1
    }
    features = ['f000', 'f202', 'f100', 'f002', 'f104', 'f205', 'f010', 'f203', 'f200', 'f110',
         'f303', 'f304', 'f050', 'f400', 'f106', 'f107', 'f108', 'f140', 'f141', 'f142',
         'f143', 'f144', 'f150', 'f052', 'f053', 'f061', 'f063', 'f361']
    _make_redshift_feature(params, features, "f601", nfolds=10)


def f603_estimate_redshift():
    params = {
        'boosting_type': 'gbdt',
        'objective': 'mse',
        'learning_rate': 0.05,
        'subsample': .9,
        'colsample_bytree': .7,
        'reg_alpha': 1.0e-2,
        'reg_lambda': 1.0e-2,
        'verbose': -1,
        'max_depth': 5,
        'importance_type': 'gain',
        'n_jobs': -1
    }
    features = ['f000', 'f202', 'f100', 'f002', 'f104', 'f205', 'f010', 'f203', 'f200', 'f110',
                'f303', 'f304', 'f050', 'f400', 'f106', 'f107', 'f108','f140','f141','f142','f143', 'f144',
                'f052','f053','f061','f063','f361','f500']
    drop_features = ['ra', 'decl', 'gal_l', 'gal_b', 'ddf',
                     'hostgal_specz', 'hostgal_photoz',
                     'max(flux)_ch2', 'max(flux)_ch3', 'max(flux)_ch4', 'max(flux)_ch5',
                     'std(flux)_ch2', 'std(flux)_ch3',
                     'mean(detected)_ch0', 'mean(detected)_ch4',
                     'std(detected)_ch0', 'std(detected)_ch1', 'std(detected)_ch2', 'std(detected)_ch4',
                     'timescale_th0.5_min_ch3', 'timescale_th0.35_min_ch3',
                     'diff(min(flux))_1_2', 'diff(min(flux))_3_4',
                     'delta(max(flux), last(detected))_ch0', 'delta(first(detected), max(flux))_ch0',
                     'detected_median(flux)_ch0',
                     '2__fft_coefficient__coeff_0__attr_"abs"', '3__fft_coefficient__coeff_1__attr_"abs"',
                     'sn_salt2_ncall', 'distmod']
    _make_redshift_feature(params, features, "f603", nfolds=10, drop_features=drop_features)


def f701_redshift_difference():
    f601_estimate_redshift()
    estimated = common.load_feature("f601")
    meta = common.load_metadata()
    dst = pd.merge(meta[['object_id', 'hostgal_photoz']], estimated, on='object_id', how='left')
    dst['hostgal_photoz_predicted_diff'] = dst['hostgal_photoz'] - dst['hostgal_z_predicted']

    common.save_feature(dst[['object_id', 'hostgal_photoz_predicted_diff']], "f701")


def f1010_redshift_difference_perch():
    meta = common.load_metadata()
    meta = pd.merge(meta, common.load_feature('f603'), on='object_id', how='left')
    meta = pd.merge(meta, common.load_feature('f000'), on='object_id', how='left')

    meta['Mpc'] = meta['hostgal_z_predicted'].apply(z2pc)
    meta['Gpc'] = meta['Mpc'] / 1000.0

    features = []
    for i in range(6):
        ch = i
        meta['flux_diff_ch{}'.format(ch)] = meta['max(flux)_ch{}'.format(ch)] - meta['min(flux)_ch{}'.format(ch)]
        meta['luminosity_diff_ch{}'.format(ch)] = meta['flux_diff_ch{}'.format(ch)] * meta['Gpc'] * meta['Gpc']
        features.append('luminosity_diff_ch{}'.format(ch))

    common.save_feature(meta[['object_id'] + features], "f1010")


def _make_redshift_feature(params, src_features: List[str], feature_id: str, nfolds: int,
                           remove_galactic_test_data: bool = True,
                           drop_features: List[str] = None):
    x_train, x_test, y_train = _make_df(src_features, remove_galactic_test_data, drop_features)

    feature = _estimate_redshift(params, x_train, x_test, y_train, nfolds=nfolds)
    common.save_feature(feature, feature_id)


def _make_df(input_feature_list: List[str], remove_galactic_test_data: bool = True, drop_features: List[str] = None):
    df = common.load_metadata()

    for f in tqdm(input_feature_list):
        df = pd.merge(df, common.load_feature(f), on='object_id', how='left')

    if drop_features is not None:
        df.drop(drop_features, axis=1, inplace=True)

    df.set_index('object_id', inplace=True)

    x_train = df[df.hostgal_specz > 0.0]

    if remove_galactic_test_data:
        x_test = df[df.hostgal_specz.isnull() & (df.hostgal_photoz > 0.0)]
    else:
        x_test = df[df.hostgal_specz.isnull()]

    x_train.drop('target', axis=1, inplace=True)
    x_test.drop('target', axis=1, inplace=True)

    y_train = x_train.hostgal_specz
    x_train.drop('hostgal_specz', axis=1, inplace=True)
    x_test.drop('hostgal_specz', axis=1, inplace=True)

    return x_train, x_test, y_train


def _estimate_redshift(params, x_train, x_test, y_train, feature_name='hostgal_z_predicted', nfolds=10):
    folds = KFold(nfolds)

    models = []

    oof_preds = np.zeros((x_train.shape[0], 1))

    feature_importance_df = pd.DataFrame()

    print('Start training')

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(x_train, y_train)):
        train_x, train_y = x_train.iloc[train_idx], y_train.iloc[train_idx]
        valid_x, valid_y = x_train.iloc[valid_idx], y_train.iloc[valid_idx]

        train_set = lgb.Dataset(train_x, train_y)
        valid_set = lgb.Dataset(valid_x, valid_y)

        booster = lgb.train(params, train_set, num_boost_round=2000, early_stopping_rounds=50, verbose_eval=100,
                            valid_sets=[valid_set])
        models.append(booster)

        oof_preds[valid_idx, 0] = booster.predict(valid_x)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = x_train.columns.tolist()
        fold_importance_df["importance"] = booster.feature_importance(importance_type='gain')
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    test_preds = np.zeros((x_test.shape[0], len(models)))

    for i, clf in enumerate(models):
        print(i)
        test_preds[:, i] = clf.predict(x_test)

    df_oof = pd.DataFrame({'actual':y_train,
                  'photoz': x_train['hostgal_photoz'],
                  'photoz_err': x_train['hostgal_photoz_err'],
                  'predicted':oof_preds.flatten()})

    print('MSE of photoz - spcez (original)')
    print(mean_squared_error(df_oof['actual'], df_oof['photoz']))

    print('MSE of photoz - spcez (trained)')
    print(mean_squared_error(df_oof['actual'], df_oof['predicted']))

    f_test = pd.DataFrame({'predicted': test_preds.mean(axis=1),
                            'object_id': x_test.index})

    f_train = pd.DataFrame({'predicted': oof_preds.flatten(),
                            'object_id': x_train.index})

    f_all = pd.concat([f_train, f_test])

    f_all.rename(columns={'predicted': feature_name}, inplace=True)

    return f_all.reset_index(drop=True)
