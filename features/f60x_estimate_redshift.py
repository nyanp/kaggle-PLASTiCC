from typing import List

import lightgbm as lgb
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm

import common
import config


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

    # NOTE: remove_galactic_test_data=False is meaningless. It is enabled just to reproduce original feature value.
    _make_redshift_feature(params, "f600", nfolds=5, remove_galactic_test_data=False)


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
    _make_redshift_feature(params, "f601", nfolds=10)


def f701_redshift_difference():
    f601_estimate_redshift()
    estimated = common.load_feature("f601")
    meta = common.load_metadata()
    dst = pd.merge(meta[['object_id', 'hostgal_photoz']], estimated, on='object_id', how='left')
    dst['hostgal_photoz_predicted_diff'] = dst['hostgal_photoz'] - dst['hostgal_z_predicted']

    common.save_feature(dst[['object_id', 'hostgal_photoz_predicted_diff']], "f701")


def _make_redshift_feature(params, feature_id: str, nfolds: int, remove_galactic_test_data: bool = True):
    x_train, x_test, y_train = _make_df(
        ['f000', 'f202', 'f100', 'f002', 'f104', 'f205', 'f010', 'f203', 'f200', 'f110',
         'f303', 'f304', 'f050', 'f400', 'f106', 'f107', 'f108', 'f140', 'f141', 'f142',
         'f143', 'f144', 'f150', 'f052', 'f053', 'f061', 'f063', 'f361'], remove_galactic_test_data)

    feature = _estimate_redshift(params, x_train, x_test, y_train, nfolds=nfolds)
    common.save_feature(feature, feature_id)


def _make_df(input_feature_list: List[str], remove_galactic_test_data: bool = True):
    df = common.load_metadata()

    for f in tqdm(input_feature_list):
        df = pd.merge(df, common.load_feature(f), on='object_id', how='left')

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
