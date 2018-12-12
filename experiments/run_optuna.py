import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import LGBMClassifier
import gc
import time
import functools
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
import optuna
from tqdm import tqdm
import pandas as pd
import numpy as np

classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}

classes_with_other = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95, 99]
class_weight_with_other = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1, 99: 2}

classes_in = [6, 16, 53, 65, 92]
class_weight_in = {6: 1, 16: 1, 53: 1, 65: 1, 92: 1}

classes_out = [15, 42, 52, 62, 64, 67, 88, 90, 95]
class_weight_out = {15: 2, 42: 1, 52: 1, 62: 1, 64: 2, 67: 1, 88: 1, 90: 1, 95: 1}

class_extra_galaxtic = [90,42,15,62,88,67,52,95,64]
class_inner_galaxtic = [65,16,92,6,53]


def lgb_multi_weighted_logloss(y_true, y_preds):
    weight = {15: 2, 42: 1, 52: 1, 62: 1, 64: 2, 67: 1, 88: 1, 90: 1, 95: 1}
    cls = [15, 42, 52, 62, 64, 67, 88, 90, 95]

    y_p = y_preds.reshape(y_true.shape[0], len(cls), order='F')
    y_ohe = pd.get_dummies(y_true)
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    y_p_log = np.log(y_p)
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    class_arr = np.array([weight[k] for k in sorted(weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos
    loss = - np.sum(y_w) / np.sum(class_arr)

    return 'wloss', loss, False


def multi_weighted_logloss_(y_true, y_preds):
    if len(np.unique(y_true)) == 15:
        weight = class_weight_with_other
    elif len(np.unique(y_true)) == 14:
        weight = class_weight
    elif len(np.unique(y_true)) == 9:
        weight = class_weight_out
    else:
        weight = class_weight_in

    y_p = y_preds
    y_ohe = pd.get_dummies(y_true)
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)

    y_p_log = np.log(y_p)
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    class_arr = np.array([weight[k] for k in sorted(weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss

blacklist = ['2__fft_coefficient__coeff_0__attr_"abs"',
 '3__fft_coefficient__coeff_1__attr_"abs"',
 'ddf',
 'delta(first(detected), max(flux))_ch0',
 'delta(max(flux), last(detected))_ch0',
 'detected_median(flux)_ch0',
 'diff(min(flux))_1_2',
 'diff(min(flux))_3_4',
 'extra',
 'max(flux)_ch2',
 'max(flux)_ch3',
 'max(flux)_ch4',
 'max(flux)_ch5',
 'mean(detected)_ch0',
 'mean(detected)_ch4',
 'std(detected)_ch0',
 'std(detected)_ch1',
 'std(detected)_ch2',
 'std(detected)_ch4',
 'std(flux)_ch2',
 'std(flux)_ch3',
 'timescale_th0.35_min_ch3',
 'timescale_th0.5_min_ch3',
 'sn_salt2_ncall']

features = ['f000', 'f202', 'f100', 'f002', 'f104', 'f205', 'f010', 'f203', 'f200', 'f110',
                                         'f303', 'f304', 'f050', 'f400', 'f106', 'f107', 'f108','f140','f141','f142','f143',
                                         'f144',
                                         'f052','f053','f061','f063','f361','f600','f500','f1003','f1080','f1086','f1087']

if False:
    df = pd.read_feather('../input/meta.f')

    for f in tqdm(features):
        tmp = pd.read_feather('../features_tr/' + str(f) + '.f')
        tmp['object_id'] = tmp['object_id'].astype(np.int32)
        df = pd.merge(df, tmp, on='object_id', how='left')

    drop_ = [d for d in blacklist if d in df]
    print('dropped: {}'.format(drop_))
    df.drop(drop_, axis=1, inplace=True)

    for c in df:
        if df[c].dtype == 'object':
            df[c] = df[c].astype('category')

    if 'index' in df:
        df.drop('index', axis=1, inplace=True)

    x = df[~df['target'].isnull()]
    x_test = df[df['target'].isnull()]
    y = x['target'].astype(np.int32)
    x.drop('target', axis=1, inplace=True)
    x_test.drop('target', axis=1, inplace=True)
    x.set_index('object_id', inplace=True)
    x_test.set_index('object_id', inplace=True)

    n_folds = 5
    print('prep done.')
    x.reset_index().to_feather('x.f')
    y.reset_index().to_feather('y.f')

x = pd.read_feather('x.f')
x.set_index('object_id', inplace=True)
y = pd.read_feather('y.f')['target']
n_folds = 5

def objective(trial):

    x = pd.read_feather('x.f')
    x.set_index('object_id', inplace=True)
    y = pd.read_feather('y.f')['target']
    n_folds = 5

    x = x.reset_index(drop=True)[y.isin(classes_out)].reset_index(drop=True)
    y = y[y.isin(classes_out)].reset_index(drop=True)

    param = {
        'objective': 'ova',
        'num_class': 9,
        'verbose': -1,
        'boosting_type': 'gbdt',
        'max_depth': trial.suggest_categorical('max_depth', [16, 32]),
        'learning_rate': 0.001,
        'subsample': trial.suggest_uniform('subsample', 0.05, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.05, 1.0),
        'reg_alpha': 0,
        'reg_lambda': 0,
        'min_split_gain': 0,
        'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-3, 10),
        'max_bin': 256,
        'min_data_in_leaf': trial.suggest_categorical('min_data_in_leaf', [1, 5, 10]),
        'n_estimators': 10000,
    }

    if param['boosting_type'] == 'dart':
        param['drop_rate'] = trial.suggest_loguniform('drop_rate', 1e-8, 1.0)
        param['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)
        param['xgboost_dart_mode'] = trial.suggest_categorical('xgboost_dart_mode', [False, True])
    if param['boosting_type'] == 'goss':
        param['top_rate'] = trial.suggest_uniform('top_rate', 0.0, 1.0)
        param['other_rate'] = trial.suggest_uniform('other_rate', 0.0, 1.0 - param['top_rate'])

    #if param['boosting_type'] != 'goss':
    #    param['bagging_fraction'] = trial.suggest_uniform('bagging_fraction', 0.05, 1.0)
    #    param['bagging_freq'] = trial.suggest_int('bagging_freq', 0, 10)
    return run(param, x, y)


def run(param, x, y):
    print('param: {}'.format(param))

    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    n_classes = y.nunique()
    param['num_class'] = n_classes
    values = y.value_counts()
    n = {v: values[v] for v in values.index}
    w_train = np.array([class_weight[i] / n[i] for i in y])
    w_train /= w_train.mean()

    oof_preds = np.zeros((x.shape[0], n_classes))

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(x, y)):
        train_x, train_y = x.iloc[train_idx], y.iloc[train_idx]
        valid_x, valid_y = x.iloc[valid_idx], y.iloc[valid_idx]

        sample_weight = w_train[train_idx]

        print(valid_x.shape)
        print(valid_y.shape)

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(**param)

        print('fold {}'.format(n_fold))

        try:
            clf.fit(train_x, train_y, sample_weight=sample_weight, eval_set=[(valid_x, valid_y)],
                    eval_metric=lgb_multi_weighted_logloss,
                    verbose=20, early_stopping_rounds=500)
        except Exception as e:
            print('############## EXCEPTION ##################')
            print(e)

        oof_preds[valid_idx, :] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)

        loss = multi_weighted_logloss_(valid_y, oof_preds[valid_idx])

        if loss > 0.9:
            raise optuna.structs.TrialPruned()

        print('Fold {} loss: {}'.format(n_fold, loss))
        # print('Fold {} AUC : {:.6f}'.format(n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))

        del train_x, train_y, valid_x, valid_y
        gc.collect()

    full_auc = multi_weighted_logloss_(y, oof_preds)
    print('full loss: {}'.format(full_auc))

    return full_auc

if __name__ == '__main__':
    study = optuna.create_study()

    for i in range(100):
        study.optimize(objective, n_trials=5)

        print('Number of finished trials: {}'.format(len(study.trials)))

        print('Best trial:')
        trial = study.best_trial

        print('  Value: {}'.format(trial.value))

        print('  Params: ')
        for key, value in trial.params.items():
            print('    {}: {}'.format(key, value))