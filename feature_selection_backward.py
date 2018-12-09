from model.lgbm import LGBMModel
from experiments.experiments import Experiment
from experiments.experiments_dual import ExperimentDualModel
import pandas as pd
import gc
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from model.problem import class_weight
from model.loss import lgb_multi_weighted_logloss, multi_weighted_logloss
from lightgbm import LGBMClassifier
from tqdm import tqdm


baseline_features = ['f000', 'f202', 'f100', 'f002', 'f104', 'f205', 'f010', 'f203', 'f200', 'f110',
                                         'f303', 'f304', 'f050', 'f400', 'f106', 'f107', 'f108','f140','f141','f142','f143',
                                         'f144',
                                         'f052','f053','f061','f063','f361','f600','f500','f1003','f1080','f1086','f1087','f509',"f510_hsiao",]

drop_feat=['hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b']

param = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 14,
    'metric': 'multi_logloss',
    'subsample': .9,
    'colsample_bytree': .9,
    'reg_alpha': 0,
    'reg_lambda': 3,
    'min_split_gain': 0,
    'min_child_weight': 10,
    'silent': True,
    'verbosity': -1,
    'learning_rate': 0.1,
    'max_depth': 3,
    'min_data_in_leaf': 1,
    'n_estimators': 10000,
    'max_bin': 128,
    'bagging_fraction': 0.66,
    'verbose': -1
}

logdir = 'log_fsb_181208'
n_loop = 10


df = pd.read_feather('input/meta.f')
x = df[~df.target.isnull()]
x = x[x.hostgal_photoz > 0]
y = x.target.astype(np.int32)
x.drop('target', axis=1, inplace=True)

for f in tqdm(baseline_features):
    tmp = pd.read_feather('features_tr/{}.f'.format(f))
    x = pd.merge(x, tmp, on='object_id', how='left')

for c in drop_feat:
    if c in x:
        x.drop(c, axis=1, inplace=True)
x.set_index('object_id', inplace=True)

print('x: {}'.format(x.shape))

try:
    os.mkdir(logdir)
except:
    pass


def loop(x, y, param, folds, logger):
    baseline = trial(x, y, param, folds, logger, 'baseline')

    best_score = baseline
    best_features_to_drop = None

    for c in x:
        score = trial(x.drop(c, axis=1), y, param, folds, logger, trial_name='{}'.format(c))
        if score < best_score:
            print('!! best score updated from {} to {} on {}'.format(best_score, score, c))
            best_score = score
            best_features_to_drop = c

    return best_score, best_features_to_drop


def trial(x, y, param, nfolds, logger, trial_name) -> float:
    folds = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=0)
    feature_importance_df = pd.DataFrame()
    n_classes = y.nunique()

    oof_preds = np.zeros((x.shape[0], n_classes))

    score = []

    n_classes = len(np.unique(y))

    param['num_class'] = n_classes


    values = y.value_counts()
    n = {v: values[v] for v in values.index}
    w_train = np.array([class_weight[i] / n[i] for i in y])
    w_train /= w_train.mean()

    best_iterations = []

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(x, y)):
        train_x, train_y = x.iloc[train_idx], y.iloc[train_idx]
        valid_x, valid_y = x.iloc[valid_idx], y.iloc[valid_idx]

        if w_train is not None:
            sample_weight = w_train[train_idx]
        else:
            sample_weight = None

        print(train_x.shape)

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(**param)

        clf.fit(train_x, train_y, sample_weight=sample_weight, eval_set=[(valid_x, valid_y)],
                eval_metric=lgb_multi_weighted_logloss, verbose=-1, early_stopping_rounds=100)

        oof_preds[valid_idx, :] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)
        best_iterations.append(clf.best_iteration_)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = x.columns.tolist()
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        loss = multi_weighted_logloss(valid_y, oof_preds[valid_idx])
        print('Fold {} loss: {}'.format(n_fold + 1, loss))
        score.append(loss)

        del train_x, train_y, valid_x, valid_y
        gc.collect()

    # print('Full AUC score %.6f' % roc_auc_score(y, oof_preds))
    # display_importances(feature_importance_df)
    full_auc = multi_weighted_logloss(y, oof_preds)
    print('*** full auc: {}'.format(full_auc))

    logger.write('{},{},{}\n'.format(trial_name, full_auc, ','.join([str(s) for s in score])))
    logger.flush()

    return full_auc


with open(logdir + '/summary.csv', 'w') as logger:
    dropped = []
    for i in range(n_loop):
        print('{}-th loop. baseline dropped:{}'.format(i, dropped))
        score, c = loop(x, y, param, 5, logger)

        if c is None:
            print('No features to select. finish')
            break

        dropped.append(c)
