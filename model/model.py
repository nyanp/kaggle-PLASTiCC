from typing import List
import pandas as pd
import numpy as np
import time
import gc

import lightgbm as lgb
from lightgbm import LGBMClassifier
from contextlib import contextmanager
from sklearn.model_selection import KFold, StratifiedKFold

from .problem import *


def lgb_multi_weighted_logloss(y_true, y_preds):

    if len(np.unique(y_true)) == 15:
        weight = class_weight_with_other
        cls = classes_with_other
    elif len(np.unique(y_true)) == 14:
        weight = class_weight
        cls = classes
    elif len(np.unique(y_true)) == 9:
        weight = class_weight_out
        cls = classes_out
    else:
        weight = class_weight_in
        cls = classes_in

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


def multi_weighted_logloss(y_true, y_preds):

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


class Model:
    def __int__(self):
        pass

    def prep(self, df, target = 'target', index = 'object_id'):
        for c in df:
            if df[c].dtype == 'object':
                df[c] = df[c].astype('category')

        if 'index' in df:
            df.drop('index', axis=1, inplace=True)

        x_train = df[~df[target].isnull()]
        x_test = df[df[target].isnull()]
        y_train = x_train[target].astype(np.int32)

        x_train.drop(target, axis=1, inplace=True)
        x_test.drop(target, axis=1, inplace=True)
        x_train.set_index(index, inplace=True)
        x_test.set_index(index, inplace=True)

        assert 'index' not in x_train

        return x_train, y_train, x_test

    def fit_predict(self, df: pd.DataFrame, logger = None) -> pd.DataFrame:
        x_train, y_train, x_test = self.prep(df)
        self.fit(x_train, y_train, logger)
        return self.predict(x_test)

    def fit(self, x, y, logger = None):
        raise NotImplementedError()

    def predict(self, x) -> pd.DataFrame:
        raise NotImplementedError()

    def feature_importances(self) -> pd.DataFrame:
        raise NotImplementedError()

    def score(self) -> float:
        raise NotImplementedError()

    def scores(self) -> List[float]:
        raise NotImplementedError()

    def name(self):
        raise NotImplementedError()

    def get_params(self):
        raise NotImplementedError()

    def get_oof_prediction(self):
        raise NotImplementedError()
