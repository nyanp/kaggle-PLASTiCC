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


class Model:
    def __int__(self):
        pass

    def prep(self, df, target = 'target', index = 'object_id', pseudo_df = None):
        for c in df:
            if df[c].dtype == 'object':
                df[c] = df[c].astype('category')

        if 'index' in df:
            df.drop('index', axis=1, inplace=True)

        if pseudo_df is not None:
            x_train = pseudo_df[~pseudo_df[target].isnull()]
        else:
            x_train = df[~df[target].isnull()]
        x_test = df[df[target].isnull()]
        y_train = x_train[target].astype(np.int32)

        x_train.drop(target, axis=1, inplace=True)
        x_test.drop(target, axis=1, inplace=True)
        x_train.set_index(index, inplace=True)
        x_test.set_index(index, inplace=True)

        if pseudo_df is not None:
            print('number of training: {}'.format(len(x_train)))
            print('number of test: {}'.format(len(x_test)))

        assert 'index' not in x_train

        return x_train, y_train, x_test

    def fit_predict(self, df: pd.DataFrame, logger = None, pseudo_df: pd.DataFrame = None, use_extra: bool = True) -> pd.DataFrame:
        x_train, y_train, x_test = self.prep(df, pseudo_df=pseudo_df)
        self.fit(x_train, y_train, logger, use_extra)
        return self.predict(x_test)

    def fit(self, x, y, logger = None, use_extra = True):
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
