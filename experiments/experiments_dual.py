import pandas as pd
from typing import List
from tqdm import tqdm
from model.model import Model
import logging
import time
import gc
from model.postproc import *
from model.lgbm import multi_weighted_logloss
from .confusion_matrix import save_confusion_matrix
from model.problem import classes
import sys
import os


class ExperimentDualModel:
    def __init__(self, basepath: str,
                 features_inner: List[str],
                 features_extra: List[str],
                 model_inner: Model,
                 model_extra: Model,
                 submit_path: str = 'output/submission.csv',
                 log_name: str = 'default',
                 drop_feat_inner = None,
                 drop_feat_extra = None,
                 logging_level = logging.DEBUG,
                 postproc_version = 1,
                 mode='both',
                 pseudo_meta_path=None):

        try:
            os.mkdir(basepath+log_name)
        except:
            pass

        df = pd.read_feather(basepath + 'input/meta.f')

        if pseudo_meta_path is not None:
            print('using pseudo label')
            pseudo_df = pd.read_feather(basepath + pseudo_meta_path)
        else:
            pseudo_df = None

        self.mode = mode
        self.logdir = basepath+log_name+"/"

        if submit_path is None:
            self.submit_path = None
            df = df[~df.target.isnull()].reset_index() # use training data only
        else:
            self.submit_path = basepath + submit_path

        df['extra'] = (df['hostgal_photoz'] > 0.0).astype(np.int32)

        self.df_inner = df[df.extra == 0].reset_index(drop=True)
        self.df_extra = df[df.extra == 1].reset_index(drop=True)

        if pseudo_df is not None:
            pseudo_df['extra'] = (pseudo_df['hostgal_photoz'] > 0.0).astype(np.int32)
            self.pseudo_df_inner = pseudo_df[pseudo_df.extra == 0].reset_index(drop=True)
            self.pseudo_df_extra = pseudo_df[pseudo_df.extra == 1].reset_index(drop=True)
        else:
            self.pseudo_df_inner = None
            self.pseudo_df_extra = None

        self.model_inner = model_inner
        self.model_extra = model_extra
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging_level)
        self.fh = logging.FileHandler(self.logdir+'log.txt')
        self.fh.setLevel(logging_level)
        if len(self.logger.handlers) == 0:
            self.logger.addHandler(self.fh)

        self.logger.info('load features...')
        if self._use_inner:
            self.df_inner = self._setup(self.df_inner, features_inner, basepath, drop_feat_inner)
            if self.pseudo_df_inner is not None:
                self.pseudo_df_inner = self._setup(self.pseudo_df_inner, features_inner, basepath, drop_feat_inner)
            gc.collect()
        if self._use_extra:
            self.df_extra = self._setup(self.df_extra, features_extra, basepath, drop_feat_extra)
            if self.pseudo_df_extra is not None:
                self.pseudo_df_extra = self._setup(self.pseudo_df_extra, features_extra, basepath, drop_feat_extra)
            gc.collect()

        self.postproc_version = postproc_version

    def _setup(self, df, features, basepath, drop) -> pd.DataFrame:
        for f in tqdm(features):
            if self.submit_path is None:
                tmp = pd.read_feather(basepath + 'features_tr/' + str(f) + '.f')
            else:
                tmp = pd.read_feather(basepath + 'features/' + str(f) + '.f')

            df = pd.merge(df, tmp, on='object_id', how='left')
        if drop is not None:
            df.drop(drop, axis=1, inplace=True)
        return df

    def _exec(self, name, df, model, pseudo_df=None):
        self.logger.info(name)
        self.logger.info('features: {}'.format(df.columns.tolist()))
        self.logger.info('model: {}'.format(model.name()))
        self.logger.info('param: {}'.format(model.get_params()))
        s = time.time()

        if self.submit_path is None:
            x_train, y_train, _ = model.prep(df)
            model.fit(x_train, y_train, self.logger)
            pred = None
        else:
            pred = model.fit_predict(df, self.logger, pseudo_df=pseudo_df)

        self.logger.info('training time: {}'.format(time.time() - s))

        importance = model.feature_importances()

        fi = importance.groupby('feature')['importance'].mean().reset_index()
        fi.sort_values(by='importance', ascending=False, inplace=True)
        fi = fi.reset_index(drop=True)
        self.logger.debug('importance:')
        for i in range(30):
            self.logger.debug('{} : {}'.format(fi.loc[i, 'feature'], fi.loc[i, 'importance']))

        oof, y = model.get_oof_prediction()
        return pred, oof, y

    @property
    def _use_inner(self):
        return self.mode == 'inner-only' or self.mode == 'both'

    @property
    def _use_extra(self):
        return self.mode == 'extra-only' or self.mode == 'both'

    def _make_df(self, oof, model, df):
        classes = ['class_' + str(c) for c in model.clfs[0].classes_]
        d = pd.DataFrame(oof, columns=classes)
        d['object_id'] = df['object_id']
        d['target'] = df['target']
        return d[['object_id', 'target'] + classes]

    def _merge_oof(self, oof_inner, oof_outer):
        inner = self._make_df(oof_inner, self.model_inner, self.df_inner)
        outer = self._make_df(oof_outer, self.model_extra, self.df_extra)
        df = pd.concat([inner, outer]).sort_values(by='object_id').fillna(0).reset_index(drop=True)
        df.set_index('object_id', inplace=True)
        df['target'] = df['target'].astype(np.int32)
        return df[['target']+['class_'+str(i) for i in classes]]

    def execute(self):
        if self._use_inner:
            print('exec-inner')
            pred_inner, oof_inner, y_inner = self._exec('inner', self.df_inner, self.model_inner, self.pseudo_df_inner)

        if self._use_extra:
            print('exec-outer')
            pred_extra, oof_outer, y_outer = self._exec('extra', self.df_extra, self.model_extra, self.pseudo_df_extra)

        if self._use_extra and self._use_inner:
            self.oof = self._merge_oof(oof_inner, oof_outer)
            save_confusion_matrix(self.oof.drop('target', axis=1).values, self.oof['target'], self.logdir+'oof_dual.png')

        if self.submit_path is not None:
            pred_all = pd.concat([pred_inner, pred_extra]).fillna(0)
            if self.postproc_version == 1:
                pred_all = add_class99(pred_all)
            elif self.postproc_version == 2:
                pred_all = add_class99_2(pred_all)
            else:
                raise NotImplementedError()
            submit(pred_all, self.submit_path)


    def score(self, type='extra'):
        if type == 'inner':
            return self.model_inner.score()
        else:
            return self.model_extra.score()

    def scores(self, type='extra'):
        if type == 'inner':
            return self.model_inner.scores()
        else:
            return self.model_extra.scores()


