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
from typing import Dict


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
                 pseudo_n_loop=0,
                 pseudo_th=0.97,
                 pseudo_classes=[90],
                 save_pseudo_label=True,
                 cache_path_inner=None,
                 cache_path_extra=None,
                 pl_labels: Dict[str, str] = None,
                 use_cache=False):

        try:
            os.mkdir(basepath+log_name)
        except:
            pass

        df = pd.read_feather(basepath + 'input/meta.f')

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
            self.df_inner = self._setup(self.df_inner, features_inner, basepath, drop_feat_inner, cache_path_inner, use_cache)
            gc.collect()
        if self._use_extra:
            self.df_extra = self._setup(self.df_extra, features_extra, basepath, drop_feat_extra, cache_path_extra, use_cache)
            gc.collect()
            self.df_extra_pseudo = self.df_extra.copy()

        self.postproc_version = postproc_version
        self.pseudo_n_loop = pseudo_n_loop
        self.pseudo_classes = pseudo_classes
        self.pseudo_th = pseudo_th
        self.save_pseudo_label = save_pseudo_label
        self.pl_labels = pl_labels

    def _setup(self, df, features, basepath, drop, cache_path=None, use_cache=False) -> pd.DataFrame:

        if use_cache and cache_path is not None:
            try:
                print('load from cache: {}'.format(cache_path))
                return pd.read_feather(cache_path)
            except:
                pass

        for f in tqdm(features):
            if self.submit_path is None:
                tmp = pd.read_feather(basepath + 'features_tr/' + str(f) + '.f')
            else:
                tmp = pd.read_feather(basepath + 'features_all/' + str(f) + '.f')

            tmp['object_id'] = tmp['object_id'].astype(np.int32)

            df = pd.merge(df, tmp, on='object_id', how='left')

        if drop is not None:
            drop_ = [d for d in drop if d in df]
            print('dropped: {}'.format(drop_))
            df.drop(drop_, axis=1, inplace=True)

        if use_cache and cache_path:
            df.to_feather(cache_path)
        return df

    def _exec(self, name, df, model, pseudo_df=None, last_loop:bool=True):
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
            pred = model.fit_predict(df, self.logger, pseudo_df=pseudo_df, use_extra=last_loop)

        self.logger.info('training time: {}'.format(time.time() - s))

        importance = model.feature_importances()
        importance.reset_index(drop=True).to_feather(self.logdir+'importance_{}.f'.format(name))

        fi = importance.groupby('feature')['importance'].mean().reset_index()
        fi.sort_values(by='importance', ascending=False, inplace=True)
        fi = fi.reset_index(drop=True)
        self.logger.debug('importance:')
        for i in range(min(len(fi),30)):
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

    def _merge_oof(self, oof_inner, oof_outer, df_inner, df_extra):
        inner = self._make_df(oof_inner, self.model_inner, df_inner)
        outer = self._make_df(oof_outer, self.model_extra, df_extra)
        df = pd.concat([inner, outer]).sort_values(by='object_id').fillna(0).reset_index(drop=True)
        df.set_index('object_id', inplace=True)
        df['target'] = df['target'].astype(np.int32)
        return df[['target']+['class_'+str(i) for i in classes]]

    def _merge_variance(self, df_inner, df_extra):
        inner = pd.DataFrame(self.model_inner.variance,
                             columns=['class_{}'.format(c) for c in self.model_inner.clfs[0].classes_],
                             index=df_inner[df_inner.target.isnull()].object_id)

        extra = pd.DataFrame(self.model_extra.variance,
                             columns=['class_{}'.format(c) for c in self.model_extra.clfs[0].classes_],
                             index=df_extra[df_extra.target.isnull()].object_id)
        df = pd.concat([inner, extra]).sort_values(by='object_id').fillna(0).reset_index(drop=True)
        print(df.head())
        if 'object_id' in df:
            df.set_index('object_id', inplace=True)
        return df[['class_'+str(i) for i in classes]]

    def _update_pseudo_label(self, pred_extra: pd.DataFrame, round: int):
        print('before update: {} training samples'.format(
            self.df_extra_pseudo[~self.df_extra_pseudo.target.isnull()].shape))

        print(pred_extra.head(10))

        for cls in self.pseudo_classes:
            c = 'class_{}'.format(cls)
            print(c)
            obj_ids = pred_extra[pred_extra[c] > self.pseudo_th].object_id

            print('total {} samples exceeds threshold'.format(len(obj_ids)))

            df = self.df_extra_pseudo # train + test
            pseudo = df[df.object_id.isin(obj_ids)] # test
            non_pseudo = df[~df.object_id.isin(obj_ids)] # train + test

            pseudo.target = cls
            self.df_extra_pseudo = pd.concat([pseudo, non_pseudo]).reset_index(drop=True)

            if self.save_pseudo_label:
                tmp = pd.DataFrame({'object_id':pseudo.object_id})
                tmp.reset_index(drop=True).to_feather(self.logdir+'pseudo_label_class{}_round{}.f'.format(cls, round))

        print('after update: {} training samples'.format(
            self.df_extra_pseudo[~self.df_extra_pseudo.target.isnull()].shape))

    def _train_ids(self):
        extra_idx = self.df_extra[~self.df_extra.target.isnull()].reset_index().object_id.tolist()
        inner_idx = self.df_inner[~self.df_inner.target.isnull()].reset_index().object_id.tolist()
        return extra_idx + inner_idx

    def execute(self):
        if self._use_inner:
            print('exec-inner')
            pred_inner, oof_inner, y_inner = self._exec('inner', self.df_inner, self.model_inner)

        if self._use_extra:
            print('exec-outer')
            if self.pl_labels:
                df = self.df_extra_pseudo
                for c in self.pl_labels:
                    lbl = pd.read_feather(self.pl_labels[c])['object_id']
                    pseudo = df[df.object_id.isin(lbl)]  # test
                    non_pseudo = df[~df.object_id.isin(lbl)]  # train + test
                    pseudo.target = c
                    print('class {} : total {} samples for pseudo target'.format(c, len(pseudo)))
                    df = pd.concat([pseudo, non_pseudo]).reset_index(drop=True)
                self.df_extra_pseudo = df
                pred_extra, oof_outer, y_outer = self._exec('extra', self.df_extra, self.model_extra, self.df_extra_pseudo)
            elif self.pseudo_n_loop > 0 and self.submit_path:
                pred_extra = None
                for i in range(self.pseudo_n_loop):
                    if i > 0:
                        self._update_pseudo_label(pred_extra, i)
                    pred_extra, oof_outer, y_outer = self._exec('extra', self.df_extra, self.model_extra, self.df_extra_pseudo)
            else:
                pred_extra, oof_outer, y_outer = self._exec('extra', self.df_extra, self.model_extra, None)

        if self._use_extra and self._use_inner:
            self.oof = self._merge_oof(oof_inner, oof_outer, self.df_inner, self.df_extra_pseudo)
            save_confusion_matrix(self.oof.drop('target', axis=1).values, self.oof['target'], self.logdir+'oof_dual.png')
            self.oof.reset_index().to_feather(self.logdir+'oof.f')

            object_ids = self._train_ids()
            print('using data: {}'.format(len(object_ids)))
            oof_ = self.oof.reset_index(drop=False)
            self.oof_cv = oof_[oof_.object_id.isin(object_ids)].reset_index(drop=True)

            print(self.oof.shape)
            print(self.oof_cv.shape)


            save_confusion_matrix(self.oof_cv.drop(['target', 'object_id'], axis=1).values, self.oof_cv['target'],
                                  self.logdir + 'oof_dual_wo_pseudo.png')
            self.oof_cv.to_feather(self.logdir + 'oof_wo_pseudo.f')

            self.logger.debug('totalCV (with pseudo): {}'.format(multi_weighted_logloss(self.oof.target, self.oof.reset_index().drop(['object_id', 'target'], axis=1))))
            self.logger.debug('totalCV (w/o pseudo):  {}'.format(multi_weighted_logloss(self.oof_cv.target, self.oof_cv.drop(['object_id', 'target'], axis=1))))

            try:
                variance = self._merge_variance(self.df_inner, self.df_extra)
                variance.reset_index().to_feather(self.logdir + 'variance_over_folds.f')
            except:
                pass

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


