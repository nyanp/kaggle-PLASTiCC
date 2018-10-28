import pandas as pd
from typing import List
from tqdm import tqdm
from model.model import Model
import logging
import time
import gc
from model.postproc import *
from model.model import multi_weighted_logloss

class ExperimentDualModel:
    def __init__(self, basepath: str,
                 features_inner: List[str],
                 features_extra: List[str],
                 model_inner: Model,
                 model_extra: Model,
                 submit_path: str = 'output/submission.csv',
                 log_name: str = 'default',
                 drop_feat_inner = None,
                 drop_feat_extra = None):

        df = pd.read_feather(basepath + 'input/meta.f')

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
        self.logger.setLevel(logging.DEBUG)
        self.fh = logging.FileHandler(basepath+log_name+'.log')
        self.fh.setLevel(logging.DEBUG)
        self.logger.addHandler(self.fh)

        self.logger.info('load features...')
        self.df_inner = self._setup(self.df_inner, features_inner, basepath, drop_feat_inner)
        gc.collect()
        self.df_extra = self._setup(self.df_extra, features_extra, basepath, drop_feat_extra)
        gc.collect()

    def _setup(self, df, features, basepath, drop) -> pd.DataFrame:
        for f in tqdm(features):
            tmp = pd.read_feather(basepath + 'features/' + str(f) + '.f')

            df = pd.merge(df, tmp, on='object_id', how='left')
        if drop is not None:
            df.drop(drop, axis=1, inplace=True)
        return df

    def _exec(self, name, df, model):
        self.logger.info(name)
        self.logger.info('features: {}'.format(df.columns.tolist()))
        self.logger.info('model: {}'.format(model.name()))
        self.logger.info('param: {}'.format(model.get_params()))
        s = time.time()

        if self.submit_path is None:
            x_train, y_train, _ = model.prep(df)
            model.fit(x_train, y_train)
            pred = None
        else:
            pred = model.fit_predict(df, self.logger)

        self.logger.info('training time: {}'.format(time.time() - s))

        oof, y = model.get_oof_prediction()
        return pred, oof, y

    def execute(self):
        pred_inner, oof_inner, y_inner = self._exec('inner', self.df_inner, self.model_inner)
        pred_extra, oof_outer, y_outer = self._exec('extra', self.df_extra, self.model_extra)

        pred_all = pd.concat([pred_inner, pred_extra]).fillna(0)
        pred_all = add_class99(pred_all)

        #oof_score = multi_weighted_logloss(pd.concat([y_inner, y_outer]), pd.concat([oof_inner, oof_outer]))
        #self.logger.info('oof score(inner + extra): {}'.format(oof_score))

        submit(pred_all, self.submit_path)



