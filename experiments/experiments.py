import pandas as pd
from typing import List, Dict
from tqdm import tqdm
from model.model import Model
import logging
import time
from model.postproc import *
from .confusion_matrix import save_confusion_matrix


class Experiment:
    def __init__(self, basepath: str,
                 features: List[str] = None,
                 model: Model = None,
                 submit_path: str = 'output/submission.csv',
                 log_name: str = 'default',
                 use_feat: Dict[str, List[str]] = None,
                 drop_feat: List[str] = None,
                 df: pd.DataFrame = None,
                 postproc_version: int = 1):

        if df is None:
            self.df = pd.read_feather(basepath + 'input/meta.f')
        else:
            self.df = df

        if submit_path is None:
            self.submit_path = None
            self.df = self.df[~self.df.target.isnull()].reset_index(drop=True) # use training data only
        else:
            self.submit_path = basepath + submit_path

        self.len = len(self.df)
        self.model = model
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.DEBUG)
        self.fh = logging.FileHandler(basepath+log_name+'.log')
        self.fh.setLevel(logging.DEBUG)
        self.postproc_version = postproc_version

        if len(self.logger.handlers) == 0:
            self.logger.addHandler(self.fh)

        if df is None:
            self.logger.info('load features...')
            for f in tqdm(features):
                if submit_path is None:
                    tmp = pd.read_feather(basepath + 'features_tr/' + str(f) + '.f')
                else:
                    tmp = pd.read_feather(basepath + 'features/' + str(f) + '.f')

                if use_feat is not None and f in use_feat:
                    if isinstance(tmp[use_feat[f]], list):
                        tmp = tmp[use_feat[f]]
                    else:
                        tmp = tmp[[use_feat[f]]]
                self.df = pd.merge(self.df, tmp, on='object_id', how='left')

                if len(self.df) != self.len:
                    raise RuntimeError('Error on merging {}: size of metadata changed from {} to {}'.format(f, self.len, len(self.df)))

        if drop_feat is not None:
            self.df.drop(drop_feat, axis=1, inplace=True)


    def execute(self):
        self.logger.info('features: {}'.format(self.df.columns.tolist()))
        self.logger.info('model: {}'.format(self.model.name()))
        self.logger.info('param: {}'.format(self.model.get_params()))

        print(self.df.columns.tolist())

        s = time.time()

        if self.submit_path is None:
            x_train, y_train, _ = self.model.prep(self.df)
            self.model.fit(x_train, y_train, self.logger)
        else:
            pred = self.model.fit_predict(self.df, self.logger)
            pred = self.post_proc(pred)
            self.logger.info('output data: {}(shape:{})'.format(self.submit_path, pred.shape))
            submit(pred, self.submit_path)

        self.logger.info('training time: {}'.format(time.time() - s))
        self.score = self.model.score()
        self.scores = self.model.scores()

        oof, y = self.model.get_oof_prediction()
        save_confusion_matrix(oof, y, 'matrix.png')

    def post_proc(self, pred):

        if self.postproc_version == 1:
            pred = add_class99(pred)
        elif self.postproc_version == 2:
            pred = add_class99_2(pred)
        else:
            raise NotImplementedError()
        pred = filter_by_galactic_vc_extra_galactic(pred, self.df)
        return pred
