import pandas as pd
from typing import List
from tqdm import tqdm
from model.model import Model
import logging
import time
from model.postproc import *

class Experiment:
    def __init__(self, basepath: str,
                 features: List[str],
                 model: Model,
                 submit_path: str = 'output/submission.csv',
                 log_name: str = 'default'):
        self.df = pd.read_feather(basepath + 'input/meta.f')
        self.len = len(self.df)
        self.model = model
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.DEBUG)
        self.fh = logging.FileHandler(basepath+log_name+'.log')
        self.fh.setLevel(logging.DEBUG)
        self.logger.addHandler(self.fh)
        self.submit_path = basepath + submit_path

        self.logger.info('load features...')
        for f in tqdm(features):
            tmp = pd.read_feather(basepath + 'features/' + str(f) + '.f')

            self.df = pd.merge(self.df, tmp, on='object_id', how='left')

            if len(self.df) != self.len:
                raise RuntimeError('Error on merging {}: size of metadata changed from {} to {}'.format(f, self.len, len(self.df)))

    def execute(self):
        self.logger.info('features: {}'.format(self.df.columns.tolist()))
        self.logger.info('model: {}'.format(self.model.name()))
        self.logger.info('param: {}'.format(self.model.get_params()))

        s = time.time()

        if self.submit_path is None:
            x_train, y_train, _ = self.model.prep(self.df)
            self.model.fit(x_train, y_train)
        else:
            pred = self.model.fit_predict(self.df, self.logger)
            pred = self.post_proc(pred)
            self.logger.info('output data: {}(shape:{})'.format(self.submit_path, pred.shape))
            submit(pred, self.submit_path)

        self.logger.info('training time: {}'.format(time.time() - s))

    def post_proc(self, pred):
        pred = add_class99(pred)
        pred = filter_by_galactic_vc_extra_galactic(pred, self.df)
        return pred
