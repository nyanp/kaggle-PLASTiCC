from experiments.experiments import Experiment
from model.lgbm import LGBMModel

class Experiment1(Experiment):
    def __init__(self, basepath):
        super().__init__(basepath=basepath,
                         features=['f000'],
                         model=LGBMModel(),
                         submit_path='output/experiment1.csv',
                         log_name='experiment1')

