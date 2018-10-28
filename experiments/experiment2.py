from experiments.experiments import Experiment
from model.lgbm import LGBMModel

class Experiment2(Experiment):
    def __init__(self, basepath):
        super().__init__(basepath=basepath,
                         features=['f000', 'f202', 'f100', 'f002', 'f104', 'f205', 'f010', 'f203', 'f200', 'f110'],
                         model=LGBMModel(),
                         submit_path='output/experiment2.csv',
                         log_name='experiment2',
                         drop_feat=['hostgal_specz', 'ra', 'decl'])
