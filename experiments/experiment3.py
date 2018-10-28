from experiments.experiments_dual import ExperimentDualModel
from model.lgbm import LGBMModel

class Experiment3(ExperimentDualModel):
    def __init__(self, basepath):
        super().__init__(basepath=basepath,
                         features_inner=['f000', 'f202', 'f100', 'f002', 'f104', 'f205', 'f010', 'f203', 'f200', 'f110'],
                         features_extra=['f000', 'f202', 'f100', 'f002', 'f104', 'f205', 'f010', 'f203', 'f200', 'f110'],
                         model_inner=LGBMModel(),
                         model_extra=LGBMModel(),
                         submit_path='output/experiment3.csv',
                         log_name='experiment3',
                         drop_feat_inner=['hostgal_specz', 'ra', 'decl'],
                         drop_feat_extra = ['hostgal_specz', 'ra', 'decl'])
