from experiments.experiments_dual import ExperimentDualModel
from model.lgbm import LGBMModel

class Experiment5(ExperimentDualModel):
    def __init__(self, basepath):
        super().__init__(basepath=basepath,
                         features_inner=['f000', 'f202', 'f100', 'f002', 'f104', 'f205', 'f010', 'f203', 'f200', 'f110'],
                         features_extra=['f000', 'f202', 'f100', 'f002', 'f104', 'f205', 'f010', 'f203', 'f200', 'f110'],
                         model_inner=LGBMModel(nfolds=10),
                         model_extra=LGBMModel(nfolds=10),
                         submit_path='output/experiment5.csv',
                         log_name='experiment5',
                         drop_feat_inner=['hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b'],
                         drop_feat_extra = ['hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b'])
