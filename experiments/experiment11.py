from experiments.experiments_dual import ExperimentDualModel
from model.lgbm import LGBMModel

# experiment10 + class weight
class Experiment11(ExperimentDualModel):
    def __init__(self, basepath):
        super().__init__(basepath=basepath,
                         features_inner=['f000', 'f202', 'f100', 'f002', 'f104', 'f205', 'f010', 'f203', 'f200', 'f110',
                                         'f303', 'f304', 'f050', 'f400'],
                         features_extra=['f000', 'f202', 'f100', 'f002', 'f104', 'f205', 'f010', 'f203', 'f200', 'f110',
                                         'f303', 'f304', 'f050', 'f400'],
                         model_inner=LGBMModel(nfolds=10, use_weight=False),
                         model_extra=LGBMModel(nfolds=10, use_weight=True),
                         submit_path='output/experiment11_2.csv',
                         log_name='experiment11_2',
                         drop_feat_inner=['hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b', 'distmod', 'hostgal_photoz'],
                         drop_feat_extra=['hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b'],
                         postproc_version=2)
