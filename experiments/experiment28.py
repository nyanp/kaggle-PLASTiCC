from experiments.experiments_dual import ExperimentDualModel
from model.lgbm import LGBMModel

# experiment22 + pseudo-label
class Experiment28(ExperimentDualModel):
    def __init__(self, basepath, submit_path=None):
        super().__init__(basepath=basepath,
                         features_inner=['f000', 'f202', 'f100', 'f002', 'f104', 'f205', 'f010', 'f203', 'f200', 'f110',
                                         'f303', 'f304', 'f050', 'f400', 'f106', 'f107', 'f108','f140','f141','f142','f143','f144'],
                         features_extra=['f000', 'sncosmo', 'f202', 'f100', 'f002', 'f104', 'f205', 'f010', 'f203', 'f200', 'f110',
                                         'f303', 'f304', 'f050', 'f400', 'f106', 'f107', 'f108','f140','f141','f142','f143','f144','f150','f052','f053','f061','f063','f361'],
                         model_inner=LGBMModel(nfolds=10, weight_mode='weighted'),
                         model_extra=LGBMModel(nfolds=10, weight_mode='weighted'),
                         submit_path=submit_path,
                         log_name='experiment28',
                         drop_feat_inner=['hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b', 'distmod', 'hostgal_photoz'],
                         drop_feat_extra=['hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b', 'c_err', 't0_err', 'x0_err', 'x1_err', 'c_err', 'ncall', 'z_err'],
                         postproc_version=2,
                         pseudo_classes=[90, 42],
                         pseudo_n_loop=3,
                         pseudo_th=0.97)
