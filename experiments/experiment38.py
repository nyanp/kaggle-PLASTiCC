
blacklist = ['2__fft_coefficient__coeff_0__attr_"abs"',
 '3__fft_coefficient__coeff_1__attr_"abs"',
 'ddf',
 'delta(first(detected), max(flux))_ch0',
 'delta(max(flux), last(detected))_ch0',
 'detected_median(flux)_ch0',
 'diff(min(flux))_1_2',
 'diff(min(flux))_3_4',
 'extra',
 'max(flux)_ch2',
 'max(flux)_ch3',
 'max(flux)_ch4',
 'max(flux)_ch5',
 'mean(detected)_ch0',
 'mean(detected)_ch4',
 'std(detected)_ch0',
 'std(detected)_ch1',
 'std(detected)_ch2',
 'std(detected)_ch4',
 'std(flux)_ch2',
 'std(flux)_ch3',
 'timescale_th0.35_min_ch3',
 'timescale_th0.5_min_ch3']

from experiments.experiments_dual import ExperimentDualModel
from model.lgbm import LGBMModel

# experiment33 + drop irrelevant features + f500 + f504 mixed
class Experiment38(ExperimentDualModel):
    def __init__(self, basepath, submit_path=None):
        super().__init__(basepath=basepath,
                         features_inner=['f000', 'f202', 'f100', 'f002', 'f104', 'f205', 'f010', 'f203', 'f200', 'f110',
                                         'f303', 'f304', 'f050', 'f400', 'f106', 'f107', 'f108','f140','f141','f142','f143','f144'],
                         features_extra=['f000', 'f202', 'f100', 'f002', 'f104', 'f205', 'f010', 'f203', 'f200', 'f110',
                                         'f303', 'f304', 'f050', 'f400', 'f106', 'f107', 'f108','f140','f141','f142','f143',
                                         'f144','f150','f052','f053','f061','f063','f361','f600','f500','f504'],
                         model_inner=LGBMModel(nfolds=10, weight_mode='weighted'),
                         model_extra=LGBMModel(nfolds=10, weight_mode='weighted'),
                         submit_path=submit_path,
                         log_name='experiment38',
                         drop_feat_inner=['hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b', 'distmod', 'hostgal_photoz'] + blacklist,
                         drop_feat_extra=['hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b'] + blacklist,
                         postproc_version=2,
                         pseudo_classes=[90, 42],
                         pseudo_n_loop=3,
                         pseudo_th=0.97)
