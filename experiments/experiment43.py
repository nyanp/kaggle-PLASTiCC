from experiments.experiments_dual import ExperimentDualModel
from model.lgbm import LGBMModel

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
 'timescale_th0.5_min_ch3',
 'sn_salt2_ncall']


param_extra =  {
                'boosting_type': 'gbdt',
                'objective': 'multiclass',
                'num_class': 9,
                'metric': 'multi_logloss',
                'colsample_bytree': .9,
                'reg_alpha': .0,
                'reg_lambda': .05,
                'min_split_gain': 0.05,
                'min_child_weight': 20,
                'silent':True,
                'verbosity':-1,
                'learning_rate':0.05,
                'max_depth':-1,
                'num_leaves':8,
                'n_estimators':10000,
                'verbose':-1
            }

param_inner =  {
                'boosting_type': 'gbdt',
                'objective': 'multiclass',
                'num_class': 5,
                'metric': 'multi_logloss',
                'colsample_bytree': .7,
                'reg_alpha': .01,
                'reg_lambda': .01,
                'min_split_gain': 0.1,
                'min_child_weight': 10,
                'silent':True,
                'verbosity':-1,
                'learning_rate':0.1,
                'max_depth':4,
                'n_estimators':10000,
                'verbose':-1
            }

# experiment41 + hyperparameter tuning
class Experiment43(ExperimentDualModel):
    def __init__(self, basepath, submit_path='output/experiment43.csv'):
        super().__init__(basepath=basepath,
                         features_inner=['f000', 'f202', 'f100', 'f002', 'f104', 'f010', 'f110',
                                         'f303', 'f304', 'f050', 'f400', 'f106', 'f107', 'f108','f140','f141','f142','f143','f144'],
                         features_extra=['f000', 'f202', 'f100', 'f002', 'f104', 'f205', 'f010', 'f203', 'f200', 'f110',
                                         'f303', 'f304', 'f050', 'f400', 'f106', 'f107', 'f108','f140','f141','f142','f143',
                                         'f144',
                                         'f052','f053','f061','f063','f361','f600','f500','f1003'],
                         model_inner=LGBMModel(nfolds=10, weight_mode='weighted', param=param_inner),
                         model_extra=LGBMModel(nfolds=10, weight_mode='weighted', param=param_extra),
                         submit_path=submit_path,
                         log_name='experiment43',
                         drop_feat_inner=['hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b', 'distmod', 'hostgal_photoz'] + blacklist,
                         drop_feat_extra=['hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b', 'hostgal_photoz'] + blacklist,
                         postproc_version=2,
                         pseudo_classes=[90, 42],
                         pseudo_n_loop=3,
                         pseudo_th=0.97)
