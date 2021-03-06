import config
from model.experiments_dual import ExperimentDualModel
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
             'sn_salt2_ncall',
             'sn_salt2_t0',
             'snana-2004fe_p_sn5_snana-2004fe_t0',
             'snana-2007Y_p_sn5_snana-2007Y_t0',
             'hsiao_p_sn5_hsiao_t0',
             'nugent-sn2n_p_sn5_nugent-sn2n_t0',
             'nugent-sn1bc_p_sn5_nugent-sn1bc_t0'
             ]

extragalactic_features = {
    'original': ['f000', 'f202', 'f100', 'f002', 'f104', 'f205', 'f010', 'f203', 'f200', 'f110',
                 'f303', 'f304', 'f050', 'f400', 'f106', 'f107', 'f108', 'f140', 'f141', 'f142',
                 'f143', 'f144', 'f052', 'f053', 'f061', 'f063', 'f361', 'f600', 'f1003', 'f1080',
                 'f1086', 'f1087', 'f500', 'f509', 'f510', 'f511', 'f512', 'f513'],
    'salt2': ['f000', 'f202', 'f100', 'f002', 'f104', 'f205', 'f010', 'f203', 'f200', 'f110',
              'f303', 'f304', 'f050', 'f400', 'f106', 'f107', 'f108', 'f140', 'f141', 'f142',
              'f143', 'f144', 'f052', 'f053', 'f061', 'f063', 'f361', 'f600', 'f1003', 'f1080',
              'f1086', 'f1087', 'f500'],
    'no-template': ['f000', 'f202', 'f100', 'f002', 'f104', 'f205', 'f010', 'f203', 'f200', 'f110',
                    'f303', 'f304', 'f050', 'f400', 'f106', 'f107', 'f108', 'f140', 'f141', 'f142',
                    'f143', 'f144', 'f052', 'f053', 'f061', 'f063', 'f361', 'f600', 'f1003', 'f1080',
                    'f1086', 'f1087'],
    'small': ['f000', 'f050', 'f1003', 'f1080', 'f500','f509', 'f510', 'f511', 'f512'],
    'best': ['f000', 'f202', 'f100', 'f002', 'f104', 'f205', 'f010', 'f203', 'f200', 'f110',
             'f303', 'f304', 'f050', 'f400', 'f106', 'f107', 'f108', 'f140', 'f141', 'f142',
             'f143', 'f144', 'f052', 'f053', 'f061', 'f063', 'f361', 'f600', 'f1003', 'f1080',
             'f1086', 'f1087', 'f500', 'f509', 'f510', 'f511', 'f512', 'f513', 'f515', 'f517'],
}

galactic_features = ['f000', 'f202', 'f100', 'f002', 'f104', 'f205', 'f010', 'f203', 'f200', 'f110', 'f303',
                     'f304', 'f050', 'f400', 'f106', 'f107', 'f108', 'f140', 'f141', 'f142', 'f143', 'f144']

# experiment62 + f513
class Experiment65(ExperimentDualModel):
    def __init__(self,
                 submit_filename=config.SUBMIT_FILENAME,
                 pseudo_n_loop=3,
                 save_pseudo_label=False,
                 use_extra_classifier=False,
                 logdir='experiment65',
                 param=None,
                 seed=None,
                 cache_path_inner=None,
                 cache_path_extra=None,
                 use_cache=False,
                 pseudo_th=0.985  ### IMPORTANT
                 ):
        if param is None:
            param = {
                'boosting_type': 'gbdt',
                'objective': 'multiclass',
                'num_class': 14,
                'metric': 'multi_logloss',
                'subsample': .9,
                'colsample_bytree': .9,
                'reg_alpha': 0,
                'reg_lambda': 3,
                'min_split_gain': 0,
                'min_child_weight': 10,
                'silent': True,
                'verbosity': -1,
                'learning_rate': 0.1,
                'max_depth': 3,
                'min_data_in_leaf': 1,
                'n_estimators': 10000,
                'max_bin': 128,
                'bagging_fraction': 0.66,
                'verbose': -1
            }

        super().__init__(features_inner=galactic_features,
                         features_extra=extragalactic_features[config.MODELING_MODE],
                         model_inner=LGBMModel(nfolds=10, param=param, weight_mode='weighted',
                                               use_extra_classifier=use_extra_classifier, seed=seed),
                         model_extra=LGBMModel(nfolds=10, param=param, weight_mode='weighted',
                                               use_extra_classifier=use_extra_classifier, seed=seed),
                         submit_filename=submit_filename,
                         logdir=logdir,
                         drop_feat_inner=['hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b', 'distmod',
                                          'hostgal_photoz'] + blacklist,
                         drop_feat_extra=['hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b',
                                          'hostgal_photoz'] + blacklist,
                         postproc_version=2,
                         pseudo_classes=[90, 42],
                         pseudo_n_loop=pseudo_n_loop,
                         pseudo_th=pseudo_th,
                         save_pseudo_label=save_pseudo_label,
                         cache_path_inner=cache_path_inner,
                         cache_path_extra=cache_path_extra,
                         use_cache=use_cache)
