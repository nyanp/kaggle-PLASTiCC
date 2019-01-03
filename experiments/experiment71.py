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
'sn_salt2_ncall',
'sn_salt2_t0',
'snana-2004fe_p_sn5_snana-2004fe_t0',
'snana-2007Y_p_sn5_snana-2007Y_t0',
'hsiao_p_sn5_hsiao_t0',
'nugent-sn2n_p_sn5_nugent-sn2n_t0',
'nugent-sn1bc_p_sn5_nugent-sn1bc_t0'
]

# experiment62 + f515 + f517 (salt-2 & salt-2 averaged), pseudo-th=0.986
class Experiment71(ExperimentDualModel):
    def __init__(self, basepath,
                 submit_path='output/experiment71.csv',
                 pseudo_n_loop=3,
                 save_pseudo_label=True,
                 use_extra_classifier=True,
                 n_estimators_extra_classifier=1000,
                 log_name='experiment71',
                 param=None,
                 seed=None,
                 cache_path_inner=None,
                 cache_path_extra=None,
                 use_cache=False,
                 pseudo_th=0.986, ### IMPORTANT
                 use_pl_labels=True
                 ):

        if use_pl_labels:
            pl_labels = {
                42: 'pseudo_label_class42_round2.f',
                90: 'pseudo_label_class90_round2.f',
            }
            pseudo_n_loop=0
        else:
            pl_labels = None

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
                'silent':True,
                'verbosity':-1,
                'learning_rate':0.1,
                'max_depth':3,
                'min_data_in_leaf':1,
                'n_estimators':10000,
                'max_bin':128,
                'bagging_fraction':0.66,
                'verbose':-1
            }

        super().__init__(basepath=basepath,
                         features_inner=['f000', 'f202', 'f100', 'f002', 'f104', 'f205', 'f010', 'f203', 'f200', 'f110',
                                         'f303', 'f304', 'f050', 'f400', 'f106', 'f107', 'f108','f140','f141','f142','f143','f144'],
                         features_extra=['f000', 'f202', 'f100', 'f002', 'f104', 'f205', 'f010', 'f203', 'f200', 'f110',
                                         'f303', 'f304', 'f050', 'f400', 'f106', 'f107', 'f108','f140','f141','f142','f143',
                                         'f144',
                                         'f052','f053','f061','f063','f361','f600','f1003','f1080','f1086',
                                         'f1087','f500','f509','f510','f511','f512','f513','f515','f517'],
                         model_inner=LGBMModel(nfolds=10, param=param, weight_mode='weighted', use_extra_classifier=use_extra_classifier, n_estimators_extra_classifier=n_estimators_extra_classifier, seed=seed),
                         model_extra=LGBMModel(nfolds=10, param=param, weight_mode='weighted', use_extra_classifier=use_extra_classifier, n_estimators_extra_classifier=n_estimators_extra_classifier, seed=seed),
                         submit_path=submit_path,
                         log_name=log_name,
                         drop_feat_inner=['hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b', 'distmod', 'hostgal_photoz'] + blacklist,
                         drop_feat_extra=['hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b', 'hostgal_photoz'] + blacklist,
                         postproc_version=2,
                         pseudo_classes=[90, 42],
                         pseudo_n_loop=pseudo_n_loop,
                         pseudo_th=pseudo_th,
                         save_pseudo_label=save_pseudo_label,
                         cache_path_inner=cache_path_inner,
                         cache_path_extra=cache_path_extra,
                         use_cache=use_cache,
                         pl_labels=pl_labels)