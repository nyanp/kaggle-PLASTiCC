from model.lgbm import LGBMModel
from experiments.experiments import Experiment
from experiments.experiments_dual import ExperimentDualModel
import pandas as pd
import gc
import os
import numpy as np

n_cv = 5
n_offset = 0

baseline_features_inner = []

baseline_features_extra = ['f000', 'f202', 'f100', 'f002', 'f104', 'f205', 'f010', 'f203', 'f200', 'f110',
                                         'f303', 'f304', 'f050', 'f400', 'f106', 'f107', 'f108','f140','f141','f142','f143',
                                         'f144',
                                         'f052','f053','f061','f063','f361','f600','f500','f1003','f1080','f1086','f1087']

additional_features = [
    'f2001','f2002','f2003','f2004','f2005','f2006','f2007','f2008','f2009','f2010','f2011','f2012','f2013','f2014','f2015','f2016'
]

additional_features_ = [
    'f507', 'f1000', 'f1001', 'f1002', 'f1004', 'f1005', 'f1006',
    'f060', 'f062',
    'f052', 'f053', 'f601', 'f601a', 'f701', 'f311', 'f321', 'f109',

    'f1000','f1001','f1002','f506','f508',
    'f060','f061','f062','f210','f211','f212','f213',
    'f030','f052','f053','f601','f601a','f701','f370','f360','f311','f321',
    'f109','f151','f152','f153'
]
eltwise_feature_tables = [
    "f303", "f307", "f311", "f350", "f310",
    "f340","f302", "f306", "f301", "f305", "f309", "f330", "f370",
    "f300", "f304", "f308", "f321", "f360",
]

drop_feat_inner=['hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b', 'distmod', 'hostgal_photoz']
drop_feat_extra=['hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b']

# per-file feature selection

def beats(old_score, old_scores, new_score, new_scores):
    if new_score >= old_score:
        return False
    n_beats = 0
    for i in range(len(new_scores)):
        if new_scores[i] < old_scores[i]:
            n_beats += 1

    if n_beats > len(old_scores)/2 + n_offset:
        return True
    return False


def fs_per_file(n_loop:int = 10, log='log_fs', fs_on='extra'):
    os.mkdir(log)

    if fs_on == 'extra':
        baseline_features = baseline_features_extra
        mode = 'extra-only'
    else:
        baseline_features = baseline_features_inner
        mode = 'inner-only'

    params = {
        'basepath': './',
        'features_inner': baseline_features_inner,
        'features_extra': baseline_features_extra,
        'model_inner': LGBMModel(weight_mode='weighted', nfolds=n_cv),
        'model_extra': LGBMModel(weight_mode='weighted', nfolds=n_cv),
        'submit_path': None,
        'log_name': log,
        'drop_feat_inner': drop_feat_inner,
        'drop_feat_extra': drop_feat_extra,
        'mode': mode
    }

    summary = open(log+'/summary.csv', 'w')

    summary.write('name,n_features,features,' + ','.join(['fold{}'.format(i) for i in range(n_cv)]) + ',total\n')

    for i in range(n_loop):
        baseline = ExperimentDualModel(**params)
        baseline.execute()

        baseline_score = baseline.score(fs_on)
        baseline_scores = baseline.scores(fs_on)

        best_score = baseline_score
        best_scores = baseline_scores
        best_feat = None

        summary.write('{}/{}th baseline,'.format(i, n_loop))
        summary.write('{},"{}",{},{}\n'.format(len(baseline_features), baseline_features, baseline_scores, baseline_score))
        summary.flush()

        for additional in additional_features:
            if fs_on == 'extra':
                params['features_extra'] = baseline_features + [additional]
            else:
                assert fs_on == 'inner'
                params['features_inner'] = baseline_features + [additional]

            exp = ExperimentDualModel(**params)
            exp.execute()

            exp_score = exp.score(fs_on)
            exp_scores = exp.scores(fs_on)

            summary.write('baseline+{},'.format(additional))
            summary.write('{},"{}",{},{}\n'.format(len(baseline_features + [additional]), baseline_features + [additional], exp_scores,
                                                   exp_score))
            summary.flush()

            if beats(best_score, best_scores, exp_score, exp_scores):
                print('!!! best score !!! {} -> {}'.format(best_score, exp_score))
                best_score = exp_score
                best_scores = exp_scores
                best_feat = additional

        if best_feat is None:
            print('epoch {} : no feature candidate improved current best: {}(features: {})'.format(i, best_score, baseline_features))
            return
        else:
            print('epoch {} : improve score from {} to {}. best feature: {}'.format(i, baseline_score, best_score, best_feat))
            baseline_features.append(best_feat)
            additional_features.remove(best_feat)


def fs_per_column(n_loop:int = 100):
    features = None

    for f in eltwise_feature_tables:
        tmp = pd.read_feather('./features_tr/'+f+'.f')
        if tmp.object_id.dtype == 'object':
            tmp['object_id'] = tmp['object_id'].astype(np.int64)
        if features is None:
            features = tmp
        else:
            features = pd.merge(features, tmp, on='object_id', how='left')


    base = pd.read_feather('./input/meta.f')
    base = base[~base.target.isnull()].reset_index(drop=True)

    for f in baseline_features:
        tmp = pd.read_feather('./features_tr/'+f+'.f')
        base = pd.merge(base, tmp, on='object_id', how='left')

    print('base: {}, features: {}'.format(base.shape, features.shape))
    print('base-columns: {}'.format(base.columns.tolist()))
    print('features-columns: {}'.format(features.columns.tolist()))

    for i in range(n_loop):
        baseline = Experiment('./', None, LGBMModel(), None, 'log_fs_percol', drop_feat=drop_feat, df=base)
        baseline.execute()
        baseline_score = baseline.score
        baseline_scores = baseline.scores

        best_score = baseline_score
        best_scores = baseline_scores
        best_feat = None

        for f in features:
            if f == 'object_id':
                continue

            print('##### Trying: {} ######'.format(f))

            base_ = pd.merge(base, features[['object_id', f]], on='object_id', how='left').copy()

            print(base_.columns.tolist())

            exp = Experiment('./', None, LGBMModel(), None, 'log_fs', drop_feat=drop_feat, df=base_)
            exp.execute()

            exp_score = exp.score
            exp_scores = exp.scores

            if beats(best_score, best_scores, exp_score, exp_scores):
                best_score = exp_score
                best_scores = exp_scores
                best_feat = f

            del base_

        if best_feat is None:
            print('epoch {} : no feature candidate improved current best: {}(features: {})'.format(i, best_score, base.columns.tolist()))
            return
        else:
            print('epoch {} : improve score from {} to {}. best feature: {}'.format(i, baseline_score, best_score, base.columns.tolist()))
            base = pd.merge(base, features[['object_id', best_feat]], on='object_id', how='left')
            features.drop(best_feat, axis=1, inplace=True)


fs_per_file(16, 'log_fs_181203_mamas', fs_on='extra')
#fs_per_column(100)

