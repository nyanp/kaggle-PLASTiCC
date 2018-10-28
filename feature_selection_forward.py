from model.lgbm import LGBMModel
from experiments.experiments import Experiment
import pandas as pd
import gc
import numpy as np

baseline_features = ['f000', 'f202', 'f100', 'f002', 'f104', 'f205', 'f010', 'f203', 'f200', 'f110']

additional_features = [
    "f001", "f101",
    "f102", "f204",
    "f103", "f201",
    'f105', 'f020', 'f021', 'f022', 'f023', 'f024', 'f025', 'f026', 'f027', 'f028', 'f029',
    'f120', 'f121', 'f122', 'f123'
]

eltwise_feature_tables = [
    "f303", "f307", "f311", "f350", "f310",
    "f340","f302", "f306", "f301", "f305", "f309", "f330", "f370",
    "f300", "f304", "f308", "f321", "f360",
]


drop_feat=['hostgal_specz', 'ra', 'decl']

# per-file feature selection

def beats(old_score, old_scores, new_score, new_scores):
    if new_score >= old_score:
        return False
    n_beats = 0
    for i in range(len(new_scores)):
        if new_scores[i] < old_scores[i]:
            n_beats += 1

    if n_beats > len(old_scores)/2:
        return True
    return False


def fs_per_file(n_loop:int = 10):
    for i in range(n_loop):
        baseline = Experiment('./', baseline_features, LGBMModel(), None, 'log_fs', drop_feat=drop_feat)
        baseline.execute()

        baseline_score = baseline.score
        baseline_scores = baseline.scores

        best_score = baseline_score
        best_scores = baseline_scores
        best_feat = None

        for additional in additional_features:
            exp = Experiment('./', baseline_features + [additional], LGBMModel(), None, 'log_fs', drop_feat=drop_feat)
            exp.execute()

            exp_score = exp.score
            exp_scores = exp.scores

            if beats(best_score, best_scores, exp_score, exp_scores):
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


fs_per_file(10)
fs_per_column(100)

