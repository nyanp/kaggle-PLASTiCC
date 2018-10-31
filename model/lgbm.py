from .model import *
import pandas as pd
import numpy as np

class LGBMModel(Model):
    def __init__(self, param = None, random_state = 1, nfolds = 5):

        if param is None:
            self.param =  {
                'boosting_type': 'gbdt',
                'objective': 'multiclass',
                'num_class': 14,
                'metric': 'multi_logloss',
                'subsample': .9,
                'colsample_bytree': .7,
                'reg_alpha': .01,
                'reg_lambda': .01,
                'min_split_gain': 0.01,
                'min_child_weight': 10,
                'silent':True,
                'verbosity':-1,
                'learning_rate':0.1,
                'max_depth':4,
                'n_estimators':10000,

                'verbose':-1
            }
        else:
            self.param = param
        self.random_state = random_state
        self.n_classes = None
        self.feature_importance_ = None
        self.score_ = None
        self.nfolds = nfolds

    def get_params(self):
        d = {
            'random_state': self.random_state,
            'nfolds': self.nfolds,
            'params': self.param
        }
        return d

    def fit(self, x, y, logger = None):
        for c in x:
            if x[c].count() == 0:
                raise RuntimeError('#### column {} has no valid value!'.format(c))

        folds = StratifiedKFold(n_splits=self.nfolds, shuffle=True, random_state=self.random_state)
        feature_importance_df = pd.DataFrame()
        self.n_classes = y.nunique()

        oof_preds = np.zeros((x.shape[0], self.n_classes))

        logger.info('train: {}'.format(x.shape))
        print('train: {}'.format(x.shape))

        score = []
        clfs = []

        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(x, y)):
            train_x, train_y = x.iloc[train_idx], y.iloc[train_idx]
            valid_x, valid_y = x.iloc[valid_idx], y.iloc[valid_idx]

            # LightGBM parameters found by Bayesian optimization
            clf = LGBMClassifier(**self.param)
            clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)],
                    eval_metric=lgb_multi_weighted_logloss, verbose=-1, early_stopping_rounds=100)

            oof_preds[valid_idx, :] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)

            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = x.columns.tolist()
            fold_importance_df["importance"] = clf.feature_importances_
            fold_importance_df["fold"] = n_fold + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

            loss = multi_weighted_logloss(valid_y, oof_preds[valid_idx])
            logger.info('Fold {} loss: {}'.format(n_fold + 1, loss))

            # print('Fold {} AUC : {:.6f}'.format(n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
            score.append(loss)

            clfs.append(clf)

            del train_x, train_y, valid_x, valid_y
            gc.collect()

        # print('Full AUC score %.6f' % roc_auc_score(y, oof_preds))
        # display_importances(feature_importance_df)
        full_auc = multi_weighted_logloss(y, oof_preds)
        logger.info('*** full auc: {}'.format(full_auc))

        self.scores_ = score
        self.score_ = full_auc
        self.feature_importance_ = feature_importance_df
        self.clfs = clfs
        self.y = y
        self.oof_preds = oof_preds

    def name(self):
        return 'LGBM'

    def predict(self, x) -> pd.DataFrame:
        preds = np.zeros((len(x), self.n_classes))

        for clf in self.clfs:
            preds += clf.predict_proba(x, num_iteration=clf.best_iteration_) / len(self.clfs)

        d = pd.DataFrame(preds, columns=['class_' + str(s) for s in self.clfs[0].classes_])
        d['object_id'] = x.index
        return d

    def score(self):
        return self.score_

    def scores(self):
        return self.scores_

    def feature_importances(self):
        return self.feature_importance_

    def get_oof_prediction(self):
        return self.oof_preds, self.y

