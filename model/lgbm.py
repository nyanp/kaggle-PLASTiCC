from .model import *
import pandas as pd
import numpy as np
from .problem import class_weight
from .loss import *
import functools

class LGBMModel(Model):
    def __init__(self, param = None, random_state = 1, nfolds = 5, weight_mode='none', use_extra_classifier=False):

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
        self.weight_mode = weight_mode
        self.use_extra_classifier = use_extra_classifier

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

        n_classes = len(np.unique(y))

        self.param['num_class'] = n_classes

        if self.weight_mode == 'simple':
            w_train = np.array([class_weight[i] / 2.0 for i in y])
        elif self.weight_mode == 'weighted':
            values = y.value_counts()
            n = {v: values[v] for v in values.index}
            w_train = np.array([class_weight[i]/n[i] for i in y])
            w_train /= w_train.mean()
        else:
            assert self.weight_mode == 'none'
            w_train = None
            
        best_iterations = []

        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(x, y)):
            train_x, train_y = x.iloc[train_idx], y.iloc[train_idx]
            valid_x, valid_y = x.iloc[valid_idx], y.iloc[valid_idx]

            if w_train is not None:
                sample_weight = w_train[train_idx]
            else:
                sample_weight = None

            print(train_x.shape)

            # LightGBM parameters found by Bayesian optimization
            clf = LGBMClassifier(**self.param)

            clf.fit(train_x, train_y, sample_weight=sample_weight, eval_set=[(valid_x, valid_y)],
                    eval_metric=lgb_multi_weighted_logloss, verbose=-1, early_stopping_rounds=100)

            oof_preds[valid_idx, :] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)
            best_iterations.append(clf.best_iteration_)

            #dtrain = lgb.Dataset(train_x, label_to_code(train_y))
            #dvalid = lgb.Dataset(valid_x, label_to_code(valid_y))

            #clf = lgb.train(params=self.param, train_set=dtrain, num_boost_round=10000, valid_sets=dvalid, fobj=obj, feval=feval, early_stopping_rounds=50,
            #                verbose_eval=200)


            #oof_preds[valid_idx, :] = clf.predict(valid_x, num_iteration=clf.best_iteration)

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

        if self.use_extra_classifier:
            if w_train is not None:
                sample_weight = w_train
            else:
                sample_weight = None

            n_iterations = int(1.1 * np.mean(best_iterations))

            # LightGBM parameters found by Bayesian optimization
            param = self.param.copy()
            param['n_estimators'] = n_iterations
            clf = LGBMClassifier(**param)
            clf.fit(x, y, sample_weight=sample_weight, verbose=-1)
            self.clf = clf


        self.scores_ = score
        self.score_ = full_auc
        self.feature_importance_ = feature_importance_df
        self.clfs = clfs
        self.y = y
        self.oof_preds = oof_preds

    def name(self):
        return 'LGBM'

    def predict(self, x) -> pd.DataFrame:
        if self.use_extra_classifier:
            mean_preds = self.clf.predict_proba(x, num_iteration=-1)
            self.variance = None
        else:
            preds = np.zeros((len(x), self.n_classes, self.nfolds))

            for i, clf in enumerate(self.clfs):
                preds[:, :, i] = clf.predict_proba(x, num_iteration=clf.best_iteration_)

            # variance over folds
            self.variance = np.var(preds, axis=2)
            mean_preds = np.mean(preds, axis=2)

        #for clf in self.clfs:
        #    preds += clf.predict_proba(x, num_iteration=clf.best_iteration_) / len(self.clfs)

        d = pd.DataFrame(mean_preds, columns=['class_' + str(s) for s in self.clfs[0].classes_])
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

