import pandas as pd

from model.problem import *

class_dict = {c: i for i, c in enumerate(classes)}


def lgb_multi_weighted_logloss(y_true, y_preds):

    if len(np.unique(y_true)) == 15:
        weight = class_weight_with_other
        cls = classes_with_other
    elif len(np.unique(y_true)) == 14:
        weight = class_weight
        cls = classes
    elif len(np.unique(y_true)) == 9:
        weight = class_weight_out
        cls = classes_out
    else:
        weight = class_weight_in
        cls = classes_in

    y_p = y_preds.reshape(y_true.shape[0], len(cls), order='F')
    y_ohe = pd.get_dummies(y_true)
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    y_p_log = np.log(y_p)
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    class_arr = np.array([weight[k] for k in sorted(weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos
    loss = - np.sum(y_w) / np.sum(class_arr)

    return 'wloss', loss, False


def multi_weighted_logloss(y_true, y_preds):

    if len(np.unique(y_true)) == 15:
        weight = class_weight_with_other
    elif len(np.unique(y_true)) == 14:
        weight = class_weight
    elif len(np.unique(y_true)) == 9:
        weight = class_weight_out
    else:
        weight = class_weight_in

    y_p = y_preds
    y_ohe = pd.get_dummies(y_true)
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)

    y_p_log = np.log(y_p)
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    class_arr = np.array([weight[k] for k in sorted(weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss

