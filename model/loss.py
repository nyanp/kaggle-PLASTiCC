import numpy as np
from model.problem import *
import pandas as pd
import torch
import torch.nn.functional as F
from torch.autograd import grad
from sklearn.preprocessing import OneHotEncoder

class_dict = {c: i for i, c in enumerate(classes)}



def lgb_multi_weighted_logloss_2(y_preds, train_data):
    y_true = train_data.get_label()

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


weight_tensor = torch.tensor(list(class_weight.values()),
                             requires_grad=False).type(torch.FloatTensor)
weight_tensor_in = torch.tensor(list(class_weight_in.values()),
                             requires_grad=False).type(torch.FloatTensor)
weight_tensor_out = torch.tensor(list(class_weight_out.values()),
                             requires_grad=False).type(torch.FloatTensor)

# this is the simplified original loss function by Olivier. It works excellently as an
# evaluation function, but we won't be able to use it in training
def torch_multi_weighted_logloss(y_true, y_preds):
    enc = OneHotEncoder(sparse=False)
    if len(y_true.shape) == 1:
        y_true = np.expand_dims(y_true, 1)
    y_ohe = enc.fit_transform(y_true)
    y_p = np.clip(a=y_preds, a_min=1e-15, a_max=1 - 1e-15)
    if y_p.shape[0] > y_true.shape[0]:
        y_p = y_p.reshape(y_true.shape[0], len(classes), order='F')
        if y_p.shape[0] != y_true.shape[0]:
            raise ValueError(
                'Dimension Mismatch for y_p {0} and y_true {1}!'.format(
                    y_p.shape, y_true.shape))
    y_p_log = np.log(y_p)
    y_log_ones = np.sum(np.multiply(y_ohe, y_p_log), axis=0)
    nb_pos = np.sum(y_ohe, axis=0).astype(float)
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos
    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss

# this is a reimplementation of the above loss function using pytorch expressions.
# Alternatively this can be done in pure numpy (not important here)
# note that this function takes raw output instead of probabilities from the booster
# Also be aware of the index order in LightDBM when reshaping (see LightGBM docs 'fobj')
def torch_wloss_metric(preds, train_data, classes, weight_tensor):
    y_t = torch.tensor(train_data.get_label(), requires_grad=False).type(torch.LongTensor)
    y_h = torch.zeros(
        y_t.shape[0], len(classes), requires_grad=False).scatter(1, y_t.reshape(-1, 1), 1)
    y_h /= y_h.sum(dim=0, keepdim=True)
    y_p = torch.tensor(preds, requires_grad=False).type(torch.FloatTensor)
    if len(y_p.shape) == 1:
        y_p = y_p.reshape(len(classes), -1).transpose(0, 1)
    ln_p = torch.log_softmax(y_p, dim=1)
    wll = torch.sum(y_h * ln_p, dim=0)
    loss = -torch.dot(weight_tensor, wll) / torch.sum(weight_tensor)
    return 'wloss', loss.numpy() * 1., False

# with autograd or pytorch you can pretty much come up with any loss function you want
# without worrying about implementing the gradients yourself
def torch_wloss_objective(preds, train_data, classes, weight_tensor):
    y_t = torch.tensor(train_data.get_label(), requires_grad=False).type(torch.LongTensor)
    y_h = torch.zeros(
        y_t.shape[0], len(classes), requires_grad=False).scatter(1, y_t.reshape(-1, 1), 1)
    ys = y_h.sum(dim=0, keepdim=True)
    y_h /= ys
    y_p = torch.tensor(preds, requires_grad=True).type(torch.FloatTensor)
    y_r = y_p.reshape(len(classes), -1).transpose(0, 1)
    ln_p = torch.log_softmax(y_r, dim=1)

    wll = torch.sum(y_h * ln_p, dim=0)
    loss = -torch.dot(weight_tensor, wll)
    grads = grad(loss, y_p, create_graph=True)[0]
    grads *= float(len(classes)) / torch.sum(1 / ys)  # scale up grads
    hess = torch.ones(y_p.shape)  # haven't bothered with properly doing hessian yet
    return grads.detach().numpy(), hess.detach().numpy()


def torch_wloss_obj_in(preds, train_data):
    return torch_wloss_objective(preds, train_data, classes_in, weight_tensor_in)


def torch_wloss_obj_out(preds, train_data):
    return torch_wloss_objective(preds, train_data, classes_out, weight_tensor_out)


def torch_wloss_obj_all(preds, train_data):
    return torch_wloss_objective(preds, train_data, classes, weight_tensor)


def torch_wloss_metric_in(preds, train_data):
    return torch_wloss_metric(preds, train_data, classes_in, weight_tensor_in)


def torch_wloss_metric_out(preds, train_data):
    return torch_wloss_metric(preds, train_data, classes_out, weight_tensor_out)


def torch_wloss_metric_all(preds, train_data):
    return torch_wloss_metric(preds, train_data, classes, weight_tensor)
