import pandas as pd
import numpy as np
from .problem import class_inner_galaxtic, class_extra_galaxtic


def filter_by_galactic_vc_extra_galactic(pred: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    meta['extra'] = (meta['hostgal_photoz'] > 0.0).astype(np.int32)

    if 'object_id' not in meta:
        meta.reset_index(inplace=True)
        assert 'object_id' in meta

    pred = pd.merge(pred, meta[['extra', 'object_id']], on='object_id', how='left')
    df_extra = pred[pred.extra == 1]
    df_inner = pred[pred.extra == 0]
    df_extra[['class_{}'.format(c) for c in class_inner_galaxtic]] = 0
    df_inner[['class_{}'.format(c) for c in class_extra_galaxtic]] = 0

    pred = pd.concat([df_extra, df_inner]).reset_index(drop=True).drop('extra', axis=1).set_index('object_id')
    return pred


def add_class99(pred: pd.DataFrame) -> pd.DataFrame:
    pred['class_99'] = 1
    for c in pred:
        if c == 'class_99' or c == 'object_id':
            continue
        pred['class_99'] *= (1 - pred[c])

    return pred


def add_class99_2(pred: pd.DataFrame) -> pd.DataFrame:
    pred = add_class99(pred)
    pred['class_99'] *= 1.0 - (pred['class_15'] == 0.0) * 0.8  # inner
    pred['class_99'] *= 1.0 - (pred['class_15'] > 0.0) * 0.1  # extra

    return pred

