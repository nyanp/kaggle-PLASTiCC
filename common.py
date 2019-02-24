import os
import time
from contextlib import contextmanager

import pandas as pd

import config


def load_lightcurve() -> pd.DataFrame:
    if config.TRAINING_ONLY:
        path = config.DATA_DIR + "train.f"
    else:
        path = config.DATA_DIR + "all.f"

    return pd.read_feather(path)


def load_partial_lightcurve(index: int) -> pd.DataFrame:
    assert index >= 0 and index <= 29
    path = config.DATA_DIR + "all_{}.f".format(index)

    return pd.read_feather(path)


def load_metadata() -> pd.DataFrame:
    meta = pd.read_feather(config.DATA_DIR + "meta.f")

    if config.TRAINING_ONLY:
        meta = meta[~meta.target.isnull()].reset_index(drop=True)

    return meta


def load_passband_metadata() -> pd.DataFrame:
    meta = pd.read_feather(config.DATA_DIR + "passband_meta.f")
    return meta


# "f210" => pd.DataFrame
def load_feature(feature_id: str) -> pd.DataFrame:
    path = config.FEATURE_LOAD_DIR + feature_id + ".f"

    return pd.read_feather(path)


def save_feature(df: pd.DataFrame, feature_id: str, with_csv_dump: bool = False):
    path = config.FEATURE_SAVE_DIR + feature_id + ".f"
    df.to_feather(path)

    if with_csv_dump:
        df.head(1000).to_csv(config.DEBUG_CSV_SAVE_DIR + feature_id + ".csv")


def save_submit_file(pred: pd.DataFrame, filename: str):
    path = os.path.join(config.SUBMIT_DIR, filename)

    if 'object_id' in pred:
        pred.to_csv(path, index=False)
    else:
        pred.to_csv(path, index=True)


def save_shared_file(features: pd.DataFrame, filename: str):
    path = os.path.join(config.SHARE_DIR, filename)
    features.to_feather(path)


@contextmanager
def timer(name):
    try:
        s = time.time()
        yield
    finally:
        print("[{:5g}sec] {}".format(time.time() - s, name))