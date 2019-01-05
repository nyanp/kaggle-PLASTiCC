import os

import pandas as pd

import config
from .util import timer


def concat_to_feather(train, test, dst):
    tr = pd.read_csv(train, index=False)
    tt = pd.read_csv(test, index=False)

    if tt:
        df = pd.concat([tr, tt]).reset_index(drop=True)
    else:
        df = tr

    df.to_feather(dst)


def split_lightcurve(src, dst, n_split=30):
    df = pd.read_feather(src)

    for i in range(n_split):
        partial = df[df.object_id % n_split == i].reset_index(drop=True)
        partial.to_feather(dst.format(i))


def mkdir(dir):
    try:
        os.mkdir(dir)
    except:
        pass


if __name__ == "__main__":
    with timer("Make directory"):
        mkdir(config.DATA_DIR)
        mkdir(config.DEBUG_CSV_DIR)
        mkdir(config.FEATURE_DIR)
        mkdir(config.SHARE_DIR)
        mkdir(config.SUBMIT_DIR)


    with timer("Convert metadata"):
        concat_to_feather(config.DATA_DIR + "training_set_metadata.csv",
                          config.DATA_DIR + "test_set_metadata.csv",
                          config.DATA_DIR + "meta.f")

    with timer("Convert light curves"):
        concat_to_feather(config.DATA_DIR + "training_set.csv",
                          config.DATA_DIR + "test_set.csv",
                          config.DATA_DIR + "all.f")

    with timer("Convert light curves"):
        concat_to_feather(config.DATA_DIR + "training_set.csv",
                          None,
                          config.DATA_DIR + "train.f")

    with timer("Split light curves"):
        split_lightcurve(config.DATA_DIR + "all.f", config.DATA_DIR + "all_{}.f")
