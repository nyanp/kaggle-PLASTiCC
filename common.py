import pandas as pd

import config


def load_lightcurve() -> pd.DataFrame:
    if config.TRAINING_ONLY:
        path = config.DATA_DIR + "train.f"
    else:
        path = config.DATA_DIR + "all.f"

    return pd.read_feather(path)


def load_metadata() -> pd.DataFrame:
    meta = pd.read_feather(config.DATA_DIR + "meta.f")

    if config.TRAINING_ONLY:
        meta = meta[~meta.target.isnull()].reset_index(drop=True)

    return meta
