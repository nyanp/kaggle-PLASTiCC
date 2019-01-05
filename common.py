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


def load_passband_metadata() -> pd.DataFrame:
    meta = pd.read_feather(config.DATA_DIR + "passband_meta.f")
    return meta


# "f210" => pd.DataFrame
def load_feature(feature_id: str) -> pd.DataFrame:
    path = config.FEATURE_DIR + feature_id + ".f"

    return pd.read_feather(path)


def save_feature(df: pd.DataFrame, feature_id: str, with_csv_dump: bool = False):
    if config.REPLICA_MODE:
        path = config.FEATURE_DIR + feature_id + "_replica.f"
    else:
        path = config.FEATURE_DIR + feature_id + ".f"

    df.to_feather(path)

    if with_csv_dump:
        df.head(1000).to_csv(config.DEBUG_CSV_DIR + feature_id + ".csv")
