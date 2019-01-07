import os

import pandas as pd
from tqdm import tqdm

import config
from util import timer


def concat_to_feather(train, test, dst, column_order=None):
    tr = pd.read_csv(train)

    if test is not None:
        tt = pd.read_csv(test)
        df = pd.concat([tr, tt]).reset_index(drop=True)
    else:
        df = tr

    if column_order is not None:
        df[column_order].to_feather(dst)
    else:
        df.to_feather(dst)


def split_lightcurve(src, dst, n_split=30):
    df = pd.read_feather(src)

    for i in tqdm(range(n_split)):
        partial = df[df.object_id % n_split == i].reset_index(drop=True)
        partial.to_feather(dst.format(i))


def mkdir(dir):
    try:
        os.mkdir(dir)
    except:
        pass


def make_passband_meta():
    data = pd.read_feather(config.DATA_DIR + 'all.f')
    print('')
    max_flux = data.groupby(['object_id', 'passband'])['flux'].max().reset_index()
    max_flux = pd.merge(max_flux, data[['object_id', 'passband', 'flux', 'mjd']],
                        on=['object_id', 'passband', 'flux'],
                        how='left')
    max_flux.columns = ['object_id', 'passband', 'max(flux)', 'time(max(flux))']
    max_flux.drop_duplicates(subset=['object_id', 'passband'], inplace=True)
    max_flux.reset_index(drop=True).to_feather(config.DATA_DIR + 'passband_meta.f')


if __name__ == "__main__":
    with timer("Make directory"):
        mkdir(config.DATA_DIR)
        mkdir(config.DEBUG_CSV_DIR)
        mkdir(config.FEATURE_DIR)
        mkdir(config.MODEL_DIR)
        mkdir(config.SHARE_DIR)
        mkdir(config.SUBMIT_DIR)

    with timer("Convert metadata"):
        concat_to_feather(config.DATA_DIR + "training_set_metadata.csv",
                          config.DATA_DIR + "test_set_metadata.csv",
                          config.DATA_DIR + "meta.f",
                          column_order=['object_id', 'ra', 'decl', 'gal_l', 'gal_b', 'ddf',
                                        'hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err',
                                        'distmod', 'mwebv', 'target'])

    with timer("Convert light curves"):
        concat_to_feather(config.DATA_DIR + "training_set.csv",
                          config.DATA_DIR + "test_set.csv",
                          config.DATA_DIR + "all.f")

    with timer("Convert light curves(training data)"):
        concat_to_feather(config.DATA_DIR + "training_set.csv",
                          None,
                          config.DATA_DIR + "train.f")

    if config.USE_TEMPLATE_FIT_FEATURES:
        with timer("Split light curves into 30 chunks"):
            split_lightcurve(config.DATA_DIR + "all.f", config.DATA_DIR + "all_{}.f")

    with timer("Cache time(max-flux)"):
        make_passband_meta()
