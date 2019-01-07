import sys
from typing import List

import pandas as pd
import subprocess

import common
import config
from pathlib import Path
from tqdm import tqdm

features_to_run = [500, 505, 506, 507, 509, 510, 511, 512, 513, 515, 516]

# step05_template_features.py  : create all features (it takes 6+ months in 64 core machine)
# step05_template_features.py n : create features where object_id % 30 == n (it takes 10 days)


def merge_features():
    p = Path('features_all/')
    for feature in features_to_run:
        print('Merge feature {}'.format(feature))
        files = [str(f) for f in list(p.glob('f{}_*.f'.format(feature)))]
        print('Total {} files found'.format(len(files)))

        dfs = []
        for file in tqdm(files):
            dfs.append(pd.read_feather(file))
        df = pd.concat(dfs).reset_index(drop=True)
        df.drop_duplicates(inplace=True)
        common.save_feature(df, 'f{}'.format(feature))


def extract_feature(meta: pd.DataFrame, feature: int, data_index: List[int]):
    for d in data_index:
        n = len(meta[meta.object_id % 30 == d])

        print('total {} rows.'.format(n))

        start = 0
        end = n
        script = 'run_template.py'

        print('script: {}, index: {}, {}-{}, chunk:{}'.format(script, d, start, end, chunksize))

        for i in range(start, end, chunksize):
            chunk_start = i
            chunk_end = min(end, i + chunksize)

            try:
                subprocess.call(["python", script, str(feature), str(d), str(chunk_start), str(chunk_end)])
            except:
                print('###### catch exception: {}-{} ######'.format(chunk_start, chunk_end))
                raise

            if config.USE_FIRST_CHUNK_FOR_TEMPLATE_FITTING:
                return


if __name__ == "__main__":
    data_index = list(range(30))

    if len(sys.argv) == 2:
        if sys.argv[1] == 'merge':
            merge_features()
            exit(0)
        else:
            data_index = [int(sys.argv[1])]

    meta = common.load_metadata()
    meta = meta[meta.hostgal_photoz > 0].reset_index(drop=True)

    chunksize = 300

    for f in features_to_run:
        print('######### create feature: {} ########'.format(f))
        extract_feature(meta, f, data_index)
