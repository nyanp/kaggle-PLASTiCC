import pandas as pd
import common


def f517_blending_salts():
    meta = common.load_metadata()
    f500 = common.load_feature('f500')
    f515 = common.load_feature('f515')
    f516 = common.load_feature('f516')
    df = pd.merge(meta[['object_id','target','hostgal_photoz','ddf']], f500, on='object_id', how='left')
    df = pd.merge(df, f515, on='object_id', how='left')
    df = pd.merge(df, f516, on='object_id', how='left')

    prefix = ['sn_salt2_', 'salt2-extended_p_sn3_salt2-extended_', 'salt2_p_sn3_salt2_']
    params = ['x0', 't0', 'z', 'c', 'x1']

    for p in params:
        print('param: {}'.format(p))

        # weighted average based on error
        weights = []
        weighted_sum = []
        for m in prefix:
            col = 'w_{}{}'.format(p, m)
            df[col] = 1 / (df['{}{}_err'.format(m, p)] * df['{}{}_err'.format(m, p)])
            weights.append(col)
            df[col + '_s'] = df[col] * df[m + p]
            weighted_sum.append(col + '_s')

        df['salt2-{}-weighted-avg'.format(p)] = df[weighted_sum].sum(axis=1)
        df['tmp'] = df[weights].sum(axis=1)
        df['salt2-{}-weighted-avg'.format(p)] = df['salt2-{}-weighted-avg'.format(p)] / df['tmp']
        df.drop('tmp', axis=1, inplace=True)
        df.drop(weighted_sum, axis=1, inplace=True)
        df.drop(weights, axis=1, inplace=True)

        common.save_feature(df[['object_id'] + ['salt2-{}-weighted-avg'.format(p) for p in params]], 'f517')
