import sys
import sncosmo
from sncosmo.models import Model
import pandas as pd
import time
from astropy.table import Table
from contextlib import contextmanager
from tqdm import tqdm

training_only = False
debug =False

@contextmanager
def timer(name):
    s = time.time()
    yield

    print('[{}] {}'.format(time.time() - s, name))


def fitting(model, meta, data, object_id):
    z = meta.loc[object_id, 'hostgal_photoz']
    zerr = meta.loc[object_id, 'hostgal_photoz_err']
    table = Table.from_pandas(data[data.object_id == object_id])

    # run the fit
    result, fitted_model = sncosmo.fit_lc(
        table, model,
        ['z', 't0', 'x0', 'x1', 'c'],  # parameters of model to vary
        bounds={'z': (z - zerr, z + zerr)})  # bounds on parameters (if any)

    return [result.chisq] + [result.ncall] + list(result.parameters) + list(result.errors.values())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError('Specify Data Index')

    data_index = int(sys.argv[1])

    print('index: {}'.format(data_index))
    with timer('load data'):
        meta = pd.read_feather('../input/meta.f')
        if training_only:
            meta = meta[~meta.target.isnull()]
        meta = meta[meta.hostgal_photoz > 0].reset_index(drop=True)
        meta.set_index('object_id', inplace=True)

        lc = pd.read_feather('../input/all_{}.f'.format(data_index))
        if training_only:
            lc = lc[lc.object_id.isin(meta.index)].reset_index(drop=True)

    print('shape(lc): {}'.format(lc.shape))
    print('shape(meta): {}'.format(meta.shape))

    with timer('dropping meta'):
        meta = meta[meta.object_id.isin(lc.object_id)].reset_index(drop=True)
        print('shape(meta): {}'.format(meta.shape))

    passbands = ['lsstu','lsstg','lsstr','lssti','lsstz','lssty']

    with timer('prep'):
        s = time.time()
        lc['band'] = lc['passband'].apply(lambda x: passbands[x])
        lc['zpsys'] = 'ab'
        lc['zp'] = 25.0

    ret = pd.DataFrame(
        columns=['chisq', 'ncall', 'z', 't0', 'x0', 'x1', 'c', 'z_err', 't0_err', 'x0_err', 'x1_err', 'c_err'])

    # create a model
    model = Model(source='salt2')

    n_errors = 0

    n_loop = len(meta)
    if debug:
        n_loop = 100

    for i in tqdm(range(n_loop)):
        object_id = meta.index[i]
        try:
            ret.loc[object_id] = fitting(model, meta, lc, object_id)
        except:
            n_errors += 1
            pass

    print('total {} data processed. {} data was skipped'.format(len(meta), n_errors))

    ret.reset_index(inplace=True)
    ret.rename(columns={'index': 'object_id'}, inplace=True)
    ret.to_feather('../features/f500_{}.f'.format(data_index))
