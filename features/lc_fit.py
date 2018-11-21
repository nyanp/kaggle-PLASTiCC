import sys
import sncosmo
from sncosmo.models import Model
import pandas as pd
import time
from astropy.table import Table
from contextlib import contextmanager
from tqdm import tqdm
from astropy import wcs, units as u
from sncosmo.bandpasses import read_bandpass

training_only = False
debug =False
checkpoint = 500
skip = 0
end = -1

for band in ['g','i','r','u','y','z']:
    b = read_bandpass('../lsst/total_{}.dat'.format(band), wave_unit=u.nm, trim_level=0.001, name='lsst{}_n'.format(band), normalize=True)
    sncosmo.register(b, 'lsst{}_n'.format(band))

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
    if len(sys.argv) < 2:
        raise RuntimeError('Specify Data Index')

    data_index = int(sys.argv[1])

    if len(sys.argv) >= 3:
        skip = int(sys.argv[2])
    if len(sys.argv) >= 4:
        end = int(sys.argv[3])

    print('index: {}'.format(data_index))
    print('skip: {}'.format(skip))
    print('end: {}'.format(end))

    with timer('load data'):
        meta = pd.read_feather('../input/meta.f')
        if training_only:
            meta = meta[~meta.target.isnull()]
        meta = meta[meta.hostgal_photoz > 0].reset_index(drop=True)

        lc = pd.read_feather('../input/all_{}.f'.format(data_index))
        if training_only:
            lc = lc[lc.object_id.isin(meta.index)].reset_index(drop=True)

    print('shape(lc): {}'.format(lc.shape))
    print('shape(meta): {}'.format(meta.shape))

    with timer('dropping meta'):
        meta = meta[meta.object_id.isin(lc.object_id)].reset_index(drop=True)
        print('shape(meta): {}'.format(meta.shape))

    meta.set_index('object_id', inplace=True)

    #passbands = ['lsstu','lsstg','lsstr','lssti','lsstz','lssty']
    passbands = ['lsstu_n', 'lsstg_n', 'lsstr_n', 'lssti_n', 'lsstz_n', 'lssty_n']

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

    if end > 0:
        n_loop = end

    for i in tqdm(range(skip, n_loop)):
        object_id = meta.index[i]
        try:
            ret.loc[object_id] = fitting(model, meta, lc, object_id)
            #if i % checkpoint == 0 and len(ret) > 0:
            #    ret.reset_index(drop=True).to_feather('../features/f500_{}_checkpoint{}_{}.f'.format(data_index, skip, i))
        except:
            n_errors += 1
            pass

        if i == 30 and n_errors == 30:
            raise RuntimeError('All 30 first attempts were failed. stopped')

    print('total {} data processed. {} data was skipped'.format(len(meta), n_errors))

    ret.reset_index(inplace=True)
    ret.rename(columns={'index': 'object_id'}, inplace=True)
    ret.to_feather('../features/f501_{}_{}_{}.f'.format(data_index, skip, end))
