from typing import List

import sncosmo
import pandas as pd
from astropy import units as u
from astropy.table import Table
from sncosmo.bandpasses import read_bandpass
from sncosmo.models import Model
from tqdm import tqdm

import config
from util import timer


def fit_lc(model, meta, data, object_id, zbounds='estimated', clip_bounds=False, t_bounds=False, snr=5):
    table = Table.from_pandas(data[data.object_id == object_id])

    if zbounds == 'fixed':
        z = 0.7
        zerr = 0.7
    else:
        z = meta.loc[object_id, 'hostgal_photoz']
        zerr = meta.loc[object_id, 'hostgal_photoz_err']

    zmin = z - zerr
    zmax = z + zerr
    if clip_bounds:
        zmin = max(0.001, z - zerr)

    bounds = {
        'z': (zmin, zmax)
    }

    if t_bounds:
        tmin = data[data.object_id == object_id].mjd.min() - 50
        tmax = data[data.object_id == object_id].mjd.max()
        bounds['t0'] = (tmin, tmax)

    # run the fit
    result, fitted_model = sncosmo.fit_lc(
        table, model,
        model.param_names,  # parameters of model to vary
        bounds=bounds, minsnr=snr)  # bounds on parameters (if any)

    return [result.chisq] + [result.ncall] + list(result.parameters) + list(result.errors.values())


def extract_features(meta: pd.DataFrame,
                     lc: pd.DataFrame,
                     source: str,
                     normalize: bool = False,
                     snr: int = 3,
                     zbounds: str = 'estimated',
                     skip: int = 0,
                     end: int = -1,
                     clip_bounds: bool = False,
                     t_bounds: bool = False,
                     columns: List[str] = None):
    if normalize:
        try:
            for band in ['g', 'i', 'r', 'u', 'y', 'z']:
                b = read_bandpass('lsst/total_{}.dat'.format(band), wave_unit=u.nm, trim_level=0.001,
                                  name='lsst{}_n'.format(band), normalize=True)
                sncosmo.register(b, 'lsst{}_n'.format(band))
        except:
            raise
            pass

    with timer('dropping meta'):
        meta = meta[meta.object_id.isin(lc.object_id)].reset_index(drop=True)
        print('shape(meta): {}'.format(meta.shape))

    if 'object_id' in meta:
        meta.set_index('object_id', inplace=True)

    if normalize:
        passbands = ['lsstu_n', 'lsstg_n', 'lsstr_n', 'lssti_n', 'lsstz_n', 'lssty_n']
    else:
        passbands = ['lsstu', 'lsstg', 'lsstr', 'lssti', 'lsstz', 'lssty']

    with timer('prep'):
        lc['band'] = lc['passband'].apply(lambda x: passbands[x])
        lc['zpsys'] = 'ab'
        lc['zp'] = 25.0

    # create a model
    model = Model(source=source)

    params = model.param_names

    if columns:
        ret = pd.DataFrame(columns=columns)
    else:
        columns = ['chisq', 'ncall'] + [source + '_' + c for c in params] + [source + '_' + c + '_err' for c in params]
        ret = pd.DataFrame(columns=columns)

        prefix = source
        if zbounds == 'fixed':
            prefix += '_f_'
        else:
            prefix += '_p_'

        prefix += 'sn{}_'.format(snr)

        if normalize:
            prefix += 'n_'

        ret.columns = [prefix + c for c in ret.columns]

    n_errors = 0

    n_loop = len(meta)

    if end > 0:
        n_loop = end

    for i in tqdm(range(skip, n_loop)):
        object_id = meta.index[i]
        try:
            ret.loc[object_id] = fit_lc(model, meta, lc, object_id, zbounds, clip_bounds, t_bounds, snr)
        except:
            n_errors += 1

            if i == 30 and n_errors == 31:
                print('All 30 first attempts were failed. stopped')
                raise

    print('total {} data processed. {} data was skipped'.format(len(meta), n_errors))

    ret.reset_index(inplace=True)
    ret.rename(columns={'index': 'object_id'}, inplace=True)
    return ret
