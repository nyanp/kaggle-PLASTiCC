import pandas as pd
from .common import *
import numpy as np

# https://www.kaggle.com/scirpus/plasticc-pipeline-starter
@feature('f040')
def f040_flux_ratio_sq(input: Input, debug=True, target_dir='.'):

    input.lc['flux_ratio_sq'] = np.power(input.lc['flux'] / input.lc['flux_err'], 2.0)

    return aggregate_by_id_passbands(input.lc, 'flux_ratio_sq', ['sum'])


@feature('f041')
def f041_flux_by_flux_ratio_sq(input: Input, debug=True, target_dir='.'):

    input.lc['flux_ratio_sq'] = np.power(input.lc['flux'] / input.lc['flux_err'], 2.0)
    input.lc['flux_by_flux_ratio_sq'] = input.lc['flux'] * input.lc['flux_ratio_sq']

    return aggregate_by_id_passbands(input.lc, 'flux_by_flux_ratio_sq', ['sum'])

