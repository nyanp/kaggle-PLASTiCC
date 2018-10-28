import pandas as pd
from .common import *
import numpy as np


@feature('f130', required_feature='max(flux)_ch1', required_feature_in='f000.f')
def f130_flux_minmax(input: Input, debug=True, target_dir='.'):
    meta = input.meta

    cols = []
    for c in range(6):
        n = 'max(flux)_ch{}'.format(c)
        p = 'min(flux)_ch{}'.format(c)
        dst = 'max(flux) - min(flux)_ch{}'.format(c)
        meta[dst] = meta[n] - meta[p]
        cols.append(dst)

    return meta[['object_id'] + cols]
