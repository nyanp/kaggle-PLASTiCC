import pandas as pd
from .common import *


@feature('f000')
def f000_agg_ch_flux(input: Input, debug=True, target_dir='.'):
    aggs = aggregate_by_id_passbands(input.lc, 'flux', ['mean', 'max', 'min', 'median', 'std'])

    return aggs


@feature('f001')
def f001_agg_ch_flux_err(input: Input, debug=True, target_dir='.'):
    aggs = aggregate_by_id_passbands(input.lc, 'flux_err', ['mean', 'max', 'min', 'median', 'std'])

    return aggs



@feature('f002')
def f002_agg_ch_detected(input: Input, debug=True, target_dir='.'):
    aggs = aggregate_by_id_passbands(input.lc, 'detected', ['mean', 'std'])

    return aggs

