import pandas as pd
from .common import *


@feature('f010')
def f010_agg_global_flux(input: Input, debug=True, target_dir='.'):
    aggs = aggregate_by_id(input.lc, 'flux', ['mean', 'max', 'min'])

    return aggs

