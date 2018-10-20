import pandas as pd
from .common import *

def f001_agg_ch_flux_err(meta: pd.DataFrame, lc: pd.DataFrame, debug=True, target_dir='../features'):
    aggs = aggregate(lc, 'flux_err', ['mean','max','min','median','std'])

    if debug:
        aggs.head(1000).to_csv(target_dir+'/f001.csv')
    aggs.to_feather(target_dir+'/f001.f')
