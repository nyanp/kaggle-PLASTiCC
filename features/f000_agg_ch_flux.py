import pandas as pd
from .common import *

def f000_agg_ch_flux(meta: pd.DataFrame, lc: pd.DataFrame, debug=True, target_dir='../features'):
    aggs = aggregate(lc, 'flux', ['mean','max','min','median','std'])

    if debug:
        aggs.head(1000).to_csv(target_dir+'/f000.csv')
    aggs.to_feather(target_dir+'/f000.f')