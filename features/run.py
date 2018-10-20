from .f000_agg_ch_flux import *
from .f001_agg_ch_flux_err import *

debug=True

meta = pd.read_feather('../input/meta.f')
lc = pd.read_feather('../input/all.f')

f000_agg_ch_flux(meta, lc, debug)
f001_agg_ch_flux_err(meta, lc, debug)
