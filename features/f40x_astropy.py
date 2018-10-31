from astropy.stats import LombScargle
from .common import *

def extract(g):
    try:
        freq, power = LombScargle(g.mjd, g.flux, g.flux_err).autopower()
        return pd.Series([power.max(), freq[power.argmax()]])
    except:
        return pd.Series([0.0, 0.0])


@feature('f400')
def f400_lombscargle(input: Input, debug=True, target_dir='.'):
    d =input.lc.groupby(['object_id', 'passband']).apply(extract)
    d.columns = ['max(astropy.lombscargle.power)','astropy.lombscargle.timescale']
    return unstack(d)