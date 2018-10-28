from .common import *


@feature('f030')
def f030_agg_ch_flux_skew(input: Input, debug=True, target_dir='.'):
    aggs = aggregate_by_id_passbands(input.lc, 'flux', ['skew'])

    return aggs


@feature('f031')
def f031_agg_ch_flux_kurtosis(input: Input, debug=True, target_dir='.'):
    aggs = aggregate_by_id_passbands(input.lc, 'flux', ['kurt'])

    return aggs
