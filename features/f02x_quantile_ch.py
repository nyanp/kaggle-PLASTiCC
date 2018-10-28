import pandas as pd
import numpy as np
from .common import *

@feature('f020')
def f020_percentile95(input: Input, debug=True, target_dir='.'):
    aggs = aggregate_by_id_passbands(input.lc, 'flux', [percentile(95)])

    return aggs

@feature('f021')
def f021_percentile05(input: Input, debug=True, target_dir='.'):
    aggs = aggregate_by_id_passbands(input.lc, 'flux', [percentile(5)])

    return aggs

@feature('f022')
def f022_percentile60(input: Input, debug=True, target_dir='.'):
    aggs = aggregate_by_id_passbands(input.lc, 'flux', [percentile(60)])

    return aggs

@feature('f023')
def f023_percentile40(input: Input, debug=True, target_dir='.'):
    aggs = aggregate_by_id_passbands(input.lc, 'flux', [percentile(40)])

    return aggs


@feature('f024')
def f024_percentile675(input: Input, debug=True, target_dir='.'):
    aggs = aggregate_by_id_passbands(input.lc, 'flux', [percentile(67.5)])

    return aggs

@feature('f025')
def f025_percentile325(input: Input, debug=True, target_dir='.'):
    aggs = aggregate_by_id_passbands(input.lc, 'flux', [percentile(32.5)])

    return aggs

@feature('f026')
def f026_percentile75(input: Input, debug=True, target_dir='.'):
    aggs = aggregate_by_id_passbands(input.lc, 'flux', [percentile(75)])

    return aggs

@feature('f027')
def f027_percentile25(input: Input, debug=True, target_dir='.'):
    aggs = aggregate_by_id_passbands(input.lc, 'flux', [percentile(25)])

    return aggs

@feature('f028')
def f028_percentile825(input: Input, debug=True, target_dir='.'):
    aggs = aggregate_by_id_passbands(input.lc, 'flux', [percentile(82.5)])

    return aggs

@feature('f029')
def f029_percentile175(input: Input, debug=True, target_dir='.'):
    aggs = aggregate_by_id_passbands(input.lc, 'flux', [percentile(17.5)])

    return aggs
