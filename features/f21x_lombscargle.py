import pandas as pd
from .common import *
import numpy as np
import gc



@feature('f210')
def f210_lombscargle_max_power(input: Input, debug=True, target_dir='.'):
    return aggregate_by_id_passbands(input.ls, 'power', ['max'], ['max(lombscargle_power)'])


@feature('f211')
def f211_lombscargle_max_power_freq(input: Input, debug=True, target_dir='.'):
    max_power = input.ls.groupby(['object_id','passband'])['power'].max()
    max_power.name = 'max(power)'

    ls_with_max = pd.merge(input.ls, max_power.reset_index(), on=['object_id', 'passband'])
    ls_with_max = ls_with_max[ls_with_max['max(power)'] == ls_with_max['power']].drop_duplicates(subset=['object_id', 'passband'])

    ls_with_max.rename(columns={'freq': 'lombscargle_timescale_days'}, inplace=True)

    return unstack(ls_with_max[['object_id','passband','lombscargle_timescale_days']])

