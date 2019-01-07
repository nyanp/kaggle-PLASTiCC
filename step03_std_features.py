from features.f3xx_tsfresh import *
from features.f40x_astropy import *

import common
import config

debug = config.TRAINING_ONLY
output = config.FEATURE_DIR
cv_only = config.TRAINING_ONLY

meta = common.load_metadata()
lc = common.load_lightcurve()
pb = common.load_passband_metadata()

lc['id_passband'] = lc['object_id'].astype(str) + lc['passband'].astype(str)
input = Input(meta, pb, lc)

f300_num_peaks(input=input, debug=debug, target_dir=output)
f301_quantile2(input=input, debug=debug, target_dir=output)
f302_quantile8(input=input, debug=debug, target_dir=output)
f303_c3(input=input, debug=debug, target_dir=output)
f304_autocorr1(input=input, debug=debug, target_dir=output)
f305_autocorr2(input=input, debug=debug, target_dir=output)
f306_autocorr3(input=input, debug=debug, target_dir=output)
f307_autocorr4(input=input, debug=debug, target_dir=output)
f308_autocorr5(input=input, debug=debug, target_dir=output)
f309_autocorr_mean(input=input, debug=debug, target_dir=output)
f310_autocorr_median(input=input, debug=debug, target_dir=output)
f311_autocorr_var(input=input, debug=debug, target_dir=output)
f321_partial_autocorr_lag10(input=input, debug=debug, target_dir=output)
f330_number_cwt_peaks(input=input, debug=debug, target_dir=output)
f340_number_crossing_m(input=input, debug=debug, target_dir=output)
f350_linear_trend(input=input, debug=debug, target_dir=output)
f361_fft_coefficient(input=input, debug=debug, target_dir=output)
f370_fft_aggregated(input=input, debug=debug, target_dir=output)

f400_lombscargle(input=input, debug=debug, target_dir=output)
