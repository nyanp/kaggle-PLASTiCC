from features.f00x_agg_ch import *
from features.f01x_agg_global_flux import *
from features.f02x_quantile_ch import *
from features.f05x_mjd_delta import *
from features.f06x_detected_agg_ch import *
from features.f10x_flux_diff_ch import *
from features.f11x_flux_slope import *
from features.f14x_flux_ratio_ch import *
from features.f15x_luminosity import *
from features.f20x_timescale import *

import common
import config

debug = config.TRAINING_ONLY
output = config.FEATURE_SAVE_DIR
cv_only = config.TRAINING_ONLY

meta = common.load_metadata()
lc = common.load_lightcurve()
pb = common.load_passband_metadata()

input = Input(meta, pb, lc)

f000_agg_ch_flux(input=input, debug=debug, target_dir=output)
f001_agg_ch_flux_err(input=input, debug=debug, target_dir=output)
f002_agg_ch_detected(input=input, debug=debug, target_dir=output)

f010_agg_global_flux(input=input, debug=debug, target_dir=output)

f026_percentile75(input=input, debug=debug, target_dir=output)

f050_mjddelta(input=input, debug=debug, target_dir=output)
f051_mjddelta_per_ch(input=input, debug=debug, target_dir=output)
f052_max_to_detected_time(input=input, debug=debug, target_dir=output)
f053_max_to_detected_time_ch(input=input, debug=debug, target_dir=output)
f054_mjddelta_per_ch(input=input, debug=debug, target_dir=output)

f061_median_ch_flux_detected(input=input, debug=debug, target_dir=output)
f063_median_flux_diff1_ch_detected(input=input, debug=debug, target_dir=output)

f100_max_flux_diff1_ch(input=input, debug=debug, target_dir=output)
f101_mean_flux_diff1_ch(input=input, debug=debug, target_dir=output)
f102_median_flux_diff1_ch(input=input, debug=debug, target_dir=output)
f103_max_flux_diff2_ch(input=input, debug=debug, target_dir=output)
f104_max_flux_diff3_ch(input=input, debug=debug, target_dir=output)

f106_max_flux_diff4_ch(input=input, debug=debug, target_dir=output)
f107_max_flux_diff5_ch(input=input, debug=debug, target_dir=output)
f108_min_flux_diff1_ch(input=input, debug=debug, target_dir=output)
f109_min_flux_diff2_ch(input=input, debug=debug, target_dir=output)
f110_flux_slope_minmax(input=input, debug=debug, target_dir=output)

f140_flux_amplitude_ratio1_ch(input=input, debug=debug, target_dir=output)
f141_flux_amplitude_ratio2_ch(input=input, debug=debug, target_dir=output)
f142_flux_amplitude_ratio3_ch(input=input, debug=debug, target_dir=output)
f143_flux_amplitude_ratio4_ch(input=input, debug=debug, target_dir=output)
f144_flux_amplitude_ratio5_ch(input=input, debug=debug, target_dir=output)

f150_luminosity_minmax_ch(input=input, debug=debug, target_dir=output)
f151_luminosity_minmax_ch_ratio(input=input, debug=debug, target_dir=output)
f152_luminosity_minmax_ch(input=input, debug=debug, target_dir=output)
f153_luminosity_minmax_ch(input=input, debug=debug, target_dir=output)

f200_timescale_th015_max(input=input, debug=debug, target_dir=output)
f201_timescale_th015_min(input=input, debug=debug, target_dir=output)
f202_timescale_th035_max(input=input, debug=debug, target_dir=output)
f203_timescale_th035_min(input=input, debug=debug, target_dir=output)
f204_timescale_th035_max(input=input, debug=debug, target_dir=output)
f205_timescale_th035_min(input=input, debug=debug, target_dir=output)
