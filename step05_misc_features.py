import config
from features.f10xx_misc import *

debug = config.TRAINING_ONLY
output = config.FEATURE_DIR

meta = common.load_metadata()
lc = common.load_lightcurve()
pb = common.load_passband_metadata()

input = Input(meta, pb, lc)

f1000_salt2_normalized_chisq(input, debug=debug, target_dir=output)
f1001_detected_to_risetime_ratio(input, debug=debug, target_dir=output)
f1002_detected_to_falltime_ratio(input, debug=debug, target_dir=output)
f1003_luminosity_by_estimated_redshift(input, debug=debug, target_dir=output)
f1004_tsfresh_flux(input, debug=debug, target_dir=output)
f1005_tsfresh_flux_per_passband(input, debug=debug, target_dir=output)
f1006_tsfresh_mjd(input, debug=debug, target_dir=output)
f1080_snr3_minmax_diff(input, debug=debug, target_dir=output)
f1081_first_is_detected(input, debug=debug, target_dir=output)
f1082_last_is_detected(input, debug=debug, target_dir=output)
f1083_max_flux_within_snr3(input, debug=debug, target_dir=output)
f1084_min_flux_within_snr3(input, debug=debug, target_dir=output)
f1085_luminosity_diff_within_snr3(input, debug=debug, target_dir=output)
f1086_first_detected_to_prev_mjd_diff(input, debug=debug, target_dir=output)
f1087_last_detected_to_next_mjd_diff(input, debug=debug, target_dir=output)
f1088_first_detected_to_prev_mjd_diff_perch(input, debug=debug, target_dir=output)
f1089_last_detected_to_next_mjd_diff_perch(input, debug=debug, target_dir=output)
