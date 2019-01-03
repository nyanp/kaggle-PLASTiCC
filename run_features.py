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
from features.f3xx_tsfresh import *
from features.f40x_astropy import *

debug=True
cv_only=False

if cv_only:
    output = 'features_tr/'
else:
    output = 'features/'

meta = pd.read_feather('input/meta.f') # for each object_id

if cv_only:
    meta = meta[~meta.target.isnull()].reset_index(drop=True)

print('read all light curves...')

if cv_only:
    lc = pd.read_feather('input/train.f')
    ls = pd.read_feather('input/lombscale_train.f')
else:
    lc = pd.read_feather('input/all.f') # for each observation
    ls = None

#lc['id_passband'] = lc['object_id'].astype(str) + '_' + lc['passband'].astype(str)

input = Input(meta, None, lc, ls)

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
