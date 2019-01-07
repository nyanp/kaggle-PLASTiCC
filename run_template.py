import sys

import common
import config
from features.lc_fit_common import extract_features

# run template.py [feature_id] [data_index] [start_row] [end_row]

if __name__ == "__main__":

    parameters = {
        500: {
            'source': 'salt2-extended',
            'normalize': False,
            'snr': 5,
            'zbounds': 'default'
        },
        505: {
            'source': 'salt2-extended',
            'normalize': True,
            'snr': 3,
            'zbounds': 'fixed'
        },
        506: {
            'source': 'nugent-sn2n',
            'normalize': True,
            'snr': 3,
            'zbounds': 'fixed'
        },
        507: {
            'source': 'nugent-sn1bc',
            'normalize': True,
            'snr': 3,
            'zbounds': 'fixed'
        },
        509: {
            'source': 'snana-2004fe',
            'normalize': False,
            'snr': 5,
            'zbounds': 'default'
        },
        510: {
            'source': 'snana-2007Y',
            'normalize': False,
            'snr': 5,
            'zbounds': 'default'
        },
        511: {
            'source': 'hsiao',
            'normalize': False,
            'snr': 5,
            'zbounds': 'default'
        },
        512: {
            'source': 'nugent-sn2n',
            'normalize': False,
            'snr': 5,
            'zbounds': 'default'
        },
        513: {
            'source': 'nugent-sn1bc',
            'normalize': False,
            'snr': 5,
            'zbounds': 'default'
        },
        515: {
            'source': 'salt2-extended',
            'normalize': False,
            'snr': 3,
            'zbounds': 'default',
            'clip_bounds': True,
            't_bounds': True,
        },
        516: {
            'source': 'salt2',
            'normalize': False,
            'snr': 3,
            'zbounds': 'default',
            'clip_bounds': True,
            't_bounds': True,
        },
    }

    param = parameters[type]
    print('param: {}'.format(param))

    meta = common.load_metadata()
    meta = meta[meta.hostgal_photoz > 0].reset_index(drop=True)

    if len(sys.argv) < 6:
        raise RuntimeError('Specify Data Index')

    type = int(sys.argv[1])
    data_index = int(sys.argv[2])
    skip = int(sys.argv[3])
    end = int(sys.argv[4])
    lc = common.load_partial_lightcurve(data_index)
    dst_id = 'f{}_{}_{}_{}'.format(type, data_index, skip, end)

    dst = extract_features(meta, lc, skip=skip, end=end, **param)
    common.save_feature(dst, dst_id, with_csv_dump=config.USE_FIRST_CHUNK_FOR_TEMPLATE_FITTING)
