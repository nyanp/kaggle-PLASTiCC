from subprocess import Popen
import sys
import pandas as pd
import subprocess

if __name__ == "__main__":
    meta = int(sys.argv[1])

    #n = len(pd.read_feather('../input/all_{}.f'.format(meta)))
    meta_ = pd.read_feather('../input/meta.f')
    meta_ = meta_[meta_.hostgal_photoz > 0].reset_index(drop=True)
    n = len(meta_[meta_.object_id % 30 == meta])

    chunksize = 300
    print('total {} rows.'.format(n))

    start = 0
    end = n
    script = 'lc_fit.py'

    if len(sys.argv) >= 4:
        start = int(sys.argv[2])
        end = int(sys.argv[3])
    if len(sys.argv) >= 5:
        chunksize = int(sys.argv[4])
    if len(sys.argv) >= 6:
        script = 'lc_fit_' + sys.argv[5] + '.py'

    print('script: {}, index: {}, {}-{}, chunk:{}'.format(script, meta, start, end, chunksize))

    for i in range(start, end, chunksize):
        chunk_start = i
        chunk_end = min(end, i + chunksize)

        try:
            subprocess.call(["python", script, str(meta), str(chunk_start), str(chunk_end)])
        except:
            print('###### catch exception: {}-{} ######'.format(chunk_start, chunk_end))
