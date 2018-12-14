from subprocess import Popen
import sys
import pandas as pd
import subprocess
import os

if __name__ == "__main__":
    meta = int(sys.argv[1])
    index = sys.argv[2]
    script = 'lc_fit_' + index + '.py'

    #n = len(pd.read_feather('../input/all_{}.f'.format(meta)))
    meta_ = pd.read_feather('../input/meta.f')
    meta_ = meta_[meta_.hostgal_photoz > 0].reset_index(drop=True)
    n = len(meta_[meta_.object_id % 30 == meta])

    print('script: {}, index: {}'.format(script, meta))

    chunksize = 30
    end = n
    print('total {} rows.'.format(n))

    for i in range(0, 103800, 300):
        n = 'f{}_{}_{}_{}.f'.format(index,meta,i,i+300)
        if not os.path.exists(n):
            print("{} doesn't exist".format(n))

            for j in range(i, i+300, chunksize):
                chunk_start = j
                chunk_end = min(end, j + chunksize)
                try:
                    subprocess.call(["python", script, str(meta), str(chunk_start), str(chunk_end)])
                except:
                    print('###### catch exception: {}-{} ######'.format(chunk_start, chunk_end))
