from subprocess import Popen
import sys
import pandas as pd
import subprocess

if __name__ == "__main__":
    meta = sys.argv[1]

    n = len(pd.read_feather('../input/all_{}.f'.format(meta)))

    chunksize = 300
    print('total {} rows.'.format(n))

    start = 0
    end = n

    if len(sys.argv) >= 4:
        start = int(sys.argv[2])
        end = int(sys.argv[3])
    if len(sys.argv) >= 5:
        chunksize = int(sys.argv[4])

    print('index: {}, {}-{}, chunk:{}'.format(meta, start, end, chunksize))

    for i in range(start, end, chunksize):
        chunk_start = i
        chunk_end = i + chunksize

        try:
            subprocess.call(["python", "lc_fit.py", str(meta), str(chunk_start), str(chunk_end)])
        except:
            print('###### catch exception: {}-{} ######'.format(chunk_start, chunk_end))
