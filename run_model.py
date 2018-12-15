import sys
from experiments.experiment71 import *


#exp = Experiment70(basepath='./')
#exp.execute()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    else:
        seed = 0

    print('Training start. seed: {}'.format(seed))
    exp = Experiment71(basepath='./', seed=seed, submit_path='experiment71_seed{}.csv'.format(seed))
    exp.execute()
