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
    #exp = Experiment71(basepath='./', seed=seed, submit_path='experiment71_seed{}.csv'.format(seed),
    #                   use_extra_classifier=False, n_estimators_extra_classifier=None)
    exp = Experiment71(basepath='./', seed=seed, submit_path='experiment71_r500_seed{}.csv'.format(seed), log_name='experiment71_pl',
                       use_extra_classifier=True, n_estimators_extra_classifier=500, use_pl_labels=True)
    exp.execute()
