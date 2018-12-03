from experiments.experiment50 import *
from experiments.experiment51 import *
from experiments.experiment53 import *

exp = Experiment53(basepath='./',
                   submit_path='output/experiment53_2.csv',
                   pseudo_n_loop=0,
                   use_extra_classifier=True)

exp.execute()