from experiments.experiment50 import *
from experiments.experiment51 import *
from experiments.experiment53 import *
from experiments.experiment56 import *


exp = Experiment56(basepath='./',
                   submit_path='output/experiment56.csv',
                   log_name='experiment56',
                   use_extra_classifier=False)
exp.execute()


exp = Experiment56(basepath='./',
                   submit_path='output/experiment56_th975.csv',
                   log_name='experiment56_th975',
                   use_extra_classifier=False,
                   pseudo_th=0.975)
exp.execute()

exp = Experiment56(basepath='./',
                   submit_path='output/experiment56_th98.csv',
                   log_name='experiment56_th98',
                   use_extra_classifier=False,
                   pseudo_th=0.98)
exp.execute()
