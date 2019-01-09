import config
from experiments.experiment65 import *

n_loop = 3 if config.USE_PSEUDO_LABEL else 0
exp = Experiment65(logdir=config.MODEL_DIR, pseudo_n_loop=n_loop)
exp.execute()
