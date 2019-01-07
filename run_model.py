import sys

import config
from experiments.experiment65 import *

exp = Experiment65(logdir=config.MODEL_DIR)
exp.execute()
