from model.lgbm import LGBMModel
import pandas as pd
import time
from model.postproc import *
from experiments.experiment1 import Experiment1


exp = Experiment1(basepath='./')

exp.execute()
