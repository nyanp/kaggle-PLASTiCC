import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import grad
from sklearn.preprocessing import OneHotEncoder


classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1,
                64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}

class_dict = {c: i for i, c in enumerate(classes)}
