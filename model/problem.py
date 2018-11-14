
classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}

classes_with_other = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95, 99]
class_weight_with_other = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1, 99: 2}

classes_in = [6, 16, 53, 65, 92]
class_weight_in = {6: 1, 16: 1, 53: 1, 65: 1, 92: 1}

classes_out = [15, 42, 52, 62, 64, 67, 88, 90, 95]
class_weight_out = {15: 2, 42: 1, 52: 1, 62: 1, 64: 2, 67: 1, 88: 1, 90: 1, 95: 1}

class_extra_galaxtic = [90,42,15,62,88,67,52,95,64]
class_inner_galaxtic = [65,16,92,6,53]

class_dict = {c: i for i, c in enumerate(classes)}
class_dict_out = {c: i for i, c in enumerate(classes_out)}
class_dict_in = {c: i for i, c in enumerate(classes_in)}

import numpy as np

def label_to_code(labels):
    if len(np.unique(labels)) == 14:
        return np.array([class_dict[c] for c in labels])
    elif len(np.unique(labels)) == 9:
        return np.array([class_dict_out[c] for c in labels])
    else:
        assert len(np.unique(labels)) == 5
        return np.array([class_dict_in[c] for c in labels])
