from collections import OrderedDict

import numpy as np
from menpo.landmark.labels.base import (validate_input,
                                        connectivity_from_array, labeller_func)
from menpo.shape import LabelledPointUndirectedGraph


def lv_index_to_label(index):
    out_indices = np.arange(0, 17)
    in_indices = np.arange(17, 34)
     

    if index in in_indices:
        return 'Endocardium'

    if index in out_indices:
        return 'Epicardium'

