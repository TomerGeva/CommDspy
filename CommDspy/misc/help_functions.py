import numpy as np

def check_binary(pattern):
    data_in = np.unique(pattern)
    if len(data_in) == 1 and (0 not in data_in and 1 not in data_in):
        raise ValueError('Data in is not binary, please consider other encoding methods')
    elif len(data_in) == 2 and (0 not in data_in and 1 not in data_in):
        raise ValueError('Data in is not binary, please consider other encoding methods')
    elif len(data_in) > 2:
        raise ValueError('Data in is not binary, please consider other encoding methods')