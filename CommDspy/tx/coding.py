import numpy as np
from CommDspy.constants import ConstellationEnum
from CommDspy.auxiliary import get_levels, get_gray_level_vec
from scipy.signal import lfilter

def coding_gray(pattern, constellation=ConstellationEnum.PAM4):
    """
        :param pattern: Uncoded pattern, should be a numpy array of non-negative integers stating the index in the
         constellation point. Examples:
                1. 1-bit patterns will be '0' and '1'
                2. 2-bit patterns will be '0', '1', '2' and '3'
        :param constellation: Enumeration stating the constellation. Should be taken from:
                              CommDspy.constants.ConstellationEnum
        :return: Gray coded pattern
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    level_num       = len(get_levels(constellation))
    bits_per_symbol = int(np.ceil(np.log2(level_num)))
    # ==================================================================================================================
    # Gray coding
    # ==================================================================================================================
    if bits_per_symbol > 1:
        levels = get_gray_level_vec(level_num)
        return levels[pattern]
    else:
        return pattern

def coding_differential(pattern, constellation=ConstellationEnum.PAM4):
    """
    :param pattern: symbols we would like to perform differential encoding
    :param constellation: the constellation we are working with
    :return: Function performs differential encoding to the input signal. The flow is as follows:

                     m
    x_n -----------> + --------------------------> y_n
                     ^                       |
                     |  |-------------|      |
                     |--|      D      |<------
                        |-------------|
    """
    lvl_num = len(get_levels(constellation))
    return (lfilter([1], [1, -1], pattern.astype(float)) % lvl_num).astype(int)

def coding_manchester(pattern):
    """
    :param pattern: pattern to perform manchester encoding on, should be a numpy array
    :return: pattern after manchester encoding, note that the length of the pattern will be double due to the nature of
    the encoding process. Example for manchester encoding:
    pattern = [1,    0,    1,    0,    0,    1,    1,    1,    0,    0,    1]
    encoded = [1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1 ,0]
    NOTE, this encoding scheme assumes binary data input
    """
    # ==================================================================================================================
    # Checking data validity
    # ==================================================================================================================
    data_in = np.unique(pattern)
    if len(data_in) > 2 or (0 not in data_in and 1 not in data_in):
        raise ValueError('Data in is not binary, please consider other encoding methods')
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    pattern_shape          = pattern.shape
    new_pattern_shape      = list(pattern_shape)
    new_pattern_shape[-1] *= 2
    # ==================================================================================================================
    # Encoding
    # ==================================================================================================================
    coding_dict   = {0:[0, 1], 1:[1, 0]}
    coded_pattern = []
    for symbol in np.reshape(pattern, -1): coded_pattern += coding_dict[symbol]

    return np.reshape(np.array(coded_pattern), new_pattern_shape)
