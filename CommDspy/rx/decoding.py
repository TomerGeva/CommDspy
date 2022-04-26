import numpy as np
from CommDspy.constants import CodingEnum, ConstellationEnum
from CommDspy.auxiliary import get_levels, get_gray_level_vec
from scipy.signal import lfilter


def decoding_gray(pattern, constellation=ConstellationEnum.PAM4):
    """
    :param pattern: Numpy array of gray coded symbols.
    :param constellation: Enumeration stating the constellation. Should be taken from:
                          CommDspy.constants.ConstellationEnum
    :return: Function performs decoding from gray to uncoded
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    level_num = len(get_levels(constellation))
    bits_per_symbol = int(np.ceil(np.log2(level_num)))
    # ==================================================================================================================
    # Gray coding
    # ==================================================================================================================
    if bits_per_symbol > 1:
        levels = get_gray_level_vec(level_num)
    else:
        levels = np.array([0, 1])
    # ==================================================================================================================
    # Converting symbols to indices, i.e. performing decoding
    # ==================================================================================================================
    symbol_idx_vec = levels[pattern]
    return symbol_idx_vec

def decoding_differential(pattern, constellation=ConstellationEnum.PAM4):
    """
        :param pattern: symbols we would like to perform differential encoding
        :param constellation: the constellation we are working with
        :return: Function performs differential encoding to the input signal. The flow is as follows:

                            |-------------|        m
             y_n----------->|      D      |------> +---------> x_n
                   |        |-------------|   -    ^
                   |                              +|
                   --------------------------------
        """
    lvl_num = len(get_levels(constellation))
    return (lfilter([1, -1], 1, pattern) % lvl_num).astype(int)