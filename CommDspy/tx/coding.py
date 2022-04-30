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
    shape   = pattern.shape
    lvl_num = len(get_levels(constellation))
    return np.reshape(lfilter([1], [1, -1], np.reshape(pattern, -1).astype(float)) % lvl_num, shape).astype(int)

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
    coding_list  = np.array([[0, 1], [1, 0]])
    coded_pattern = np.reshape(coding_list[pattern], -1)

    return np.reshape(np.array(coded_pattern), new_pattern_shape)

def coding_bipolar(pattern):
    """
   :param pattern: pattern to perform bipolar encoding on, should be a numpy array
   :return: pattern after bipolar encoding, alternating +- 1 for the "marks" (1) where "spaces" (0) remain 0. Since this
   code has three levels, the encoded vector will result in numbers between 0 and 2, where:
    * 0 bits will be encoded to 1 values (mapped to 0 by the cdsp.tx.mapping function)
    * 1 bits will be encoded to 0,2 values (mapped to +- 'x' by the cdsp.tx.mapping function)
   Example:
   pattern = [1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1]
   encoded = [0, 1, 2, 1, 1, 0, 2, 0, 1, 1, 2]
   NOTE, this encoding scheme assumes binary data input
   """
    sign_vec = coding_differential(pattern, ConstellationEnum.OOK)
    return 1 + pattern * (-1) ** sign_vec

def coding_mlt3(pattern):
    """
    :param pattern: pattern to perform MLT-3 encoding on, should be a numpy array
    :return: pattern after MLT-3 encoding, alternating -1,0,1,0 for the '1' where '0' do not change the level. Since this
    code has three levels, the encoded vector will result in numbers between 0 and 2, where:
    * 1 bits will change the levels from 0,1,2,1 cycle
    * 0 bits will remain in the same level
    Aassuming initial level of '1' in case the data starts with zeros
    Example:
    pattern = [1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1]
    encoded = [1, 1, 2, 2, 2, 1, 0, 1, 1, 1, 2]
    NOTE, this encoding scheme assumes binary data input
    """
    # ==================================================================================================================
    # Converting to 4 levels using differential encoding
    # ==================================================================================================================
    mlt3_enc = coding_differential(pattern, constellation=ConstellationEnum.PAM4)
    # ==================================================================================================================
    # Changing level '3' to '1' forming the cycle of 0,1,2,1
    # ==================================================================================================================
    mlt3_enc[mlt3_enc == 3] = 1
    return mlt3_enc
