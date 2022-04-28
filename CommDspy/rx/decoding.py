import numpy as np
from CommDspy.constants import ConstellationEnum
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
    return (lfilter([1, -1], [1], pattern.astype(float)) % lvl_num).astype(int)

def decoding_manchester(pattern):
    """
    :param pattern: pattern to perform manchester decoding on, should be a numpy array
    :return: pattern after manchester decoding, note that the length of the pattern will be half due to the nature of
    the encoding process. Example for manchester encoding:
    pattern = [1,    0,    1,    0,    0,    1,    1,    1,    0,    0,    1]
    encoded = [1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1 ,0]
    NOTE, this decoding scheme assumes binary data input
    ERROR DETECTION:
    Manchester encoding implements the data in the transitions, therefore we will look at the differences between
    pairs of bits. Where there are no differences there MUST be a slicing mistake. This can not be used for error
    correction since we do not know which of the two symbols was inverted, but we can mark this place as a guessed bit
    and guess the value due to it being a HARD decision process. when there are not transitions, 0.5 will be returned,
    indicating a guess value in this location
    Using the pre-slicing values one would be able to have a better prediction of the mistake than a simple guess.
    """
    # ==================================================================================================================
    # Checking data validity
    # ==================================================================================================================
    data_in = np.unique(pattern)
    if len(data_in) > 2 or (0 not in data_in and 1 not in data_in):
        raise ValueError('Data in is not binary, please consider other decoding methods')
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    pattern_shape = pattern.shape
    new_pattern_shape = list(pattern_shape)
    new_pattern_shape[-1]= int(new_pattern_shape[-1] / 2)
    # ==================================================================================================================
    # Decoding
    # ==================================================================================================================
    pattern_flat    = np.reshape(pattern, [-1, 2])
    transitions     = -1 * np.diff(pattern_flat, axis=1) # should be -1, 1 or 0 for mistakes
    decoded_pattern = (transitions + 1) / 2

    return np.reshape(np.array(decoded_pattern), new_pattern_shape)

def decoding_bipolar(pattern, error_deterction):
    """
    :param pattern: pattern to perform manchester decoding on, should be a numpy array
    :param error_deterction: flag indicating if we want the decoding to perform error detection.
        * If False, simply maps the zeros to 0 and the +-1 to 1
        * If True, checks to see if there are adjacent marks with similar signs, indicating there was a mistake
          somewhere in this region. the error detection will be as follows:
          ** If there are two consecutive '1' values or more, they will be replaced with '0.5' values
          ** If there are teo consecutive '-1' values or more, they will be replaced with '-0.5' values
        NOTE that the errors are not constrained to the location of the '0.5' and '-0.5' values, but can be in the neighboring '0' values as well
    :return:
    """
    if not error_deterction:
        return np.abs(pattern)
    else:
        # ----------------------------------------------------------------------------------------------------------
        # Detecting the mark locations in the pattern
        # ----------------------------------------------------------------------------------------------------------
        pattern_flat    = np.reshape(pattern, -1)
        decoded_pattern = np.abs(pattern).astype(float)
        idx_vec         = np.arange(0, len(decoded_pattern))
        mark_location   = idx_vec[pattern_flat != 0]
        mark_change     = np.concatenate((np.array([1]), np.diff(pattern_flat[mark_location])))
        running_sign    = 1
        for ii, mark in enumerate(mark_change):
            if mark == 0:
                decoded_pattern[mark_location[ii]] = decoded_pattern[mark_location[ii-1]] = 0.5 * np.sign(running_sign)
            else:
                running_sign = np.sign(mark)
        return decoded_pattern
