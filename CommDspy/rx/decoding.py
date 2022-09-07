import numpy as np
from CommDspy.misc.help_functions import check_binary
from CommDspy.constants import ConstellationEnum
from CommDspy.auxiliary import get_levels, get_gray_level_vec, get_bin_perm
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
    and guess the value due to it being a HARD decision process. when there aren't any transitions, 0.5 will be returned,
    indicating a guess value in this location
    Using the pre-slicing values one would be able to have a better prediction of the mistake than a simple guess.
    """
    # ==================================================================================================================
    # Checking data validity
    # ==================================================================================================================
    check_binary(pattern)
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

def decoding_bipolar(pattern, error_detection=False):
    """
    :param pattern: pattern to perform bipolar decoding on, should be a numpy array
    :param error_detection: flag indicating if we want the decoding to perform simple error detection.
        * If False, simply maps the zeros to 0 and the +-1 to 1
        * If True, checks to see if there are adjacent marks with similar signs, indicating there was a mistake
          somewhere in this region. the error detection will be as follows:
          ** If there are two consecutive '1' values or more, they will be replaced with '0.5' values
          ** If there are two consecutive '-1' values or more, they will be replaced with '-0.5' values
        NOTE that the errors are not constrained to the location of the '0.5' and '-0.5' values, but can be in the
        neighboring '0' values as well.
    :return:
    """
    if not error_detection:
        return np.abs(pattern - 1)
    else:
        # ----------------------------------------------------------------------------------------------------------
        # Detecting the mark locations in the pattern
        # ----------------------------------------------------------------------------------------------------------
        pattern_flat    = np.reshape(pattern - 1, -1)
        decoded_pattern = np.abs(pattern - 1).astype(float)
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

def decoding_mlt3(pattern):
    pattern_flat = np.reshape(pattern, -1)
    # ==================================================================================================================
    # Decoding
    # ==================================================================================================================
    if pattern[0] == 1:
        transitions = np.concatenate((np.array([1]), np.abs(np.diff(pattern_flat))))
    else:
        transitions = np.concatenate((np.array([0]), np.abs(np.diff(pattern_flat))))
    return np.reshape(transitions, pattern.shape)

def decoding_differential_manchester(pattern):
    # ==================================================================================================================
    # Checking data validity
    # ==================================================================================================================
    check_binary(pattern)
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    pattern_shape         = pattern.shape
    new_pattern_shape     = list(pattern_shape)
    new_pattern_shape[-1] = int(new_pattern_shape[-1] / 2)
    # ==================================================================================================================
    # Decoding
    # ==================================================================================================================
    pattern_flat = np.reshape(pattern, [-1, 2])
    transitions  = 1 -  np.abs(np.diff(pattern_flat, axis=1))  # should be -1, 1 or 0 for mistakes
    return np.reshape(transitions, new_pattern_shape).astype(int)

def decoding_linear(pattern, G, error_correction=False):
    """
    :param pattern: binary array to perform linear block decoding on. pattern length must be divisible by the block
                    length, otherwise, ignoring the last bits
    :param G: Generating matrix used to encode the pattern
    :param error_correction: If True, checks for block which are not in the codebook, and replaces them with the
                             codeword with the closest hamming distance. If there is more than 1 codeword with minimal
                             distance, chooses one of them as we can not know which 1 it was.
    :return: Function performs block decoding according to the following procedure:
                1. Computes the codebook according to the generating matrix G (assuming full codebook)
                2. Computes tha hamming distance for each block from all the codes in the codebook
                3. allocates the original data matching the codeword
             If we use error correction, also returns the error probability as computed from the hamming distance, and
             is equal to 1 over the number of codewords with minimal hamming distance
       Examples:
        1. decoded_pattern = decoding_linear(coded_pattern, G)
        2. decoded_pattern = decoding_linear(coded_pattern, G, False)
        3. decoded_pattern, p_err = decoding_linear(coded_pattern, G, True)
    """
    # ==================================================================================================================
    # Checking data validity
    # ==================================================================================================================
    check_binary(pattern)
    check_binary(G)
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    k, n = G.shape
    # ==================================================================================================================
    # Computing the codebook
    # ==================================================================================================================
    perm_bin = get_bin_perm(k)
    codebook = perm_bin.dot(G) % 2
    # ==================================================================================================================
    # Reshaping
    # ==================================================================================================================
    if len(pattern) % n != 0:
        residual = len(pattern) % n
        pattern_block = np.reshape(pattern[:-1*residual], [-1, n])
    else:
        pattern_block = np.reshape(pattern, [-1, n])
    # ==================================================================================================================
    # Decoding
    # ==================================================================================================================
    hamming         = np.sum(np.abs(codebook[:, None, :] - pattern_block[None, :, :]), axis=2)
    if error_correction:
        # ----------------------------------------------------------------------------------------------------------
        # Creating the error_prob vector
        # ----------------------------------------------------------------------------------------------------------
        p_err = np.zeros(pattern_block.shape[0])
        # ----------------------------------------------------------------------------------------------------------
        # Error correction
        # ----------------------------------------------------------------------------------------------------------
        min_hamming  = np.min(hamming, axis=0)
        not_codeword = min_hamming > 0
        correct_idx  = np.argmin(hamming[:, not_codeword], axis=0)
        pattern_block[not_codeword] = codebook[correct_idx]
        # ----------------------------------------------------------------------------------------------------------
        # Fixing the hamming matrix, filling p_err
        # ----------------------------------------------------------------------------------------------------------
        p_err[not_codeword]                = 1 / np.sum(hamming[:, not_codeword] == min_hamming[not_codeword], axis=0)
        hamming[correct_idx, not_codeword] = 0
    decoded_idx     = np.argmin(hamming, axis=0)
    decoded_blocks  = perm_bin[decoded_idx]
    if not error_correction:
        return np.reshape(decoded_blocks, -1)
    else:
        return np.reshape(decoded_blocks, -1), p_err
