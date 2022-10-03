import numpy as np
from CommDspy.misc.help_functions import check_binary, check_valid_conv
from CommDspy.misc.ml_decoding import ml_decoding, Trellis
from CommDspy.constants import ConstellationEnum
from CommDspy.auxiliary import get_levels, get_gray_level_vec, get_bin_perm, hamming, bin2uint, uint2bin
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

def decoding_linear(pattern, G, error_prob=False):
    """
    :param pattern: binary array to perform linear block decoding on. pattern length must be divisible by the block
                    length, otherwise, ignoring the last bits
    :param G: Generating matrix used to encode the pattern
    :param error_prob: If True, checks for block which are not in the codebook, and replaces them with the codeword with
                       the closest hamming distance.
    :return: Function performs MAP block decoding according to the following procedure:
                1. Computes the codebook according to the generating matrix G (assuming full codebook)
                2. Computes tha hamming distance for each block from all the codes in the codebook
                3. Allocates the original data matching the codeword, i.e. performing error correction if possible. If
                   there is more than 1 codeword with minimal distance, chooses one of them as we can not know which 1
                   it was.
             If we set error_prob to True, also returns the error probability as computed from the hamming distance, and
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
    return ml_decoding(perm_bin, codebook, pattern_block, error_prob)

def decoding_conv_ml(pattern, G, tb_len, feedback=None, use_feedback=None, error_prob=False):
    """
    :param pattern: Convolution coded pattern we need to perform decoding on
    :param G: Generating matrix from the convolution code. Read the CommDspy.tx.coding_conv for more description
    :param tb_len: Traceback length, how far we should go for each block to decode. This is usually set as 5 times the
                   constraint length, i.e. n_out * 5 * (m+1) where m is the longest memory. Note, this value should be
                   divisible by the number of outputs per input "chunck". Example: if the code has 2 inputs and 5
                    outputs, tb_len should be divisible by 5
    :param feedback: Feedback polynomial for convolution encoder. Read the CommDspy.tx.coding_conv for more description
    :param use_feedback: 2d numpy array stating if the usage of the feedback. Read the CommDspy.tx.coding_conv for more
                         description.
    :param error_prob: If True, checks for block which are not in the codebook, and replaces them with the codeword with
                       the closest hamming distance.
    :return: Function performs MAP block decoding according to the following procedure:
                1. Computes the codebook according to the generating matrix G (assuming full codebook)
                2. Computes tha hamming distance for each block from all the codes in the codebook
                3. Allocates the original data matching the codeword, i.e. performing error correction if possible. If
                   there is more than 1 codeword with minimal distance, chooses one of them as we can not know which 1
                   it was.
             If we set error_prob to True, also returns the error probability as computed from the hamming distance, and
             is equal to 1 over the number of codewords with minimal hamming distance
    """
    # ==================================================================================================================
    # Basic checking of data validity
    # ==================================================================================================================
    check_valid_conv(pattern, G, feedback, use_feedback)
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    n_in  = len(G)
    n_out = G[0].shape[0]
    # ==================================================================================================================
    # Creating the trellis dictionary and total input possibilities
    # ==================================================================================================================
    n_chunks    = tb_len // n_out
    n_bits_in   = n_in * n_chunks
    trellis_obj = Trellis(G, feedback, use_feedback)
    # ==================================================================================================================
    # Creating the codebook
    # ==================================================================================================================
    # Each start state produces 2 ** n_bits_in codewords, thus we will have a total of num_states * 2 ** n_bits_in
    # codewords total. The input index will be output index modulo (2 ** n_bits_in codewords)
    inputs   = get_bin_perm(n_bits_in)
    codebook = np.zeros([len(trellis_obj.states) * 2**n_bits_in, tb_len], dtype=int)
    for ii, start_state in enumerate(trellis_obj.states):
        # ----------------------------------------------------------------------------------------------------------
        # Creating the codeword for each input and each possible starting state
        # ----------------------------------------------------------------------------------------------------------
        for jj, input_vec in enumerate(inputs):
            input_chunks = np.reshape(input_vec, [-1, n_in])
            prev_state   = start_state
            # **************************************************************************************************
            # Getting the output + next_state for each chunk, creating the codeword
            # **************************************************************************************************
            for kk, chunk in enumerate(input_chunks):
                key = (tuple(chunk), tuple(prev_state))
                output, prev_state = trellis_obj.trellis[key]
                codebook[ii*2**n_bits_in+jj, n_out*kk:n_out*(kk+1)] = output
    # ==================================================================================================================
    # Reshaping
    # ==================================================================================================================
    pattern_block = _trunc_reshape(pattern, tb_len)
    # ==================================================================================================================
    # Decoding
    # ==================================================================================================================
    return ml_decoding(np.tile(inputs, [len(trellis_obj.states), 1]), codebook, pattern_block, error_prob)

def decoding_conv_viterbi(pattern, G, tb_len, feedback=None, use_feedback=None, error_prob=False):
    """
    :param pattern: Convolution coded pattern we need to perform decoding on
    :param G: Generating matrix from the convolution code. Read the CommDspy.tx.coding_conv for more description
    :param tb_len: Traceback length, how far we should go for each block to decode. This is usually set as 5 times the
                   constraint length, i.e. n_out * 5 * (m+1) where m is the longest memory. Note, this value should be
                   divisible by the number of outputs per input "chunck". Example: if the code has 2 inputs and 5
                    outputs, tb_len should be divisible by 5
    :param feedback: Feedback polynomial for convolution encoder. Read the CommDspy.tx.coding_conv for more description
    :param use_feedback: 2d numpy array stating if the usage of the feedback. Read the CommDspy.tx.coding_conv for more
                         description.
    :param error_prob: If True, checks for block which are not in the codebook, and replaces them with the codeword with
                       the closest hamming distance.
    :return: Function performs hard viterbi decoding over binary state memory and data. If there are 4 states, they will
    (probably) be {(0,0), (0,1), (1,0), (1,1)} and the input/output are binary vectors (or scalars).
    """
    # ==================================================================================================================
    # Basic checking of data validity
    # ==================================================================================================================
    check_valid_conv(pattern, G, feedback, use_feedback)
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    n_in      = len(G)
    n_out     = G[0].shape[0]
    n_chunks  = tb_len // n_out
    input_mat = np.zeros([len(pattern) // tb_len, n_in*n_chunks], dtype=int)
    trellis_obj   = Trellis(G, feedback, use_feedback)  # Trellis object, holding all the important things
    hamming_dist  = np.zeros([len(trellis_obj.states), n_chunks], dtype=int)
    hamming_state = np.zeros([len(trellis_obj.states), n_chunks], dtype=int)
    decoded_mat   = np.zeros([len(trellis_obj.states), n_chunks * n_in], dtype=int)  # holds the decoded input 'chunk' for each state
    # ==================================================================================================================
    # Reshaping
    # ==================================================================================================================
    pattern_block = _trunc_reshape(pattern, tb_len)
    # ==================================================================================================================
    # Starting the viterbi algorithm block by block
    # ==================================================================================================================
    output_tensor_reshape = np.reshape(trellis_obj.output_tensor, [-1, n_out])
    in_state_int_vec      = np.arange(trellis_obj.num_states)
    for ii in range(pattern_block.shape[0]):
        # ----------------------------------------------------------------------------------------------------------
        # Forward propagating - Computing the accumulative hamming along each path, selecting the best
        # ----------------------------------------------------------------------------------------------------------
        for chunk_idx in range(n_chunks):
            pattern_chunk = pattern_block[ii, n_out*chunk_idx:n_out*(chunk_idx+1)]
            # ----------------------------------------------------------------------------------------------------------
            # For each chunk compute all possible hamming distances
            # ----------------------------------------------------------------------------------------------------------
            # step_hamming[ii,jj] = hamming from state ii to state jj, computing for all transitions in one step
            step_hamming, _ = hamming(output_tensor_reshape, pattern_chunk[None, :])
            step_hamming    = np.reshape(step_hamming, trellis_obj.output_tensor.shape[:-1])
            # ----------------------------------------------------------------------------------------------------------
            # Noting the best for this step, the return path and the input for the best per state
            # ----------------------------------------------------------------------------------------------------------
            hamming_state[:, chunk_idx] = np.argmin(step_hamming, axis=0)
            if chunk_idx > 0:
                hamming_dist[:, chunk_idx] = np.min(step_hamming, axis=0) + hamming_dist[hamming_state[:,chunk_idx], chunk_idx-1]
            else:
                hamming_dist[:, chunk_idx] = np.min(step_hamming, axis=0)
            decoded_mat_prev = decoded_mat.copy()
            out_state = hamming_state[in_state_int_vec, chunk_idx]
            if chunk_idx > 0:
                decoded_mat[in_state_int_vec, :n_in*(chunk_idx+1)] = np.concatenate([decoded_mat_prev[out_state, :n_in * chunk_idx], trellis_obj.input_tensor[out_state, in_state_int_vec]], axis=1)
            else:
                decoded_mat[in_state_int_vec, :n_in*(chunk_idx+1)] = trellis_obj.input_tensor[out_state, in_state_int_vec]
        # ----------------------------------------------------------------------------------------------------------
        # Backward propagating - Going along the selected path, recovering the input
        # ----------------------------------------------------------------------------------------------------------
        last_state = np.argmin(hamming_dist[:, -1])
        input_mat[ii] = decoded_mat[last_state]

    return np.reshape(input_mat, -1), last_state


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Auxiliary
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def _trunc_reshape(input_vec, block_len):
    if len(input_vec) % block_len != 0:
        residual  = len(input_vec) % block_len
        input_mat = np.reshape(input_vec[:-1 * residual], [-1, block_len])
    else:
        input_mat = np.reshape(input_vec, [-1, block_len])
    return input_mat