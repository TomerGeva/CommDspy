import numpy as np
from scipy import linalg
from CommDspy.constants import PrbsEnum, ConstellationEnum


def get_polynomial(prbs_type):
    """
    :param prbs_type: Enumeration of the wanted polynomial
    :return: The PRBS polynomial
    """
    poly_coeff = np.array([0] * prbs_type.value)
    if prbs_type == PrbsEnum.PRBS7:
        poly_coeff[[5, 6]] = 1
    elif prbs_type == PrbsEnum.PRBS9:
        poly_coeff[[4, 8]] = 1
    elif prbs_type == PrbsEnum.PRBS11:
        poly_coeff[[8, 10]] = 1
    elif prbs_type == PrbsEnum.PRBS13:
        poly_coeff[[0, 1, 11, 12]] = 1
    elif prbs_type == PrbsEnum.PRBS15:
        poly_coeff[[13, 14]] = 1
    elif prbs_type == PrbsEnum.PRBS31:
        poly_coeff[[27, 30]] = 1
    else:
        print("PRBS type not supported :)")
        return np.array([None])
    return poly_coeff

def get_levels(constellation, full_scale=False):
    """
    :param constellation: Enumeration of the wanted constellation
    :param full_scale: Boolean stating if we want the levels to be scaled such that the mean power of the levels will be
                       1 (0 dB)
    :return: The constellation as written in the documentation
    """
    if constellation == ConstellationEnum.NRZ:
        levels = np.array([-1, 1])
    elif constellation == ConstellationEnum.OOK:
        levels = np.array([0, 1])
    elif constellation == ConstellationEnum.PAM3:
        levels = np.array([-1, 0, 1])
    elif constellation == ConstellationEnum.PAM4:
        levels = np.array([-3, -1, 1, 3])
    else:
        raise ValueError('Constellation type not supported')
    if full_scale:
        return levels / np.sqrt(np.mean(levels ** 2))
    else:
        return levels

def get_gray_level_vec(level_num):
    """
    :param level_num: number of levels
    :return: the gray coded level numbering
    """
    bits_per_symbol = int(np.ceil(np.log2(level_num)))
    lvls = np.array([0, 1])[:, None]
    for ii in range(1, bits_per_symbol):
        lvls = np.vstack((lvls, lvls[::-1]))
        lvls = np.hstack((np.repeat(np.array([0, 1])[:, None], 2 ** ii, axis=0), lvls))
    return lvls.dot([1 << ii for ii in np.arange(lvls.shape[1] - 1, -1, -1)])[:level_num]

def power(signal):
    """
    :param signal:
    :return: Computes the mean power of the signal
    """
    return np.mean(signal * np.conj(signal))

def rms(signal):
    """
    :param signal:
    :return: Computes the RMS of the signal
    """
    return np.sqrt(power(signal))

def buffer(signal, length, overlap=0, delay=0, clip=False):
    """
    :param signal: Input signal
    :param length: Length of the output rows
    :param overlap: Integer value which controls the amount of overlap or underlap in the buffered data frames.
                       - If >0, there will be P samples of data from the end of one frame
                         (column) that will be repeated at the start of the next data frame.
                       - If <0, the buffering operation will skip P samples of data after each
                         frame, effectively skipping over data in X, and thus reducing the
                         buffer "frame rate".
                       - If empty or omitted, P is assumed to be zero (no overlap or underlap).
    :param delay: Initial condition options. default (0) begins filling the buffer immediately. delay = d > 0 sets the
                  first `d` values to zero
    :param clip: Indicating if we want to clip the not-complete rows from the matrix. If False, returns all the signal
                 where the final row is padded with zeros to match the row length. If true, clips the last row.
    :return: Buffer a signal vector into a matrix of data frames. This function mimics the MATLAB buffer function
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    full_len = delay + len(signal) + length
    stride   = length-overlap
    # ==================================================================================================================
    # Taking delay into consideration
    # ==================================================================================================================
    signal_full = np.concatenate((np.zeros(delay), signal, np.zeros(length))) if delay > 0 else np.concatenate((signal, np.zeros(length)))
    # ==================================================================================================================
    # Creating the full overlapping hankel matrix
    # ==================================================================================================================
    hankel_mat = linalg.hankel(signal_full, signal_full[::-1][:length])
    # ==================================================================================================================
    # Extracting the wanted columns - overlaps of 0 means jumps of "length"
    # ==================================================================================================================
    rows = np.arange(0, full_len, stride)
    matrix = hankel_mat[rows, :].astype(type(signal[0]))
    # ==================================================================================================================
    # Clipping if needed
    # ==================================================================================================================
    rows_to_remove = np.sum(np.any(np.array([np.arange(full_len - length - max(overlap, 0), full_len)]).T - rows == 0, axis=1))
    if rows_to_remove > 0:
        matrix = matrix[:-1*rows_to_remove]
    if clip and matrix[-1,:] not in linalg.hankel(signal, signal[::-1][:length])[:-1]:
        matrix = matrix[:-1]
    return matrix

def upsample(signal, rate):
    """
    :param signal:input signal, should be a 1D numpy array
    :param rate: over sampling ratio ; should be an integer value!
    :return: fills `0` values between each signal. the number of `0` values between each two adjacent values is (OSR - 1)
    """
    zero_vec = np.zeros_like(signal)
    zero_mat = np.tile(zero_vec, [rate-1, 1])
    return np.reshape(np.vstack((signal, zero_mat)).T, -1)

def zoh(signal, hold_idx):
    """
    :param signal: input to preform zero order hold on
    :param hold_idx: number of indices to hold
    :return: signal after zero order hold
    """
    return np.reshape(np.tile(signal[:, None], [1, hold_idx]), -1)

def get_bin_perm(k):
    """
    :param k: number of bits
    :return: returns a numpy 2D array with all the length 'k' binary vector permutations
    """
    perm = np.arange(2 ** k)
    return ((perm[:, None] & (1 << np.arange(k - 1, -1, -1))) > 0).astype(int)

def hamming(pattern_block, codebook):
    """
    :param pattern_block: 2d numpy array with dimension N X l holding the pattern in blocks with length 'l'
    :param codebook: 2d numpy array holding the codebook with M codewords, with size M x l
    :return: For a codeword of length 'l', computing the hamming distance from each word in the codebook, returning a 2d
             num array where the (ii,jj) location hold for the hamming distance of pattern ii from codeword jj. Also
             returns the minimal hamming distance index for each block in the pattern
    """
    hamming     = np.sum(np.abs(codebook[:, None, :] - pattern_block[None, :, :]), axis=2)
    decoded_idx = np.argmin(hamming, axis=0)
    return hamming, decoded_idx

def bin2uint(bin_tensor):
    weights = [1<< ii for ii in range(bin_tensor.shape[-1]-1, -1, -1)]
    return bin_tensor.dot(weights)

def uint2bin(int_tensor, n_bits):
    dims     = int_tensor.shape
    input_1d = np.reshape(int_tensor, -1)
    bin_2d   = np.squeeze(((input_1d[:, None] & (1 << np.arange(n_bits - 1, -1, -1))) > 0).astype(int))
    dims_new = np.concatenate([dims, [n_bits]]).astype(int)
    return np.reshape(bin_2d, dims_new)