import numpy as np
from CommDspy.constants import CodingEnum, ConstellationEnum
from CommDspy.auxiliary import get_levels, get_gray_level_vec

def decoding(pattern, constellation=ConstellationEnum.PAM4, coding=CodingEnum.UNCODED, pn_inv=False, full_scale=False):
    """
    :param pattern: Numpy array of coded symbols.
                * If PAM4 assuming that the constellation is [-3x,-x,x,3x]
                * If NRZ assuming that the constellation is [-x,x]
                * If OOK assuming that the constellation is [0, x]
    :param constellation: Enumeration stating the constellation. Should be taken from:
                          CommDspy.constants.ConstellationEnum
    :param coding: Enumeration stating the wanted coding, only effective if constellation has more than 2 constellation
                   points. Should be taken from CommDspy.constants.CodingEnum
    :param pn_inv: If True the P-N were inverted in the creation of the symbol stream
    :param full_scale: Boolean stating if the levels were scaled such that the mean power of the levels will be 1 (0 dB)
    :return: Function performs decoding and then converts the symbols to binary. Note that the function supports OOK,
             NRZ and PAM4.
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    coded_levels    = get_levels(constellation, full_scale)
    bits_per_symbol = int(np.log2(len(coded_levels)))
    # ==================================================================================================================
    # Setting base levels
    # ==================================================================================================================
    if bits_per_symbol == 2:
        if coding == CodingEnum.GRAY:
            levels = np.array([0, 1, 3, 2])
        else:
            levels = np.array([0, 1, 2, 3])
    else:
        levels = np.array([0, 1])
    # ==================================================================================================================
    # PN-inv
    # ==================================================================================================================
    if pn_inv:
        pattern *= -1
    # ==================================================================================================================
    # Converting symbols to indices, i.e. performing decoding
    # ==================================================================================================================
    idx_mat = np.round((pattern - np.min(coded_levels)) / np.diff(coded_levels)[0]).astype(int)
    idx_vec = np.reshape(idx_mat, (-1, 1))
    symbol_idx_vec = levels[idx_vec]
    return np.reshape(symbol_idx_vec, pattern.shape)

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