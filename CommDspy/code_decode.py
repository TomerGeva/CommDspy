import numpy as np
from CommDspy.constants import CodingEnum, ConstellationEnum
from CommDspy.auxiliary import get_levels


def coding(pattern, constellation=ConstellationEnum.PAM4, coding=CodingEnum.UNCODED, pn_inv=False, full_scale=False):
    """
    :param pattern: Uncoded pattern, should be a numpy array of non-negative integers stating the index in the
     constellation point. Examples:
                            1. 1-bit patterns will be '0' and '1'
                            2. 2-bit patterns will be '0', '1', '2' and '3'
    :param constellation: Enumeration stating the constellation. Should be taken from:
                          CommDspy.constants.ConstellationEnum
    :param coding: Enumeration stating the wanted coding, only effective if constellation has more than 2 constellation
                   points. Should be taken from CommDspy.constants.CodingEnum
    :param pn_inv: Boolean stating if the pattern should be inverted after the coding
    :param full_scale: Boolean stating if we want the levels to be scaled such that the mean power of the levels will be
                       1 (0 dB)
    :return: Coded pattern, meaning the pattern at the constellation points
                1. After gray coding if needed
                2. Inverted if needed
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    levels          = get_levels(constellation, full_scale)
    bits_per_symbol = int(np.log2(len(levels)))
    # ==================================================================================================================
    # Gray coding
    # ==================================================================================================================
    if bits_per_symbol > 1 and coding == CodingEnum.GRAY:
        levels[-2:] = levels[-1:-3:-1]
    # ==================================================================================================================
    # PN inv
    # ==================================================================================================================
    if pn_inv:
        levels = -1 * levels

    return levels[pattern]

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
