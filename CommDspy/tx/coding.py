import numpy as np
from CommDspy.constants import CodingEnum, ConstellationEnum
from CommDspy.auxiliary import get_levels, get_gray_level_vec


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
