import numpy as np
from CommDspy.constants import PrbsEnum, CodingEnum, ConstellationEnum


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

def get_constellation(constellation):
    """
    :param constellation: Enumeration of the wanted constellation
    :return: The constellation as written in the documentation
    """
    if constellation == ConstellationEnum.NRZ:
        return np.array([-1, 1])
    elif constellation == ConstellationEnum.OOK:
        return np.array([0, 1])
    elif constellation == ConstellationEnum.PAM4:
        return np.array([-3, -1, 1, 3])
    else:
        print("Constellation type not supported :)")
        return None

def code_pattern(pattern, constellation=ConstellationEnum.PAM4, coding=CodingEnum.UNCODED, pn_inv=False):
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
    :return: Coded pattern, meaning the pattern at the constellation points
                1. After gray coding if needed
                2. Inverted if needed
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    levels          = get_constellation(constellation)
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