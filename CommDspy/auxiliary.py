import numpy as np
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
    elif constellation == ConstellationEnum.PAM4:
        levels = np.array([-3, -1, 1, 3])
    else:
        print("Constellation type not supported :)")
        return None
    if full_scale:
        return levels / np.sqrt(np.mean(levels ** 2))
    else:
        return levels

def power(signal):
    """
    :param signal:
    :return: Computes the mean power of the signal
    """
    return np.mean(signal ** 2)

def rms(signal):
    """
    :param signal:
    :return: Computes the RMS of the signal
    """
    return np.sqrt(np.mean(signal ** 2))