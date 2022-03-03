import numpy as np
import CommDspy as cdsp
from CommDspy.constants import CodingEnum


def coding_pattern_test(constellation, coding, pn_inv):
    """
    :param constellation: Enumeration stating the constellation. Should be taken from:
                          CommDspy.constants.ConstellationEnum
    :param coding: Enumeration stating the coding. Should be taken from CommDspy.constants.CodingEnum
    :param pn_inv: Boolean stating if the pattern should be inverted after the coding
    :return: Testing the code_pattern function with these inputs
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    levels  = cdsp.get_constellation(constellation)
    assert levels.any() is not None, 'Constellation ' + constellation.name + ' type not supported'
    bits       = int(np.log2(len(levels)))
    pattern    = np.random.randint(0, 2**bits, 10)
    assert_str = '|{0:^6s}|{1:^5s}| bits = {2:1d} ; pn_inv = {3:^6s} Falied!!!!'.format(
        constellation.name,
        coding.name,
        bits,
        'True' if pn_inv else 'False'
    )
    # ==================================================================================================================
    # Getting DUT coded pattern
    # ==================================================================================================================
    coded_dut = cdsp.code_pattern(pattern, constellation, coding, pn_inv)
    # ==================================================================================================================
    # Computing the coding in a different way
    # ==================================================================================================================
    coded_ref = pattern
    if bits == 2:
        # ----------------------------------------------------------------------------------------------------------
        # Gray coding
        # ----------------------------------------------------------------------------------------------------------
        if coding == CodingEnum.GRAY:
            coded_temp = pattern.copy()
            coded_temp[pattern == 3] = 2
            coded_temp[pattern == 2] = 3
            coded_ref = coded_temp
    # --------------------------------------------------------------------------------------------------------------
    # pn-inv
    # --------------------------------------------------------------------------------------------------------------
    if pn_inv:
        levels *= -1
    coded_ref = levels[coded_ref]
    assert np.all(coded_ref == coded_dut), assert_str

