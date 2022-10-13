import numpy as np
import os
from CommDspy import get_polynomial
from CommDspy.tx import prbs_gen
from test.auxiliary import read_1line_csv


def prbs_gen_test(prbs_type, pattern_length):
    """
    :param prbs_type: wanted PRBS to be generated, Should be taken from CommDspy.constants.PrbsEnum
    :param pattern_length: length of the test pattern to generate
    :return: Testing the PRBS generator function according to the saved data. Not testing PRBS31 in this test, takes too
             long to generate the complete pattern
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    poly_coeff      = get_polynomial(prbs_type)
    init_seed       = np.array([1] * prbs_type.value)
    prbs_len        = 2 ** len(init_seed) - 1
    assert poly_coeff.any() is not None, prbs_type.name + ' type not supported'
    assert_str = '|{0:^6s}| wanted length = {1:d}| ' .format(
        prbs_type.name,
        pattern_length,
    )
    ref_filename = os.path.join(os.getcwd(),'test_data',prbs_type.name + '_seed_ones.csv')
    # ==================================================================================================================
    # Getting DUT PRBS and bit manipulation
    # ==================================================================================================================
    prbs_seq, seed_dut = prbs_gen(poly_coeff, init_seed, pattern_length)
    # ==================================================================================================================
    # Creating reference pattern
    # ==================================================================================================================
    # --------------------------------------------------------------------------------------------------------------
    # Loading pattern
    # --------------------------------------------------------------------------------------------------------------
    ref_prbs_bin = read_1line_csv(ref_filename)
    # --------------------------------------------------------------------------------------------------------------
    # Duplicating if needed
    # --------------------------------------------------------------------------------------------------------------
    rep_number = int(np.ceil(pattern_length / prbs_len))
    ref_prbs_bin_mult = np.tile(ref_prbs_bin, rep_number) if rep_number > 1 else ref_prbs_bin
    # --------------------------------------------------------------------------------------------------------------
    # Extracting reference pattern
    # --------------------------------------------------------------------------------------------------------------
    ref_pattern = ref_prbs_bin_mult[:pattern_length]
    if pattern_length >= prbs_type.value:
        ref_seed = np.flip(ref_pattern[-1*prbs_type.value:])
    else:
        seed_temp = np.concatenate((np.flip(ref_pattern), np.array([1] * prbs_type.value)))
        ref_seed = seed_temp[:prbs_type.value]

    # ==================================================================================================================
    # Checking that all is fine
    # ==================================================================================================================
    assert np.all(ref_pattern == prbs_seq), assert_str + "patterns are not the same!"
    assert np.all(ref_seed == seed_dut), assert_str + "seeds are not equal!"
    return 1