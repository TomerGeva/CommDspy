import numpy as np
import os
from CommDspy import get_polynomial
from CommDspy import prbs_gen
from test.auxiliary import read_1line_csv


def prbs_gen_test(prbs_type, pattern_length, bits_per_symbol, bit_order_inv, pn_inv):
    """
    :param prbs_type: wanted PRBS to be generated, Should be taken from CommDspy.constants.PrbsEnum
    :param pattern_length: length of the test pattern to generate
    :param bits_per_symbol: Integer indicating how many bits should be generated per symbol. By default this input is
                            equal to 1, meaning that a binary PRBS will be generated. If this input is higher than 1,
                            adjacent bits will be coupled together to form a symbol. Example: bits_per_symbol = 2:
                                00 --> 0
                                01 --> 1
                                10 --> 2
                                11 --> 3
    :param bit_order_inv: Boolean indicating if the bit order in the signal generation is flipped.
    :param pn_inv: Boolean stating if the pattern should be inverted after the coding
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
    assert_str = '|{0:^6s}| bits_per_symbol = {1:1d}; bit_order_inv = {2:^6s}; pn_inv = {3:^6s} Falied!!!!' .format(
        prbs_type.name,
        bits_per_symbol,
        'True' if bit_order_inv else 'False',
        'True' if pn_inv else 'False'
    )
    ref_filename = os.path.join(os.getcwd(),'test_data',prbs_type.name + '_seed_ones.csv')
    # ==================================================================================================================
    # Getting DUT PRBS and bit manipulation
    # ==================================================================================================================
    prbs_seq, _ = prbs_gen(poly_coeff, init_seed, pattern_length,
                           bits_per_symbol=bits_per_symbol,
                           bit_order_inv=bit_order_inv,
                           pn_inv=pn_inv)
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
    rep_number = int(np.ceil(pattern_length*bits_per_symbol / prbs_len))
    ref_prbs_bin_mult = np.tile(ref_prbs_bin, rep_number) if rep_number > 1 else ref_prbs_bin
    if bits_per_symbol > 1:
        bit_weights       = np.arange(bits_per_symbol)
        ref_prbs_bin_mult = np.reshape(ref_prbs_bin_mult[:pattern_length*bits_per_symbol], [-1, bits_per_symbol])
        # ----------------------------------------------------------------------------------------------------------
        # Modifying the pattern according to the flags, and converting binary to UN-CODED symbols
        # ----------------------------------------------------------------------------------------------------------
        if bit_order_inv:
            bit_weights = np.fliplr(bit_weights)
        if pn_inv:
            ref_prbs_bin_mult = 1 - ref_prbs_bin_mult
        ref_pattern = ref_prbs_bin_mult.dot(2 ** bit_weights)
    else:
        ref_pattern = ref_prbs_bin
    # --------------------------------------------------------------------------------------------------------------
    # Extracting reference pattern
    # --------------------------------------------------------------------------------------------------------------
    ref_pattern = ref_pattern[:pattern_length]
    # ==================================================================================================================
    # Checking that all is fine
    # ==================================================================================================================
    assert np.all(ref_pattern == prbs_seq), assert_str
    return 1