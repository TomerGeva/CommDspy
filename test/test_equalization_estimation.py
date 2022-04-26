import numpy as np
import os
import CommDspy as cdsp
import random
from test.auxiliary import read_1line_csv


def equalization_prbs_test(prbs_type):
    """
    :param prbs_type:
    :return: Testing the equalization function. This is dine via inserting the PRBS through an IIR, which the inverse of
             is a FIR and can be thus be checked according to the reference IIR. Nevertheless, the IIR should have all
             poles inside the unit circle, or else the resulting signal will be unstable
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    bits_per_symbol = random.randint(1, 2)
    num_poles       = random.randint(1, 8)
    constellation = cdsp.constants.ConstellationEnum.PAM4 if bits_per_symbol > 1 else (
        cdsp.constants.ConstellationEnum.NRZ if random.random() > 0.5 else cdsp.constants.ConstellationEnum.OOK)
    coding = cdsp.constants.CodingEnum.UNCODED  # if random.random() > 0.5 else cdsp.constants.CodingEnum.GRAY
    poly_coeff = cdsp.get_polynomial(prbs_type)
    init_seed = np.array([1] * prbs_type.value)
    prbs_len = 2 ** len(init_seed) - 1
    assert poly_coeff.any() is not None, prbs_type.name + ' type not supported'
    ref_filename = os.path.join(os.getcwd(), 'test_data', prbs_type.name + '_seed_ones.csv')
    assert_str = '|{0:^6s}| Constellation = {1:^6s} | Coding = {2:^6s} | {3:^3d} poles  ; {4:^3d} DFE taps '.format(
        prbs_type.name,
        constellation.name,
        coding.name,
        num_poles,
        0
    )
    # ==================================================================================================================
    # Creating reference pattern
    # ==================================================================================================================
    # --------------------------------------------------------------------------------------------------------------
    # Loading pattern
    # --------------------------------------------------------------------------------------------------------------
    ref_prbs_bin = read_1line_csv(ref_filename)
    # --------------------------------------------------------------------------------------------------------------
    # Duplicating if needed and coding
    # --------------------------------------------------------------------------------------------------------------
    if bits_per_symbol > 1:
        ref_prbs_bin_mult = np.tile(ref_prbs_bin, bits_per_symbol)
        ref_pattern = cdsp.tx.bin2symbol(ref_prbs_bin_mult, 2 ** bits_per_symbol, False, False, False, False)
    else:
        ref_pattern = ref_prbs_bin
    ref_pattern = cdsp.tx.coding(ref_pattern, constellation, coding)
    # --------------------------------------------------------------------------------------------------------------
    # Creating repetitions of the pattern
    # --------------------------------------------------------------------------------------------------------------
    reps        = random.randint(2, 5)
    cutoff      = random.randint(1, int(prbs_len / 2))
    ref_pattern = np.tile(ref_pattern, reps)[:-1 * cutoff]
    # ==================================================================================================================
    # Creating reference channel
    # ==================================================================================================================
    while True:
        roots = np.around(np.random.random(num_poles), decimals=3) - 0.5
        channel_ref = np.poly(roots)
        if np.max(channel_ref) == channel_ref[0]:
            break
    # ==================================================================================================================
    # Passing data through the channel
    # ==================================================================================================================
    channel_out = cdsp.channel.awgn_channel(ref_pattern, [1], channel_ref)[len(channel_ref)+1:]
    # ==================================================================================================================
    # Running DUT
    # ==================================================================================================================
    precursors        = np.argmax(abs(channel_ref))
    pn_inv_postcoding = channel_ref[precursors] != np.max(channel_ref)
    postcursors       = len(channel_ref) - precursors - 1
    equ_dut = cdsp.equalization_estimation_prbs(prbs_type, channel_out, constellation,
                                                prbs_full_scale=False,
                                                ffe_postcursor=postcursors, ffe_precursor=precursors, dfe_taps=0,
                                                normalize=False,
                                                bit_order_inv=False,
                                                pn_inv_precoding=False,
                                                gray_coded=False,
                                                pn_inv_postmapping=pn_inv_postcoding)
    # ==================================================================================================================
    # Comparing results
    # ==================================================================================================================
    assert np.all(np.abs(channel_ref - equ_dut[0]) < 5e-3), assert_str

