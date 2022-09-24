import numpy as np
import os
import CommDspy as cdsp
import random
from test.auxiliary import read_1line_csv


def channel_estimation_prbs_test(prbs_type):
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    bits_per_symbol = random.randint(1, 2)
    num_precursors  = random.randint(0, 8)
    num_postcursors = random.randint(0, 32)
    constellation   = cdsp.constants.ConstellationEnum.PAM4 if bits_per_symbol > 1 else (cdsp.constants.ConstellationEnum.NRZ if random.random() > 0.5 else cdsp.constants.ConstellationEnum.OOK)
    gray_coding     = False # random.random() > 0.5
    poly_coeff      = cdsp.get_polynomial(prbs_type)
    init_seed       = np.array([1] * prbs_type.value)
    prbs_len        = 2 ** len(init_seed) - 1
    assert poly_coeff.any() is not None, prbs_type.name + ' type not supported'
    ref_filename = os.path.join(os.getcwd(),'test_data',prbs_type.name + '_seed_ones.csv')
    assert_str   = '|{0:^6s}| Constellation = {1:^6s} | Coding = {2:^6s} | {3:^3d} precursors ; {4:^3d} postcursors ;'.format(
        prbs_type.name,
        constellation.name,
        str(gray_coding),
        num_precursors,
        num_postcursors
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
        ref_pattern       = cdsp.tx.bin2symbol(ref_prbs_bin_mult, 2**bits_per_symbol, False, False, False, False)
    else:
        ref_pattern = ref_prbs_bin
    ref_pattern = cdsp.tx.mapping(ref_pattern, constellation) if not gray_coding else cdsp.tx.mapping(cdsp.tx.coding_gray(ref_pattern, constellation), constellation)
    # --------------------------------------------------------------------------------------------------------------
    # Creating repetitions of the pattern
    # --------------------------------------------------------------------------------------------------------------
    reps   = random.randint(2, 5)
    cutoff = random.randint(1, int(prbs_len/2))
    ref_pattern = np.tile(ref_pattern, reps)[:-1*cutoff]
    # ==================================================================================================================
    # Creating reference channel
    # ==================================================================================================================
    precursors  = np.around(np.random.random(num_precursors), decimals=6) - 0.5
    postcursors = np.around(np.random.random(num_postcursors), decimals=6) - 0.5
    channel_ref = np.concatenate((precursors, [1], postcursors))
    # ==================================================================================================================
    # Passing signal through the channel
    # ==================================================================================================================
    channel_out, _ = cdsp.channel.awgn_channel(ref_pattern, channel_ref, [1])
    channel_out    = channel_out[len(channel_ref)+1:]
    # ==================================================================================================================
    # Passing through DUT
    # ==================================================================================================================
    channel_dut = cdsp.channel_estimation_prbs(prbs_type, channel_out, constellation,
                                               channel_postcursor=num_postcursors,
                                               channel_precursor=num_precursors,
                                               normalize=True)

    assert np.all(np.abs(channel_dut[0] - channel_ref) < 1e-10), assert_str + ' channel estimation Failed!'
    assert np.allclose(channel_dut[0], channel_ref), assert_str + ' channel estimation Failed!'

