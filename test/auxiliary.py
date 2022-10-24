import csv
import os
import numpy as np
import CommDspy as cdsp
import random

def read_1line_csv(filename, delimiter=','):
    """
    :param filename: Input filename
    :param delimiter: The delimiter in the file
    :return: the first line of the file
    """
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        ref_prbs_bin = np.array(next(reader)).astype(int)
    return ref_prbs_bin

def generate_pattern(prbs_type):
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    bits_per_symbol = random.randint(1, 2)
    constellation = cdsp.constants.ConstellationEnum.PAM4 if bits_per_symbol > 1 else (cdsp.constants.ConstellationEnum.NRZ if random.random() > 0.5 else cdsp.constants.ConstellationEnum.OOK)
    gray_coding = False  # random.random() > 0.5
    poly_coeff = cdsp.get_polynomial(prbs_type)
    init_seed = np.array([1] * prbs_type.value)
    prbs_len = 2 ** len(init_seed) - 1
    assert poly_coeff.any() is not None, prbs_type.name + ' type not supported'
    ref_filename = os.path.join(os.getcwd(), 'test_data', prbs_type.name + '_seed_ones.csv')
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
    ref_pattern = cdsp.tx.mapping(ref_pattern, constellation) if not gray_coding else cdsp.tx.mapping(cdsp.tx.coding_gray(ref_pattern, constellation), constellation)
    # --------------------------------------------------------------------------------------------------------------
    # Creating repetitions of the pattern
    # --------------------------------------------------------------------------------------------------------------
    reps = random.randint(2, 5)
    cutoff = random.randint(1, int(prbs_len / 2))
    ref_pattern = np.tile(ref_pattern, reps)[:-1 * cutoff]

    return ref_pattern, constellation, gray_coding

def generate_and_pass_channel(ref_pattern, prbs_type, constellation, gray_coding):
    num_poles = random.randint(1, 8)
    assert_str = '|{0:^6s}| Constellation = {1:^6s} | Gray Coding = {2:^6s} | {3:^3d} poles  ; {4:^3d} DFE taps '.format(
        prbs_type.name,
        constellation.name,
        str(gray_coding),
        num_poles,
        0
    )
    # ==================================================================================================================
    # Creating reference channel
    # ==================================================================================================================
    while True:
        roots       = np.around(np.random.random(num_poles), decimals=3) - 0.5
        channel_ref = np.poly(roots)
        if np.max(np.abs(channel_ref)) == channel_ref[0]:
            break
    # ==================================================================================================================
    # Passing data through the channel
    # ==================================================================================================================
    channel_out, _ = cdsp.channel.awgn_channel(ref_pattern, [1], channel_ref)
    channel_out    = channel_out[len(channel_ref) + 1:]

    return channel_ref, channel_out, constellation, assert_str

