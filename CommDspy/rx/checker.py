import numpy as np
from CommDspy.tx.prbs_generator import prbs_generator
from CommDspy.auxiliary import get_polynomial
from CommDspy.rx.lock_pattern import lock_pattern_to_signal_binary

def prbs_checker(prbs_type, data_in, init_lock, loss_th=100, prbs_seq=None):
    """
    :param prbs_type: should be an enumeration from the toolbox
    :param data_in: received bits after slicing - 1 dimensional binary array
    :param init_lock: Boolean indicating weather the data_in is locked to the all ones seed
    :param loss_th: if there are over this number of consecutive errors, losing lock
    :param prbs_seq: If None, generating the PRBS sequence, else, assumes that the input is the reference PRBS sequence
    :return: If init_lock is False, lock on the PRBS pattern and computes the error. If already locked, simply computing
             the erred bits
         poly_coeff: Row vector of generating polynomial coefficients excluding the leading 1. Note that size of
                   ploy_coeff should be equal to the degree of the generating polynomial g(x)
                   example - for g(x) = 1+x+x^2+x^12+x^13,
                   use poly_coeff = [1 1 0 0 0 0 0 0 0 0 0 1 1]
        param init_seed: row vector of the same size as poly_coeff. Set to the initial state of the prbs delay line.
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    init_seed = np.array([1] * prbs_type.value)
    prbs_len  = 2 ** len(init_seed) - 1
    # ==================================================================================================================
    # Getting reference PRBS
    # ==================================================================================================================
    if prbs_seq is None:
        poly_coeff = get_polynomial(prbs_type)
        if poly_coeff[0] == -1:
            return True, 0, [0]
        prbs_seq, _ = prbs_generator(poly_coeff, init_seed, prbs_len)
    # ==================================================================================================================
    # Locking on the pattern beginning
    # ==================================================================================================================
    if not init_lock:
        prbs_seq, _ = lock_pattern_to_signal_binary(prbs_seq, data_in)
    # ==================================================================================================================
    # Checking prbs errors
    # ==================================================================================================================
    prbs_rep = int(np.ceil(len(data_in) / (2**len(init_seed)-1) ))
    prbs_seq = np.tile(prbs_seq, prbs_rep)
    prbs_seq = prbs_seq[:len(data_in)]
    # ==================================================================================================================
    # Checking prbs errors
    # ==================================================================================================================
    error_bit           = data_in != prbs_seq
    correct_bit_count   = len(data_in) - np.sum(error_bit)
    lost_lock           = np.sum(error_bit) >= loss_th
    return lost_lock, correct_bit_count, error_bit

def prbs_checker_economy(prbs_type, data_in, loss_th=100):
    """
    :param prbs_type: Should be an enumeration from the toolbox
    :param data_in: Received bits after slicing
    :param loss_th: number of total erroneous bits to lose lock
    :return:computing the erred bits and their location, indicating if the data is good for locking. This is done by
    using the first bits in the pattern as the intial seed for the PRBS, generating the rest of the pattern from that
    point.
    """
    # ==================================================================================================================
    # Getting generating polynomial for the PRBS
    # ==================================================================================================================
    poly_coeff = get_polynomial(prbs_type)
    # ==================================================================================================================
    # Performing the check
    # ==================================================================================================================
    for seed_start in range(prbs_type.value):
        # ----------------------------------------------------------------------------------------------------------
        # Extracting the seed
        # ----------------------------------------------------------------------------------------------------------
        seed = data_in[seed_start:seed_start+prbs_type.value]
        # ----------------------------------------------------------------------------------------------------------
        # Generating the PRBS for the comparison
        # ----------------------------------------------------------------------------------------------------------
        prbs_seq, _ = prbs_generator(poly_coeff, seed[::-1], len(data_in) - seed_start - prbs_type.value)
        # ----------------------------------------------------------------------------------------------------------
        # checking errors
        # ----------------------------------------------------------------------------------------------------------
        error_bit = data_in[seed_start:] != np.concatenate([data_in[seed_start:seed_start+prbs_type.value], prbs_seq])
        correct_bit_count = len(data_in) - np.sum(error_bit)
        loss_lock = np.sum(error_bit) >= loss_th
        if not loss_lock:
            break
    return loss_lock, correct_bit_count, error_bit
