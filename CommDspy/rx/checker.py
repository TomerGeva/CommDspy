import numpy as np
from CommDspy.tx.prbs_generator import prbs_generator
from CommDspy.tx.prbs_iterator import PrbsIterator
from CommDspy.auxiliary import get_polynomial
from CommDspy.rx.lock_pattern import lock_pattern_to_signal_binary

def prbs_checker(prbs_type, data_in, init_lock, loss_th=100):
    """
    :param prbs_type: should be an enumeration from the toolbox
    :param data_in: received bits after slicing - 1 dimensional binary array
    :param init_lock: Boolean indicating weather the data_in is locked to the all ones seed
    :param loss_th: if there are over this number of consecutive errors, losing lock
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

def prbs_checker_economy(prbs_type, data_in, init_lock, lock_th=13, loss_th=100):
    """
    :param prbs_type: Should be an enumeration from the toolbox
    :param data_in: Received bits after slicing
    :param init_lock: Boolean indicating weather the data_in is locked to the all ones seed
    :param lock_th: If init_lock is False, indicating the streak of correct bits needed to be considered as locked
    :param loss_th: number of total erroneous bits to lose lock
    :return:computing the erred bits and their location, indicating if the data is good for locking
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    prbs_len        = 2**prbs_type.value - 1
    correct_count   = 0
    error_count     = np.array([0] * len(data_in))
    seed            = np.array([1] * prbs_type.value)
    start_seed      = None
    wrap_limit      = int(np.ceil(prbs_len / lock_th))
    data_in_temp    = data_in.copy()
    # ==================================================================================================================
    # Getting generating polynomial for the PRBS
    # ==================================================================================================================
    poly_coeff = get_polynomial(prbs_type)
    if poly_coeff[0] == -1:
        return True, 0, [0]
    # ==================================================================================================================
    # Locking on the pattern beginning
    # ==================================================================================================================
    if init_lock:
        start_seed = seed
    else:
        locked = False
        prbs_iter = iter(PrbsIterator(poly_coeff, seed))
        idx = 0
        for wraps in range(wrap_limit):
            data_in_temp = np.concatenate((data_in_temp[lock_th:], data_in_temp[:lock_th])) if wraps > 0 else data_in_temp
            while not locked:
                prbs_bit = next(prbs_iter)
                if prbs_bit == data_in[correct_count]:
                    if correct_count == 0:
                        start_seed = prbs_iter.get_prev_seed()
                        seed = prbs_iter.get_seed()
                    correct_count += 1
                    if correct_count == lock_th:
                        locked = True
                else:
                    idx += 1
                    if idx == prbs_len:
                        break
                    if correct_count > 0:
                        prbs_iter = iter(PrbsIterator(poly_coeff, seed))
                        correct_count = 0
            if locked:
                break
        if not locked:
            print('Unable to lock on pattern')
            return 0, 0, [1]
    # ==================================================================================================================
    # Setting PRBS to the correct location
    # ==================================================================================================================
    correct_count = 0
    prbs_iter = iter(PrbsIterator(poly_coeff, start_seed))
    for ii, data_bit in enumerate(data_in_temp):
        prbs_bit = next(prbs_iter)
        if prbs_bit == data_bit:
            correct_count += 1
        else:
            error_count[ii] = 1

    loss_lock = np.sum(error_count) >= loss_th
    return loss_lock, correct_count, error_count

