import numpy as np
import os
import random
from test.auxiliary import read_1line_csv
from CommDspy import prbs_ana, prbs_ana_econ

def prbs_analysis_test(prbs_type, loss_th, lock_th):
    """
    :param prbs_type:
    :param loss_th:
    :param lock_th:
    :return:
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    ref_filename = os.path.join(os.getcwd(),'test_data',prbs_type.name + '_seed_ones.csv')
    ref_prbs_bin = read_1line_csv(ref_filename)
    prbs_len     = len(ref_prbs_bin)
    shift_idx    = random.randint(0, prbs_len)
    p_err        = 1e-2
    assert_str  = '|{0:^6s}|'.format(prbs_type.name)
    # ==================================================================================================================
    # Shifting PRBS and injecting errors
    # ==================================================================================================================
    prob                           = np.random.random(prbs_len)
    prbs_shifted_ref               = np.concatenate((ref_prbs_bin[shift_idx:], ref_prbs_bin[:shift_idx]))
    prbs_shifted_ref[prob < p_err] = 1 - prbs_shifted_ref[prob < p_err]
    ref_prbs_bin[prob < p_err]     = 1 - ref_prbs_bin[prob < p_err]
    lost_lock_ref                  = np.sum(prob < p_err) >= loss_th
    correct_bit_count_ref          = prbs_len - np.sum(prob < p_err)
    error_bit_ref                  = prob < p_err
    # ==================================================================================================================
    # Running DUT with init_lock = False
    # ==================================================================================================================
    lost_lock, correct_bit_count, error_bit = prbs_ana(prbs_type, prbs_shifted_ref, init_lock=False, loss_th=loss_th)
    # ==================================================================================================================
    # Checking
    # ==================================================================================================================
    assert lost_lock_ref         == lost_lock        , assert_str + ' prbs_ana |Init lock = False | shift_idx = ' + shift_idx + ' | lost_lock failed !!!'
    assert correct_bit_count_ref == correct_bit_count, assert_str + ' prbs_ana |Init lock = False | shift_idx = ' + shift_idx + ' | correct_bit_count failed !!!'
    assert np.all(error_bit_ref  == error_bit)       , assert_str + ' prbs_ana |Init lock = False | shift_idx = ' + shift_idx + ' | error_bit  failed !!!'
    # ==================================================================================================================
    # Running DUT with init_lock = True
    # ==================================================================================================================
    lost_lock, correct_bit_count, error_bit = prbs_ana(prbs_type, ref_prbs_bin, init_lock=True, loss_th=loss_th)
    # ==================================================================================================================
    # Checking
    # ==================================================================================================================
    assert lost_lock_ref         == lost_lock        , assert_str + ' prbs_ana | Init lock = True | shift_idx = ' + str(shift_idx) + ' | lost_lock failed !!!'
    assert correct_bit_count_ref == correct_bit_count, assert_str + ' prbs_ana | Init lock = True | shift_idx = ' + str(shift_idx) + ' | correct_bit_count failed !!!'
    assert np.all(error_bit_ref  == error_bit)       , assert_str + ' prbs_ana | Init lock = True | shift_idx = ' + str(shift_idx) + ' | error_bit  failed !!!'
    # ==================================================================================================================
    # Running DUT with init_lock = False
    # ==================================================================================================================
    lost_lock, correct_bit_count, error_bit = prbs_ana_econ(prbs_type, prbs_shifted_ref, init_lock=False, loss_th=loss_th, lock_th=lock_th)
    # ==================================================================================================================
    # Checking
    # ==================================================================================================================
    assert lost_lock_ref == lost_lock                , assert_str + ' prbs_ana_econ | Init lock = False | shift_idx = ' + str(shift_idx) + ' | lost_lock failed !!!'
    assert correct_bit_count_ref == correct_bit_count, assert_str + ' prbs_ana_econ | Init lock = False | shift_idx = ' + str(shift_idx) + ' | correct_bit_count failed !!!'
    assert np.all(error_bit_ref == error_bit)        , assert_str + ' prbs_ana_econ | Init lock = False | shift_idx = ' + str(shift_idx) + ' | error_bit  failed !!!'
    # ==================================================================================================================
    # Running DUT with init_lock = True
    # ==================================================================================================================
    lost_lock, correct_bit_count, error_bit = prbs_ana_econ(prbs_type, ref_prbs_bin, init_lock=True, loss_th=loss_th, lock_th=lock_th)
    # ==================================================================================================================
    # Checking
    # ==================================================================================================================
    assert lost_lock_ref == lost_lock                , assert_str + ' prbs_ana_econ | Init lock = True | shift_idx = ' + str(shift_idx) + ' | lost_lock failed !!!'
    assert correct_bit_count_ref == correct_bit_count, assert_str + ' prbs_ana_econ | Init lock = True | shift_idx = ' + str(shift_idx) + ' | correct_bit_count failed !!!'
    assert np.all(error_bit_ref == error_bit)        , assert_str + ' prbs_ana_econ | Init lock = True | shift_idx = ' + str(shift_idx) + ' | error_bit  failed !!!'