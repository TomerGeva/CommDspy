import numpy as np
import random
import csv
import os
import test
from CommDspy.constants import PrbsEnum, ConstellationEnum
import CommDspy as cdsp
import json

prbs_types      = [PrbsEnum.PRBS7, PrbsEnum.PRBS9, PrbsEnum.PRBS11, PrbsEnum.PRBS13, PrbsEnum.PRBS15]
constellations  = [ConstellationEnum.OOK, ConstellationEnum.NRZ, ConstellationEnum.PAM4]

def test_init():
    random.seed(None)
    seed = random.randint(0, int(1e9))
    print('Seed = ' + str(seed))
    random.seed(seed)
    np.random.seed(seed)

def test_dig_delay_fir_coeffs():
    assert np.allclose(cdsp.dig_delay_lagrange_coeffs(1, 0), [1, 0])
    assert np.allclose(cdsp.dig_delay_lagrange_coeffs(1, 1), [0, 1])
    assert np.allclose(cdsp.dig_delay_lagrange_coeffs(2, 0), [0, 1, 0])
    assert np.allclose(cdsp.dig_delay_lagrange_coeffs(2, 1), [0, 0, 1])
    assert np.allclose(cdsp.dig_delay_lagrange_coeffs(3, 0), [0, 1, 0, 0])
    assert np.allclose(cdsp.dig_delay_lagrange_coeffs(3, 0, forward=False), [0, 0, 1, 0])

def test_equalization():
    for prbs_type in [PrbsEnum.PRBS7, PrbsEnum.PRBS9, PrbsEnum.PRBS11, PrbsEnum.PRBS13]:
        test.equalization_prbs_test(prbs_type)

def test_buffer():
    a = np.array([1, 2, 3, 4, 5, 6, 7])
    matrix_dut = cdsp.buffer(a, 2, delay=2, overlap=0, clip=False)
    matrix_ref = np.array([[0, 0], [1, 2], [3, 4], [5, 6], [7, 0]])
    assert np.all(matrix_dut == matrix_ref), 'Does not work'
    matrix_dut = cdsp.buffer(a, 2, delay=2, overlap=0, clip=True)
    matrix_ref = np.array([[0, 0], [1, 2], [3, 4], [5, 6]])
    assert np.all(matrix_dut == matrix_ref), 'Does not work'
    matrix_dut = cdsp.buffer(a, 2, delay=0, overlap=1, clip=True)
    matrix_ref = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    assert np.all(matrix_dut == matrix_ref), 'Does not work'
    matrix_dut = cdsp.buffer(a, 2, delay=0, overlap=1, clip=False)
    matrix_ref = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    assert np.all(matrix_dut == matrix_ref), 'Does not work'
    matrix_dut = cdsp.buffer(a, 2, delay=0, overlap=-1, clip=False)
    matrix_ref = np.array([[1, 2], [4, 5], [7, 0]])
    assert np.all(matrix_dut == matrix_ref), 'Does not work'
    matrix_dut = cdsp.buffer(a, 2, delay=0, overlap=-1, clip=True)
    matrix_ref = np.array([[1, 2], [4, 5]])
    assert np.all(matrix_dut == matrix_ref), 'Does not work'
    matrix_dut = cdsp.buffer(a, 2, delay=0, overlap=-2, clip=False)
    matrix_ref = np.array([[1, 2], [5, 6]])
    assert np.all(matrix_dut == matrix_ref), 'Does not work'
    matrix_dut = cdsp.buffer(a, 2, delay=0, overlap=-2, clip=True)
    matrix_ref = np.array([[1, 2], [5, 6]])
    assert np.all(matrix_dut == matrix_ref), 'Does not work'

def test_channel_estimation_prbs():
    """
    :return: Testing the channel_estimation_prbs function
    """
    for prbs_type in [PrbsEnum.PRBS7, PrbsEnum.PRBS9, PrbsEnum.PRBS11, PrbsEnum.PRBS13]:
        test.channel_estimation_prbs_test(prbs_type)

def test_noise_awgn():
    """
    :return: Testing the AWGN function
    """
    for prbs_type in prbs_types:
        for constellation in constellations:
            test.awgn_test(prbs_type, constellation)

def test_prbs_analisys():
    """
    :return: Testing both the prbs_ana and the prbs_ana_econ
    """
    loss_th = 10
    for prbs_type in prbs_types:
        test.prbs_analysis_test(prbs_type, loss_th, lock_th=15, econ=False)
        # test.prbs_analysis_test(prbs_type, loss_th, lock_th=15, econ=True)

def test_lock_pattern_to_signal_1():
    """
    :return:Testing the locking of the pattern on a signal length of 2 PRBS cycles with a small amount of errors
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    p_err           = 1e-4
    for prbs_type in prbs_types:
        pattern_length = 2 ** prbs_type.value - 1
        # ----------------------------------------------------------------------------------------------------------
        # Generating pattern
        # ----------------------------------------------------------------------------------------------------------
        poly_coeff = cdsp.get_polynomial(prbs_type)
        init_seed  = np.array([1] * prbs_type.value)
        prbs_seq, _ = cdsp.tx.prbs_gen(poly_coeff, init_seed, pattern_length)
        # ----------------------------------------------------------------------------------------------------------
        # Injecting errors to signal
        # ----------------------------------------------------------------------------------------------------------
        signal = np.tile(prbs_seq, 2)
        prob = np.random.random(signal.shape)
        signal[prob < p_err] = 1 - signal[prob < p_err]
        # ----------------------------------------------------------------------------------------------------------
        # Shifting the pattern and checking
        # ----------------------------------------------------------------------------------------------------------
        shift_idx = random.randint(0, len(prbs_seq)-1)
        pattern = np.concatenate((prbs_seq[shift_idx:], prbs_seq[:shift_idx]))
        test.lock_pattern_to_signal_test(pattern, signal, shift_idx)

def test_lock_pattern_to_signal_2():
    """
    :return:Testing the locking of the pattern on a signal length of 1 PRBS cycles with a small amount of errors
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    p_err           = 1e-4
    for prbs_type in prbs_types:
        pattern_length = 2 ** prbs_type.value - 1
        # ----------------------------------------------------------------------------------------------------------
        # Generating pattern
        # ----------------------------------------------------------------------------------------------------------
        poly_coeff = cdsp.get_polynomial(prbs_type)
        init_seed  = np.array([1] * prbs_type.value)
        prbs_seq, _ = cdsp.tx.prbs_gen(poly_coeff, init_seed, pattern_length)
        # ----------------------------------------------------------------------------------------------------------
        # Injecting aerror to signal
        # ----------------------------------------------------------------------------------------------------------
        signal = prbs_seq
        prob = np.random.random(signal.shape)
        signal[prob < p_err] = 1 - signal[prob < p_err]
        # ----------------------------------------------------------------------------------------------------------
        # Shifting the pattern and checking
        # ----------------------------------------------------------------------------------------------------------
        shift_idx = random.randint(0, len(prbs_seq)-1)
        pattern = np.concatenate((prbs_seq[shift_idx:], prbs_seq[:shift_idx]))
        test.lock_pattern_to_signal_test(pattern, signal, shift_idx)

def test_lock_pattern_to_signal_3():
    """
    :return:Testing the locking of the pattern on a signal length of less than 1 PRBS cycle with a small amount of errors
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    p_err           = 1e-4
    for prbs_type in prbs_types:
        pattern_length = 2 ** prbs_type.value - 1
        # ----------------------------------------------------------------------------------------------------------
        # Generating pattern
        # ----------------------------------------------------------------------------------------------------------
        poly_coeff = cdsp.get_polynomial(prbs_type)
        init_seed  = np.array([1] * prbs_type.value)
        prbs_seq, _ = cdsp.tx.prbs_gen(poly_coeff, init_seed, pattern_length)
        # ----------------------------------------------------------------------------------------------------------
        # Injecting error to signal
        # ----------------------------------------------------------------------------------------------------------
        cutoff_idx = random.randint(int(len(prbs_seq) / 4), int(len(prbs_seq) * 3 / 4))
        signal = prbs_seq[:cutoff_idx]
        prob = np.random.random(signal.shape)
        signal[prob < p_err] = 1 - signal[prob < p_err]
        # ----------------------------------------------------------------------------------------------------------
        # Shifting the pattern and checking
        # ----------------------------------------------------------------------------------------------------------
        shift_idx = random.randint(0, len(prbs_seq)-1)
        pattern = np.concatenate((prbs_seq[shift_idx:], prbs_seq[:shift_idx]))
        test.lock_pattern_to_signal_test(pattern, signal, shift_idx)

def test_lock_pattern_to_signal_binary_1():
    """
    :return:Testing the locking of the pattern on a signal length of 2 PRBS cycles with a small amount of errors
    """
    p_err = 1e-4
    for prbs_type in prbs_types:
        # ----------------------------------------------------------------------------------------------------------
        # Loading pattern
        # ----------------------------------------------------------------------------------------------------------
        ref_filename = os.path.join(os.getcwd(), 'test_data', prbs_type.name + '_seed_ones.csv')
        with open(ref_filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            ref_prbs_bin = np.array(next(reader)).astype(int)
        # ----------------------------------------------------------------------------------------------------------
        # Injecting error to signal
        # ----------------------------------------------------------------------------------------------------------
        signal    = np.tile(ref_prbs_bin, 2)
        prob      = np.random.random(signal.shape)
        signal[prob < p_err] = 1 - signal[prob < p_err]
        # ----------------------------------------------------------------------------------------------------------
        # Shifting the pattern and checking
        # ----------------------------------------------------------------------------------------------------------
        shift_idx = random.randint(0, len(ref_prbs_bin)-1)
        pattern   = np.concatenate((ref_prbs_bin[shift_idx:], ref_prbs_bin[:shift_idx]))
        test.lock_pattern_to_signal_binary_test(pattern, signal, shift_idx)

def test_lock_pattern_to_signal_binary_2():
    """
    :return:Testing the locking of the pattern on a signal length of 1 PRBS cycles with a small amount of errors
    """
    p_err = 1e-4
    for prbs_type in prbs_types:
        # ----------------------------------------------------------------------------------------------------------
        # Loading pattern
        # ----------------------------------------------------------------------------------------------------------
        ref_filename = os.path.join(os.getcwd(), 'test_data', prbs_type.name + '_seed_ones.csv')
        with open(ref_filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            ref_prbs_bin = np.array(next(reader)).astype(int)
        # ----------------------------------------------------------------------------------------------------------
        # Injecting error to signal
        # ----------------------------------------------------------------------------------------------------------
        signal = ref_prbs_bin
        prob = np.random.random(signal.shape)
        signal[prob < p_err] = 1 - signal[prob < p_err]
        # ----------------------------------------------------------------------------------------------------------
        # Shifting the pattern and checking
        # ----------------------------------------------------------------------------------------------------------
        shift_idx = random.randint(0, len(ref_prbs_bin)-1)
        pattern = np.concatenate((ref_prbs_bin[shift_idx:], ref_prbs_bin[:shift_idx]))
        test.lock_pattern_to_signal_binary_test(pattern, signal, shift_idx)

def test_lock_pattern_to_signal_binary_3():
    """
    :return:Testing the locking of the pattern on a signal length of less than 1 PRBS cycles with a small amount of errors
    """
    p_err = 1e-4
    for prbs_type in prbs_types:
        # ----------------------------------------------------------------------------------------------------------
        # Loading pattern
        # ----------------------------------------------------------------------------------------------------------
        ref_filename = os.path.join(os.getcwd(), 'test_data', prbs_type.name + '_seed_ones.csv')
        with open(ref_filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            ref_prbs_bin = np.array(next(reader)).astype(int)
        # ----------------------------------------------------------------------------------------------------------
        # Injecting error to signal
        # ----------------------------------------------------------------------------------------------------------
        cutoff_idx = random.randint(int(len(ref_prbs_bin)/4), int(len(ref_prbs_bin)*3/4))
        signal = ref_prbs_bin[:cutoff_idx]
        prob = np.random.random(signal.shape)
        signal[prob < p_err] = 1 - signal[prob < p_err]
        # ----------------------------------------------------------------------------------------------------------
        # Shifting the pattern and checking
        # ----------------------------------------------------------------------------------------------------------
        shift_idx = random.randint(0, len(ref_prbs_bin)-1)
        pattern = np.concatenate((ref_prbs_bin[shift_idx:], ref_prbs_bin[:shift_idx]))
        test.lock_pattern_to_signal_binary_test(pattern, signal, shift_idx)

def test_symbol2bin():
    test.symbol2bin_test()

def test_demapping_plus_decoding_gray():
    """
    :return: Testing the coding function
    """
    pattern_2bit = np.array([0, 1, 2, 3, 0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    pattern_1bit = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1])
    # ==================================================================================================================
    # 2 bit PAM4 test
    # ==================================================================================================================
    coded_2bit_pam4          = np.array([-3, -1, 1, 3, -3, -3, -1, -1, 1, 1, 3, 3, -3, -3, -3, -1, -1, -1, 1, 1, 1, 3, 3, 3])
    coded_2bit_pam4_gray     = np.array([-3, -1, 3, 1, -3, -3, -1, -1, 3, 3, 1, 1, -3, -3, -3, -1, -1, -1, 3, 3, 3, 1, 1, 1])
    coded_2bit_pam4_inv      = -1 * coded_2bit_pam4
    coded_2bit_pam4_gray_inv = -1 * coded_2bit_pam4_gray
    assert np.all(pattern_2bit ==                       cdsp.rx.demapping(coded_2bit_pam4,          ConstellationEnum.PAM4, pn_inv=False)),                          'PAM4 UNCODED '
    assert np.all(pattern_2bit == cdsp.rx.decoding_gray(cdsp.rx.demapping(coded_2bit_pam4_gray,     ConstellationEnum.PAM4, pn_inv=False), ConstellationEnum.PAM4)), 'PAM4 GRAY '
    assert np.all(pattern_2bit ==                       cdsp.rx.demapping(coded_2bit_pam4_inv,      ConstellationEnum.PAM4, pn_inv=True)),                           'PAM4 UNCODED inverted '
    assert np.all(pattern_2bit == cdsp.rx.decoding_gray(cdsp.rx.demapping(coded_2bit_pam4_gray_inv, ConstellationEnum.PAM4, pn_inv=True), ConstellationEnum.PAM4)),  'PAM4 GRAY inverted '
    # ==================================================================================================================
    # 1 bit PAM4 test
    # ==================================================================================================================
    coded_1bit_pam4          = np.array([-3, -1, -3, -1, -3, -3, -1, -1, -3, -1, -3, -3, -3, -1, -1, -1])
    coded_1bit_pam4_gray     = coded_1bit_pam4
    coded_1bit_pam4_inv      = -1 * coded_1bit_pam4
    coded_1bit_pam4_gray_inv = -1 * coded_1bit_pam4_gray
    assert np.all(pattern_1bit ==                       cdsp.rx.demapping(coded_1bit_pam4,          ConstellationEnum.PAM4, pn_inv=False)),                          'PAM4 UNCODED - 1 bit'
    assert np.all(pattern_1bit == cdsp.rx.decoding_gray(cdsp.rx.demapping(coded_1bit_pam4_gray,     ConstellationEnum.PAM4, pn_inv=False), ConstellationEnum.PAM4)), 'PAM4 GRAY - 1 bit'
    assert np.all(pattern_1bit ==                       cdsp.rx.demapping(coded_1bit_pam4_inv,      ConstellationEnum.PAM4, pn_inv=True)),                           'PAM4 UNCODED inverted - 1 bit'
    assert np.all(pattern_1bit == cdsp.rx.decoding_gray(cdsp.rx.demapping(coded_1bit_pam4_gray_inv, ConstellationEnum.PAM4, pn_inv=True), ConstellationEnum.PAM4)),  'PAM4 GRAY inverted - 1 bit'
    # ==================================================================================================================
    # 1 bit NRZ test
    # ==================================================================================================================
    coded_1bit_nrz          = coded_1bit_pam4 + 2
    coded_1bit_nrz_gray     = coded_1bit_nrz
    coded_1bit_nrz_inv      = -1 * coded_1bit_nrz
    coded_1bit_nrz_gray_inv = -1 * coded_1bit_nrz_gray
    assert np.all(pattern_1bit ==                       cdsp.rx.demapping(coded_1bit_nrz,          ConstellationEnum.NRZ, pn_inv=False)),                         'NRZ UNCODED'
    assert np.all(pattern_1bit == cdsp.rx.decoding_gray(cdsp.rx.demapping(coded_1bit_nrz_gray,     ConstellationEnum.NRZ, pn_inv=False), ConstellationEnum.NRZ)), 'NRZ GRAY'
    assert np.all(pattern_1bit ==                       cdsp.rx.demapping(coded_1bit_nrz_inv,      ConstellationEnum.NRZ, pn_inv=True)),                          'NRZ UNCODED inverted'
    assert np.all(pattern_1bit == cdsp.rx.decoding_gray(cdsp.rx.demapping(coded_1bit_nrz_gray_inv, ConstellationEnum.NRZ, pn_inv=True), ConstellationEnum.NRZ)),  'NRZ GRAY inverted'
    # ==================================================================================================================
    # 1 bit OOK test
    # ==================================================================================================================
    coded_1bit_ook          = (coded_1bit_nrz + 1) / 2
    coded_1bit_ook_gray     = coded_1bit_ook
    coded_1bit_ook_inv      = -1 * coded_1bit_ook
    coded_1bit_ook_gray_inv = -1 * coded_1bit_ook
    assert np.all(pattern_1bit ==                       cdsp.rx.demapping(coded_1bit_ook,          ConstellationEnum.OOK, False)),                         'OOK UNCODED'
    assert np.all(pattern_1bit == cdsp.rx.decoding_gray(cdsp.rx.demapping(coded_1bit_ook_gray,     ConstellationEnum.OOK, False), ConstellationEnum.OOK)), 'OOK GRAY'
    assert np.all(pattern_1bit ==                       cdsp.rx.demapping(coded_1bit_ook_inv,      ConstellationEnum.OOK, True)),                          'OOK UNCODED inverted'
    assert np.all(pattern_1bit == cdsp.rx.decoding_gray(cdsp.rx.demapping(coded_1bit_ook_gray_inv, ConstellationEnum.OOK, True), ConstellationEnum.OOK)),  'OOK GRAY inverted'

def test_bin2symbol():
    test.bin2symbol_test()

def test_prbs_gen_1():
    """
    :return: Testing the generation of all PRBS values except PRBS 31 in all constellations for 1 complete pattern
    generation
    """
    # ==============================================================================================================
    # Partial length PRBS generator tests
    # ==============================================================================================================
    for prbs_type in prbs_types:
        pattern_length = 2 ** prbs_type.value - 1
        test.prbs_gen_test(prbs_type, pattern_length)

def test_prbs_gen_2():
    """
    :return: Testing the generation of all PRBS values except PRBS 31 in all constellations for 1 partial pattern length
    """
    # ==============================================================================================================
    # Partial length PRBS generator tests
    # ==============================================================================================================
    for prbs_type in prbs_types:
        pattern_length = 2 ** prbs_type.value - 1
        required_length = random.randint(0, pattern_length)
        test.prbs_gen_test(prbs_type, required_length)

def test_prbs_gen_3():
    """
    :return: Testing the generation of all PRBS values except PRBS 31 in all constellations for 1 full pattern and
             another partial length
    """
    # ==============================================================================================================
    # Partial length PRBS generator tests
    # ==============================================================================================================
    for prbs_type in prbs_types:
        pattern_length = 2 ** prbs_type.value - 1
        required_length = random.randint(pattern_length+1, 2*pattern_length)
        test.prbs_gen_test(prbs_type, required_length)

def test_prbs_gen_4():
    """
    :return: Testing the generation of all PRBS values except PRBS 31 in all constellations for very small pattern lengths
    """
    # ==============================================================================================================
    # Partial length PRBS generator tests
    # ==============================================================================================================
    for prbs_type in prbs_types:
        required_length = random.randint(1, prbs_type.value)
        test.prbs_gen_test(prbs_type, required_length)

def test_coding_diffferential_manchester():
    test.coding_differential_manchester_test()

def test_decoding_diffferential_manchester():
    test.decoding_differential_manchester_test()

def test_coding_mlt3():
    test.coding_mlt3_test()

def test_decoding_mlt3():
    test.decoding_mlt3_test()

def test_coding_bipolar():
    test.coding_bipolar_test()

def test_decoding_bipolar():
    test.decoding_bipolar_test()

def test_coding_manchester():
    test.coding_manchester_test()

def test_decoding_manchester():
    test.decoding_manchester_test()

def test_coding_diff():
    for constellation in constellations:
        test.coding_differential_test(constellation)

def test_decoding_diff():
    for constellation in constellations:
        test.decoding_differential_test(constellation)

def test_coding_gray():
    for constellation in constellations:
        test.coding_gray_test(constellation)

def test_decoding_gray():
    for constellation in constellations:
        test.decoding_gray_test(constellation)

def test_coding_gray_plus_mapping():
    """
    :return: Testing the coding function
    """
    pattern_2bit = np.array([0, 1, 2, 3, 0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    pattern_1bit = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1])
    # ==================================================================================================================
    # 2 bit PAM4 test
    # ==================================================================================================================
    coded_2bit_pam4          = np.array([-3,-1,1,3,-3,-3,-1,-1,1,1,3,3,-3,-3,-3,-1,-1,-1,1,1,1,3,3,3])
    coded_2bit_pam4_gray     = np.array([-3,-1,3,1,-3,-3,-1,-1,3,3,1,1,-3,-3,-3,-1,-1,-1,3,3,3,1,1,1])
    coded_2bit_pam4_inv      = -1 * coded_2bit_pam4
    coded_2bit_pam4_gray_inv = -1 * coded_2bit_pam4_gray
    assert np.all(coded_2bit_pam4           == cdsp.tx.mapping(pattern_2bit,                                              ConstellationEnum.PAM4, pn_inv=False)), 'PAM4 UNCODED '
    assert np.all(coded_2bit_pam4_gray      == cdsp.tx.mapping(cdsp.tx.coding_gray(pattern_2bit, ConstellationEnum.PAM4), ConstellationEnum.PAM4, pn_inv=False)), 'PAM4 GRAY '
    assert np.all(coded_2bit_pam4_inv       == cdsp.tx.mapping(pattern_2bit,                                              ConstellationEnum.PAM4, pn_inv=True)),  'PAM4 UNCODED inverted '
    assert np.all(coded_2bit_pam4_gray_inv  == cdsp.tx.mapping(cdsp.tx.coding_gray(pattern_2bit, ConstellationEnum.PAM4), ConstellationEnum.PAM4, pn_inv=True)),  'PAM4 GRAY inverted '
    # ==================================================================================================================
    # 1 bit PAM4 test
    # ==================================================================================================================
    coded_1bit_pam4          = np.array([-3,-1,-3,-1,-3,-3,-1,-1,-3,-1,-3,-3,-3,-1,-1,-1])
    coded_1bit_pam4_gray     = coded_1bit_pam4
    coded_1bit_pam4_inv      = -1 * coded_1bit_pam4
    coded_1bit_pam4_gray_inv = -1 * coded_1bit_pam4_gray
    assert np.all(coded_1bit_pam4           == cdsp.tx.mapping(pattern_1bit,                                              ConstellationEnum.PAM4, pn_inv=False)), 'PAM4 UNCODED - 1 bit'
    assert np.all(coded_1bit_pam4_gray      == cdsp.tx.mapping(cdsp.tx.coding_gray(pattern_1bit, ConstellationEnum.PAM4), ConstellationEnum.PAM4, pn_inv=False)), 'PAM4 GRAY - 1 bit'
    assert np.all(coded_1bit_pam4_inv       == cdsp.tx.mapping(pattern_1bit,                                              ConstellationEnum.PAM4, pn_inv=True)),  'PAM4 UNCODED inverted - 1 bit'
    assert np.all(coded_1bit_pam4_gray_inv  == cdsp.tx.mapping(cdsp.tx.coding_gray(pattern_1bit, ConstellationEnum.PAM4), ConstellationEnum.PAM4, pn_inv=True)),  'PAM4 GRAY inverted - 1 bit'
    # ==================================================================================================================
    # 1 bit NRZ test
    # ==================================================================================================================
    coded_1bit_nrz      = coded_1bit_pam4 + 2
    coded_1bit_nrz_gray = coded_1bit_nrz
    coded_1bit_nrz_inv  = -1 * coded_1bit_nrz
    coded_1bit_nrz_gray_inv = -1 * coded_1bit_nrz_gray
    assert np.all(coded_1bit_nrz            == cdsp.tx.mapping(pattern_1bit,                                             ConstellationEnum.NRZ, pn_inv=False)), 'NRZ UNCODED'
    assert np.all(coded_1bit_nrz_gray       == cdsp.tx.mapping(cdsp.tx.coding_gray(pattern_1bit, ConstellationEnum.NRZ), ConstellationEnum.NRZ, pn_inv=False)), 'NRZ GRAY'
    assert np.all(coded_1bit_nrz_inv        == cdsp.tx.mapping(pattern_1bit,                                             ConstellationEnum.NRZ, pn_inv=True)), 'NRZ UNCODED inverted'
    assert np.all(coded_1bit_nrz_gray_inv   == cdsp.tx.mapping(cdsp.tx.coding_gray(pattern_1bit, ConstellationEnum.NRZ), ConstellationEnum.NRZ, pn_inv=True)), 'NRZ GRAY inverted'
    # ==================================================================================================================
    # 1 bit OOK test
    # ==================================================================================================================
    coded_1bit_ook          = (coded_1bit_nrz + 1) / 2
    coded_1bit_ook_gray     = coded_1bit_ook
    coded_1bit_ook_inv      = -1 * coded_1bit_ook
    coded_1bit_ook_gray_inv = -1 * coded_1bit_ook
    assert np.all(coded_1bit_ook            == cdsp.tx.mapping(pattern_1bit,                                             ConstellationEnum.OOK, pn_inv=False)), 'OOK UNCODED'
    assert np.all(coded_1bit_ook_gray       == cdsp.tx.mapping(cdsp.tx.coding_gray(pattern_1bit, ConstellationEnum.OOK), ConstellationEnum.OOK, pn_inv=False)), 'OOK GRAY'
    assert np.all(coded_1bit_ook_inv        == cdsp.tx.mapping(pattern_1bit,                                             ConstellationEnum.OOK, pn_inv=True)), 'OOK UNCODED inverted'
    assert np.all(coded_1bit_ook_gray_inv   == cdsp.tx.mapping(cdsp.tx.coding_gray(pattern_1bit, ConstellationEnum.OOK), ConstellationEnum.OOK, pn_inv=True)), 'OOK GRAY inverted'


if __name__ == '__main__':
    pass
