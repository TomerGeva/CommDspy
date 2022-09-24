import numpy as np
import CommDspy as cdsp


def coding_gray_test(constellation):
    """
    :param constellation: Enumeration stating the constellation. Should be taken from:
                          CommDspy.constants.ConstellationEnum
    :return:
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    levels = cdsp.get_levels(constellation)
    assert levels.any() is not None, 'Constellation ' + constellation.name + ' type not supported'
    bits = int(np.log2(len(levels)))
    pattern = np.random.randint(0, 2 ** bits, 100)
    assert_str = '|{0:^6s}| bits = {1:1d} Falied!!!!'.format( constellation.name, bits)
    # ==================================================================================================================
    # Getting DUT coded pattern
    # ==================================================================================================================
    coded_dut = cdsp.tx.coding_gray(pattern, constellation)
    # ==================================================================================================================
    # Computing the coding in a different way
    # ==================================================================================================================
    coded_ref = pattern
    if bits == 2:
        coded_temp = pattern.copy()
        coded_temp[pattern == 3] = 2
        coded_temp[pattern == 2] = 3
        coded_ref = coded_temp
    assert np.all(coded_ref == coded_dut), assert_str

def decoding_gray_test(constellation):
    """
    :param constellation: Enumeration stating the constellation. Should be taken from:
                          CommDspy.constants.ConstellationEnum
    :return:
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    levels = cdsp.get_levels(constellation)
    assert levels.any() is not None, 'Constellation ' + constellation.name + ' type not supported'
    bits    = int(np.log2(len(levels)))
    pattern = np.random.randint(0, 2 ** bits, 100)
    assert_str = '|{0:^6s}| bits = {1:1d} Falied!!!!'.format(constellation.name, bits)
    # ==================================================================================================================
    # Getting DUT coded pattern
    # ==================================================================================================================
    coded_dut   = cdsp.tx.coding_gray(pattern, constellation)
    decoded_dut = cdsp.rx.decoding_gray(coded_dut, constellation)
    assert np.all(pattern == decoded_dut), assert_str

def coding_differential_test(constellation):
    """
    :param constellation: Enumeration stating the constellation. Should be taken from:
                          CommDspy.constants.ConstellationEnum
    :return:
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    levels = cdsp.get_levels(constellation)
    assert levels.any() is not None, 'Constellation ' + constellation.name + ' type not supported'
    bits = int(np.log2(len(levels)))
    pattern = np.random.randint(0, 2 ** bits, 100)
    assert_str = '|{0:^6s}| bits = {1:1d} Falied!!!!'.format(constellation.name, bits)
    # ==================================================================================================================
    # Getting DUT coded pattern
    # ==================================================================================================================
    coded_dut = cdsp.tx.coding_differential(pattern, constellation)
    # ==================================================================================================================
    # Computing the coding in a different way
    # ==================================================================================================================
    coded_ref = np.zeros_like(pattern)
    init_symbol = 0
    for ii, symbol in enumerate(pattern):
        coded_ref[ii] = int((init_symbol + symbol) % len(levels))
        init_symbol = coded_ref[ii]
    assert np.all(coded_ref == coded_dut), assert_str

def decoding_differential_test(constellation):
    """
    :param constellation: Enumeration stating the constellation. Should be taken from:
                          CommDspy.constants.ConstellationEnum
    :return:
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    levels = cdsp.get_levels(constellation)
    assert levels.any() is not None, 'Constellation ' + constellation.name + ' type not supported'
    bits = int(np.log2(len(levels)))
    pattern = np.random.randint(0, 2 ** bits, 100)
    assert_str = '|{0:^6s}| bits = {1:1d} Falied!!!!'.format(constellation.name, bits)
    # ==================================================================================================================
    # Getting DUT coded pattern
    # ==================================================================================================================
    coded_dut = cdsp.tx.coding_differential(pattern, constellation)
    decoded_dut = cdsp.rx.decoding_differential(coded_dut, constellation)
    assert np.all(pattern == decoded_dut), assert_str

def coding_manchester_test():
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    pattern = np.random.randint(0, 2 , [100, 3])
    # ==================================================================================================================
    # Getting DUT coded pattern
    # ==================================================================================================================
    coded_dut = cdsp.tx.coding_manchester(pattern)
    # ==================================================================================================================
    # Computing the coding in a different way
    # ==================================================================================================================
    coded_ref = np.zeros(600)
    for ii, symbol in enumerate(np.reshape(pattern, -1)):
        coded_ref[2*ii]     = symbol
        coded_ref[2*ii + 1] = 1 - symbol
    coded_ref = np.reshape(coded_ref, [100, 6])
    assert np.all(coded_ref == coded_dut), 'Manchester encoding failed!'

def decoding_manchester_test():
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    pattern = np.random.randint(0, 2, 100)
    # ==================================================================================================================
    # Getting DUT coded pattern
    # ==================================================================================================================
    coded_dut = cdsp.tx.coding_manchester(pattern)
    decoded_dut = cdsp.rx.decoding_manchester(coded_dut)
    assert np.all(pattern == decoded_dut), 'Manchester decoding failed!'

def coding_bipolar_test():
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    pattern = np.random.randint(0, 2, [100, 3])
    # ==================================================================================================================
    # Getting DUT coded pattern
    # ==================================================================================================================
    coded_dut = cdsp.tx.coding_bipolar(pattern)
    # ==================================================================================================================
    # Computing the coding in a different way
    # ==================================================================================================================
    coded_ref = np.zeros_like(np.reshape(pattern, -1))
    running_sign = -1
    for ii, symbol in enumerate(np.reshape(pattern, -1)):
        if symbol == 1:
            coded_ref[ii] = 1 + running_sign
            running_sign *= -1
        else:
            coded_ref[ii] = 1
    coded_ref = np.reshape(coded_ref, [100, 3])

    assert np.all(coded_ref == coded_dut), 'Bi-polar encoding failed!'

def decoding_bipolar_test():
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    pattern = np.random.randint(0, 2, 100)
    # ==================================================================================================================
    # Getting DUT coded pattern
    # ==================================================================================================================
    coded_dut = cdsp.tx.coding_bipolar(pattern)
    decoded_dut = cdsp.rx.decoding_bipolar(coded_dut, error_detection=False)
    assert np.allclose(pattern, decoded_dut), 'Bipolar decoding failed!'
    decoded_dut = cdsp.rx.decoding_bipolar(coded_dut, error_detection=True)
    assert np.allclose(pattern, decoded_dut), 'Bipolar decoding failed!'

def coding_mlt3_test():
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    pattern = np.random.randint(0, 2, [100, 3])
    # ==================================================================================================================
    # Getting DUT coded pattern
    # ==================================================================================================================
    coded_dut = cdsp.tx.coding_mlt3(pattern)
    # ==================================================================================================================
    # Computing the coding in a different way
    # ==================================================================================================================
    coded_ref = np.zeros_like(np.reshape(pattern, -1))
    values = np.array([0,1,2,1])
    running_idx = 0
    for ii, symbol in enumerate(np.reshape(pattern, -1)):
        if symbol == 1:
            running_idx = (running_idx + 1) % 4
        coded_ref[ii] = values[running_idx]
    coded_ref = np.reshape(coded_ref, [100, 3])
    assert np.all(coded_ref == coded_dut), 'MLT-3 encoding failed!'

def decoding_mlt3_test():
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    pattern = np.random.randint(0, 2, 100)
    # ==================================================================================================================
    # Getting DUT coded pattern
    # ==================================================================================================================
    coded_dut = cdsp.tx.coding_mlt3(pattern)
    decoded_dut = cdsp.rx.decoding_mlt3(coded_dut)
    assert np.allclose(pattern, decoded_dut), 'MLT-3 decoding failed!'

def coding_differential_manchester_test():
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    pattern = np.random.randint(0, 2, [100, 3])
    # ==================================================================================================================
    # Getting DUT coded pattern
    # ==================================================================================================================
    coded_dut = cdsp.tx.coding_differential_manchester(pattern)
    # ==================================================================================================================
    # Computing the coding in a different way
    # ==================================================================================================================
    coded_ref = np.zeros(600)
    running_idx = 0
    coding_list = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    for ii, symbol in enumerate(np.reshape(pattern, -1)):
        if symbol == 1:
            coded_ref[2 * ii]     = coding_list[2 * running_idx][0]
            coded_ref[2 * ii + 1] = coding_list[2 * running_idx][1]
        else:
            coded_ref[2 * ii]     = coding_list[2 * running_idx + 1][0]
            coded_ref[2 * ii + 1] = coding_list[2 * running_idx + 1][1]
            running_idx += 1
        running_idx = (running_idx + 1) % 2
    coded_ref = np.reshape(coded_ref, [100, 6]).astype(int)
    assert np.all(coded_ref == coded_dut), 'Differential Manchester encoding failed!'

def decoding_differential_manchester_test():
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    pattern = np.random.randint(0, 2, 100)
    # ==================================================================================================================
    # Getting DUT coded pattern
    # ==================================================================================================================
    coded_dut = cdsp.tx.coding_differential_manchester(pattern)
    decoded_dut = cdsp.rx.decoding_differential_manchester(coded_dut)
    assert np.allclose(pattern, decoded_dut), 'Differential Manchester decoding failed!'

def coding_linear_block_test():
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    pattern = np.random.randint(0, 2, 100)
    k       = np.random.randint(1, 6, 1)[0]
    n       = k + 1
    G       = np.concatenate([np.eye(k), np.ones([k, 1])], axis=1).astype(int)  # creating parity matrix
    # ==================================================================================================================
    # Getting DUT coded pattern
    # ==================================================================================================================
    coded_dut = cdsp.tx.coding_linear(pattern, G)
    # ==================================================================================================================
    # Computing the coding in a different way
    # ==================================================================================================================
    padding = k - (len(pattern) % k) if len(pattern) % k > 0 else 0
    pattern_padded = np.concatenate([pattern, np.array([0] * padding)]).astype(int)
    pattern_blocks = np.reshape(pattern_padded, [-1, k])
    block_parity   = np.sum(pattern_blocks, axis=1) % 2
    pattern_blocks_coded = np.concatenate([pattern_blocks, block_parity[:, None]], axis=1)
    coded_ref = np.reshape(pattern_blocks_coded, -1)

    assert np.all(coded_ref == coded_dut), 'Linear block encoding failed!'

def decoding_linear_block_test():
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    pattern = np.random.randint(0, 2, 100)
    k = np.random.randint(1, 6, 1)[0]
    n = k + 1
    G = np.concatenate([np.eye(k), np.ones([k, 1])], axis=1).astype(int)  # creating parity matrix
    # ==================================================================================================================
    # Getting DUT coded pattern
    # ==================================================================================================================
    coded_dut    = cdsp.tx.coding_linear(pattern, G)
    decoded_dut  = cdsp.rx.decoding_linear(coded_dut, G, error_prob=False)
    assert np.allclose(pattern, decoded_dut[:len(pattern)]), 'Linear block decoding failed!'
    # coded_dut[1] = 1 - coded_dut[1]
    decoded_dut, p_err = cdsp.rx.decoding_linear(coded_dut, G, error_prob=True)
    assert np.allclose(pattern, decoded_dut[:len(pattern)]), 'Linear block decoding failed!'

def coding_conv_basic_test():
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    pat_len = 100
    pattern = np.random.randint(0, 2, pat_len)
    n_in    = np.random.randint(1, 4)
    n_out   = np.random.randint(n_in+1, n_in + 4)
    G       = {}
    # --------------------------------------------------------------------------------------------------------------
    # Creating the generating matrix, dictionary representation
    # --------------------------------------------------------------------------------------------------------------
    for ii in range(n_in):
        G_ii = []
        constraint_length = np.random.randint(1, 5)
        for jj in range(n_out):
            transfer_function = np.random.randint(0, 2, [constraint_length])
            G_ii.append(transfer_function)
        G[ii] = np.array(G_ii)
    # ==================================================================================================================
    # Getting DUT coded pattern
    # ==================================================================================================================
    coded_dut = cdsp.tx.coding_conv(pattern, G)
    # ==================================================================================================================
    # Computing the coding in a different way
    # ==================================================================================================================
    pattern_reshape = cdsp.buffer(pattern, n_in)
    coded_ref       = np.zeros([n_out, pattern_reshape.shape[0]], dtype=int)
    # -------------------------------------------------------
    # Creating the memory for each of the inputs
    # -------------------------------------------------------
    memory = {}  # memory plus the current input
    for kk in range(n_in):
        memory[kk] = np.zeros(G[kk].shape[1], dtype=int)
    # -----------------------------------------------------------------------------------
    # Running through each of the input chunks, computing the output w.r.t to all inputs
    # -----------------------------------------------------------------------------------
    for kk, input_chuck in enumerate(pattern_reshape):
        for ii, kk_i in enumerate(input_chuck):  # Updating the constraint
            memory[ii] = np.concatenate([[kk_i], memory[ii][:-1]]) if len(memory[ii]) > 1 else np.array([kk_i])
        for jj in range(n_out):
            for ii in range(n_in):  # the ith input in the kth chunk
                G_ii_jj           = G[ii][jj]
                coded_ref[jj, kk] = coded_ref[jj, kk] ^ (G_ii_jj.dot(memory[ii]) % 2)
    # ==================================================================================================================
    # Reshaping
    # ==================================================================================================================
    coded_ref_tot = np.reshape(coded_ref.T, -1)
    assert all(coded_ref_tot == coded_dut), 'Convolution encoding with FIR only failed!'

