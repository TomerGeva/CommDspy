import numpy as np
import CommDspy as cdsp
from CommDspy.misc.ml_decoding import Trellis
from time import time


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

def decoding_conv_basic_test():
    # np.random.seed(21)
    # np.random.seed(61)
    # np.random.seed(51)
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    pat_len = 100
    pattern = np.random.randint(0, 2, pat_len)
    n_in  = np.random.randint(1, 4)
    n_out = np.random.randint(n_in + 2, n_in + 5)
    G = {}
    # --------------------------------------------------------------------------------------------------------------
    # Creating the generating matrix, dictionary representation
    # --------------------------------------------------------------------------------------------------------------
    condition = False
    while not condition:
        for ii in range(n_in):
            G_ii = []
            constraint_length = np.random.randint(2, 5)
            for jj in range(n_out):
                transfer_function = np.random.randint(0, 2, [constraint_length])
                G_ii.append(transfer_function)
            G[ii] = np.array(G_ii)
        # **************************************************************************************************
        # Checking the conditions:
        # 1. If only the zero word produces the 0 codeword
        # 2. From each state, each transition produces a different output
        # 3. Each output has to be dependent on the input, meaning the constant 0 output is not valid
        # 4. only the transition from state '0' to state '0' produces the '0' output
        # **************************************************************************************************
        trellis_obj = Trellis(G, None, None)
        # 1.
        state_transition_set = set()
        for key in trellis_obj.trellis:
            out, state = trellis_obj.trellis[key]
            state_transition_set.add(tuple([key[1], out, state]))
        condition = len(state_transition_set) == len(trellis_obj.trellis)
        # 2.
        sets_dict = {}
        # Creating the sets
        if condition:
            for key in trellis_obj.trellis:
                out, state = trellis_obj.trellis[key]
                if key[1] in sets_dict:
                    sets_dict[key[1]].add(out)
                else:
                    sets_dict[key[1]] = set()
                    sets_dict[key[1]].add(out)
            # Checking
            for in_state in sets_dict:
                if len(sets_dict[in_state]) != len(trellis_obj.inputs):
                    condition = False
        # 3.
        if condition:
            for ii in G:
                if np.sum(G[ii]) == 0:
                    condition = False
        # 4.
        if condition:
            for key in trellis_obj.trellis:
                out, in_state = trellis_obj.trellis[key]
                if sum(out) == 0:
                    # if (sum(key[1]) > 0 and sum(in_state) == 0) or (sum(key[1]) == 0 and sum(in_state) > 0):
                    if sum(key[1]) > 0 or sum(in_state) > 0:
                        condition = False
                        break
    # ==================================================================================================================
    # Getting DUT coded pattern
    # ==================================================================================================================
    coded_dut   = cdsp.tx.coding_conv(pattern, G)
    t1 = time()
    decoded_dut = cdsp.rx.decoding_conv_map(coded_dut, G, tb_len=5*n_out, error_prob=False)
    print(f'MAP decoding done in {time() - t1:.5f} seconds')
    assert np.allclose(pattern[:len(decoded_dut)], decoded_dut), 'Basic convolution MAP decoding failed!'
    t1 = time()
    decoded_dut2, _ = cdsp.rx.decoding_conv_viterbi(coded_dut, G, 5*n_out, feedback=None, use_feedback=None)
    print(f'Slow viterbi decoding done in {time() - t1:.5f} seconds')
    assert np.allclose(pattern[:len(decoded_dut2)], decoded_dut2), 'Basic convolution VITERBI decoding failed!'

def coding_conv_feedback_test():
    # np.random.seed(2)
    # np.random.seed(200)
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    pat_len = 100
    pattern = np.random.randint(0, 2, pat_len)
    n_in    = np.random.randint(1, 4)
    n_out   = np.random.randint(n_in+1, n_in + 4)
    G        = {}
    feedback = {}
    use_feedback = np.zeros([n_in, n_out], dtype=int)
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
    # --------------------------------------------------------------------------------------------------------------
    # Creating the feedback polynomials, dictionary representation
    # --------------------------------------------------------------------------------------------------------------
    for ii in range(n_in):
        if np.random.rand() < 0.5 and G[ii].shape[1] > 1: # creating feedback for this input
            memory = G[ii].shape[1] - 1
            feedback[ii] = np.concatenate([[1], np.random.randint(0, 2, memory)])
            for jj, G_ii_jj in enumerate(G[ii]):
                if sum(G_ii_jj) > 1:
                    use_feedback[ii, jj] = 1
                elif G_ii_jj[0] == 0:
                    use_feedback[ii, jj] = 1
                else:
                    use_feedback[ii, jj] = int(np.random.rand() < 0.5)  # 50% chance to use the feedback
    # ==================================================================================================================
    # Getting DUT coded pattern
    # ==================================================================================================================
    coded_dut = cdsp.tx.coding_conv(pattern, G, feedback, use_feedback)
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
            if ii in feedback.keys():
                kk_i_fb = (kk_i + memory[ii][:-1].dot(feedback[ii][1:])) % 2
            else:
                kk_i_fb = kk_i
            memory[ii] = np.concatenate([[kk_i_fb], memory[ii][:-1]]) if len(memory[ii]) > 1 else np.array([kk_i_fb])
        for jj in range(n_out):
            for ii in range(n_in):  # the ith input in the kth chunk
                G_ii_jj = G[ii][jj]
                if len(G_ii_jj) == 1 or use_feedback[ii, jj] == 0:
                    # no memory, either systematic or always 0 output, depending on G_ii_jj[0] value.
                    # 1 - systematic ; 0 - zero output
                    coded_ref[jj, kk] = coded_ref[jj, kk] ^ (input_chuck[ii] * G_ii_jj[0])
                else:
                    coded_ref[jj, kk] = coded_ref[jj, kk] ^ (G_ii_jj.dot(memory[ii]) % 2)
    # ==================================================================================================================
    # Reshaping
    # ==================================================================================================================
    coded_ref_tot = np.reshape(coded_ref.T, -1)
    assert all(coded_ref_tot == coded_dut), 'Recursive convolution encoding failed!'

def decoding_conv_feedback_test():
    # np.random.seed(50)
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    pat_len = 100
    pattern = np.random.randint(0, 2, pat_len)
    n_in  =  np.random.randint(1, 4)
    n_out =  np.random.randint(n_in + 2, n_in + 5)
    condition = False
    while not condition:
        G            = {}
        feedback     = {}
        use_feedback = np.zeros([n_in, n_out], dtype=int)
        # ----------------------------------------------------------------------------------------------------------
        # Creating the generating matrix, dictionary representation
        # ----------------------------------------------------------------------------------------------------------
        for ii in range(n_in):
            G_ii = []
            constraint_length = np.random.randint(2, 5)
            for jj in range(n_out):
                transfer_function = np.random.randint(0, 2, [constraint_length])
                G_ii.append(transfer_function)
            G[ii] = np.array(G_ii)
        # ----------------------------------------------------------------------------------------------------------
        # Creating the feedback polynomials, dictionary representation
        # ----------------------------------------------------------------------------------------------------------
        for ii in range(n_in):
            if np.random.rand() < 0.5 and G[ii].shape[1] > 1:  # creating feedback for this input
                memory = G[ii].shape[1] - 1
                feedback[ii] = np.concatenate([[1], np.random.randint(0, 2, memory)])
                for jj, G_ii_jj in enumerate(G[ii]):
                    if sum(G_ii_jj) > 1:
                        use_feedback[ii, jj] = 1
                    elif G_ii_jj[0] == 0:
                        use_feedback[ii, jj] = 1
                    else:
                        use_feedback[ii, jj] = int(np.random.rand() < 0.5)  # 50% chance to use the feedback
        # **************************************************************************************************
        # Checking the conditions:
        # 1. If only the zero word produces the 0 codeword
        # 2. From each state, each transition produces a different output
        # 3. Each output has to be dependent on the input, meaning the constant 0 output is not valid
        # 4. only the transition from state '0' to state '0' produces the '0' output
        # **************************************************************************************************
        trellis_obj = Trellis(G, feedback, use_feedback)
        # 1.
        state_transition_set = set()
        for key in trellis_obj.trellis:
            out, state = trellis_obj.trellis[key]
            state_transition_set.add(tuple([key[1], out, state]))
        condition = len(state_transition_set) == len(trellis_obj.trellis)
        # 2.
        sets_dict = {}
        # Creating the sets
        if condition:
            for key in trellis_obj.trellis:
                out, state = trellis_obj.trellis[key]
                if key[1] in sets_dict:
                    sets_dict[key[1]].add(out)
                else:
                    sets_dict[key[1]] = set()
                    sets_dict[key[1]].add(out)
            # Checking
            for in_state in sets_dict:
                if len(sets_dict[in_state]) != len(trellis_obj.inputs):
                    condition = False
        # 3.
        if condition:
            for ii in G:
                if np.sum(G[ii]) == 0:
                    condition = False
        # 4.
        if condition:
            for key in trellis_obj.trellis:
                out, in_state = trellis_obj.trellis[key]
                if sum(out) == 0:
                    # if (sum(key[1]) > 0 and sum(in_state) == 0) or (sum(key[1]) == 0 and sum(in_state) > 0):
                    if sum(key[1]) > 0 or sum(in_state) > 0:
                        condition = False
                        break
    # ==================================================================================================================
    # Getting DUT coded pattern
    # ==================================================================================================================
    print('found valid encoder')
    coded_dut   = cdsp.tx.coding_conv(pattern, G, feedback, use_feedback)
    t1 = time()
    decoded_dut = cdsp.rx.decoding_conv_map(coded_dut, G, tb_len=5*n_out, feedback=feedback, use_feedback=use_feedback, error_prob=False)
    print(f'MAP decoding done in {time() - t1:.5f} seconds')
    assert np.allclose(pattern[:len(decoded_dut)], decoded_dut), 'Recursive convolution MAP decoding failed!'
    t1 = time()
    decoded_dut2, _ = cdsp.rx.decoding_conv_viterbi(coded_dut, G, 5*n_out, feedback=feedback, use_feedback=use_feedback)
    print(f'Slow viterbi decoding done in {time() - t1:.5f} seconds')
    assert np.allclose(pattern[:len(decoded_dut2)], decoded_dut2), 'Recursive convolution VITERBI decoding failed!'