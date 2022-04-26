import numpy as np
import random
import CommDspy as cdsp

def bin2symbol_test():
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    test_pattern_len = random.randint(1, 100)
    bit_vec          = np.random.randint(0, 2, test_pattern_len)
    num_of_symbols   = 2 ** random.randint(1, int(np.floor(np.log2(test_pattern_len))))
    bit_order_inv    = random.random() > 0.5
    inv_msb          = random.random() > 0.5
    inv_lsb          = random.random() > 0.5
    pn_inv           = random.random() > 0.5
    assert_str       = '|Pattern len {0:^5d}| symbol number {1:3d} ; bit order inv {2:^6s} ; msb_inv = {3:^6s} ; lsb_inv = {4:^6s} ; pn_inv = {5:^6s} Falied!!!!'.format(
        test_pattern_len,
        num_of_symbols,
        'True' if bit_order_inv else 'False',
        'True' if inv_msb else 'False',
        'True' if inv_lsb else 'False',
        'True' if pn_inv else 'False'
    )
    # ==================================================================================================================
    # Calling DUT
    # ==================================================================================================================
    sym_dut = cdsp.tx.bin2symbol(bit_vec, num_of_symbols, bit_order_inv, inv_msb, inv_lsb, pn_inv)
    # ==================================================================================================================
    # Computing reference
    # ==================================================================================================================
    if num_of_symbols == 2:
        sym_vec_ref = 1-bit_vec if pn_inv else bit_vec
    else:
        ref_symbol_num  = int(np.log2(num_of_symbols))
        limit           = int(np.floor(test_pattern_len / ref_symbol_num))
        bit_vec_ref     = np.reshape(bit_vec[:ref_symbol_num*limit], [-1, ref_symbol_num])
        if bit_order_inv:
            bit_vec_ref = np.fliplr(bit_vec_ref)
        if inv_msb:
            bit_vec_ref[:, -1] = 1 - bit_vec_ref[:, -1]
        if inv_lsb:
            bit_vec_ref[:, 0]  = 1 - bit_vec_ref[:, 0]
        if pn_inv:
            bit_vec_ref = 1 - bit_vec_ref
        sym_vec_ref = bit_vec_ref.dot(2 ** np.arange(np.log2(num_of_symbols)))
    assert np.all(sym_vec_ref == sym_dut), assert_str

def symbol2bin_test():
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    test_pattern_len = random.randint(2, 100)
    bit_vec      = np.random.randint(0, 2, test_pattern_len)
    num_of_symbols   = 2 ** random.randint(1, int(np.floor(np.log2(test_pattern_len))))
    bit_order_inv = random.random() > 0.5
    inv_msb = random.random() > 0.5
    inv_lsb = random.random() > 0.5
    pn_inv  = random.random() > 0.5
    assert_str = '|Pattern len {0:^5d}| symbol number {1:3d} ; bit order inv {2:^6s} ; msb_inv = {3:^6s} ; lsb_inv = {4:^6s} ; pn_inv = {5:^6s} Falied!!!!'.format(
        test_pattern_len,
        num_of_symbols,
        'True' if bit_order_inv else 'False',
        'True' if inv_msb else 'False',
        'True' if inv_lsb else 'False',
        'True' if pn_inv else 'False'
    )
    # ==================================================================================================================
    # Calling DUT
    # ==================================================================================================================
    sym_dut     = cdsp.tx.bin2symbol(bit_vec, num_of_symbols, bit_order_inv, inv_msb, inv_lsb, pn_inv)
    bit_vec_dut = cdsp.rx.symbol2bin(sym_dut, num_of_symbols, bit_order_inv, inv_msb, inv_lsb, pn_inv)

    if num_of_symbols > 2:
        ref_symbol_num = int(np.log2(num_of_symbols))
        limit = int(np.floor(test_pattern_len / ref_symbol_num))
        bit_vec_ref    = bit_vec[:limit*ref_symbol_num]
    else:
        bit_vec_ref = bit_vec

    assert np.all(bit_vec_ref == bit_vec_dut), assert_str
