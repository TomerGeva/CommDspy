import numpy as np

def symbol2bin(symbol_mat, num_of_symbols, bit_order_inv=False, inv_msb=False, inv_lsb=False, pn_inv=False):
    """
    :param symbol_mat: Numpy array of UNCODED symbols.
                * If PAM4 assuming that the constellation is [-3x,-x,x,3x]
                * If NRZ assuming that the constellation is [-x,x]
                * If OOK assuming that the constellation is [0, x]
    :param num_of_symbols: number of UNCODED symbols wanted. CURRETLY SUPPORTS 2, 4
    :param bit_order_inv: If True, the bit order is swapped in the creation of the symbol stream:
                          01 <--> 10
                          00 <--> 00
                          11 <--> 11
    :param inv_msb:
    :param inv_lsb:
    :param pn_inv: If True the P-N were inverted in the creation of the symbol stream
    :return: Function performs decoding and then converts the symbols to binary. Note that the function supports OOK,
             NRZ and PAM4.
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    bits_per_symbol = int(np.log2(num_of_symbols))
    symbol_vec      = np.reshape(symbol_mat, -1)
    # ==================================================================================================================
    # Converting to binary representation
    # ==================================================================================================================
    symbol_mat_binary = np.squeeze(((symbol_vec[:, None] & (1 << np.arange(0, bits_per_symbol, 1))) > 0).astype(int))
    # ==================================================================================================================
    # Bit manipulations according to the flags
    # ==================================================================================================================
    if pn_inv:
        symbol_mat_binary = 1 - symbol_mat_binary
    if bits_per_symbol > 1:
        if inv_msb:
            symbol_mat_binary[:, -1] = 1 - symbol_mat_binary[:, -1]
        if inv_lsb:
            symbol_mat_binary[:, 0]  = 1 - symbol_mat_binary[:, 0]
        if bit_order_inv:
            symbol_mat_binary = np.fliplr(symbol_mat_binary)
    return np.reshape(symbol_mat_binary, [-1]).astype(int)