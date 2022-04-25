import numpy as np


def bin2symbol(bin_mat, num_of_symbols, bit_order_inv=False, inv_msb=False, inv_lsb=False, pn_inv=False):
    """
    :param bin_mat: Matrix of binary numbers, ir not 1 1-d vector, flattens it in row major order. THIS should match the
                    fit the wanted symbols completely, otherwise the redundant bits will be removed
    :param num_of_symbols: number of UNCODED symbols wanted. CURRETLY SUPPORTS POWERS OF 2 (2, 4, 8, etc)
    :param bit_order_inv: If num_of_symbols > 2 and this boolean is True, the bit order is swapped in the creation of
                          the symbol stream:
                          01 <--> 10
                          00 <--> 00
                          11 <--> 11
    :param inv_msb: Only relevant for num_of_symbols > 2
    :param inv_lsb: Only relevant for num_of_symbols > 2
    :param pn_inv: If True the P-N were inverted in the creation of the symbol stream
    :return: Function converts the binary stream to symbol stream. Example, for the case of 4 symbols this function
             returns an array with [0,1,2,3] values
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    bin_vec         = np.reshape(bin_mat, -1)
    bits_per_symbol = int(np.ceil(np.log2(num_of_symbols)))
    trim            = len(bin_vec) % bits_per_symbol
    # ==================================================================================================================
    # Taking into consideration the bits_per_symbol
    # ==================================================================================================================
    if num_of_symbols > 2:
        trimmed_bin = bin_vec[:-1*trim].copy() if trim > 0 else bin_vec.copy()
        trimmed_bin = np.reshape(trimmed_bin, [-1, bits_per_symbol])
        # ----------------------------------------------------------------------------------------------------------
        # Modifying the pattern according to the flags, and converting binary to UN-CODED symbols
        # ----------------------------------------------------------------------------------------------------------
        if bit_order_inv:
            trimmed_bin = np.fliplr(trimmed_bin)
        if inv_msb:
            trimmed_bin[:, -1] = 1 - trimmed_bin[:, -1]
        if inv_lsb:
            trimmed_bin[:, 0] = 1 - trimmed_bin[:, 0]
        if pn_inv:
            trimmed_bin = 1 - trimmed_bin
        pattern = trimmed_bin.dot(2 ** np.arange(bits_per_symbol))
    else:
        pattern = (1-bin_vec).copy() if pn_inv else bin_vec.copy()
    return pattern.astype(int)