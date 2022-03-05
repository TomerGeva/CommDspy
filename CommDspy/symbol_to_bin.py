import numpy as np
from CommDspy.auxiliary import get_levels
from CommDspy.constants import CodingEnum, ConstellationEnum


def bin2symbol(bin_mat, num_of_symbols, bit_order_inv=False, inv_msb=False, inv_lsb=False, pn_inv=False):
    """
    :param bin_mat: Matrix of binary numbers, ir not 1 1-d vector, flattens it in row major order. THIS should match the
                    fit the wanted symbols completely, otherwise the redundant bits will be removed
    :param num_of_symbols: number of UNCODED symbols wanted. CURRETLY SUPPORTS MULTIPLICATIONS OF 2 (2, 4, 8, etc)
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
    bits_per_symbol = int(np.log2(num_of_symbols))
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
            trimmed_bin[-1, :] = 1 - trimmed_bin[-1, :]
        if inv_lsb:
            trimmed_bin[0, :] = 1 - trimmed_bin[0, :]
        if pn_inv:
            trimmed_bin = 1 - trimmed_bin
        pattern = trimmed_bin.dot(2 ** np.arange(bits_per_symbol))
    else:
        pattern = (1-bin_vec).copy() if pn_inv else bin_vec.copy()
    return pattern.astype(int)

def symbol2bin(symbol_mat, constellation, coding, bit_order_inv=False, inv_msb=False, inv_lsb=False, pn_inv=False):
    """
    :param symbol_mat: Numpy array of coded symbols.
                * If PAM4 assuming that the constellation is [-3x,-x,x,3x]
                * If NRZ assuming that the constellation is [-x,x]
                * If OOK assuming that the constellation is [0, x]
    :param constellation: Enumeration stating the constellation. Should be taken from:
                          CommDspy.constants.ConstellationEnum
    :param coding: Enumeration stating the wanted coding, only effective if constellation has more than 2 constellation
                   points. Should be taken from CommDspy.constants.CodingEnum
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
    bits_per_symbol = int(np.log2(len(get_levels(constellation))))
    # ==================================================================================================================
    # Setting base levels
    # ==================================================================================================================
    if bits_per_symbol == 2:
        if coding == CodingEnum.GRAY:
            levels = np.array([0, 1, 3, 2])
        else:
            levels = np.array([0, 1, 2, 3])
    else:
        bits = 1
        levels = np.array([0, 1])
    # ==================================================================================================================
    # PN-inv
    # ==================================================================================================================
    if pn_inv:
        symbol_mat *= -1
    # ==================================================================================================================
    # Converting symbols to indices, i.e. performing decoding
    # ==================================================================================================================
    if constellation != ConstellationEnum.OOK:
        idx_mat = ((symbol_mat + abs(np.min(symbol_mat))) / (2*np.min(abs(symbol_mat)))).astype(int)
    elif constellation == ConstellationEnum.OOK:
        idx_mat = (symbol_mat  / (np.max(symbol_mat))).astype(int)
    idx_vec = np.reshape(idx_mat, (-1, 1))
    symbol_idx_vec = levels[idx_vec]
    # ==================================================================================================================
    # Converting to binary representation
    # ==================================================================================================================
    symbol_mat_binary = np.squeeze(((symbol_idx_vec[:, None] & (1 << np.arange(0, bits, 1))) > 0).astype(int))
    # ==================================================================================================================
    # Bit manipulations according to the flags
    # ==================================================================================================================
    if constellation not in [ConstellationEnum.OOK, ConstellationEnum.NRZ]:
        if bit_order_inv:
            symbol_mat_binary = np.fliplr(symbol_mat_binary)
        if inv_msb:
            symbol_mat_binary[0, :] = 1 - symbol_mat_binary[0, :]
        if inv_lsb:
            symbol_mat_binary[1, :] = 1 - symbol_mat_binary[-1, :]

    return np.reshape(symbol_mat_binary, [-1])
