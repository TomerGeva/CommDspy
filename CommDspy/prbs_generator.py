import numpy as np


def prbs_generator(poly_coeff, init_seed, output_length, bits_per_symbol=1, bit_order_inv=True, pn_inv=False,):
    """
    :param poly_coeff: Row vector of generating polynomial coefficients excluding the leading 1. Note that size of
                       ploy_coeff should be equal to the degree of the generating polynomial g(x)
                       example - for g(x) = 1+x+x^2+x^12+x^13,
                       use poly_coeff = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    :param init_seed: row vector of the same size as poly_coeff. Set to the initial state of the prbs delay line.
    :param output_length: returned sequence length
    :param bits_per_symbol: Integer indicating how many bits should be generated per symbol. By default this input is
                            equal to 1, meaning that a binary PRBS will be generated. If this input is higher than 1,
                            adjacent bits will be coupled together to form a symbol. Example: bits_per_symbol = 2:
                                00 --> 0
                                01 --> 1
                                10 --> 2
                                11 --> 3
                    The following parameters are only relevant for bits_per_symbol > 1 values
    :param bit_order_inv: Boolean indicating if the bit order in the signal generation is flipped.
    :param pn_inv: Boolean indicating if we want to invert the PRBS: 1 <--> 0
    :return: Function computes and returns a list of symbols for the PRBS wanted, as well as the seed
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    poly_size   = len(poly_coeff)
    prbs_length = 2**poly_size - 1
    seed        = init_seed
    loop_length = min((output_length*bits_per_symbol, prbs_length))
    rep_number  = int(np.ceil(output_length*bits_per_symbol/prbs_length))
    single_bin  = np.array([0] * loop_length)
    # ==================================================================================================================
    # Generating single interval of binary PRBS
    # ==================================================================================================================
    for ii in range(loop_length):
        single_bin[ii] = np.mod(poly_coeff.dot(seed), 2)
        seed = np.concatenate(([single_bin[ii]], seed[:-1]))
    # ==================================================================================================================
    # Taking into consideration the bits_per_symbol
    # ==================================================================================================================
    multiple_bin = np.tile(single_bin, rep_number) if rep_number > 1 else single_bin
    if bits_per_symbol > 1:
        multiple_bin = np.reshape(multiple_bin[:output_length*bits_per_symbol], [-1, bits_per_symbol])
        # ----------------------------------------------------------------------------------------------------------
        # Modifying the pattern according to the flags, and converting binary to UN-CODED symbols
        # ----------------------------------------------------------------------------------------------------------
        if bit_order_inv:
            multiple_bin = np.fliplr(multiple_bin)
        if pn_inv:
            multiple_bin = 1 - multiple_bin
        pattern = multiple_bin.dot(2 ** np.arange(bits_per_symbol))
    else:
        pattern = single_bin
    # ==================================================================================================================
    # Organizing
    # ==================================================================================================================
    sequence = pattern[:output_length]
    seed = np.flip(sequence[-1*poly_size:])
    return sequence, seed
