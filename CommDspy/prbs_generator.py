import numpy as np


def prbs_generator(poly_coeff, init_seed, output_length):
    """
    :param poly_coeff: Row vector of generating polynomial coefficients excluding the leading 1. Note that size of
                       ploy_coeff should be equal to the degree of the generating polynomial g(x)
                       example - for g(x) = 1+x+x^2+x^12+x^13,
                       use poly_coeff = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    :param init_seed: row vector of the same size as poly_coeff. Set to the initial state of the prbs delay line.
    :param output_length: returned sequence length
    :return: Function computes and returns a list of bits for the PRBS wanted, as well as the seed
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    poly_size   = len(poly_coeff)
    prbs_length = 2**poly_size - 1
    seed        = init_seed
    loop_length = min((output_length, prbs_length))
    rep_number  = int(np.ceil(output_length/prbs_length))
    single_bin  = np.array([0] * loop_length)
    # ==================================================================================================================
    # Generating single interval of binary PRBS
    # ==================================================================================================================
    for ii in range(loop_length):
        single_bin[ii] = np.mod(poly_coeff.dot(seed), 2)
        seed = np.concatenate(([single_bin[ii]], seed[:-1]))
    # ==================================================================================================================
    # Taking into consideration the repetitions
    # ==================================================================================================================
    multiple_bin = np.tile(single_bin, rep_number) if rep_number > 1 else single_bin
    # ==================================================================================================================
    # Organizing
    # ==================================================================================================================
    sequence = multiple_bin[:output_length]
    seed = np.flip(sequence[-1*poly_size:]) if output_length >= len(seed) else seed
    return sequence, seed
