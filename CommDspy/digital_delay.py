import numpy as np


def dig_delay_fir_coeffs(n, alpha, forward=True):
    """
    :param n: The order of the interpolation polynomial
    :param alpha: Fractional delay value, should be between 0 and 1
    :param forward: Only used for even values of n.
                If True, assumes the 0-delay coefficient is n//2
                If False, assumes the 0-delay coefficient is n//2 + 1
    :return: Function uses Lagrange interpolation polynomials to produce digital delay filter coefficients.
    The Lagrange interpolation function: given sets of (x_i, y_i)_{i=0}^{n=1} the lagrange interpolation polynomial is
    given by:


                           --- n-1                                 n-1
                           \\                                    -------     (x - x_j)
               poly(x) =   >       y_i * l_i(x)     ;   l_i(x) =   | |     -------------
                          //                                    j=0,j!=i    (x_i - x_j)
                          --- i=0

    for odd values of n, we assume n//2 is the 0-delay coefficient, otherwise we decide according to "forward".
    Assuming the 0-delay value, noted as z_0 to be 0, and the rest of the samples are complete multiplications of some
    cycle T, then:
    z_1 = T, z_2 = 2T, z_-1 = -T, z_-2 = -2T etc. In addition, we assume that the wanted interpolation point lis inside
    the section [0, T) and is represented via alpha*T where alpha in [0, 1). Therefore, the Lagrange polynomials can be
    written as:

                           --- n//2                                n//2
                           \\                                    -------      (a - j)
               poly(a) =   >       y_i * l_i(a)     ;   l_i(a) =   | |     -------------
                          //                                     j=-n//2,    (i - j)
                          --- i=-n//2                            j != i

    We can use the lagrange coefficients l_i(a) as the filter taps for the digital delay
    """
    # ==================================================================================================================
    # Setting vector of indices
    # ==================================================================================================================
    if n % 2 == 0:
        ii_vec = np.arange(-(n // 2), n // 2 + 1, 1)
    elif forward:
        ii_vec = np.arange(-(n // 2), n // 2 + 2, 1)
    else:
        ii_vec = np.arange(-(n // 2) - 1, n // 2 + 1, 1)
    # ==================================================================================================================
    # Creating the nominator component vector and the denominator component matrix
    # ==================================================================================================================
    nom_mat = np.tile(alpha - ii_vec, [len(ii_vec), 1])
    den_mat = ii_vec - ii_vec[:, None]
    # ==================================================================================================================
    # Creating the filter parameters
    # ==================================================================================================================
    indexing_mat = (1 - np.eye(len(ii_vec))).astype(bool)
    nom_vec = np.prod(np.reshape(nom_mat[indexing_mat], [-1, len(ii_vec)-1]), axis=1)
    den_vec = np.prod(np.reshape(den_mat.T[indexing_mat], [-1, len(ii_vec) - 1]), axis=1)

    return nom_vec / den_vec
