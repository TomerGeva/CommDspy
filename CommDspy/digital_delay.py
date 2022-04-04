import numpy as np
from scipy.signal import lfilter

def dig_delay_lagrange_coeffs(n, alpha, forward=True):
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

def dig_delay_sinc_coeffs(n, alpha, forward=True):
    """
    :param n: The order of the interpolation polynomial
    :param alpha: Fractional delay value, should be between 0 and 1
    :param forward: Only used for even values of n.
                If True, assumes the 0-delay coefficient is n//2
                If False, assumes the 0-delay coefficient is n//2 + 1
    :return: The function computes the FIR coefficients based on the normalized "sinc" function. The since function is
    equal to 0 for all integer values except 0, where the limit value is 1. This means that for 0 delay we will get the
    original sequence. For a delay value between 0 and 1 we will use sinc interpolation coefficients to re-create the
    oversampled channel.
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
    # Creating the filter parameters
    # ==================================================================================================================
    return np.sinc(ii_vec + alpha)

def digital_oversample(signal_vec, osr, order=16):
    """
    :param signal_vec: Input signal vector
    :param osr: Over Sampling Rate
    :param order: Lagrange interpolation order, also determines the trim from the end of the vector. The trimming logic
     is as follows:
    The Lagrange polynomials are computed such that the symbol will be at the middle of the polynomial. Therefroe, the
    trimming is:
     - For an even order, we get the current symbol exactly in the middle and the trimming is even at the beginning and
       the end. Example: when the order is 2  we have [pre, curr, post] and we get a trim of 1 symbol at the beginning
       and 1 symbol at the end.
     - For an odd order, we always use the default of `forward=True` in the `dig_delay_fir_coeffs` meaning that there is
       1 extra coefficient at theright side, meaning that the trimming will be order//2 at the beginning and order//2+1
       at the end
    :return: Function uses the digital delay to up-sample the input signal. Note that in the process of up-sampling we
    trim the beginning of the signal. Function returns:
    - The new up-sampled signal after trimming
    - The new time axis after the up-sampling and trimming
    - The old time axis after the trimming
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    alphas           = np.arange(1, osr) / osr
    trim_pre         = (order // 2)
    trim_post        = (order // 2) + (order % 2)
    output_mat       = np.zeros((len(signal_vec) - trim_pre - trim_post, osr))
    try:
        output_mat[:, 0] = signal_vec[trim_pre:-1*trim_post]
    except ValueError: # trim_post = 0
        output_mat[:, 0] = signal_vec[trim_pre:]
    # ==================================================================================================================
    # Creating the digital delay samples
    # ==================================================================================================================
    for ii, alpha in enumerate(alphas):
        fir_coeffs        = dig_delay_lagrange_coeffs(order, alpha)
        output_mat[:, ii+1] = lfilter(fir_coeffs[::-1], 1, signal_vec)[order:]
    # ==================================================================================================================
    # Flattening and adding the last symbol
    # ==================================================================================================================
    temp = np.hstack((np.reshape(output_mat, -1), signal_vec[-1*trim_post]))
    x1 = np.arange(0, len(signal_vec) - trim_pre - trim_post + 1)
    x2 = np.arange(0, len(temp)) / osr
    return temp, x2, x1
