import numpy as np
from scipy import linalg
from CommDspy.prbs_generator import prbs_generator
from CommDspy.symbol_to_bin import bin2symbol
from CommDspy.code_decode import coding
from CommDspy.auxiliary import get_polynomial, get_levels
from CommDspy.lock_pattern import lock_pattern_to_signal


def channel_estimation_prbs(prbs_type, signal, constellation,
                            prbs_full_scale=False,
                            channel_postcursor=500, channel_precursor=19,
                            normalize=False,
                            bit_order_inv=False,
                            pn_inv_precoding=False,
                            gray_coded=True,
                            pn_inv_postcoding=False):
    """
    :param prbs_type: Type of PRBS used. This variable should be an enumeration from the toolbox. In the case of PRBSxQ
                      patterns, use the bits_per_symbol to generate the pattern
    :param signal: The signal we want to use to estimate the channel
    :param constellation: Enumeration stating the constellation. Should be taken from:
                          CommDspy.constants.ConstellationEnum
    :param prbs_full_scale: Boolean stating if we want the levels to be scaled such that the mean power of the levels
                            at the transmitter will be 1 (0 [dB]), i.e. that the PRBS pattern will be coded to power of
                            0 [dB]
    :param channel_postcursor: Number of postcursors in the channel estimation
    :param channel_precursor: Number of precursors in the channel estimation
    :param normalize: Boolean stating if the user wants to normalize the channel estimation such that the peak will have
                      a value of 1
    %
    The Following flags are only relevant for constellation with multiple bits per symbol and are used to manipulate the
    PRBS to match the signal, thus allowing good channel estimation:
    :param bit_order_inv: Boolean indicating if the bit order in the signal generation is flipped.
    :param pn_inv_precoding: Boolean indicating if the P and N were flipped in the signal capture process before the
                             coding.
    :param gray_coded: Boolean indicating if the signal is GRAY coded, if False, UNCODED
    :param pn_inv_postcoding: Boolean indicating if the P and N were flipped in the signal capture process after the
                              coding.
    :return:
        ch_est: A vector with length of 'channel_fir_len' with the FIR of the channel estimation.
        err: The sum of square residuals, i.e. sum of square differences between the estimted channel output and the
             input signal
        CURRENTLY, this function is reliable only for OOK, NRZ and PAM4 constellations
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    levels          = get_levels(constellation, prbs_full_scale)
    init_seed       = np.array([1] * prbs_type.value)
    prbs_len        = 2 ** len(init_seed) - 1
    bits_per_symbol = int(np.log2(len(levels)))
    # ==================================================================================================================
    # Getting reference PRBS and coding
    # ==================================================================================================================
    poly_coeff = get_polynomial(prbs_type)
    if poly_coeff[0] == -1:
        return True, 0, [0]
    prbs_seq, _ = prbs_generator(poly_coeff, init_seed, 2 * prbs_len)
    prbsq       = bin2symbol(prbs_seq, len(levels), bit_order_inv, False, False, pn_inv_precoding)
    prbs_coded  = coding(prbsq, levels, gray_coded, pn_inv_postcoding)
    # ==================================================================================================================
    # Locking on the pattern beginning
    # ==================================================================================================================
    prbs_coded_aligned, xcorr = lock_pattern_to_signal(prbs_coded, signal)
    prbs_coded_aligned = np.concatenate((prbs_coded_aligned, prbs_coded_aligned[:channel_postcursor + channel_precursor + 1]))
    # ==================================================================================================================
    # Averaging to help remove noise from captured signal + shifting to account for post-cursors
    # ==================================================================================================================
    shift_idx    = np.concatenate([np.arange(channel_postcursor, prbs_len, 1), np.arange(0, channel_postcursor, 1)])
    reps         = int(np.floor(len(signal) / prbs_len))
    signal_shift = np.mean(np.reshape(signal[:(prbs_len*reps)], [-1, prbs_len]), axis=0)
    signal_shift = signal_shift[shift_idx] / np.var(signal_shift)
    # ==================================================================================================================
    # Performing least squares to find the channel estimation
    # ==================================================================================================================
    # --------------------------------------------------------------------------------------------------------------
    # Finding The matrix A for the least squares
    # --------------------------------------------------------------------------------------------------------------
    full_hankel    = linalg.hankel(prbs_coded_aligned, prbs_coded_aligned[::-1])
    partial_hankel = full_hankel[:prbs_len, :channel_postcursor + channel_precursor + 1].astype(float)
    ls_result = np.linalg.lstsq(partial_hankel, signal_shift, rcond=-1)
    # A = partial_hankel ; b = signal_shift
    # x = (A^T * A)^{-1} * A^T * b --> least squares solution : ch_est = np.linalg.inv(A.T @ A) @ A.T @ b
    ch_est = ls_result[0][::-1] / np.max(ls_result[0]) if normalize else ls_result[0][::-1]
    err    = ls_result[1]
    return ch_est, err
