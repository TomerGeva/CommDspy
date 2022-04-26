import numpy as np
from scipy import linalg
from CommDspy.tx.prbs_generator import prbs_generator
from CommDspy.tx.bin2symbol import bin2symbol
from CommDspy.tx.coding import coding_gray
from CommDspy.tx.mapping import mapping
from CommDspy.auxiliary import get_polynomial, get_levels, power
from CommDspy.rx.lock_pattern import lock_pattern_to_signal
from CommDspy.misc.least_squares import least_squares

def equalization_estimation_prbs(prbs_type, signal, constellation,
                                 ffe_postcursor=23, ffe_precursor=4, dfe_taps=0, normalize=False,
                                 regularization='None', reg_lambda=0,
                                 prbs_full_scale=False,
                                 bit_order_inv=False,
                                 pn_inv_precoding=False,
                                 gray_coded=True,
                                 pn_inv_postmapping=False):
    """
    Function which estimats the MMSE equalizer to be used to invert the ISI in the signal and recover the original data,
     using either an FFE or/and a DFE with controllable number of taps
    :param prbs_type: Type of PRBS used. This variable should be an enumeration from the toolbox. In the case of PRBSxQ
                      patterns, use the bits_per_symbol to generate the pattern
    :param signal: The signal we want to use to estimate the channel
    :param constellation: Enumeration stating the constellation. Should be taken from:
                          CommDspy.constants.ConstellationEnum
    :param ffe_postcursor: Number of postcursors in the FFE estimation
    :param ffe_precursor: Number of precursors in the FFE estimation
    :param dfe_taps: Number of postcursors in the DFE estimation
    :param normalize: Boolean stating if the user wants to normalize the Rx FFE such that the peak will have value of 1
    :param regularization: String indicating the regularization in the computation of the equalizer. Can be either:
            - 'None' - Ordinary Least Squares (OLS) solving without regularization
            - 'ridge' - Applying ridge regression, L2 regularization
            - 'lasso' - Applying lasso regression, L1 regularization
    :param reg_lambda: If regularization is not 'None', and reg_lambda != 0, applies the wanted regularization with a
                       regularization factor of reg_lambda
    :param prbs_full_scale: Boolean stating if we want the levels to be scaled such that the mean power of the levels
                            at the transmitter will be 1 (0 [dB]), i.e. that the PRBS pattern will be coded to power of
                            0 [dB]
    %
    The Following flags are used to construct the PRBS data used as reference:
    :param bit_order_inv: Boolean indicating if the bit order in the signal generation is flipped.
    :param pn_inv_precoding: Boolean indicating if the P and N were flipped in the signal capture process before the
                             coding.
    :param gray_coded: Boolean indicating if the signal is GRAY coded, if False, UNCODED
    :param pn_inv_postmapping: Boolean indicating if the P and N were flipped in the signal capture process after the
                              mapping.
    :return:
        ffe: The equalization FFE, normalized such that the cursor will have a value of 1
        dfe: The equalization DFE
        dig_gain: The digital gain of the system. Note that this is highly dependant on the constellation
        ls_err: Sum of squared residuals
        mse: normalized MSE, meaning the MSE divided by the variance of the constellation, in dB units

    CURRENTLY this function is reliable only for NRZ and PAM4 constellations
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    init_seed       = np.array([1] * prbs_type.value)
    prbs_len        = 2 ** len(init_seed) - 1
    levels          = get_levels(constellation, prbs_full_scale)
    postcursor_num  = np.max((ffe_postcursor, dfe_taps))
    # ==================================================================================================================
    # Getting reference PRBS and coding
    # ==================================================================================================================
    poly_coeff = get_polynomial(prbs_type)
    if poly_coeff[0] == -1:
        return [0], [0], 0, 0, 0
    prbs_seq, _ = prbs_generator(poly_coeff, init_seed, 2 * prbs_len)
    prbsq       = bin2symbol(prbs_seq, len(levels), bit_order_inv, False, False, pn_inv_precoding)
    prbs_coded  = mapping(prbsq, constellation, prbs_full_scale, pn_inv=pn_inv_postmapping) if not gray_coded else mapping(coding_gray(prbsq, constellation), constellation, prbs_full_scale, pn_inv=pn_inv_postmapping)
    # ==================================================================================================================
    # Locking on the pattern beginning and then shifting to account for post-cursors
    # ==================================================================================================================
    prbs_coded_aligned, _ = lock_pattern_to_signal(prbs_coded, signal)
    shift_idx = np.concatenate([np.arange(postcursor_num, prbs_len, 1), np.arange(0, postcursor_num, 1)])
    prbs_coded_shift = prbs_coded_aligned[shift_idx]
    # ==================================================================================================================
    # Averaging to help remove noise from captured signal
    # ==================================================================================================================
    reps        = int(np.floor(len(signal) / prbs_len))
    if reps == 0:  # edge case when the signal is less than a full cycle
        signal_mean = signal
        mat_row_num = len(signal)
        prbs_coded_shift = prbs_coded_shift[:mat_row_num]
    else:
        signal_mean = np.mean(np.reshape(signal[:(prbs_len * reps)], [-1, prbs_len]), axis=0)
        mat_row_num = prbs_len
    # ==================================================================================================================
    # Building the FFE part of the matrix in the Ax=b system
    # ==================================================================================================================
    return equalization_estimation(prbs_coded_shift, signal_mean, ffe_postcursor, ffe_precursor, dfe_taps, normalize, regularization, reg_lambda)

def equalization_estimation(reference_signal, signal, ffe_postcursor=23, ffe_precursor=4, dfe_taps=0, normalize=False, regularization='None', reg_lambda=0):
    """
    :param reference_signal: reference signal used for the equalization
    :param signal: input signal we want to pass through the equalizer
                1. signal and reference_signal MUST have the same length
                2. signal and reference_signal MUST be synchronized
    :param ffe_postcursor: Number of postcursors in the FFE estimation
    :param ffe_precursor: Number of precursors in the FFE estimation
    :param dfe_taps: Number of postcursors in the DFE estimation
    :param normalize: Boolean stating if the user wants to normalize the Rx FFE such that the peak will have value of 1
    :param regularization: String indicating the regularization in the computation of the equalizer. Can be either:
            - 'None' - Ordinary Least Squares (OLS) solving without regularization
            - 'ridge' - Applying ridge regression, L2 regularization
            - 'lasso' - Applying lasso regression, L1 regularization
    :param reg_lambda: If regularization is not 'None', and reg_lambda != 0, applies the wanted regularization with a
                       regularization factor of reg_lambda
    :return:
        ffe: The equalization FFE, normalized such that the cursor will have a value of 1
        dfe: The equalization DFE
        dig_gain: The digital gain of the system. Note that this is highly dependant on the constellation
        ls_err: Sum of squared residuals
        mse: normalized MSE, meaning the MSE divided by the variance of the constellation, in dB units
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    mat_row_num    = len(signal)
    postcursor_num = np.max((ffe_postcursor, dfe_taps))
    # ==================================================================================================================
    # Building the FFE part of the matrix in the Ax=b system
    # ==================================================================================================================
    signal_cat = np.concatenate((signal, signal[:postcursor_num + ffe_precursor + 1]))
    full_hankel = linalg.hankel(signal_cat, signal_cat[::-1][:postcursor_num + ffe_precursor + 1])
    partial_hankel = full_hankel[:mat_row_num, :]
    # ==================================================================================================================
    # Building the DFE part of the matrix in the Ax=b system and creating the matrix A for the least squares Ax=b
    # ==================================================================================================================
    if dfe_taps > 0:
        temp = np.concatenate((reference_signal, reference_signal[:postcursor_num + ffe_precursor + 1]))
        dfe_mat = linalg.hankel(temp, temp[::-1][:postcursor_num])
        dfe_mat = -1 * dfe_mat[:mat_row_num, postcursor_num - dfe_taps:].astype(float)
        mat_ffe_dfe = np.concatenate((partial_hankel, dfe_mat), axis=1)
    else:
        mat_ffe_dfe = partial_hankel
    # ==================================================================================================================
    # Performing least squares to find the optimal equalization
    # ==================================================================================================================
    ls_result = least_squares(mat_ffe_dfe, reference_signal, regularization, reg_lambda) if regularization != 'None' and reg_lambda > 0 else least_squares(mat_ffe_dfe, reference_signal)
    # ==================================================================================================================
    # Extracting results
    # ==================================================================================================================
    full_taps = ls_result[0][::-1]
    ffe = full_taps[dfe_taps:]
    dig_gain = np.max(ffe)
    dfe = ls_result[0][-1 * dfe_taps:][::-1] if dfe_taps > 0 else None
    ls_err = ls_result[1]
    err = mat_ffe_dfe @ ls_result[0] - reference_signal
    mse = 10 * np.log10(np.var(err) / power(reference_signal))
    if normalize:
        return ffe / dig_gain, dfe, dig_gain, ls_err, mse
    else:
        return ffe, dfe, 1, ls_err, mse

