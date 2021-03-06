import numpy as np
from CommDspy.auxiliary import get_levels

def demapping(signal, constellation, full_scale=False, pn_inv=False):
    """
    :param signal: Numpy array of constellation symbols.
                * If PAM4 assuming that the constellation is [-3x,-x,x,3x]
                * If PAM3 assuming that the constellation is [-x, 0, x]
                * If NRZ assuming that the constellation is [-x,x]
                * If OOK assuming that the constellation is [0, x]
    :param constellation: The constellation we want to map to signal to
    :param full_scale: Boolean stating if we want the levels to be scaled such that the mean power of the levels will be
                       1 (0 dB). The de-mapping is dependant on this and for a correct de-mapping we should enter a
                       similar value as entered in the generation of the data
    :param pn_inv: indicating if we want to invert the signal prior to de-mapping
    :return: Function returns the signals after mapping to the constellation levels
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    coded_levels    = get_levels(constellation, full_scale=full_scale)
    demapped_levels = np.arange(len(coded_levels))
    # ==================================================================================================================
    # PN-inv
    # ==================================================================================================================
    if pn_inv:
        signal *= -1
    # ==================================================================================================================
    # Converting symbols to indices, i.e. performing decoding
    # ==================================================================================================================
    idx_mat = np.round((signal - np.min(coded_levels)) / np.diff(coded_levels)[0]).astype(int)
    idx_vec = np.reshape(idx_mat, (-1, 1))
    symbol_idx_vec = demapped_levels[idx_vec]
    return np.reshape(symbol_idx_vec, signal.shape)
