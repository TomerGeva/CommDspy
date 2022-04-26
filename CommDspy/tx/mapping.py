from CommDspy.auxiliary import get_levels

def mapping(signal, constellation, full_scale=False, levels=None, pn_inv=False):
    """
    :param signal: input signal to be mapped. Should be non-negative integer array
    :param constellation: The constellation we want to map to signal to
    :param full_scale: indicating if we want to use default levels such that the mean power of the signal will be 1 (0 [dB])
    :param levels: Optional, if not None uses the levels given instead of the default levels
    :param pn_inv: indicating if we want to invert the signal after the mapping
    :return: Function returns the signals after mapping to the constellation levels
    """
    # ==================================================================================================================
    # Checking levels validity if given
    # ==================================================================================================================
    default_lvls = get_levels(constellation, full_scale)
    if levels is not None:
        if len(levels) != len(default_lvls):
            raise ValueError('Custom levels vector is invalid, not enough levels for the given constellation')
        default_lvls = levels
    # ==================================================================================================================
    # PN inv
    # ==================================================================================================================
    if pn_inv:
        default_lvls = -1 * default_lvls
    # ==================================================================================================================
    # mapping
    # ==================================================================================================================
    return default_lvls[signal]