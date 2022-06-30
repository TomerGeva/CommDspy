import numpy as np
from scipy import signal
from CommDspy.auxiliary import buffer
from CommDspy.rx.slicer import slicer

def ffe_dfe(input_signal, ffe_taps=np.array([1]), dfe_taps=None, levels=None, osr=1, phase=0):
    """
    :param input_signal: input signal to pass through the FFE-DFE
    :param ffe_taps: Numpy array containing the FFE taps to be used. If None, without any FFE
    :param dfe_taps: Numpy array containing the DFE taps to be used. If None, without and DFE taps
    :param levels: Levels used in the transmission. if None assuming levels of [-3,-1,1,3]
    :param osr: Over Sampling Rate w.r.t the signal. This is needed only for the DFE buffer calculations
    :param phase: Indicates at which the signal will be sampled for the DFE. Assuming that the first input is at phase 0
                  and there are OSR phases in total
    :return: Function passes the input signal through the FFE and DFE, returns the "slicer input" as indicated in the
             sketch below.
             Usage notes:
                1. if the input signal has OSR larger than 1, use zero padding in the FFE only
                2. when using DFE taps note:
                    2.1. The DFE taps do not need to be zero padded when using OSR of more than 1
                    2.2. If the input signal has OSR larger than 1, indicate the sampling phase to be used for the DFE in
                         the phase variable

            The model of the ffe_dfe filter is as follows:

                --------------     +         --------------
        in  --->|    FFE     | -----> + ---->|    Slicer  | ------------->
                --------------        ^      --------------       |
                                     -|                           |
                                      |      --------------       |
                                      -------|    DFE     | <-----|
                                             --------------
    """
    # ==================================================================================================================
    # If both FFE and DFE are None, this block is doing nothing
    # ==================================================================================================================
    if len(ffe_taps) == 1 and dfe_taps is None:
        return input_signal * ffe_taps
    # ==================================================================================================================
    # Local Variables
    # ==================================================================================================================
    ffe_len    = len(ffe_taps)
    dfe_buffer = np.zeros([len(dfe_taps)]) if dfe_taps is not None else 0
    slicer_in  = np.zeros([len(input_signal) - ffe_len + 1])
    slicer_out = np.zeros([len(input_signal) - ffe_len + 1])
    # ==================================================================================================================
    # If we only have FFE and without DFE we can use convolution for speed
    # ==================================================================================================================
    if dfe_taps is None:
        rx_ffe_out = signal.lfilter(ffe_taps, 1, input_signal)[ffe_len:]
        return rx_ffe_out
    # ==================================================================================================================
    # In case we hae DFE taps we need to make a decision, therefore we can not use convolution
    # ==================================================================================================================
    bufferred_signal = buffer(input_signal, ffe_len, ffe_len-1)
    for ii, buffed_signal in enumerate(bufferred_signal):
        # ----------------------------------------------------------------------------------------------------------
        # Getting the slicer in value
        # ----------------------------------------------------------------------------------------------------------
        rx_ffe_out_single = buffed_signal.dot(ffe_taps[::-1])
        rx_dfe_out_single = dfe_buffer.dot(dfe_taps)
        slicer_in[ii]     = rx_ffe_out_single - rx_dfe_out_single
        # ----------------------------------------------------------------------------------------------------------
        # Getting the slicer out value
        # ----------------------------------------------------------------------------------------------------------
        slicer_out[ii]    = slicer(np.array([slicer_in[ii]]), levels=levels)
        # ----------------------------------------------------------------------------------------------------------
        # Updating the dfe buffer
        # ----------------------------------------------------------------------------------------------------------
        if ii % osr == (phase +  osr // 2) % osr:
            idx = ii - osr // 2
            if idx >= 0:
                dfe_buffer = np.concatenate([np.array([slicer_out[idx]]), dfe_buffer[:-1]]) if len(dfe_taps) > 1 else np.array([slicer_out[idx]])

    return slicer_in[osr * len(dfe_taps):]
