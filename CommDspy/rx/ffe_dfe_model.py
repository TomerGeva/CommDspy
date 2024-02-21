import numpy as np
from scipy import signal
from CommDspy.auxiliary import buffer
from CommDspy.rx.slicer import slicer


def ffe_dfe(input_signal, ffe_taps=np.array([1]), dfe_taps=None, levels=None, osr=1, phase=0, zi_ffe=None, zi_dfe=None):
    """
    :param input_signal: input signal to pass through the FFE-DFE
    :param ffe_taps: Numpy array containing the FFE taps to be used. If None, without any FFE
    :param dfe_taps: Numpy array containing the DFE taps to be used. If None, without and DFE taps
    :param levels: Levels used in the transmission. if None assuming levels of [-3,-1,1,3]
    :param osr: Over Sampling Rate w.r.t the signal. This is needed only for the DFE buffer calculations
    :param phase: Indicates at which the signal will be sampled for the DFE. Assuming that the first input is at phase 0
                  and there are OSR phases in total
    :param zi_ffe: Initial memory state for the FFE, must have the same size as the FFE filter
    :param zi_dfe: Initial memory state for the DFE, must have the same size as the DFE filter
    :return: Function passes the input signal through the FFE and DFE, returns the "slicer input" as indicated in the
             sketch below, as well as the memory states of the FFE and DFE, if used.

                                                ---------------------------------->
                                                |
                      --------------     +      |  --------------
            in ------>|    FFE     | -----> + ---->|    Slicer  | -------
                      --------------        ^      --------------       |
                                           -|                           |
                                            |      --------------       |
                                            -------|    DFE     | <-----|
                                                   --------------
             Usage notes:
                1. if the input signal has OSR larger than 1, use up-sampling in the FFE only
                2. when using DFE taps note:
                    2.1. The DFE taps do not need to be zero padded when using OSR of more than 1
                    2.2. If the input signal has OSR larger than 1, indicate the sampling phase to be used for the DFE in
                         the phase variable
    :examples:
        1. slicer_in = ffe_dfe(signal_vec, ffe_taps=np.array([0.1, 1, -0.2])
                Function returns the slicer in vector, without returning any memory vectors
        2. slicer_in, zo_ffe = ffe_dfe(signal_vec, ffe_taps=np.array([0.1, 1, -0.2], zi_ffe=np.zeros(3))
                Function returns the slicer in vector as well as the memory state of the FFE at the end of the filterring
        3. slicer_in, zo_ffe, zo_dfe = ffe_dfe(signal_vec, ffe_taps=np.array([0.1, 1, -0.2]), dfe_taps = np.array([-0.25, 0.1]))
                Function returns the slicer in vector, as well as the memory states for both the FFE and DFE
        4. slicer_in, zo_ffe, zo_dfe = ffe_dfe(signal_vec, ffe_taps=np.array([0.1, 1, -0.2]), dfe_taps=np.array([-0.25, 0.1]), zi_ffe=np.zeros(3))
                Function returns the slicer in vector, as well as the memory states for both the FFE and DFE
        5. slicer_in, zo_ffe, zo_dfe = ffe_dfe(signal_vec, ffe_taps=np.array([0.1, 1, -0.2]), dfe_taps=np.array([-0.25, 0.1]), zi_ffe=np.zeros(3), zi_dfe=np.zeros(2))
                Function returns the slicer in vector, as well as the memory states for both the FFE and DFE
        6. slicer_in, zo_dfe = ffe_dfe(signal_vec, dfe_taps=np.array([-0.25, 0.1]), zi_dfe=np.zeros(2))
                Function returns the slicer in vector, as well as the memory states for the DFE
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
    slicer_in  = np.zeros([len(input_signal)])
    slicer_out = np.zeros([len(input_signal)])
    if dfe_taps is not None:
        if zi_dfe is None:
            dfe_memory = np.zeros([len(dfe_taps)])
        elif len(zi_dfe) == len(dfe_taps):
            dfe_memory = zi_dfe
        else:
            raise ValueError(f'zi_dfe is not in the same size as the DFE, length is {len(zi_dfe):d} instead of {dfe_taps:d}')
    if zi_ffe is None:
        zi_ffe = np.zeros(ffe_len - 1)
    elif len(zi_ffe) == ffe_len:
        zi_ffe = zi_ffe[1:]
    elif len(zi_ffe) != len(ffe_len) - 1:
        raise ValueError(f'zi_ffe is not in the same size as the FFE, length is {len(zi_ffe):d} instead of {ffe_len - 1:d}')
    # ==================================================================================================================
    # If we only have FFE and without DFE we can use convolution for speed
    # ==================================================================================================================
    if dfe_taps is None:
        rx_ffe_out = signal.convolve(ffe_taps, np.concatenate([zi_ffe, input_signal]), mode='valid')
        return rx_ffe_out, input_signal[-1*ffe_len:]
    # ==================================================================================================================
    # In case we hae DFE taps we need to make a decision, therefore we can not use convolution
    # ==================================================================================================================
    # if zi_ffe is None:
    #     zi_ffe = np.zeros(ffe_len)
    #     # bufferred_signal = buffer(input_signal, ffe_len, ffe_len-1)
    # elif len(zi_ffe) != ffe_len:
    #     raise ValueError(f'zi_ffe is not in the same size as the FFE, length is {len(zi_ffe):d} instead of {ffe_len:d}')
    bufferred_signal = buffer(np.concatenate([zi_ffe, input_signal]), ffe_len, ffe_len-1)
    for ii, buffed_signal in enumerate(bufferred_signal):
        # ----------------------------------------------------------------------------------------------------------
        # Getting the slicer in value
        # ----------------------------------------------------------------------------------------------------------
        rx_ffe_out_single = buffed_signal.dot(ffe_taps[::-1])
        rx_dfe_out_single = dfe_memory.dot(dfe_taps)
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
                dfe_memory = np.concatenate([np.array([slicer_out[idx]]), dfe_memory[:-1]]) if len(dfe_taps) > 1 else np.array([slicer_out[idx]])
    if len(ffe_taps) == 1:
        return slicer_in, dfe_memory
    else:
        return slicer_in, bufferred_signal[-1], dfe_memory
