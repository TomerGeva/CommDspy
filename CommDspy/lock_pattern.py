import numpy as np
from scipy.signal import correlate

def lock_pattern_to_signal_binary(pattern, signal):
    """
    :param pattern: Binary numpy array
    :param signal: Binary numpy array
    :return: Function performs correlation and returns the pattern at the point where it is aligned with the data
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    pattern_len = len(pattern)
    data_len    = len(signal)
    pattern_rep = np.tile(pattern, 2)
    pattern_nrz = pattern_rep * 2 - 1
    signal_nrz = signal[:np.min((pattern_len, data_len))] * 2 - 1
    # ==================================================================================================================
    # Correlating the NRZ sequences to find the alignment
    # ==================================================================================================================
    xcorr = correlate(pattern_nrz, signal_nrz, mode='valid')
    start_idx = np.argmax(xcorr)
    return pattern_rep[start_idx:start_idx + pattern_len], xcorr

def lock_pattern_to_signal(pattern, signal):
    """
    :param pattern: Numpy array of coded symbols
    :param signal: Numpy array of coded symbols
    :return: Function performs correlation and returns the pattern at the point where it is aligned with the signal
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    pattern_len = len(pattern)
    data_len    = len(signal)
    pattern_rep = np.tile(pattern, 2)
    signal_part = signal[:np.min((pattern_len, data_len))]
    # ==================================================================================================================
    # Correlating the NRZ sequences to find the alignment
    # ==================================================================================================================
    xcorr = correlate(pattern_rep, signal_part, mode='valid')
    # start_idx = np.argmax(xcorr)
    start_idx = np.argmax(np.abs(xcorr))
    return pattern_rep[start_idx:start_idx + pattern_len], xcorr