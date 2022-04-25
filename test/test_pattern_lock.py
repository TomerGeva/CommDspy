import numpy as np
import CommDspy as cdsp


def lock_pattern_to_signal_binary_test(pattern, signal, shift_idx):
    """
    :param pattern: Binary numpy array
    :param signal: Binary numpy array
    :param shift_idx: Skew value between the pattern and the signal
    :return: Testing the function lock_pattern_to_signal_binary
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    pattern_len = len(pattern)
    # ==================================================================================================================
    # Running the function
    # ==================================================================================================================
    pattern_aliged_dut, xcorr = cdsp.rx.lock_pattern_to_signal_binary(pattern, signal)
    # ==================================================================================================================
    # Creating reference
    # ==================================================================================================================
    pattern_aligned_ref = np.concatenate((pattern[-1*shift_idx:], pattern[:-1*shift_idx]))
    # ==================================================================================================================
    # Checking
    # ==================================================================================================================
    assert np.all(pattern_aligned_ref == pattern_aliged_dut), 'Patterns are not aligned'
    assert (pattern_len - shift_idx) % pattern_len == np.argmax(xcorr), f'Correlation maximum is not in the right location, shift if {shift_idx}'

def lock_pattern_to_signal_test(pattern, signal, shift_idx):
    """
    :param pattern: Binary numpy array
    :param signal: Binary numpy array
    :param shift_idx: Skew value between the pattern and the signal
    :return: Testing the function lock_pattern_to_signal_binary
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    pattern_len = len(pattern)
    # ==================================================================================================================
    # Running the function
    # ==================================================================================================================
    pattern_aliged_dut, xcorr = cdsp.rx.lock_pattern_to_signal(pattern, signal)
    # ==================================================================================================================
    # Creating reference
    # ==================================================================================================================
    pattern_aligned_ref = np.concatenate((pattern[-1*shift_idx:], pattern[:-1*shift_idx]))
    # ==================================================================================================================
    # Checking
    # ==================================================================================================================
    assert np.all(pattern_aligned_ref == pattern_aliged_dut), 'Patterns are not aligned'
    assert (pattern_len - shift_idx) == np.argmax(xcorr), f'Correlation maximum is not in the right location, shift if {shift_idx}'

