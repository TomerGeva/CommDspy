import numpy as np
from CommDspy import digital_oversample

def eye_diagram(signal, osr_in, osr_diag, fs_value=1, quantization=256, logscale=False):
    """
    :param signal: Input signal to create the eye diagram from
    :param osr_in: Over Sampling Rate of the input signal
    :param osr_diag: Over Sampling Rate of the eye diagram. ONLY SUPPORT COMPLETE MULTIPLICATIONS OF OSR_IN ATM
    :param fs_value: clipping value og the eye diagram, range in [-fs_value, fs_value]
    :param quantization: number of quantization locations for the eye
    :param logscale: if True returns the log of the histogram in the eye diagram
    :return: The function computes the eye diagram and returns a matrix of the input signal.
    Note that if an interpolation is needed, a few of the symbols will be lost at the beginning and at the end of the
    signal in the final eye plot.
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    eye_d = np.zeros([quantization, osr_diag])
    # ==================================================================================================================
    # Interpolating if needed
    # ==================================================================================================================
    if osr_in < osr_diag:
        upsample_rate = osr_diag // osr_in
        sig_upsampled, x2, _ = digital_oversample(signal, osr=upsample_rate, order=osr_in, method='sinc')
    else:
        sig_upsampled = signal
    # ==================================================================================================================
    # Dividing the signal into 1-UI blocks
    # ==================================================================================================================
    sig_mat = np.reshape(sig_upsampled, [-1, osr_diag])
    # ==================================================================================================================
    # Taking a histogram for each of the wanted intervals
    # ==================================================================================================================
    for ii in range(osr_diag):
        eye_d[:, ii], _ = np.histogram(sig_mat[:, ii], bins=quantization, range=[-1*fs_value, fs_value])
    # ==================================================================================================================
    # Generating final parameters
    # ==================================================================================================================
    eye_d = np.hstack((eye_d, eye_d)) if not logscale else np.log10(np.hstack((eye_d, eye_d)) + 1e-1)
    amp   = np.linspace(-1*fs_value, fs_value, quantization)
    return eye_d, amp
