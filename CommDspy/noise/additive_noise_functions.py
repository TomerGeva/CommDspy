import numpy as np


def awgn(signal, snr):
    """
    :param signal:numpy array of signal which we want to add AWGN to
    :param snr: Signal to Noise power ratio, i.e. what is the power ratio between the signal and the inputted noise.
                Assuming the snr is given in dB
    :return: signal dipped in AWGN with the wanted SNR
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    sig_power_hat = np.var(signal, ddof=1) + np.mean(signal) ** 2
    # ==================================================================================================================
    # Computing AWGN std to match the SNR
    # ==================================================================================================================
    snr_lin     = 10 ** (snr / 10)
    noise_power = sig_power_hat / snr_lin
    # ==================================================================================================================
    # Creating the noise and adding it to the signal
    # ==================================================================================================================
    return signal + np.random.normal(0, np.sqrt(noise_power), signal.shape)