import numpy as np
from scipy.signal import lfilter

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

def awgn_channel(signal, b, a, snr=None):
    """
    :param signal: the input signal you want to pass through the channel
    :param b: nominator polynomial values (FIR).
    :param a: denominator polynomial values (IIR) if a[0] is not 0, normelizes all parameters by a[0]
    :param snr: SNR of the AWGN signal if the SNR is None, does not add noise
    :return: The signal after passing through the channel and added the AWGN. We assume that the input signal is clean.
             Assuming initial conditions for the channel are zero
                                                noise
                                                  |
                            |---------------|     v
                signal ---> |   channel     | --> + ---> output
                            |---------------|
    """
    ch_out = lfilter(b, a, signal)
    if snr is not None:
        return awgn(ch_out)
    else:
        return ch_out

