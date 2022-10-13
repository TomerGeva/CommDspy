import numpy as np
from scipy.signal import lfilter
from CommDspy.channel import pulse_shape


def awgn(signal, snr):
    """
    :param signal:numpy array of signal which we want to add AWGN to
    :param snr: Signal to Noise power ratio, i.e. what is the power ratio between the signal and the inputted noise.
                Assuming the **snr is given in dB**
    :return: signal dipped in AWGN with the wanted SNR

                              noise
                               |
                               v
                signal ------> + ---> output

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
    ch_out = signal + np.random.normal(0, np.sqrt(noise_power), signal.shape)

    return ch_out

def awgn_channel(signal, b, a, snr=None, zi=None):
    """
    :param signal: The input signal you want to pass through the channel
    :param b: Nominator polynomial values (FIR). Assuming that the taps are set to the inputted osr
    :param a: Denominator polynomial values (IIR).
                1. If a[0] is not 1, normalizes all parameters by a[0]
                2. Assuming that the taps are set to the inputted osr
    :param snr: Signal to Noise power ratio, i.e. what is the power ratio between the signal and the inputted noise.
                Assuming the **snr is given in dB**
    :param zi: Initial condition for the channel, i.e. the memory of the channel at the beginning of the filtering.
               Should have a length of {max(len(a), len(b)) - 1} if provided. If None, assumes zeros as initial
               conditions
    :return: The signal after passing through the channel and added the AWGN. We assume that the input signal is clean.
             Assuming initial conditions for the channel are zero

                                                 noise
                                                   |
                             |---------------|     v
                signal ----->|    channel    | --> + ---> output
                             |---------------|

            In addition, returns the memory of the channel at the end of the signal passed
    """
    # ==================================================================================================================
    # Passing through the signal
    # ==================================================================================================================
    if zi is None:
        if type(a) in [list, np.ndarray]:
            zi = np.zeros(max([len(a), len(b)])-1)
        else:  # a is int or float
            zi = np.zeros(max([1, len(b)])-1)
    ch_out, zo = lfilter(b, a, signal, zi=zi)
    # ==================================================================================================================
    # Adding noise if needed
    # ==================================================================================================================
    if snr is not None:
        ch_out = awgn(ch_out, snr)

    return ch_out, zo
