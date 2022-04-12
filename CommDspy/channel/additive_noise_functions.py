import numpy as np
from scipy.signal import lfilter
from CommDspy.channel import pulse_shape


def awgn(signal, snr, osr=1, span=1, method='rect', beta=0.5):
    """
    :param signal:numpy array of signal which we want to add AWGN to
    :param snr: Signal to Noise power ratio, i.e. what is the power ratio between the signal and the inputted noise.
                Assuming the **snr is given in dB**
    :param osr: the wanted OSR after the shaping
    :param span: the span of the pulse, the span is symmetrical, i.e. a span of 8 means 8 symbols back and 8 symbols
                 forward
    :param method: the shape of the pulse. can be either:
                1. 'rect' - rectangular pulse
                2. 'sinc' - sinc pulse
                3. 'rcos' - raised cosine pulse with roll-off parameter beta
                4. 'rrc' - root raised cosine pulse with rolloff parameter beta
    :param beta: roll-off factor in case the raised cosine or RRC
    :return: signal dipped in AWGN with the wanted SNR
                                               noise
                                                |
                          |---------------|     v
                signal -->|  pulse shape  | --> + ---> output
                          |---------------|
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
    # Pulse shaping
    # ==================================================================================================================
    # ch_out_pulse = pulse_shape(signal_noise, osr=osr, span=span, method=method) if osr > 1 else signal_noise.copy()
    ch_out_pulse = pulse_shape(signal, osr=osr, span=span, method=method, beta=beta) if osr > 1 else signal.copy()
    # ==================================================================================================================
    # Creating the noise and adding it to the signal
    # ==================================================================================================================
    ch_out_pulse = ch_out_pulse + np.random.normal(0, np.sqrt(noise_power), ch_out_pulse.shape)

    return ch_out_pulse

def awgn_channel(signal, b, a, osr=1, span=1, method='rect', zi=None, snr=None):
    """
    :param signal: The input signal you want to pass through the channel
    :param b: Nominator polynomial values (FIR).
    :param a: Denominator polynomial values (IIR) if a[0] is not 0, normalizes all parameters by a[0]
    :param zi: Initial condition for the channel, i.e. the memory of the channel at the beginning of the filtering.
               Should have a length of {max(len(a), len(b)) - 1} if provided. If None, assumes zeros as initial
               conditions
    :param osr: the wanted OSR after the shaping
    :param span: the span of the pulse, the span is symmetrical, i.e. a span of 8 means 8 symbols back and 8 symbols
                 forward
    :param method: the shape of the pulse. can be either:
                1. 'rect' - rectangular pulse
                2. 'sinc' - sinc pulse
                3. 'rcos' - raised cosine pulse with roll-off parameter beta
                4. 'rrc' - root raised cosine pulse with rolloff parameter beta
    :param snr: SNR of the AWGN signal if the SNR is None, does not add noise. Assuming the **snr is given in dB**
    :return: The signal after passing through the channel and added the AWGN. We assume that the input signal is clean.
             Assuming initial conditions for the channel are zero
                                                                     noise
                                                                       |
                            |---------------|    |---------------|     v
                signal ---> |    channel    | -->|  pulse shape  | --> + ---> output
                            |---------------|    |---------------|
    """
    # ==================================================================================================================
    # Passing through the channel
    # ==================================================================================================================
    ch_out = lfilter(b, a, signal, zi=zi)
    # ==================================================================================================================
    # Adding noise if needed
    # ==================================================================================================================
    if snr is not None:
        ch_out += awgn(ch_out, snr)
    # ==================================================================================================================
    # Pulse shaping
    # ==================================================================================================================
    ch_out_pulse = pulse_shape(ch_out, osr=osr, span=span, method=method)

    return ch_out_pulse
