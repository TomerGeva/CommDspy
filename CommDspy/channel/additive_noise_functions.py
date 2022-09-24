import numpy as np
from scipy.signal import lfilter
from CommDspy.channel import pulse_shape


def awgn(signal, snr, pulse=None, osr=1, span=1, beta=0.5, rj_sigma=0.0):
    """
    :param signal:numpy array of signal which we want to add AWGN to
    :param snr: Signal to Noise power ratio, i.e. what is the power ratio between the signal and the inputted noise.
                Assuming the **snr is given in dB**
    :param pulse: the shape of the pulse. can be either:
                1. 'rect' - rectangular pulse
                2. 'sinc' - sinc pulse
                3. 'rcos' - raised cosine pulse with roll-off parameter beta
                4. 'rrc' - root raised cosine pulse with rolloff parameter beta
                5. 'imp' - impulse response, simply doing the up-sampling
                6. None - not applying any pulse shaping
    :param osr: the wanted OSR after the shaping
    :param span: the span of the pulse, the span is symmetrical, i.e. for span=8, 8 symbols back and 8 symbols forward
    :param beta: roll-off factor in case the raised cosine or RRC
    :param rj_sigma: In case we want to generate a pulse, the pulse can be added with a random jitter. This parameter
                     holds the value of the standard deviation of the random jitter applied
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
    if pulse is not None:
        ch_out_pulse = pulse_shape(signal, osr=osr, span=span, method=pulse, beta=beta, rj_sigma=rj_sigma) if osr > 1 else signal.copy()
    else:
        ch_out_pulse = signal
    # ==================================================================================================================
    # Creating the noise and adding it to the signal
    # ==================================================================================================================
    ch_out_pulse = ch_out_pulse + np.random.normal(0, np.sqrt(noise_power), ch_out_pulse.shape)

    return ch_out_pulse

def awgn_channel(signal, b, a, pulse=None, osr=1, span=1, beta=0.5, rj_sigma=0.0, zi=None, snr=None):
    """
    :param signal: The input signal you want to pass through the channel
    :param b: Nominator polynomial values (FIR). Assuming that the taps are set to the inputted osr
    :param a: Denominator polynomial values (IIR).
                1. If a[0] is not 1, normalizes all parameters by a[0]
                2. Assuming that the taps are set to the inputted osr
    :param pulse: the shape of the pulse. can be either:
                1. 'rect' - rectangular pulse
                2. 'sinc' - sinc pulse
                3. 'rcos' - raised cosine pulse with roll-off parameter beta
                4. 'rrc' - root raised cosine pulse with rolloff parameter beta
                5. 'imp' - impulse response, simply doing the up-sampling
                6. None - not applying any pulse shaping
    :param osr: the wanted OSR after the shaping
    :param span: the span of the pulse, the span is symmetrical, i.e. a span of 8 means 8 symbols back and 8 symbols
                 forward
    :param beta: roll-off factor for the raised cosine or RRC pulses
    :param rj_sigma: In case we want to generate a pulse, the pulse can be added with a random jitter. This parameter
                     holds the value of the standard deviation of the random jitter applied
    :param zi: Initial condition for the channel, i.e. the memory of the channel at the beginning of the filtering.
               Should have a length of {max(len(a), len(b)) - 1} if provided. If None, assumes zeros as initial
               conditions
    :param snr: SNR of the AWGN signal if the SNR is None, does not add noise. Assuming the **snr is given in dB**
    :return: The signal after passing through the channel and added the AWGN. We assume that the input signal is clean.
             Assuming initial conditions for the channel are zero
                                                                     noise
                                                                       |
                            |---------------|    |---------------|     v
                signal ---> |  pulse shape  | -->|    channel    | --> + ---> output
                            |---------------|    |---------------|

            In addition, returns the memory of the channel at the end of the signal passed
    """
    # ==================================================================================================================
    # Pulse shaping
    # ==================================================================================================================
    if pulse is not None:
        ch_out_pulse = pulse_shape(signal, osr=osr, span=span, method=pulse, beta=beta, rj_sigma=rj_sigma)
    else:
        ch_out_pulse = signal
    # ==================================================================================================================
    # Passing through the signal
    # ==================================================================================================================
    if zi is None:
        if type(a) in [list, np.ndarray]:
            zi = np.zeros(max([len(a), len(b)])-1)
        else:  # a is int or float
            zi = np.zeros(max([1, len(b)])-1)
    ch_out, zo = lfilter(b, a, ch_out_pulse, zi=zi)
    # ==================================================================================================================
    # Adding noise if needed
    # ==================================================================================================================
    if snr is not None:
        ch_out = awgn(ch_out, snr)

    return ch_out, zo
