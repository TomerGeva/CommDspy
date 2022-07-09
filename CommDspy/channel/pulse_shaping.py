import numpy as np
from CommDspy.auxiliary import upsample
from scipy.signal import convolve

def pulse_shape(signal, osr, span, method='rect', beta=0.5):
    """
    :param signal: Input signal in OSR 1 for the pulse shaping
    :param osr: the wanted OSR after the shaping
    :param span: the span of the pulse, the span is symmetrical, i.e. a span of 8 means 8 symbols back and 8 symbols
                 forward
    :param method: the shape of the pulse. can be either:
                1. 'rect' - rectangular pulse
                2. 'sinc' - sinc pulse
                3. 'rcos' - raised cosine pulse with roll-off parameter beta
                4. 'rrc' - root raised cosine pulse with rolloff parameter beta
                5. 'imp' - impulse response, just doing the up-sampling
    :param beta: roll-off parameter
    :return: the signal after the pulse shaping. This function simulated an ideal channel of ch[n] = delta[n] without
    noise. This is not a practical use-case but more of a feature to gain insight. For practical channels use the
    channel sub-package
    """
    # ==================================================================================================================
    # Local parameters
    # ==================================================================================================================
    sig_ups = upsample(signal, osr)
    pulse   = _get_pulse(method, osr, span, beta)
    # ==================================================================================================================
    # Convolving
    # ==================================================================================================================
    return convolve(sig_ups, pulse, mode='valid')

def rect_pulse(osr, span):
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    t = _time_vec(osr, span)
    # ==================================================================================================================
    # Creating the pulse
    # ==================================================================================================================
    pulse = np.zeros_like(t)
    pulse[np.all([0.5 > t, t >= -0.5], axis=0)] = 1
    return pulse

def sinc_pulse(osr, span):
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    t = _time_vec(osr, span)
    # ==================================================================================================================
    # Creating the pulse
    # ==================================================================================================================
    pulse = np.sinc(t)
    return pulse

def rcos_pulse(osr, span, beta):
    """
    :param osr:
    :param span:
    :param beta: Roll-off factor of the raised cosine
    :return:
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    t    = _time_vec(osr, span)
    tneq = t[np.abs(2*beta*t) != 1]
    # ==================================================================================================================
    # Creating the pulse
    # ==================================================================================================================
    pulse = np.zeros_like(t)
    pulse[np.abs(2*beta*t) == 1] = np.pi/4 * np.sinc(1/(2*beta))
    pulse[np.abs(2*beta*t) != 1] = np.sinc(tneq) * np.cos(np.pi * beta * tneq) / (1 - (2 * beta * tneq) ** 2)
    return pulse

def rrc_pulse(osr, span, beta):
    """
    :param osr:
    :param span:
    :param beta: Roll-off factor of the raised cosine
    :return:
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    t    = _time_vec(osr, span)
    neq0 = ~np.isclose(t, np.zeros_like(t))
    neqb = ~np.isclose(np.abs(t) - 1/(4*beta), np.zeros_like(t))
    # ==================================================================================================================
    # Creating the pulse
    # ==================================================================================================================
    pulse = np.zeros_like(t)
    pulse[~neq0] = 1 + beta * (4/np.pi - 1)
    pulse[~neqb] = beta / np.sqrt(2) * ((1 + 2/np.pi)*np.sin(np.pi/(4*beta)) + (1 - 2/np.pi)*np.cos(np.pi/(4*beta)))
    pulse[neqb & neq0] = 4 * beta / np.pi * (np.cos((1 + beta) * np.pi * t[neqb & neq0]) + (np.sin((1 - beta)*np.pi*t[neqb & neq0]) / (4*beta*t[neqb & neq0]))) / (1 - (4*beta*t[neqb & neq0])**2)
    return pulse

def _time_vec(osr, span):
    """
    :param osr: Over Sampling Rate of the pulse shaped signal
    :param span: number of symbols to span the pulse
    Example: if osr = 16 ; span = 8 ;
        creates a 16*(8*2) vector of the pulse, centered around 0. In this case rectanular between [-1/2, 1/2]
    :return:
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    time_len = osr * 2 * span + 1
    t = np.arange(-1 * time_len // 2, time_len // 2, 1) / osr
    return t

def _get_pulse(method, osr, span, beta):
    if method == 'rect':
        return rect_pulse(osr, span)
    elif method == 'sinc':
        return sinc_pulse(osr, span)
    elif method == 'rcos':
        return rcos_pulse(osr, span, beta)
    elif method == 'rrc':
        return rrc_pulse(osr, span, beta)
    elif method == 'imp':
        return np.array([1])
    else:
        raise ValueError('Method is not available, please try another')



