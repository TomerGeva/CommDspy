import numpy as np
from CommDspy.auxiliary import upsample
from scipy.signal import lfilter
from scipy import interpolate

def pulse_shape(signal, osr=1, span=1, pulse='rect', beta=0.5, rj_sigma=0.0, zi=None):
    """
    :param signal: Input signal in OSR 1 for the pulse shaping
    :param osr: the wanted OSR after the shaping
    :param span: the span of the pulse, the span is symmetrical, i.e. a span of 8 means 8 symbols back and 8 symbols
                 forward
    :param pulse: the shape of the pulse. can be either:
                1. 'rect' - rectangular pulse
                2. 'sinc' - sinc pulse
                3. 'rcos' - raised cosine pulse with roll-off parameter beta
                4. 'rrc' - root raised cosine pulse with rolloff parameter beta
                5. 'imp' - impulse response, just doing the up-sampling
    :param beta: roll-off parameter
    :param rj_sigma: Random Jitter std value. If 0, no Random Jitter is added to the signal. the unit of the RJ is in UI
                     Example: for Baud rate of 53.125 [GHz] UI is ~18.8[psec]. Using rj_sigma=0.05 [UI] means:
                      rj_sigma = 0.05*18.8e-12 = 0.94e-12 = 940[fsec]
    :param zi: memory for the pulse shaping. If None, assuming reset, i.e. all '0' memory. MUST be with length of:
               'osr' * 'span' * 2
    :return: the signal after the pulse shaping. This function simulated an ideal channel of ch[n] = delta[n] without
    noise. This is not a practical use-case but more of a feature to gain insight. For practical channels use the
    channel sub-package
    """
    # ==================================================================================================================
    # Local parameters
    # ==================================================================================================================
    sig_ups   = upsample(signal, osr)
    pulse_vec = _get_pulse(pulse, osr, span, beta)
    if zi is None:
        zi = np.zeros(osr * span * 2)
    elif len(zi) != osr * span:
        raise ValueError(f'Memory length does not match the requested pulse, length should be {osr*span*2:d}')
    # ==================================================================================================================
    # Convolving
    # ==================================================================================================================
    sig_conv, zo = lfilter(pulse_vec, [1], sig_ups, zi=zi)
    if np.isclose(rj_sigma, 0):
        return sig_conv, zo
    else:
        # ----------------------------------------------------------------------------------------------------------
        # Creating the time vector with and without the jitter
        # ----------------------------------------------------------------------------------------------------------
        rj       = np.random.randn(len(sig_conv)//osr) * rj_sigma
        t        = np.arange(len(sig_conv)) / osr
        t_mat    = t.reshape(-1, osr)
        t_mat_rj = t_mat + rj[:, None]
        t_rj     = t_mat_rj.reshape(-1)
        # ----------------------------------------------------------------------------------------------------------
        # interpolating to the wanted time with the jitter
        # ----------------------------------------------------------------------------------------------------------
        f = interpolate.interp1d(t, sig_conv, kind='cubic', fill_value='extrapolate')
        return f(t_rj), zo

def rect_pulse(osr, span, t=None):
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    if t is None:
        t = _time_vec_pulse(osr, span)
    # ==================================================================================================================
    # Creating the pulse
    # ==================================================================================================================
    pulse = np.zeros_like(t)
    pulse[np.all([0.5 > t, t >= -0.5], axis=0)] = 1
    return pulse

def sinc_pulse(osr, span, t=None):
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    if t is None:
        t = _time_vec_pulse(osr, span)
    # ==================================================================================================================
    # Creating the pulse
    # ==================================================================================================================
    pulse = np.sinc(t)
    return pulse

def rcos_pulse(osr, span, beta, t=None):
    """
    :param osr:
    :param span:
    :param beta: Roll-off factor of the raised cosine
    :param t:
    :return:
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    if t is None:
        t = _time_vec_pulse(osr, span)
    tneq = t[np.abs(2*beta*t) != 1]
    # ==================================================================================================================
    # Creating the pulse
    # ==================================================================================================================
    pulse = np.zeros_like(t)
    pulse[np.abs(2*beta*t) == 1] = np.pi/4 * np.sinc(1/(2*beta))
    pulse[np.abs(2*beta*t) != 1] = np.sinc(tneq) * np.cos(np.pi * beta * tneq) / (1 - (2 * beta * tneq) ** 2)
    return pulse

def rrc_pulse(osr, span, beta, t=None):
    """
    :param osr:
    :param span:
    :param beta: Roll-off factor of the raised cosine
    :param t:
    :return:
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    if t is None:
        t = _time_vec_pulse(osr, span)
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

def _time_vec_pulse(osr, span):
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
    time_len = osr * 2 * span
    t = np.arange(-1 * (time_len // 2), (time_len // 2) + 1, 1) / osr
    return t

def _get_pulse(method, osr=1, span=1, beta=0.5, t=None):
    if method == 'rect':
        return rect_pulse(osr, span, t)
    elif method == 'sinc':
        return sinc_pulse(osr, span, t)
    elif method == 'rcos':
        return rcos_pulse(osr, span, beta, t)
    elif method == 'rrc':
        return rrc_pulse(osr, span, beta, t)
    elif method == 'imp':
        return np.array([1])
    else:
        raise ValueError('Method is not available, please try another')



