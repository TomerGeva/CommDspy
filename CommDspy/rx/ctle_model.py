import numpy as np
from scipy.signal import bilinear, lfilter
from numpy.polynomial import Polynomial

def get_ctle_filter(zeros, poles, dc_gain, fs, osr):
    """
    :param zeros: list of frequencies where there are zeros in [Hz]. If the given zeros are positive, multiplies by -1
                  to enforce stability
    :param poles: list of frequencies where there are poles in [Hz]. If the given poles are positive, multiplies by -1
                  to enforce stability
    :param dc_gain: self explanatory
    :param fs: Symbol frequency, 1/Ts
    :param osr: Over Sampling Rate the signal in working with
    :return: Function uses Tustin's method to change from a continuous transfer function described via the zeros, poles
    and DC gain to a discrete transfer function, returns the IIR coefficients
    """
    # ==================================================================================================================
    # Creating continuous time polynomials
    # ==================================================================================================================
    numerator_con   = Polynomial.fromroots([-1 * np.sign(zero) * zero for zero in zeros])
    denominator_con = Polynomial.fromroots([-1 * np.sign(pole) * pole for pole in poles])
    # ==================================================================================================================
    # Extracting coefficients and normalizing - coefficients are organized from x**0 to x**n, we need to flip them
    # ==================================================================================================================
    numerator_con_coeff   = numerator_con.coef / numerator_con.coef[0]
    denominator_con_coeff = denominator_con.coef / denominator_con.coef[0]
    # w, h = signal.freqs(numerator_con_coeff[::-1], denominator_con_coeff[::-1], worN=np.logspace(7, 11, 1000))
    # ==================================================================================================================
    # Converting to the discrete equivelant
    # ==================================================================================================================
    numerator_dis, denomenator_dis = bilinear(10**(dc_gain/20)*numerator_con_coeff[::-1], denominator_con_coeff[::-1], fs=fs*osr)
    return numerator_dis, denomenator_dis

def ctle(signal, zeros, poles, dc_gain, fs, osr, zi=None):
    """
    :param signal: input signal to pass through the CTLE
    :param zeros: list of frequencies where there are zeros in [Hz]. If the given zeros are positive, multiplies by -1
                  to enforce stability
    :param poles: list of frequencies where there are poles in [Hz]. If the given poles are positive, multiplies by -1
                  to enforce stability
    :param dc_gain: self explanatory
    :param fs: Symbol frequency, 1/Ts
    :param osr: Over Sampling Rate the input signal `sig`
    :param zi: Initial condition for the CTLE, Default is None, where we start with zeros
    :return: Function passes the input signal through the CTLE defined via the zeros, poles and DC gain
    """
    # ==================================================================================================================
    # Getting CTLE parameters
    # ==================================================================================================================
    b, a = get_ctle_filter(zeros, poles, dc_gain, fs, osr)
    # ==================================================================================================================
    # Passing signal through CTLE
    # ==================================================================================================================
    return lfilter(b, a, signal, zi=zi)