import numpy as np
from CommDspy.rx.slicer import slicer

def quantize(signal, total_bits, frac_bits=0, fix_point_type='unsigned'):
    """
    :param signal: Input signal to perform quantization on
    :param total_bits: Total bit used in the quantization
    :param frac_bits: Fractional bits used in the quantization. must be complete integers, either positive or negative
    :param fix_point_type: States the type to levels to be used with the bits. can have three options:
    - 'u': unsigned levels
    - 's': signed --> assumes the MSB for the non-fractional bit to be used on for the sign, utilizing the redundancy on the negative end of the numbers (similar to 2=s complement
    - 'ss': signed symmetric --> MSB is for sign, rest of the bits are for magnitude
    If True, assumed the MSB for the non-fractional bit to be used on for the sign, utilizing the
                   redundancy on the negative end of the numbers (similar to 2=s complement)
    :return: Performs quantization for the input signal and returns it. quantization is done according to two methods:
    1. unsigned examples:
        2.0U  - 2 bits for non-fractional and 0 bit for fractions, unsigned. This means that the quantization will be
               done for {0,1,2,3} quantization levels. Input for such a case will be quantization(signal, 2, 0, False)
        2.1U  - 2 bits for non-fractional and 1 bit for fractions, unsigned, This means that the quantization will be
               done for {0,0.5,1,1.5,2,2.5,3,3.5} quantization levels. Input for such a case will be quantization(signal, 3, 1, 'u')
        2.-1U - 2 bits for non-fractional and -1 bit for fractions, unsigned. The '-1' in the fraction place means that
                the 2 bits used will not be the 0 location bit, but the 1 and 2 location bits. This means that the
                quantization will be done for {0,2,4,6}quantization levels. Input for such a case will be quantization(signal, 2, -1, 'u')
    2. signed examples:
        2.0s  - 1 bit for sign, 1 bits for magnitude and 0 bits for fractional. This means that the quantization will be
               done for {-2,-1,0,1} quantization levels. Input for such a case will be quantization(signal, 2, 0, 's')
        3.1s  - 1 bit for sign, 1 bits for magnitude and 1 bits for fractional.This means that the quantization will be
               done for {-2,-1.5,-1,-0.5,0,0.5,1,1.5} quantization levels. Input for such a case will be quantization(signal, 3, 1, 's')
        2.-1s - 1 bit for sign, 1 bits for magnitude and -1 bits for fractional.This means that the quantization will be
               done for {-4, -2, 0, 2} quantization levels. Input for such a case will be quantization(signal, 2, -1, 's')
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    levels = np.arange(0, 2 **total_bits)
    # ==================================================================================================================
    # Applying modifications to the levels if needed
    # ==================================================================================================================
    if fix_point_type == 's':
        levels -= 2**(total_bits-1)
    elif fix_point_type == 'ss':
        levels = levels[1:] - 2**(total_bits - 1)
    elif fix_point_type != 'u':
        raise ValueError('Type used is not defined, please read function description')
    if frac_bits > 0:
        levels = levels.astype(float)
        levels /= 2**frac_bits
    # ==================================================================================================================
    # Applying Quantization
    # ==================================================================================================================
    return slicer(signal, levels)