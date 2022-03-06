# CommDspy
Repository for the communication signal processing package

Developed by: Tomer Geva

## Information of the package functions:
### prbs_gen
Function receives polynomial coefficients and an initial seed, creates binary PRBS sequences of the requested length . The function is inputted with:
* poly_coeff - a coefficent vector for the generating polynomial of the PRBS pattern
* init_seed - initial seed used to generate the pattern
* output_length - wanted pattern length
### bin2symbol
Function receives a binary sequence and computes the UNCODED symbols matching the binary sequence. The function is inputted wiith:
  * bin_mat - The binary sequence wanted to be converted 
  * num_of_symbols - The number of symbols in the UNCODED pattern. NOW ONLY SUPPORTS POWERS OF 2 (2, 4, 8, ...)
  * bit_order_inv=False - Booleans stating if we want to invert the bit order (By default, MSB is the rightmost bit and the LSB is the leftmost bits)
  * inv_msb=False - Boolean stating if we want to invert the msb
  * inv_lsb=False - Boolean stating if we want to invert the lsb
  * pn_inv=False - Boolean stating if we want to invert all bits 
### symbol2bin
Function receives an UNCODED symbol sequence, returns the binary representation of the symbol sequence
  * symbol_mat - The binary sequence wanted to be converted 
  * num_of_symbols - The number of symbols in the UNCODED pattern. NOW ONLY SUPPORTS 2 and 4
  * bit_order_inv=False - Booleans stating if we want to invert the bit order (By default, MSB is the rightmost bit and the LSB is the leftmost bits)
  * inv_msb=False - Boolean stating if we want to invert the msb
  * inv_lsb=False - Boolean stating if we want to invert the lsb
  * pn_inv=False - Boolean stating if we want to invert all bits
### coding
Function used to code the pattern. Function is inputted with:
  * pattern - Input pattern which should be coded
  * constellation=ConstellationEnum.PAM4 - Wanted constellation
  * coding=CodingEnum.UNCODED - Wanted coding, either UNCODED or GRAY
  * pn_inv=False - Boolean stating if we want to invert the levels after coding
  * full_scale=False - Boolean stating if we want to set the levels such that the mean power will be 1 (0 [dB])
### slicer
Function receives data matrix from the slicer input and performs the slicing operation. If the user does not insert levels it assumes [-3,-1,1,3]
### prbs_ana
Function receives a slicer out capture matrix (or slicer in matrix after offine slicing) and does the following:
  * builds a reference PRBS sequence
  * synchronizes on the pattern
  * checks BER
  * function returns the "lost lock" indication, nunber of correct bits and the vector with '0' in the correct locations, '1' in the error locations
### prbs_ana_econ - THIS FUNCTION IS STILL SUBJECT TO TESTING
  Does the same as prbs_ana but, this function is more memory efficient at the cost of longer runtime


## Enumeration classes
### PrbsEnum 
enumeration for the PRBS type used
  * PRBS7
  * PRBS9 
  * PRBS11
  * PRBS13
  * PRBS15
  * PRBS31
### ConstellationEnum
enumeration for the constellations used
  * NRZ - Non-Return to Zero, assuming constellation of [-1, 1]
  * OOK - On Off Keying, assuming constellation of [0, 1]
  * PAM4 - Pulse Amplitude Modulation 4, assuming constellation of [-3, -1, 1, 3]
### CodingEnum
enumeration of the different coding types
  * UNCODED
  * GRAY

## Sub-package: noise
A sub package that holds all the functions wich involve in noise generation 
### awgn
Function that adds Additive White Gaussian Noise in a power to create a wanted SNR
## Objects
* PrbsIterator - An iterable used to generate the next bit in the given PRBS. during initialization, a seed and the generating polynomial are given to the object. after calling iter(), next() can be used to pop the next bit in the PRBS


# To update the version:
 1. please run the following command from the respective directory:
        
        python3 setup.py bdist_wheel

 2. please run the following command:

        pip install dist/labsignalprocess-<VERSION>-py3-none-any.whl 
