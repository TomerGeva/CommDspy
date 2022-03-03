# CommDspy
Repository for the communication signal processing package

Developed by: Tomer Geva

##Information of the package functions:
* slicer - Function receives data matrix from the slicer input and performs the slicing operation. If the user does not insert levels it assumes [-3,-1,1,3]
* symbol2bin - Function receives symbol matrix and converts the matrix to binary stream according to the flags, For further information read function description
* prbs_gen - Function receives polynomial coefficients and an initial seed, creates binary PRBS sequences of the requested length
* prbs_ana - Function receives a slicer out capture matrix (or slicer in matrix after offine slicing) and does the following:
  * builds a reference PRBS sequence
  * synchronizes on the pattern
  * checks BER
  * function returns the "lost lock" indication, nunber of correct bits and the vector with '0' in the correct locations, '1' in the error locations
* prbs_ana_econ - Does the same as prbs_ana but, this function is more memory efficient at the cost of longer runtime - THIS FUNCTION IS STILL SUBJECT TO TESTING
* to be continued

##Enumeration classes
* PrbsEnum - enumeration for the PRBS type used
  * PRBS7
  * PRBS9 
  * PRBS11
  * PRBS13
  * PRBS15
  * PRBS31
* ConstellationEnum - enumeration for the constellations used
  * NRZ - Non-Return to Zero, assuming constellation of [-1, 1]
  * OOK - On Off Keying, assuming constellation of [0, 1]
  * PAM4 - Pulse Amplitude Modulation 4, assuming constellation of [-3, -1, 1, 3]
* CodingEnum - enumeration of the different coding types
  * UNCODED
  * GRAY

## Sub-package: noise
A sub package that holds all the functions wich involve in noise generation
* awgn - a function that adds Additive White Gaussian Noise in a power to create a wanted SNR
## Objects
* PrbsIterator - An iterable used to generate the next bit in the given PRBS. during initialization, a seed and the generating polynomial are given to the object. after calling iter(), next() can be used to pop the next bit in the PRBS


###To update the version:
 1. please run the following command from the respective directory:
        
        python3 setup.py bdist_wheel

 2. please run the following command:

        pip install dist/labsignalprocess-<VERSION>-py3-none-any.whl 
