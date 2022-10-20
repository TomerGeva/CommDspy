import numpy as np
from CommDspy.rx.slicer import slicer

def lms_grad(input_vec, levels, ffe_tap_idx=np.array([]), dfe_tap_idx=np.array([]), reference_vec=None):
    """
    :param input_vec: Numpy array of inputs used to compute the MSE and tap gradients
    :param levels: Constellation levels, should be a numpy array of floats
    :param ffe_tap_idx: numpy array containing the indices of the FFE taps for which we want to compute the gradient
    :param dfe_tap_idx: numpy array containing the indices of the DFE taps for which we want to compute the gradient
    :param reference_vec: If None, assuming that there are no slicer errors and performing slicing to recover the
                          reference data. If not None, should have the same length as "input_vec" and correspond to the
                          reference level for each input
    :return: Function computes the MSE as well as the gradient w.r.t. each tap inputted.
        1. If ffe_tap_idx and dfe_tap_idx are empty, only the MSE
        2. If ffe_tap_idx is empty and dfe_tap_idx is not empty, returns the MSE and DFE grad
        3. If ffe_tap_idx is not empty and dfe_tap_idx is empty, returns the MSE and FFE grad
        4. If neither are empty, returns the MSE, FFE grad and DFE grad

    According to the FFE DFE model, the gradient is computed as follows:


                  -----               -----
                  \                   \
    \tilde{x_n} =  > b_i * y_{n-i} -   > a_j * \hat{x_{n-j}}
                  /                   /
                  -----               -----
                    i                   j

                N-1                                 ;
               -----                                ;
           1   \                                    ;
    MSE = ---   > (\tilde{x_n} - \hat{x_n})^2       ; (\tilde{x_n} - \hat{x_n}) === e_n
           N   /                                    ;
               -----                                ;
               n=0                                  ;

                 N-1                                ;                  N-1
                -----                               ;                 -----
    dMSE      1 \                                   ;     dMSE      1 \
    ----- =  --- > e_n * y_{n-i}                    ;     ----- =  --- > e_n * hat{x_{n-j}}
    db_i     2N /                                   ;     da_j     2N /
                -----                               ;                 -----
                n=0                                 ;                 n=0

    """
    if reference_vec is None:
        reference_vec = slicer(input_vec, levels)
    elif len(reference_vec) != len(input_vec):
        raise ValueError(f'Length of the reference vector ({len(reference_vec):d} is different than the input vector ({len(input_vec)}')
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    N        = len(input_vec)
    grad_ffe = np.zeros_like(ffe_tap_idx, dtype=float)
    grad_dfe = np.zeros_like(dfe_tap_idx, dtype=float)
    # ==================================================================================================================
    # Computing the MSE
    # ==================================================================================================================
    e_n = input_vec - reference_vec
    mse = np.mean(e_n**2)
    # ==================================================================================================================
    # Computing the FFE derivatives
    # ==================================================================================================================
    for idx, ii in enumerate(ffe_tap_idx):
        start_idx = max([0, ii])
        stop_idx  = min([N, N+ii])  # stop is exclusive in python
        mask_vec  = np.arange(start_idx, stop_idx)
        grad_ffe[idx] = 0.5 * np.mean(e_n[mask_vec] * input_vec[mask_vec - ii])
    # ==================================================================================================================
    # Computing the FFE derivatives
    # ==================================================================================================================
    for idx, jj in enumerate(dfe_tap_idx):
        mask_vec  = np.arange(jj, N)  # stop is exclusive in python and DFE taps are strictly positive
        grad_dfe[idx] = -0.5 * np.mean(e_n[mask_vec] * reference_vec[mask_vec - jj])
    # ==================================================================================================================
    # Returning
    # ==================================================================================================================
    if ffe_tap_idx.size == 0 and dfe_tap_idx.size == 0:
        return mse
    elif ffe_tap_idx.size == 0:
        return mse, grad_dfe
    elif dfe_tap_idx.size == 0:
        return mse, grad_ffe
    else:
        return mse, grad_ffe, grad_dfe